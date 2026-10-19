import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

from .data import ARTIFACTS, DATA, TARGETS, TZ, digest, series, write_json
from .models import Specification, fit, predict, seasonal_naive


def train_deployment(panel):
    frozen = json.loads((ARTIFACTS/'frozen_selection.json').read_text())
    if frozen['status'] != 'complete' or frozen['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
        raise ValueError('A complete selection for the current base is required')
    folder = ARTIFACTS/'deployment'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'running', 'selection_sha256': digest(ARTIFACTS/'frozen_selection.json'),
                'base_sha256': digest(DATA/'modeling_dataset.parquet'), 'models': {}}
    for station in sorted(panel.station_id.unique()):
        for target in TARGETS:
            key = f'{station}/{target}'
            selected = frozen['series'][key]
            family = selected['deployment_family']
            y = series(panel, station, target)
            record = {'family': family, 'last_observation': str(y.index[-1])}
            if family != 'seasonal_naive':
                spec = Specification.from_dict(selected[family]['specification'])
                result, messages = fit(y, spec)
                if not result.mle_retvals.get('converged'):
                    result, retry = fit(y, spec, maxiter=200)
                    messages.extend(retry)
                if not result.mle_retvals.get('converged'):
                    raise RuntimeError(f'Deployment refit failed: {key}')
                path = folder/f'{station}_{target}.pickle'
                temporary = path.with_suffix('.tmp')
                result.save(temporary)
                temporary.replace(path)
                record.update(specification=spec.to_dict(), artifact=path.name,
                              sha256=digest(path), warnings=messages)
            manifest['models'][key] = record
    manifest['status'] = 'complete'
    write_json(folder/'manifest.json', manifest)
    return folder


def current_forecasts(panel, station, now, preview=False):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('Live forecast issuance requires a timezone')
    now = now.tz_convert('UTC')
    stop = now + pd.Timedelta(hours=24)
    index = pd.date_range(now.floor('h'), stop.ceil('h'), freq='h', inclusive='left')
    duration = np.array([(min(t+pd.Timedelta(hours=1), stop)-max(t, now)).total_seconds()/3600
                         for t in index])
    manifest = None
    if not preview:
        manifest = json.loads((ARTIFACTS/'deployment/manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
            raise ValueError('Deployment manifest is incomplete or belongs to a different base')
        if manifest['selection_sha256'] != digest(ARTIFACTS/'frozen_selection.json'):
            raise ValueError('Deployment model selection changed')
    result = pd.DataFrame(index=index)
    metadata = {'issued_at': str(now), 'mode': 'Seasonal naive preview; not model-selected' if preview
                else 'Frozen 2025 validation selection', 'families': {}}
    for target in TARGETS:
        y = series(panel, station, target)
        y = y[y.index < now.floor('h')]
        selected = manifest['models'][f'{station}/{target}'] if manifest else {'family': 'seasonal_naive'}
        family = selected['family']
        if family == 'seasonal_naive':
            values = {'prediction': seasonal_naive(y, index)}
        else:
            path = ARTIFACTS/'deployment'/selected['artifact']
            if digest(path) != selected['sha256']:
                raise ValueError('Deployment artifact hash mismatch')
            if pd.Timestamp(selected['last_observation']) >= now.floor('h'):
                raise ValueError('Deployment model uses observations after this issuance')
            model = load_pickle(path)
            values = predict(model, Specification.from_dict(selected['specification']), index)
        result[target] = values['prediction']*duration
        for name, value in values.items():
            if name.startswith(('lower', 'upper')):
                result[f'{target}_{name}'] = value*duration
        metadata['families'][target] = family
    result['net_flow'] = result.dropoffs-result.pickups
    result['duration_hours'] = duration
    metadata.update(last_trip_observation=str(y.index.max()),
                    observation_age_days=(now-y.index.max()).total_seconds()/86400,
                    actual_lead_hours=((index.max()-y.index.max()).total_seconds()/3600),
                    partial_bin_assumption='Uniform activity within each hourly bin; intervals prorated likewise.')
    return result, metadata
