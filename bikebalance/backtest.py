import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import (ARTIFACTS, DATA, TARGETS, TZ, experiment_id, local_day,
                   origins, publication_cutoff, series, write_json, write_parquet)
from .models import Specification, fit, predict, seasonal_naive, update


def naive_errors(y, year):
    observed = y[y.index.tz_convert(TZ).year == year]
    prediction = seasonal_naive(y[y.index < observed.index.max()], observed.index)
    errors = pd.Series(observed.to_numpy() - prediction, index=observed.index)
    return errors.groupby(errors.index.tz_convert(TZ).hour).quantile([.025, .1, .9, .975]).unstack()


def forecast_series(y, station, target, partition, family, spec=None, schedule='monthly',
                    regime='daily_updated', days=None, engine=None):
    availability = json.loads((DATA/'source_availability.json').read_text())
    dates = origins(partition)
    if days is not None:
        dates = dates[:days]
    baseline_intervals = naive_errors(y, 2024 if partition == 'validation' else 2025)
    high_threshold = y[y.index.tz_convert(TZ).year == 2024].quantile(.9)
    result = None
    last_seen = None
    fitted_cutoff = pd.NaT
    records = []
    logs = []
    for origin in dates:
        clock = time.perf_counter()
        index = local_day(origin)
        cutoff = origin.tz_convert('UTC') if regime == 'daily_updated' else publication_cutoff(origin, availability)
        history = y[y.index < cutoff] if cutoff is not None else y.iloc[:0]
        actual = y.reindex(index).to_numpy()
        status = 'ok'
        values = {name: np.full(len(index), np.nan) for name in
                  ['prediction', 'raw_prediction', 'lower80', 'upper80', 'lower95', 'upper95']}
        warnings = []
        refitted = False
        if len(history) < 168:
            status = 'insufficient_published_history'
        elif family == 'seasonal_naive':
            values['prediction'] = seasonal_naive(history, index)
            values['raw_prediction'] = values['prediction'].copy()
            # Calibrated only for next-day forecasts, not extrapolations across an archive gap.
            if regime == 'daily_updated':
                for level, qs in [(80, (.1, .9)), (95, (.025, .975))]:
                    for bound, q in zip(['lower', 'upper'], qs):
                        adjustment = baseline_intervals[q].reindex(index.tz_convert(TZ).hour).to_numpy()
                        values[f'{bound}{level}'] = np.maximum(0, values['prediction'] + adjustment)
        else:
            if spec is None:
                raise ValueError('A fitted family requires a frozen specification')
            refit_due = result is None or (regime == 'publication_lag' and history.index.max() != last_seen)
            if regime == 'daily_updated':
                refit_due |= origin.dayofweek == 0 if schedule == 'weekly' else origin.day == 1
            if refit_due:
                initial = result.params if result is not None else None
                failure = None
                for attempt in range(2):
                    try:
                        fitter = engine.fit if engine is not None else fit
                        candidate, messages = fitter(history, spec, start_params=initial if attempt == 0 else None)
                        warnings.extend(messages)
                        if not candidate.mle_retvals.get('converged', False):
                            raise RuntimeError('Optimizer did not converge')
                        result = candidate
                        fitted_cutoff = history.index.max()
                        last_seen = history.index.max()
                        refitted = True
                        failure = None
                        break
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        failure = str(exc)
                if failure:
                    warnings.append(f'refit_failed: {failure}')
            if result is not None:
                new = history[history.index > last_seen]
                if not new.empty:
                    result = engine.update(result, new, spec) if engine is not None else update(result, new, spec)
                    last_seen = new.index.max()
                values = engine.predict(result, spec, index, origin) if engine is not None else predict(result, spec, index)
            else:
                status = 'fit_failed'
        frame = pd.DataFrame(values)
        frame['station_id'] = station
        frame['target'] = target
        frame['family'] = family
        frame['partition'] = partition
        frame['regime'] = regime
        frame['origin'] = origin.tz_convert('UTC')
        frame['timestamp'] = index
        frame['horizon'] = np.arange(1, len(index)+1)
        frame['observation_cutoff'] = history.index.max() if len(history) else pd.NaT
        frame['parameter_fit_cutoff'] = fitted_cutoff
        frame['actual'] = actual
        frame['high_demand'] = actual >= high_threshold
        frame['disruption'] = index.tz_convert(TZ).strftime('%Y-%m-%d') == '2026-02-23'
        frame['status'] = status
        records.append(frame)
        logs.append({'origin': str(origin), 'refitted': refitted, 'status': status,
                     'seconds': time.perf_counter()-clock, 'warnings': warnings})
    return pd.concat(records, ignore_index=True), logs


def run_backtest(panel, families, selection=None, partitions=('validation', 'test'),
                 regimes=('daily_updated', 'publication_lag'), days=None):
    config = {'families': families, 'selection': selection, 'partitions': partitions,
              'regimes': regimes, 'days': days}
    key, provenance = experiment_id(config)
    folder = ARTIFACTS/'runs'/key
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'run.json', {'status': 'running', **provenance})
    pieces = []
    for partition in partitions:
        for station in sorted(panel.station_id.unique()):
            for target in TARGETS:
                y = series(panel, station, target)
                for family in families:
                    chosen = None if family == 'seasonal_naive' else selection['series'][f'{station}/{target}'][family]
                    spec = Specification.from_dict(chosen['specification']) if chosen else None
                    for regime in regimes:
                        path = folder/f'{partition}_{station}_{target}_{family}_{regime}.parquet'
                        receipt = path.with_suffix('.json')
                        if path.exists() and receipt.exists():
                            saved = json.loads(receipt.read_text())
                            from .data import digest
                            if saved['sha256'] != digest(path):
                                raise ValueError(f'Cached forecast hash mismatch: {path}')
                            frame = pd.read_parquet(path)
                        else:
                            if chosen and partition == 'validation' and regime == 'daily_updated' and days is None:
                                from .data import digest
                                cached = ARTIFACTS/chosen['validation_artifact']
                                if digest(cached) != chosen['validation_sha256']:
                                    raise ValueError('Selected validation forecasts changed')
                                frame = pd.read_parquet(cached)
                                logs = [{'reused_selected_validation': str(cached)}]
                            else:
                                frame, logs = forecast_series(y, station, target, partition,
                                    family, spec, chosen['schedule'] if chosen else 'monthly', regime, days)
                            write_parquet(path, frame)
                            from .data import digest
                            write_json(receipt, {'sha256': digest(path), 'origins': logs})
                        pieces.append(frame)
                print(f'{partition}: {station} {target} complete', flush=True)
    forecasts = pd.concat(pieces, ignore_index=True)
    write_parquet(folder/'forecasts.parquet', forecasts)
    from .evaluation import summarize, paired_bootstrap
    summarize(forecasts).to_csv(folder/'metrics.csv', index=False)
    paired_bootstrap(forecasts).to_csv(folder/'paired_comparisons.csv', index=False)
    from .data import digest
    write_json(folder/'run.json', {'status': 'complete', 'rows': len(forecasts),
        'forecasts_sha256': digest(folder/'forecasts.parquet'), **provenance})
    write_json(ARTIFACTS/('latest_smoke_run.json' if days is not None else 'latest_run.json'),
               {'path': folder.relative_to(ARTIFACTS).as_posix(), 'id': key})
    return folder
