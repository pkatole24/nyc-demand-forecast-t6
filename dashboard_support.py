import json
import math

import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

from bikebalance.data import ARTIFACTS, DATA, TZ, digest, series, write_json
from bikebalance.live import current_forecasts
from bikebalance.models import Specification, fit, predict

FAMILIES = ('seasonal_naive', 'daily_sarima', 'weather_arimax')
# Live views also offer the research SARIMAX; its historical scores come from saved
# research results, so evaluation code keeps using FAMILIES.
LIVE_FAMILIES = FAMILIES + ('sarimax_calendar', 'sarimax_weather_forecast')
LABELS = {'seasonal_naive': 'Last-week baseline', 'daily_sarima': 'Daily pattern (SARIMA)',
          'weather_arimax': 'Weather + calendar (ARIMAX)', 'sarimax_calendar': 'SARIMAX + weekly calendar',
          'sarimax_weather_forecast': 'SARIMAX + calendar + weather forecast'}
PREDICTORS = {
    'seasonal_naive': 'Repeats the same hour from last week, or the latest available week when newer trip records have not been published.',
    'daily_sarima': 'Learns repeating daily patterns from each station\'s past pickups and returns.',
    'weather_arimax': 'Uses past trips, time of day, day of week, and temperature, wind and rain recorded two days before each forecast hour.',
    'sarimax_calendar': 'Learns each station\'s daily pattern from recent days at the same hour, with weekday, weekend and holiday '
                        'adjustments. No weather inputs.',
    'sarimax_weather_forecast': 'The same SARIMAX plus Open-Meteo weather forecasts issued two days ahead (temperature, wind '
                                'and rain). Best live model in the 2026 test; unavailable if the forecasts cannot be retrieved.'}


def model_signature():
    paths = [ARTIFACTS/model/'manifest.json' for model in ('sarimax_calendar', 'sarimax_weather_forecast')]
    return _team_signature()+''.join(':'+(digest(path) if path.exists() else 'pending') for path in paths)


def _team_signature():
    recent = DATA.parent/'live/weather/receipt.json'
    recent_signature = digest(recent) if recent.exists() else 'no-live-weather'
    weather = ARTIFACTS/'latest_arimax_run.json'
    weather_signature = (digest(weather) if weather.exists() else 'weather-pending')+':'+recent_signature
    path = ARTIFACTS/'dashboard_models/manifest.json'
    if not path.exists():
        return 'pending' if not weather.exists() else weather_signature
    selection = ARTIFACTS/'frozen_selection.json'
    return digest(path)+':'+(digest(selection) if selection.exists() else 'missing-selection')+':'+weather_signature


def forecast_family(panel, station, family, now):
    if family == 'seasonal_naive':
        return current_forecasts(panel, station, now, preview=True)
    if family == 'weather_arimax':
        return arimax_forecast(panel, station, now)
    if family in ('sarimax_calendar', 'sarimax_weather_forecast'):
        return sarimax_forecast(family, station, now)
    now = pd.Timestamp(now).tz_convert('UTC')
    folder = ARTIFACTS/'dashboard_models'
    path = folder/'manifest.json'
    if not path.exists():
        raise ValueError('Fitted-model forecasts are pending completion of selection and export.')
    manifest = json.loads(path.read_text())
    if manifest['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
        raise ValueError('Forecast models belong to a different base')
    if manifest['selection_sha256'] != digest(ARTIFACTS/'frozen_selection.json'):
        raise ValueError('Model selection changed; re-export forecast models')
    stop = now+pd.Timedelta(hours=24)
    index = pd.date_range(now.floor('h'), stop.ceil('h'), freq='h', inclusive='left')
    durations = np.array([(min(t+pd.Timedelta(hours=1), stop)-max(t, now)).total_seconds()/3600 for t in index])
    result = pd.DataFrame(index=index)
    cutoff = None
    for target in ['pickups', 'dropoffs']:
        record = manifest['models'].get(f'{station}/{target}/{family}')
        if record is None:
            raise ValueError('This station/model has no completed forecast artifact.')
        cutoff = pd.Timestamp(record['last_observation'])
        if cutoff >= now.floor('h'):
            raise ValueError('Model observation cutoff is after forecast issuance')
        artifact = folder/record['artifact']
        if digest(artifact) != record['sha256']:
            raise ValueError('Forecast model hash mismatch')
        values = predict(load_pickle(artifact), Specification.from_dict(record['specification']), index)
        result[target] = values['prediction']*durations
    result['net_flow'] = result.dropoffs-result.pickups
    result['duration_hours'] = durations
    return result, {'last_trip_observation': str(cutoff.tz_convert(TZ)),
                    'observation_age_days': (now-cutoff).total_seconds()/86400}


def arimax_forecast(panel, station, now):
    from bikebalance.weather import PastWeather
    pointer = json.loads((ARTIFACTS/'latest_arimax_run.json').read_text())
    folder = ARTIFACTS/pointer['path']
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
        raise ValueError('ARIMAX model belongs to a different base')
    now = pd.Timestamp(now).tz_convert('UTC')
    stop = now+pd.Timedelta(hours=24)
    index = pd.date_range(now.floor('h'), stop.ceil('h'), freq='h', inclusive='left')
    duration = np.array([(min(t+pd.Timedelta(hours=1), stop)-max(t, now)).total_seconds()/3600 for t in index])
    engine = PastWeather.live(panel)
    frame = pd.DataFrame(index=index)
    for target in ['pickups', 'dropoffs']:
        record = manifest['models'][f'{station}/{target}']
        path = folder/record['artifact']
        cutoff = pd.Timestamp(record['last_observation'])
        if cutoff >= now.floor('h') or digest(path) != record['sha256']:
            raise ValueError('ARIMAX observation cutoff or model hash is invalid')
        values = engine.predict(load_pickle(path), Specification.from_dict(record['specification']), index, now)
        frame[target] = values['prediction']*duration
    frame['net_flow'] = frame.dropoffs-frame.pickups
    frame['duration_hours'] = duration
    return frame, {'last_trip_observation': str(cutoff),
                   'last_weather_observation': str(engine.weather.index.max()),
                   'observation_age_days': (now-cutoff).total_seconds()/86400}


def sarimax_forecast(model, station, now):
    from bikebalance import sarimax_calendar
    manifest = json.loads((sarimax_calendar.folder(model)/'manifest.json').read_text())
    if manifest['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
        raise ValueError('SARIMAX model belongs to a different base')
    now = pd.Timestamp(now).tz_convert('UTC')
    stop = now+pd.Timedelta(hours=24)
    index = pd.date_range(now.floor('h'), stop.ceil('h'), freq='h', inclusive='left')
    duration = np.array([(min(t+pd.Timedelta(hours=1), stop)-max(t, now)).total_seconds()/3600 for t in index])
    frame = pd.DataFrame(index=index)
    for target in ['pickups', 'dropoffs']:
        try:
            values, _, record = sarimax_calendar.forecast(model, station, target, index)
        except OSError as exc:  # weather service unreachable: show the model as unavailable
            raise ValueError('Weather forecasts could not be retrieved') from exc
        cutoff = pd.Timestamp(record['last_observation'])
        frame[target] = values*duration
    frame['net_flow'] = frame.dropoffs-frame.pickups
    frame['duration_hours'] = duration
    return frame, {'last_trip_observation': str(cutoff), 'observation_age_days': (now-cutoff).total_seconds()/86400}


def export_models(panel):
    frozen_path = ARTIFACTS/'frozen_selection.json'
    selected = json.loads(frozen_path.read_text())
    if selected['status'] != 'complete' or selected['validation_days'] != 365:
        raise ValueError('Full 2025 model selection must finish before dashboard export')
    base_hash = digest(DATA/'modeling_dataset.parquet')
    if selected['base_sha256'] != base_hash:
        raise ValueError('Model selection belongs to a different base')
    folder = ARTIFACTS/'dashboard_models'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'base_sha256': base_hash, 'selection_sha256': digest(frozen_path), 'models': {}}
    for station in sorted(panel.station_id.unique()):
        for target in ['pickups', 'dropoffs']:
            for family in FAMILIES[1:]:
                entry = selected['series'][f'{station}/{target}'].get(family)
                if entry is None:
                    continue
                spec = Specification.from_dict(entry['specification'])
                y = series(panel, station, target)
                result, messages = fit(y, spec)
                if not result.mle_retvals.get('converged'):
                    result, retry = fit(y, spec, maxiter=200)
                    messages.extend(retry)
                if not result.mle_retvals.get('converged'):
                    raise RuntimeError(f'Comparison fit did not converge: {station}/{target}/{family}')
                path = folder/f'{station}_{target}_{family}.pickle'
                temporary = path.with_suffix('.tmp')
                result.save(temporary)
                temporary.replace(path)
                manifest['models'][f'{station}/{target}/{family}'] = {
                    'artifact': path.name, 'sha256': digest(path), 'specification': spec.to_dict(),
                    'last_observation': str(y.index.max()), 'warnings': messages}
                print(f'Exported {station} {target} {family}', flush=True)
    write_json(folder/'manifest.json', manifest)
    return folder


def station_summary(row, forecasts, now):
    valid_age = all(-30 <= (pd.Timestamp(now)-pd.Timestamp(row[field])).total_seconds() <= 300
                    for field in ['reported_at', 'feed_at'])
    bikes, docks = float(row.num_bikes_available), float(row.num_docks_available)
    result = {'status': 'Status unavailable', 'bike_shortfall': None, 'dock_shortfall': None}
    if not bool(row.operational):
        result['status'] = 'Not operational'
        return result
    if (not bool(row.simulation_allowed) or not valid_age
            or not np.isfinite([bikes, docks, row.capacity]).all()
            or row.capacity <= 0 or bikes+docks <= 0):
        result['status'] = 'Stale or invalid report'
        return result
    threshold = min(bikes+docks, max(2, math.ceil(.1*(bikes+docks))))
    result['status'] = 'Few bikes available' if bikes < threshold else (
        'Few empty docks' if docks < threshold else 'Live availability adequate')
    if forecasts is not None:
        values = forecasts[['pickups', 'dropoffs']].to_numpy(float)
        if np.isfinite(values).all() and (values >= 0).all():
            balance = np.r_[0, (forecasts.dropoffs-forecasts.pickups).cumsum().to_numpy()]
            result['bike_shortfall'] = int(np.ceil(max(0, -(bikes+balance).min())))
            result['dock_shortfall'] = int(np.ceil(max(0, (bikes+balance).max()-(bikes+docks))))
    return result


def test_frame(panel, forecasts, station, regime):
    actual = panel[panel.station_id.eq(station) & panel.partition.eq('test')].set_index('timestamp_utc')
    selected = forecasts[forecasts.station_id.eq(station) & forecasts.partition.eq('test') & forecasts.regime.eq(regime)]
    curves = {}
    for family, group in selected.groupby('family'):
        wide = group.pivot(index='timestamp', columns='target', values='prediction')
        wide['net_flow'] = wide.dropoffs-wide.pickups
        curves[family] = wide.reindex(actual.index)
    return actual, curves


def test_metrics(actual, curves, target):
    common = actual[target].notna()
    for frame in curves.values():
        common &= frame[target].notna()
    output = {}
    for family, frame in curves.items():
        error = frame.loc[common, target]-actual.loc[common, target]
        output[family] = {'mae': float(error.abs().mean()) if len(error) else None,
                          'rmse': float(np.sqrt((error**2).mean())) if len(error) else None,
                          'scored_rows': len(error)}
    return output


def aggregate_test_metrics(panel, forecasts, regime):
    actual = panel[panel.partition.eq('test')].set_index(['station_id', 'timestamp_utc'])
    selected = forecasts[forecasts.partition.eq('test') & forecasts.regime.eq(regime)
                         & forecasts.family.isin(FAMILIES)]
    curves = {}
    for family, group in selected.groupby('family'):
        wide = group.pivot(index=['station_id', 'timestamp'], columns='target', values='prediction')
        wide['net_flow'] = wide.dropoffs-wide.pickups
        curves[family] = wide.reindex(actual.index)
    return {target: test_metrics(actual, curves, target) for target in ['net_flow', 'pickups', 'dropoffs']}
