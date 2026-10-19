"""Refresh October observations only; never rebuild historical data or fit models."""
import json
import sys
from pathlib import Path

import pandas as pd
import requests
import meteostat
from meteostat import Hourly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bikebalance.data import ROOT, DATA, digest, load_panel
from bikebalance.weather import PastWeather
from bikebalance.weather_preparation import integrate_weather, parse_iem_reports, require


def atomic_write(path, writer):
    temporary = path.with_name(path.name+'.tmp')
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

OUT = ROOT/'data/live/weather'


def main():
    panel = load_panel()
    base_hash = digest(DATA/'modeling_dataset.parquet')
    now = pd.Timestamp.now(tz='UTC')
    start = pd.Timestamp('2026-10-01', tz='America/New_York').tz_convert('UTC')
    stop = min(now.floor('h'), pd.Timestamp('2026-11-01', tz='America/New_York').tz_convert('UTC'))
    grid = pd.date_range(start, stop, freq='h', inclusive='left')
    require(len(grid) > 0, 'No completed October hours available')
    require(panel.timestamp_utc.max()+pd.Timedelta(hours=1) == start, 'Historical boundary mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    tag = now.strftime('%Y%m%dT%H%M%SZ')
    Hourly.cache_dir = str(OUT/'meteostat_cache')
    Hourly.max_age = 0
    met = Hourly('72503', start.tz_localize(None).to_pydatetime(),
                 (stop-pd.Timedelta(hours=1)).tz_localize(None).to_pydatetime(),
                 model=False, flags=True).fetch()
    require(not met.empty, 'Meteostat returned no observations')
    met.index = met.index.tz_localize('UTC') if met.index.tz is None else met.index.tz_convert('UTC')
    met = met.sort_index().loc[lambda x: ~x.index.duplicated(keep='last')]
    met_path = OUT/f'meteostat_72503_{tag}.parquet'
    met.to_parquet(met_path)
    end = stop.ceil('D')
    params = {'station':'LGA', 'data':'all', 'year1':2026, 'month1':10, 'day1':1,
              'year2':end.year, 'month2':end.month, 'day2':end.day, 'tz':'Etc/UTC',
              'format':'onlycomma', 'latlon':'yes', 'elev':'yes', 'missing':'M',
              'trace':'T', 'direct':'no', 'report_type':'3'}
    response = requests.get('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                            params=params, timeout=(30,150))
    response.raise_for_status()
    raw_path = OUT/f'iem_routine_LGA_{tag}.body'
    raw_path.write_bytes(response.content)
    rain = parse_iem_reports(pd.read_csv(raw_path, dtype=str), grid)
    require(not (rain.observation_timestamp.dropna() > now).any(), 'Future rainfall report')
    aligned = met.reindex(grid)
    baseline = pd.DataFrame({'timestamp':grid.tz_convert('America/New_York'),
        'temperature':pd.array(aligned.temp, dtype='Float64'),
        'wind_speed':pd.array(aligned.wspd, dtype='Float64'),
        'weather_station_id':'72503', 'temperature_missing':aligned.temp.isna().to_numpy(),
        'wind_speed_missing':aligned.wspd.isna().to_numpy(),
        'precipitation':float('nan'), 'precipitation_missing':True})
    weather = integrate_weather(baseline, rain)
    for column in weather:
        weather[column] = weather[column].astype(panel[column].dtype)
    weather['timestamp_utc'] = weather.timestamp.dt.tz_convert('UTC')
    require(weather.timestamp_utc.is_unique, 'Duplicate incremental UTC hours')
    report = {'retrieved_at_utc':str(now), 'base_sha256':base_hash,
        'start_utc':str(start), 'stop_utc_exclusive':str(stop), 'hours':len(grid),
        'meteostat':{'station':'72503', 'model':False, 'flags':True, 'version':meteostat.__version__,
                     'path':str(met_path.relative_to(ROOT)), 'sha256':digest(met_path)},
        'iem':{'url':response.url, 'status':response.status_code, 'path':str(raw_path.relative_to(ROOT)),
               'sha256':digest(raw_path)},
        'availability_assumption':'source hour + 6h < issuance; rain report < source hour + 6h; actual first publication times unknown',
        'coverage':{}, 'boundary_contiguous':True}
    for name, available in [('temperature',weather.temperature.notna()), ('wind_speed',weather.wind_speed.notna()),
                            ('rain_observed',weather.precipitation_observed), ('rain_numeric',weather.precipitation.notna())]:
        times = weather.loc[available, 'timestamp_utc']
        report['coverage'][name] = {'available':int(available.sum()), 'missing':int((~available).sum()),
            'last_utc':str(times.max()), 'age_hours':(now-times.max()).total_seconds()/3600,
            'missing_utc':weather.loc[~available,'timestamp_utc'].astype(str).tolist()}
    report['trace_hours'] = int(weather.precipitation_trace.sum())
    report['provider_zero_hours'] = int(weather.zero_without_explicit_P_group.sum())
    report['late_rain_reports'] = int((weather.precipitation_report_timestamp_utc >= weather.timestamp_utc+pd.Timedelta(hours=6)).sum())
    engine = PastWeather(panel, incremental=weather)
    targets = pd.date_range(now.floor('h'), (now+pd.Timedelta(hours=24)).ceil('h'), freq='h', inclusive='left')
    features = engine.features(targets, issuance=now)
    report['live_feature_check'] = {'target_hours':len(targets), 'source_start_utc':str(targets.min()-pd.Timedelta(hours=48)),
        'source_end_utc':str(targets.max()-pd.Timedelta(hours=48)),
        'missing_encodings':features.filter(regex='missing').sum().to_dict()}
    require(digest(DATA/'modeling_dataset.parquet') == base_hash, 'Historical base changed')
    atomic_write(OUT/'weather_hourly.parquet', lambda p: weather.to_parquet(p,index=False))
    report['cache_sha256'] = digest(OUT/'weather_hourly.parquet')
    atomic_write(OUT/'receipt.json', lambda p:p.write_text(json.dumps(report,indent=2),encoding='utf-8'))
    print(json.dumps({k:v for k,v in report.items() if k != 'coverage'},indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='missing_utc'} for k,v in report['coverage'].items()},indent=2))


if __name__ == '__main__':
    main()
