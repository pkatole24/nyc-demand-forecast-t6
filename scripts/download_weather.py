from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import meteostat
from meteostat import Hourly, Stations

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/weather'

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    Hourly.cache_dir = str(RAW / 'meteostat_cache')
    Stations.cache_dir = str(RAW / 'meteostat_cache')
    availability = json.loads((ROOT / 'data/raw/citibike/availability.json').read_text())
    available = [x['month'] for x in availability['months'] if x['available']]
    latest = pd.Timestamp(available[-1] + '01') + pd.offsets.MonthBegin(1)
    start = pd.Timestamp('2024-01-01', tz='America/New_York').tz_convert('UTC')
    stop = latest.tz_localize('America/New_York').tz_convert('UTC')
    grid = pd.date_range(start, stop, freq='h', inclusive='left')
    stations = Stations().nearby(40.758, -73.9855).fetch(15)
    stations.to_csv(RAW / 'nearby_stations.csv')
    scores = []
    # Central Park represents Manhattan; LGA/JFK are alternative measured stations.
    for station in ['KNYC0', 'KJRB0', '72503', '74486', '72502']:
        df = Hourly(station, start.tz_localize(None).to_pydatetime(), (stop-pd.Timedelta(hours=1)).tz_localize(None).to_pydatetime(), model=False, flags=True).fetch()
        if not df.empty:
            df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
        df.to_parquet(RAW / f'{station}_hourly_observations.parquet')
        aligned = df.reindex(grid)
        missing = {c: int(aligned[c].isna().sum()) if c in aligned else len(grid) for c in ['temp','prcp','wspd']}
        score = {'station_id': station, 'observed_rows': len(df), 'expected_hours': len(grid), 'missing': missing,
                 'all_three_complete_hours': int(aligned.reindex(columns=['temp','prcp','wspd']).notna().all(axis=1).sum()),
                 'first_observation_utc': df.index.min().isoformat() if len(df) else None,
                 'last_observation_utc': df.index.max().isoformat() if len(df) else None}
        scores.append(score)
        print(json.dumps(score), flush=True)
    selected = max(scores, key=lambda x: (x['all_three_complete_hours'], -sum(x['missing'].values())))
    if selected['observed_rows'] == 0:
        raise RuntimeError('No candidate weather station has observations')
    info = stations.loc[selected['station_id']]
    result = {'package_version': meteostat.__version__, 'model': False, 'timezone_requested': 'UTC',
              'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'selected_station': {'id': selected['station_id'], 'name': str(info['name']),
                  'latitude': float(info['latitude']), 'longitude': float(info['longitude']), 'elevation_m': float(info['elevation'])},
              'start_utc': start.isoformat(), 'stop_utc_exclusive': stop.isoformat(),
              'source': 'https://meteostat.net/', 'candidates': scores, 'selected_station_id': selected['station_id']}
    (RAW / 'weather_manifest.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
