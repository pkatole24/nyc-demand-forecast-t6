from pathlib import Path
import argparse
import csv
import json
import time
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/citibike'
INTERIM = ROOT / 'data/interim'
FIELDS = ['ride_id', 'rideable_type', 'started_at', 'ended_at', 'start_station_name',
          'start_station_id', 'end_station_name', 'end_station_id', 'start_lat',
          'start_lng', 'end_lat', 'end_lng', 'member_casual']

def sql_string(value):
    return "'" + str(value).replace('\\', '/').replace("'", "''") + "'"

def clock_grid(stop):
    index = pd.date_range('2024-01-01', stop, freq='h', tz='America/New_York', inclusive='left')
    return pd.DataFrame({'utc_hour': index.tz_convert('UTC'), 'local_hour': index.tz_localize(None)})

def process(month, history_path, stop, saved):
    receipt = json.loads((RAW / f'{month}_receipt.json').read_text())
    output = INTERIM / month
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'quality.json'
    # Commit global IDs and the completion marker together by repairing the marker
    # from stored JSON after interruption. A changed archive requires a full rebuild.
    if saved:
        if saved[0] != receipt['sha256']:
            raise RuntimeError('Archive changed. Remove data/interim and rerun preparation from the first month.')
        for name in ['hourly.parquet', 'stations.parquet', 'daily.parquet']:
            if not (output / name).exists():
                raise RuntimeError('Missing committed intermediate output; rebuild data/interim.')
        marker.write_text(saved[1])
        print(f'{month}: using completed aggregates', flush=True)
        return
    paths = [RAW / 'extracted' / month / f['name'] for f in receipt['csv_files']]
    schemas = {}
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f))
        schemas[path.name] = header
        if set(header) != set(FIELDS):
            raise RuntimeError(f'Unexpected schema in {path}: {header}')
    work = INTERIM / 'month_work.duckdb'
    for suffix in ['', '.wal']:
        p = Path(str(work) + suffix)
        if p.exists():
            p.unlink()
    db = duckdb.connect(str(work))
    db.execute("SET memory_limit='4GB'; SET threads=4")
    db.execute(f'SET temp_directory={sql_string(INTERIM / "spill")}')
    file_list = '[' + ','.join(sql_string(x) for x in paths) + ']'
    db.execute(f'CREATE TABLE raw AS SELECT * FROM read_csv({file_list}, header=true, all_varchar=true, null_padding=false)')
    n = db.execute('select count(*) from raw').fetchone()[0]
    db.execute('''CREATE TABLE duplicate_keys AS SELECT trim(ride_id) ride_id,count(*) n FROM raw
        WHERE nullif(trim(ride_id),'') IS NOT NULL GROUP BY trim(ride_id) HAVING count(*)>1''')
    duplicate_keys, duplicate_excess = db.execute('select count(*),coalesce(sum(n-1),0) from duplicate_keys').fetchone()
    conflict_keys = 0
    if duplicate_keys:
        conflict_keys = db.execute('''SELECT count(*) FROM (
            SELECT trim(r.ride_id) ride_id FROM raw r JOIN duplicate_keys d ON trim(r.ride_id)=d.ride_id
            GROUP BY trim(r.ride_id) HAVING count(distinct row(rideable_type,started_at,ended_at,start_station_name,start_station_id,
            end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual))>1)''').fetchone()[0]
    quality = {'month': month, 'raw_rows': n, 'csv_schemas': schemas,
               'within_month_duplicate_ride_ids': duplicate_keys,
               'within_month_duplicate_id_excess': duplicate_excess, 'within_month_conflicting_ids': conflict_keys}
    missing = db.execute('''select count(*) filter(where nullif(trim(ride_id),'') is null),
        count(*) filter(where nullif(trim(start_station_id),'') is null),
        count(*) filter(where nullif(trim(end_station_id),'') is null) from raw''').fetchone()
    quality.update(dict(zip(['missing_ride_ids','missing_start_station_ids','missing_end_station_ids'], missing)))
    # Unique nonmissing ride IDs prove full-row uniqueness without a wide DISTINCT.
    exact_unique = db.execute('select count(*) from (select distinct * from raw)').fetchone()[0] if duplicate_keys or missing[0] else n
    quality['exact_duplicate_excess'] = n-exact_unique
    if not duplicate_keys:
        db.execute('''CREATE TABLE canonical AS SELECT *,hash(rideable_type,started_at,ended_at,start_station_name,start_station_id,
            end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual) row_hash
            FROM raw WHERE nullif(trim(ride_id),'') IS NOT NULL''')
    else:
        db.execute('''CREATE TABLE canonical AS
        SELECT * EXCLUDE(rn) FROM (
          SELECT *, hash(rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,
            start_lat,start_lng,end_lat,end_lng,member_casual) row_hash,
            row_number() over(partition by trim(ride_id) order by started_at, ended_at, start_station_id, end_station_id,
              start_station_name, end_station_name, rideable_type, start_lat, start_lng, end_lat, end_lng, member_casual) rn
          FROM raw WHERE nullif(trim(ride_id),'') IS NOT NULL
        ) WHERE rn=1''')
    db.execute('drop table raw')
    # Read previous IDs as a disk-backed table through an attached database.
    db.execute(f'ATTACH {sql_string(INTERIM / "global_ids.duckdb")} AS history (READ_ONLY)')
    cross, cross_conflicts = db.execute('''select count(*), count(*) filter(where c.row_hash<>s.row_hash)
        from canonical c join history.seen s on trim(c.ride_id)=s.ride_id''').fetchone()
    quality.update({'cross_month_duplicate_ids': cross, 'cross_month_conflicting_ids': cross_conflicts})
    db.execute('''CREATE TABLE clean AS SELECT c.*, try_cast(started_at AS TIMESTAMP) started,
        try_cast(ended_at AS TIMESTAMP) ended FROM canonical c
        ANTI JOIN history.seen s ON trim(c.ride_id)=s.ride_id''')
    db.execute('DETACH history')
    db.execute('drop table canonical')
    utc_clock = clock_grid(stop)
    db.register('clock_frame', utc_clock)
    db.execute('CREATE TABLE clock AS SELECT local_hour, count(*) folds FROM clock_frame GROUP BY local_hour')
    db.execute('''CREATE TABLE valid AS SELECT c.*, cs.folds start_folds, ce.folds end_folds,
        coalesce((started is not null and ended is not null and
          (ended>=started or (cs.folds=2 and ce.folds=2 and ended>=started-interval '1 hour'))),false) valid_trip
        FROM clean c LEFT JOIN clock cs ON date_trunc('hour',started)=cs.local_hour
        LEFT JOIN clock ce ON date_trunc('hour',ended)=ce.local_hour''')
    metrics = db.execute(f'''select count(*),
        count(*) filter(where started is null), count(*) filter(where ended is null),
        count(*) filter(where not valid_trip),
        count(*) filter(where ended-started>interval '24 hours'),
        count(*) filter(where started < strptime('{month}01','%Y%m%d') or started >= strptime('{month}01','%Y%m%d')+interval '1 month'),
        min(started), max(started), min(ended), max(ended)
        from valid''').fetchone()
    names = ['unique_new_rides','invalid_started_at','invalid_ended_at','invalid_trip_order_or_parse',
             'duration_over_24h_retained','started_outside_archive_month','min_started','max_started','min_ended','max_ended']
    quality.update({k: str(v) if k.startswith(('min_','max_')) and v is not None else v for k,v in zip(names,metrics)})
    # Local fall-back hours are retained as unresolved aggregates, not assigned a fold.
    db.execute('''CREATE TABLE endpoints AS
        SELECT trim(start_station_id) station_id, trim(start_station_name) station_name,
          try_cast(start_lat AS DOUBLE) lat, try_cast(start_lng AS DOUBLE) lng,
          started event_time, start_folds folds, 1 pickup, 0 dropoff, member_casual
        FROM valid WHERE valid_trip
        UNION ALL
        SELECT trim(end_station_id), trim(end_station_name), try_cast(end_lat AS DOUBLE),
          try_cast(end_lng AS DOUBLE), ended, end_folds, 0, 1, member_casual
        FROM valid WHERE valid_trip''')
    counts = db.execute('''select count(*) filter(where station_id is null or station_id=''),
        count(*) filter(where folds=2),
        count(*) filter(where event_time >= timestamp '2024-01-01' and folds is null),
        count(*) filter(where event_time < timestamp '2024-01-01')
        from endpoints''').fetchone()
    quality.update(dict(zip(['missing_station_endpoints_on_valid_rides','ambiguous_dst_endpoints',
                            'outside_grid_or_nonexistent_endpoints','before_grid_endpoints'],counts)))
    db.execute('''CREATE TABLE hourly AS SELECT station_id, date_trunc('hour',event_time) local_hour,
        folds=2 ambiguous_dst, sum(pickup)::BIGINT pickups, sum(dropoff)::BIGINT dropoffs
        FROM endpoints WHERE nullif(station_id,'') IS NOT NULL AND folds IS NOT NULL
        GROUP BY station_id, local_hour, ambiguous_dst''')
    db.execute(f'''CREATE TABLE stations AS SELECT '{month}' AS "month", station_id, station_name,
        count(*) events, sum(pickup) pickups, sum(dropoff) dropoffs,
        count(distinct cast(event_time AS DATE)) active_days,
        sum(pickup) filter(where member_casual='casual') casual_pickups,
        sum(pickup) filter(where isodow(event_time)>=6) weekend_pickups,
        sum(pickup) filter(where isodow(event_time)<=5 and (hour(event_time) between 7 and 9 or hour(event_time) between 16 and 18)) rush_pickups,
        count(*) filter(where event_time<timestamp '2026-01-01') training_events,
        sum(pickup) filter(where event_time<timestamp '2026-01-01') training_pickups,
        sum(pickup) filter(where event_time<timestamp '2026-01-01' and member_casual='casual') training_casual_pickups,
        sum(pickup) filter(where event_time<timestamp '2026-01-01' and isodow(event_time)>=6) training_weekend_pickups,
        sum(pickup) filter(where event_time<timestamp '2026-01-01' and isodow(event_time)<=5 and (hour(event_time) between 7 and 9 or hour(event_time) between 16 and 18)) training_rush_pickups,
        avg(lat) lat, avg(lng) lng, min(lat) min_lat, max(lat) max_lat, min(lng) min_lng, max(lng) max_lng,
        min(event_time) first_event, max(event_time) last_event
        FROM endpoints WHERE nullif(station_id,'') IS NOT NULL AND folds IS NOT NULL
        GROUP BY station_id,station_name''')
    db.execute('''CREATE TABLE daily AS SELECT cast(started AS DATE) date, count(*) rides
        FROM valid WHERE valid_trip GROUP BY date''')
    for table in ['hourly','stations','daily']:
        db.execute(f'COPY {table} TO {sql_string(output / (table+".parquet"))} (FORMAT PARQUET, COMPRESSION ZSTD)')
    id_file = output / 'ids.parquet'
    db.execute(f'COPY (SELECT trim(ride_id) ride_id,row_hash FROM clean) TO {sql_string(id_file)} (FORMAT PARQUET, COMPRESSION ZSTD)')
    quality['station_hour_rows'] = db.execute('select count(*) from hourly').fetchone()[0]
    db.close()
    global_db = duckdb.connect(str(history_path))
    global_db.execute("SET memory_limit='2GB'; SET threads=4")
    global_db.execute('BEGIN TRANSACTION')
    global_db.execute(f'INSERT INTO seen SELECT * FROM read_parquet({sql_string(id_file)})')
    encoded = json.dumps(quality, indent=2)
    global_db.execute('INSERT INTO completed VALUES (?,?,?)', [month,receipt['sha256'],encoded])
    global_db.execute('COMMIT')
    global_db.execute('CHECKPOINT')
    global_db.close()
    marker.write_text(encoded)
    id_file.unlink()
    work.unlink()
    print(f'{month}: {n:,} raw rows; {duplicate_excess:,} duplicate-ID excess; {quality["station_hour_rows"]:,} station-hours', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-for-downloads', action='store_true')
    args = p.parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW / 'availability.json').read_text())
    months = [x['month'] for x in manifest['months'] if x['available']]
    if len(months) < 24 or any(not x['available'] for x in manifest['months'][:len(months)]):
        raise RuntimeError('Requested training period or contiguous source coverage is incomplete')
    stop = (pd.Timestamp(months[-1]+'01') + pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
    # One connection at a time is required for attaching the ID database read-only.
    path = INTERIM / 'global_ids.duckdb'
    db = duckdb.connect(str(path))
    db.execute("SET memory_limit='2GB'; SET threads=4")
    db.execute('CREATE TABLE IF NOT EXISTS seen (ride_id VARCHAR, row_hash UBIGINT)')
    db.execute('CREATE TABLE IF NOT EXISTS completed (month VARCHAR PRIMARY KEY, archive_sha256 VARCHAR, quality_json VARCHAR)')
    db.close()
    for month in months:
        while not (RAW / f'{month}_receipt.json').exists():
            if not args.wait_for_downloads:
                raise RuntimeError(f'Month {month} is not downloaded/extracted')
            time.sleep(15)
        # Closing before process permits a read-only attachment; commits follow aggregation.
        db = duckdb.connect(str(path))
        saved = db.execute('select archive_sha256,quality_json from completed where month=?',[month]).fetchone()
        db.close()
        # process opens the writer only after detaching the read-only history.
        process(month, path, stop, saved)

if __name__ == '__main__':
    main()
