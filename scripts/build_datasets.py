from pathlib import Path
import calendar
import json
import duckdb
import holidays
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from prepare_trips import ROOT, sql_string, clock_grid

OUT = ROOT / 'data/processed'
INTERIM = ROOT / 'data/interim'
TZ = 'America/New_York'

def frame_table(df):
    return '| ' + ' | '.join(map(str,df.columns)) + ' |\n| ' + ' | '.join(['---']*len(df.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(str(x).replace('|','/') for x in row) + ' |' for row in df.itertuples(index=False,name=None))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    availability = json.loads((ROOT / 'data/raw/citibike/availability.json').read_text())
    months = [x['month'] for x in availability['months'] if x['available']]
    quality = [json.loads((INTERIM / m / 'quality.json').read_text()) for m in months]
    stop = (pd.Timestamp(months[-1]+'01') + pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
    grid = clock_grid(stop)
    db = duckdb.connect()
    db.execute("SET memory_limit='4GB'; SET threads=4")
    station_glob = sql_string(INTERIM / '*/stations.parquet')
    hourly_glob = sql_string(INTERIM / '*/hourly.parquet')
    daily_glob = sql_string(INTERIM / '*/daily.parquet')
    db.execute(f'CREATE VIEW station_aliases AS SELECT * FROM read_parquet({station_glob})')
    db.execute(f'CREATE VIEW hourly AS SELECT * FROM read_parquet({hourly_glob})')
    daily = db.execute(f'SELECT date,sum(rides) rides FROM read_parquet({daily_glob}) GROUP BY date ORDER BY date').df()
    daily.to_csv(OUT / 'system_daily_coverage.csv',index=False)
    all_days = pd.date_range('2024-01-01',pd.Timestamp(stop)-pd.Timedelta(days=1),freq='D')
    missing_days = all_days.difference(pd.DatetimeIndex(daily.date))
    disruption_audit = []
    for day in missing_days:
        month = day.strftime('%Y%m')
        csv_glob = sql_string(ROOT / f'data/raw/citibike/extracted/{month}/*.csv')
        audit = db.execute(f'''SELECT count(*) filter(where cast(try_cast(started_at AS TIMESTAMP) AS DATE)=date '{day.date()}'),
            count(*) filter(where cast(try_cast(ended_at AS TIMESTAMP) AS DATE)=date '{day.date()}'),
            count(*) filter(where cast(try_cast(ended_at AS TIMESTAMP) AS DATE)=date '{day.date()}' and nullif(trim(end_station_id),'') is null)
            FROM read_csv({csv_glob},header=true,all_varchar=true)''').fetchone()
        disruption_audit.append({'date':str(day.date()),'archive_month':month,'raw_pickups_on_date':audit[0],
                                 'raw_dropoffs_on_date':audit[1],'raw_dropoffs_missing_station_id':audit[2]})
    pd.DataFrame(disruption_audit,columns=['date','archive_month','raw_pickups_on_date','raw_dropoffs_on_date','raw_dropoffs_missing_station_id']).to_csv(OUT / 'system_zero_pickup_date_audit.csv',index=False)
    aliases = db.execute('''SELECT station_id, station_name, sum(events) events,
        min(first_event) first_event, max(last_event) last_event, count(distinct month) AS "months"
        FROM station_aliases GROUP BY station_id,station_name ORDER BY station_id,events DESC''').df()
    aliases.to_csv(OUT / 'station_aliases.csv',index=False)
    changes = db.execute('''SELECT lower(trim(station_name)) normalized_name,
        count(distinct station_id) station_ids, string_agg(distinct station_id, ', ' order by station_id) ids,
        min(first_event) first_event,max(last_event) last_event
        FROM station_aliases WHERE station_name IS NOT NULL AND station_name<>''
        GROUP BY normalized_name HAVING count(distinct station_id)>1 ORDER BY station_ids DESC''').df()
    changes.to_csv(OUT / 'station_identifier_changes.csv',index=False)
    station_month = db.execute('''SELECT station_id,strftime(local_hour,'%Y%m') AS "month",
        count(distinct cast(local_hour AS DATE)) active_days,
        sum(pickups) pickups,sum(dropoffs) dropoffs FROM hourly GROUP BY station_id,month''').df()
    station_month['days_in_month'] = station_month['month'].map(lambda m: calendar.monthrange(int(m[:4]),int(m[4:]))[1])
    station_month['active_day_fraction'] = station_month.active_days/station_month.days_in_month
    station_month.to_csv(OUT / 'station_month_coverage.csv',index=False)
    profiles = db.execute('''SELECT station_id,count(distinct month) AS "months",
        sum(training_events) training_events,
        sum(coalesce(training_pickups,0)) training_pickups,
        sum(coalesce(training_casual_pickups,0)) casual_pickups,
        sum(coalesce(training_weekend_pickups,0)) weekend_pickups,
        sum(coalesce(training_rush_pickups,0)) rush_pickups,
        median(lat) lat,median(lng) lng,
        max(lat)-min(lat) monthly_lat_range,max(lng)-min(lng) monthly_lng_range
        FROM station_aliases GROUP BY station_id''').df()
    names = db.execute('''SELECT station_id,station_name FROM (
        SELECT station_id,station_name,sum(training_events) events,
        row_number() over(partition by station_id order by sum(training_events) DESC, station_name) rn
        FROM station_aliases WHERE training_events>0 GROUP BY station_id,station_name
        ) WHERE rn=1''').df()
    profiles = profiles.merge(names,on='station_id',validate='one_to_one')
    training_months = station_month[station_month.month<='202512'].groupby('station_id').month.nunique().rename('training_months')
    profiles = profiles.join(training_months,on='station_id')
    boundary = shape(json.loads((ROOT / 'data/raw/geography/manhattan_borough.json').read_text())[0]['the_geom'])
    profiles['manhattan'] = [boundary.covers(Point(lng,lat)) if np.isfinite(lat) and np.isfinite(lng) else False for lat,lng in zip(profiles.lat,profiles.lng)]
    daily_min = station_month.groupby('station_id').active_day_fraction.min().rename('min_active_day_fraction')
    monthly_min = station_month.groupby('station_id')[['pickups','dropoffs']].min().min(axis=1).rename('min_monthly_direction_count')
    profiles = profiles.join(daily_min,on='station_id').join(monthly_min,on='station_id')
    profiles['casual_share'] = profiles.casual_pickups/profiles.training_pickups
    profiles['rush_share'] = profiles.rush_pickups/profiles.training_pickups
    days = pd.date_range('2024-01-01','2025-12-31',freq='D')
    weekend_days = int((days.dayofweek>=5).sum())
    weekday_days = len(days)-weekend_days
    profiles['weekend_weekday_daily_ratio'] = (profiles.weekend_pickups/weekend_days)/((profiles.training_pickups-profiles.weekend_pickups)/weekday_days)
    # Monthly centers detect substantial relocation; coordinates are not used to merge IDs.
    profiles['monthly_center_span_m'] = np.sqrt((profiles.monthly_lat_range*111000)**2+(profiles.monthly_lng_range*84000)**2)
    profiles['eligible'] = (profiles.manhattan & (profiles.months==len(months)) & (profiles.training_months==24)
                            & (profiles.min_active_day_fraction>=0.85) & (profiles.min_monthly_direction_count>=1000)
                            & (profiles.monthly_center_span_m<=250))
    profiles.sort_values(['training_events','station_id'],ascending=[False,True]).to_csv(OUT / 'station_candidates.csv',index=False)
    pool = profiles[profiles.eligible].sort_values(['training_events','station_id'],ascending=[False,True]).head(60).copy()
    if len(pool)<8:
        raise RuntimeError('Fewer than eight stations meet station consistency requirements; inspect station_candidates.csv')
    pool['recreation_score'] = pool.casual_share * pool.weekend_weekday_daily_ratio
    recreational = pool.sort_values(['recreation_score','training_events','station_id'],ascending=[False,False,True]).head(3).copy()
    recreational['usage_label'] = 'recreational-leaning'
    commuter_pool = pool[~pool.station_id.isin(recreational.station_id)].sort_values(['training_events','station_id'],ascending=[False,True]).head(35)
    commuters = commuter_pool.sort_values(['rush_share','training_events','station_id'],ascending=[False,False,True]).head(5).copy()
    commuters['usage_label'] = 'commuter-leaning'
    selected = pd.concat([commuters,recreational],ignore_index=True)
    selected.to_csv(OUT / 'selected_stations.csv',index=False)
    ids = selected.station_id.tolist()
    id_list = '(' + ','.join(sql_string(x) for x in ids) + ')'
    aggregate = db.execute(f'SELECT station_id,local_hour,ambiguous_dst,sum(pickups)::BIGINT pickups,sum(dropoffs)::BIGINT dropoffs FROM hourly WHERE station_id IN {id_list} GROUP BY station_id,local_hour,ambiguous_dst').df()
    uncertain = aggregate[aggregate.ambiguous_dst].copy()
    uncertain.to_parquet(OUT / 'citibike_dst_unresolved.parquet',index=False)
    regular = aggregate[~aggregate.ambiguous_dst].copy()
    regular['timestamp'] = regular.local_hour.dt.tz_localize(TZ,ambiguous='raise',nonexistent='raise')
    base = pd.MultiIndex.from_product([ids,pd.DatetimeIndex(grid.utc_hour).tz_convert(TZ)],names=['station_id','timestamp']).to_frame(index=False)
    bike = base.merge(regular[['station_id','timestamp','pickups','dropoffs']],on=['station_id','timestamp'],how='left',validate='one_to_one')
    bike[['pickups','dropoffs']] = bike[['pickups','dropoffs']].fillna(0).astype('Int64')
    bike['dst_unresolved_pickups'] = False
    bike['dst_unresolved_dropoffs'] = False
    wall_hours = bike.timestamp.dt.tz_localize(None)
    for row in uncertain.itertuples(index=False):
        mask = (bike.station_id==row.station_id) & (wall_hours==row.local_hour)
        if mask.sum()!=2:
            raise AssertionError('An ambiguous wall-clock hour must map to exactly two UTC hours')
        for field in ['pickups','dropoffs']:
            if getattr(row,field)>0:
                bike.loc[mask,field] = pd.NA
                bike.loc[mask,'dst_unresolved_'+field] = True
    bike['net_flow'] = bike.dropoffs-bike.pickups
    bike = bike.sort_values(['station_id','timestamp']).reset_index(drop=True)
    bike['system_no_published_pickups_date'] = bike.timestamp.dt.tz_localize(None).dt.normalize().isin(missing_days)
    bike.to_parquet(OUT / 'citibike_hourly.parquet',index=False)
    weather_meta = json.loads((ROOT / 'data/raw/weather/weather_manifest.json').read_text())
    weather_id = weather_meta['selected_station_id']
    raw_weather = pd.read_parquet(ROOT / f'data/raw/weather/{weather_id}_hourly_observations.parquet')
    if raw_weather.index.has_duplicates:
        raise AssertionError('Raw weather has duplicate UTC timestamps')
    weather = raw_weather.reindex(pd.DatetimeIndex(grid.utc_hour))
    weather = weather[['temp','prcp','wspd']].rename(columns={'temp':'temperature','prcp':'precipitation','wspd':'wind_speed'})
    weather.index = weather.index.tz_convert(TZ)
    weather.index.name = 'timestamp'
    weather = weather.reset_index()
    weather['weather_station_id'] = weather_id
    for field in ['temperature','precipitation','wind_speed']:
        weather[field+'_missing'] = weather[field].isna()
    weather.to_parquet(OUT / 'nyc_weather_hourly.parquet',index=False)
    merged = bike.merge(weather,on='timestamp',how='left',validate='many_to_one')
    merged['hour'] = merged.timestamp.dt.hour.astype('int8')
    merged['weekday'] = merged.timestamp.dt.dayofweek.astype('int8')
    us_holidays = holidays.US(years=range(2024,2027),observed=True)
    merged['holiday'] = merged.timestamp.dt.date.map(lambda d: d in us_holidays)
    merged['period'] = np.where(merged.timestamp < pd.Timestamp('2026-01-01',tz=TZ),'train','test')
    merged['utc_offset_minutes'] = merged.timestamp.map(lambda t: int(t.utcoffset().total_seconds()/60)).astype('int16')
    merged.to_parquet(OUT / 'modeling_dataset.parquet',index=False)
    assert len(bike)==len(ids)*len(grid) and len(merged)==len(bike)
    assert not bike.duplicated(['station_id','timestamp']).any()
    assert not weather.timestamp.duplicated().any()
    assert (bike.dropoffs-bike.pickups).equals(bike.net_flow)
    assert (weather.precipitation.dropna()>=0).all() and (weather.wind_speed.dropna()>=0).all()
    assert (bike.pickups.dropna()>=0).all() and (bike.dropoffs.dropna()>=0).all()
    training_profiles = db.execute('SELECT sum(training_events),sum(coalesce(training_pickups,0)) FROM station_aliases').fetchone()
    training_hourly = db.execute("SELECT sum(pickups+dropoffs),sum(pickups) FROM hourly WHERE local_hour<timestamp '2026-01-01'").fetchone()
    assert training_profiles==training_hourly
    conservation = {}
    for field in ['pickups','dropoffs']:
        expected = int(regular[field].sum())
        # If an affected bin has known events as well as ambiguous events, mask
        # both folds. In this source all repeated-hour events are unresolved.
        assert int(bike[field].sum())==expected
        conservation[field] = {'assigned_total': expected,'unresolved_wall_hour_total': int(uncertain[field].sum())}
    validation = {'station_count':len(ids),'hours_per_station':len(grid),'station_hour_rows':len(bike),
                  'train_rows':int((merged.period=='train').sum()),'test_rows':int((merged.period=='test').sum()),
                  'system_no_published_pickup_dates':[d.strftime('%Y-%m-%d') for d in missing_days],
                  'zero_pickup_hours':int((bike.pickups==0).sum()),
                  'zero_dropoff_hours':int((bike.dropoffs==0).sum()),
                  'missing_pickup_hours':int(bike.pickups.isna().sum()),'missing_dropoff_hours':int(bike.dropoffs.isna().sum()),
                  'weather_missing':{c:int(weather[c].isna().sum()) for c in ['temperature','precipitation','wind_speed']},
                  'training_endpoint_events_reconciled':int(training_hourly[0]),
                  'count_conservation':conservation,'checks_passed':True}
    (OUT / 'validation.json').write_text(json.dumps(validation,indent=2))
    monthly = pd.DataFrame([{k:v for k,v in q.items() if k!='csv_schemas'} for q in quality])
    monthly.to_csv(OUT / 'monthly_trip_quality.csv',index=False)
    report(availability,quality,selected,changes,aliases,weather_meta,weather,merged,validation,stop)
    print(json.dumps(validation,indent=2),flush=True)

def report(availability,quality,selected,changes,aliases,weather_meta,weather,merged,validation,stop):
    raw_n = sum(q['raw_rows'] for q in quality)
    available_months = [x['month'] for x in availability['months'] if x['available']]
    last_month = pd.Timestamp(stop)-pd.Timedelta(days=1)
    def total(key):
        return sum(q[key] for q in quality)
    cols = ['station_id','station_name','usage_label','training_events','casual_share','rush_share','weekend_weekday_daily_ratio','min_active_day_fraction','monthly_center_span_m']
    station_table = selected[cols].copy()
    for c in ['casual_share','rush_share','weekend_weekday_daily_ratio','min_active_day_fraction','monthly_center_span_m']:
        station_table[c] = station_table[c].map(lambda x:f'{x:.3f}')
    candidates = pd.DataFrame([{'station':x['station_id'],'observed_rows':x['observed_rows'],
                              'missing_temp':x['missing']['temp'],'missing_prcp':x['missing']['prcp'],
                              'missing_wind':x['missing']['wspd'],'all_three_complete':x['all_three_complete_hours']} for x in weather_meta['candidates']])
    weather_months = weather.assign(month=weather.timestamp.dt.strftime('%Y-%m')).groupby('month').agg(hours=('timestamp','size'),temperature_missing=('temperature',lambda x:x.isna().sum()),precipitation_missing=('precipitation',lambda x:x.isna().sum()),wind_missing=('wind_speed',lambda x:x.isna().sum())).reset_index()
    weather_months.to_csv(OUT / 'weather_month_coverage.csv',index=False)
    weather_periods = weather.assign(period=np.where(weather.timestamp < pd.Timestamp('2026-01-01',tz=TZ),'train','test')).groupby('period').agg(hours=('timestamp','size'),temperature_missing=('temperature',lambda x:x.isna().sum()),precipitation_missing=('precipitation',lambda x:x.isna().sum()),wind_missing=('wind_speed',lambda x:x.isna().sum())).reset_index()
    weather_periods.to_csv(OUT / 'weather_period_coverage.csv',index=False)
    selected_aliases = aliases[aliases.station_id.isin(selected.station_id)]
    selected_changes = changes[changes.ids.map(lambda ids:any(s in ids.split(', ') for s in selected.station_id))]
    near_daily = pd.read_csv(OUT / 'station_month_coverage.csv',dtype={'station_id':str,'month':str})
    coverage = near_daily[near_daily.station_id.isin(selected.station_id)].groupby('station_id').agg(months=('month','size'),min_active_days=('active_days','min'),min_pickups=('pickups','min'),min_dropoffs=('dropoffs','min')).reset_index()
    disruption_audit = pd.read_csv(OUT / 'system_zero_pickup_date_audit.csv')
    text = f'''# Data quality report

## Scope and coverage

Prepared datasets only; no fitting, random splitting, or dashboard was performed.
Official bucket checked at {availability['checked_at_utc']}. {len(available_months)} of 33 requested NYC monthly archives are available, January 2024–{last_month.strftime('%B %Y')}. October 2026 is outside the requested window and incomplete at the requested project date (October 13, 2026).
Training: January 1, 2024–December 31, 2025. Test: January 1, 2026–{last_month.strftime('%B %d, %Y')}. End boundary: {stop} 00:00 America/New_York (exclusive).
{raw_n:,} raw ride records across {sum(len(q['csv_schemas']) for q in quality)} extracted CSVs. Every data CSV in every monthly ZIP was extracted; macOS resource metadata is excluded.
There are {validation['hours_per_station']:,} real hourly instants per station, {validation['station_hour_rows']:,} station-hour rows for eight stations, {validation['train_rows']:,} training rows, and {validation['test_rows']:,} test rows. Local dates without any valid published systemwide pickup: {validation['system_no_published_pickup_dates']}. These dates remain on the grid with zero published counts and `system_no_published_pickups_date=True`; the quality checks do not prove the publisher captured every actual trip.

The February 23, 2026 zero-pickup date is consistent with the [NYC Mayor's February 22 storm announcement](https://www.nyc.gov/mayors-office/news/2026/02/mayor-mamdani-declares-local-state-of-emergency--snow-day-for-ny), which said Citi Bike service would halt at 8 p.m., and the [February 23 recovery update](https://www.nyc.gov/mayors-office/news/2026/02/transcript--mayor-mamdani-announces-lift-of-travel-ban--nyc-publ), which described station snow clearing. Exact reopening times are not established by this preparation. The flag identifies absence of published pickups, not a verified hourly closure schedule or zero unconstrained rider demand. The matching monthly archive can still contain drop-off timestamps without usable station IDs; those records cannot be assigned to selected stations:

{frame_table(disruption_audit)}

## Columns and usefulness

Every source CSV has the same 13 columns: ride_id, rideable_type, started_at, ended_at, start_station_name, start_station_id, end_station_name, end_station_id, start_lat, start_lng, end_lat, end_lng, member_casual.

- Ride ID identifies duplicates; IDs are preserved as strings, including decimal-like station identifiers.
- Start/end timestamps and station IDs produce pickup/drop-off labels at their respective endpoint hours.
- Names and coordinates support station identity and geographic/relocation audits.
- Membership and rideable type are useful for usage descriptions. Same-hour trip attributes and trip duration are outcomes, not automatically available forecast predictors.

## Trip validation and decisions

Missing ride IDs: {total('missing_ride_ids'):,}. Missing start station IDs: {total('missing_start_station_ids'):,} ({total('missing_start_station_ids')/raw_n:.2%}); missing end station IDs: {total('missing_end_station_ids'):,} ({total('missing_end_station_ids')/raw_n:.2%}). A missing endpoint ID removes only that endpoint, not the opposite valid endpoint.
Exact duplicate excess rows within months: {total('exact_duplicate_excess'):,}. Repeated-ID excess within months: {total('within_month_duplicate_id_excess'):,}; conflicting repeated IDs: {total('within_month_conflicting_ids'):,}. Cross-month repeated IDs: {total('cross_month_duplicate_ids'):,}, including {total('cross_month_conflicting_ids'):,} conflicting variants. Duplicate rides contribute once globally. Deterministic within-month ordering chooses a representative, and the earliest archive takes precedence across months. Rides without IDs are excluded because reliable event deduplication is not possible.
After ID deduplication, invalid start timestamps: {total('invalid_started_at'):,}; invalid end timestamps: {total('invalid_ended_at'):,}; invalid timestamp/order trips excluded: {total('invalid_trip_order_or_parse'):,}. Negative naive durations inside the fall-back repeated hour can be consistent with a real forward trip and are retained as unresolved endpoint events. Trips longer than 24 hours are retained but flagged ({total('duration_over_24h_retained'):,}), as a long duration alone does not establish a false station event.
Started outside the named archive month: {total('started_outside_archive_month'):,}. Endpoints outside the grid or at nonexistent local hours: {total('outside_grid_or_nonexistent_endpoints'):,}; endpoints before the grid: {total('before_grid_endpoints'):,}. All endpoint aggregation uses the common global period rather than clipping at each archive's named month. Archive min/max timestamps show whether records are grouped by end month; see monthly diagnostics. Pickups near the final period boundary may be right-censored if their trips end in a later, unrequested archive. The next month's archive would be needed to quantify this boundary loss. Arrivals after the final boundary are excluded.
Raw source timestamps have no offset. America/New_York is the explicit interpretation; official column documentation does not specify an offset. Timestamps during the fall-back 01:00 hour cannot be identified uniquely. {total('ambiguous_dst_endpoints'):,} systemwide endpoints fall in these repeated wall hours. Selected-station unresolved endpoint totals: {json.dumps(validation['count_conservation'])}.

## Daylight saving and zero hours

Timestamp columns are timezone-aware America/New_York Parquet timestamps. The UTC instant is the key: spring-forward dates have 23 hours; fall-back dates have 25, with distinct -04:00 and -05:00 versions of 01:00. Weather is requested in UTC and converted, so joins never rely on ambiguous naive local time.
The grid fills observed-count gaps with zero for eligible stations, including {validation['zero_pickup_hours']:,} zero-pickup hours and {validation['zero_dropoff_hours']:,} zero-drop-off hours. This is an absence of published rides, not proof of station operation. If a station/direction has unresolved fall-back events, both repeated-hour counts are nullable, not zero: {validation['missing_pickup_hours']} pickup bins and {validation['missing_dropoff_hours']} drop-off bins. `citibike_dst_unresolved.parquet` retains combined station/local-hour counts; direction-specific flags preserve the distinction. Net flow is nullable when either direction is unresolved.
No interpolation, ride allocation, or target imputation is performed.

## Selected Manhattan stations

Membership uses the NYC Department of City Planning's official borough polygon (26b, accessed via NYC Open Data); membership is checked at the observed station center. IDs are never merged on name similarity. Stable candidates have records in all 33 months, at least 85% active local days in every month, at least 1,000 monthly pickups and 1,000 monthly drop-offs, and monthly station-center span no more than 250 m.
The high-activity pool is the top 60 eligible stations by 2024–2025 endpoint events. Three recreational-leaning stations maximize training casual share × weekend/weekday daily pickup ratio. Five commuter-leaning stations maximize training weekday 07:00–09:59 and 16:00–18:59 pickup share among the top 35 remaining stations by activity. These are descriptive proxies, not verified trip purposes. No test-period demand ranking is used. Full-window station eligibility does use test-period continuity (a retrospective survivorship filter), which must be disclosed in any later evaluation; it is not a prospective station-selection policy.

{frame_table(station_table)}

Continuity:

{frame_table(coverage)}

## Station identifier and name audit

{len(changes):,} normalized station names map to multiple IDs somewhere in the source; these may be replacements, overlapping IDs, or naming reuse, not proven renumberings. See `station_identifier_changes.csv` for IDs and first/last dates, `station_aliases.csv` for per-ID names, and `station_candidates.csv` for monthly-center movement and eligibility. Names alone do not establish physical station continuity. Selected IDs have {len(selected_aliases):,} observed name variants; {len(selected_changes)} multiple-ID name groups involve selected stations. Observed coordinates may drift; the station-center filter reduces relocation risk without claiming that all ID changes are resolved.

{frame_table(selected_aliases[['station_id','station_name','first_event','last_event']])}

## Weather source, units, and coverage

Meteostat Python {weather_meta['package_version']}, `Hourly(..., model=False, flags=True)`, fetched in UTC (retrieved at {weather_meta['retrieved_at_utc']}). Raw provider-cache downloads and raw package observations for each candidate are preserved under data/raw/weather. Selected station: {weather_meta['selected_station_id']}, {weather_meta['selected_station']['name']}, latitude {weather_meta['selected_station']['latitude']}, longitude {weather_meta['selected_station']['longitude']}, elevation {weather_meta['selected_station']['elevation_m']} m. Selection maximizes hours with all three measured variables, with total missingness as tie-breaker among current Manhattan/nearby-airport candidates. Airport exposure differs from street-level Manhattan conditions; one station is a regional covariate, not a station-specific weather measurement.

{frame_table(candidates)}

Temperature is °C, precipitation mm per hour, and wind speed km/h. Missing temperature: {validation['weather_missing']['temperature']:,}/{len(weather):,}; missing precipitation: {validation['weather_missing']['precipitation']:,}/{len(weather):,}; missing wind speed: {validation['weather_missing']['wind_speed']:,}/{len(weather):,}. Missingness is preserved, with boolean flags. No weather interpolation, zero precipitation fill, or modeled weather substitution is performed. See `weather_month_coverage.csv` for monthly completeness, including recent-month coverage.

Training/test completeness (missing hourly values, before station replication):

{frame_table(weather_periods)}

Recent precipitation gaps are substantial and must be treated as a material limitation of weather-based test-period comparisons. This dataset preparation does not establish that precipitation is sufficiently complete for a later model.
These are realized historical observations, potentially revised after the fact. **Actual future weather is not known when historical forecasts are issued.** Later modeling must use forecast archives available at forecast origin, lagged observations, or a clearly labeled oracle-weather experiment; this merged table does not make contemporaneous realized weather a deployable predictor.

## Merged schema and interpretation

Core: station_id (string), timestamp (America/New_York aware), pickups (nullable integer), dropoffs (nullable integer), net_flow = dropoffs - pickups (nullable integer), temperature, precipitation, wind_speed, hour (0–23), weekday (Monday=0), holiday (US federal holidays including observed dates).
Additional columns: DST uncertainty flags, system_no_published_pickups_date (a raw-data coverage/disruption audit flag), weather station ID, weather missing flags, period (train/test by time boundary), utc_offset_minutes (distinguishes repeated wall hours). The holiday definition does not include all NYC school closures or local events. The systemwide date flag is derived retrospectively from realized rides and must not be treated as a known-ahead forecast predictor.
**Trip counts and net flow are not actual bike inventory.** Inventory also depends on starting stock, rebalancing, maintenance, capacity and unavailable docks/bikes; none is inferred here.

## Validation and limitations

Verified complete station/hour grid, unique station/UTC-hour keys, unique weather keys, exact many-to-one join row-count preservation, nonnegative observed counts/precipitation/wind, net-flow arithmetic, explicit reporting of systemwide dates without published pickups, training station-selection counters reconciled to actual endpoint-time aggregates, and selected count reconciliation to assigned endpoints plus separately retained unresolved DST totals. Results are saved in `validation.json`; monthly trip diagnostics are in `monthly_trip_quality.csv`. All checks passed for these outputs; the weather gaps and disruption remain substantive data-quality limitations.
Principal risks for later work: retrospective station continuity filtering can make the panel easier than a prospective station universe; precipitation gaps can bias weather effects; ambiguous fall-back trip times cannot be recovered from this release; published zero counts can reflect closures or source omissions. Training and test dates are preserved, but this preparation does not claim model accuracy or operational benefits.

## Sources and reproducibility

- [Official Citi Bike documentation](https://citibikenyc.com/system-data) and [archive bucket](https://s3.amazonaws.com/tripdata/index.html). Public trip histories already apply provider filtering; these are published trip counts.
- [Meteostat hourly API documentation](https://dev.meteostat.net/python/api/meteostat.hourly). The project pins the 1.7.6 class API; newer docs expose a function API. Observation flags and caches remain inspectable. Meteostat attribution/licensing applies to reuse.
- [NYC DCP borough boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne), saved unmodified under data/raw/geography.
- Run commands and pinned dependencies: README.md and requirements.txt. Each monthly receipt preserves official size, ETag, release date, archive SHA-256 and extracted-member sizes/CRCs. Raw downloads are retained separately from interim and processed files.
'''
    (OUT / 'data_quality_report.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    main()
