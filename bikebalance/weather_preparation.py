import pandas as pd
import numpy as np

TZ = "America/New_York"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(series):
    return pd.to_numeric(series, errors="coerce")


def align(frame,source,grid):
    frame=frame.copy()
    frame['hour_utc']=frame.observation_timestamp.dt.floor('h')
    frame=frame.sort_values('observation_timestamp').drop_duplicates('hour_utc',keep='last')
    frame=frame.set_index('hour_utc').reindex(grid)
    frame.index.name='timestamp_utc'
    frame['source']=source
    frame['precipitation_trace']=frame.precipitation_trace.fillna(False).astype(bool)
    frame['precipitation_observed']=frame.precipitation.notna()|frame.precipitation_trace
    return frame.reset_index()

def parse_iem_reports(d, grid, station="LGA"):
    raw=d.p01i
    pno=d.metar.str.contains(r'\bPNO\b',na=False)
    amount=number(raw)*25.4
    # Never retain a nominal dry value when the raw report says the
    # precipitation sensor is unavailable.
    amount=amount.mask(pno)
    trace=raw.eq('T') & ~pno
    explicit=d.metar.str.contains(r'\bP\d{4}\b',na=False)
    wet_zero=amount.eq(0)&~explicit&d.metar.str.contains(r'(?:\s|[-+])(?:RA|SN|DZ|SHRA|TSRA)(?:\s|$)',na=False)
    provider_zero=amount.eq(0)&~explicit
    amount=amount.mask(wet_zero)
    result = pd.DataFrame({'observation_timestamp':pd.to_datetime(d.valid,utc=True),
        'temperature':(number(d.tmpf)-32)*5/9,'wind_speed':number(d.sknt)*1.852,
        'precipitation':amount,'precipitation_trace':trace,'raw_precipitation':raw,
        'precipitation_sensor_unavailable':pno,'metar_has_explicit_P_group':explicit,
        'zero_without_explicit_P_group':provider_zero,
        'wet_code_zero_without_P':wet_zero,
        'raw_metar':d.metar,'observation_type':'station_observation_with_provider_dry_normalization',
        'interval_semantics':'routine METAR since previous hourly precipitation reset; not specials'})
    return align(result, f"IEM_{station}_routine", grid)


def integrate_weather(baseline, rain):
    require(not baseline.timestamp.duplicated().any(), 'Duplicate baseline weather keys')
    require(not rain.timestamp_utc.duplicated().any(), 'Duplicate IEM hourly keys')
    fields = ['timestamp_utc', 'observation_timestamp', 'precipitation',
              'precipitation_trace', 'precipitation_observed', 'raw_precipitation',
              'precipitation_sensor_unavailable', 'metar_has_explicit_P_group',
              'zero_without_explicit_P_group', 'wet_code_zero_without_P',
              'raw_metar', 'interval_semantics', 'source', 'observation_type']
    rain = rain[fields].rename(columns={
        'timestamp_utc': 'timestamp', 'source': 'precipitation_source',
        'observation_timestamp': 'precipitation_report_timestamp_utc'})
    rain['timestamp'] = rain.timestamp.dt.tz_convert(TZ)
    weather = baseline.drop(columns=['precipitation', 'precipitation_missing']).merge(
        rain, on='timestamp', how='left', validate='one_to_one')
    require(pd.DatetimeIndex(baseline.timestamp).tz_convert('UTC').sort_values().equals(
        pd.DatetimeIndex(rain.timestamp).tz_convert('UTC').sort_values()),
        'IEM and baseline grids differ')
    for field in ['precipitation_trace', 'precipitation_observed',
                  'precipitation_sensor_unavailable', 'metar_has_explicit_P_group',
                  'zero_without_explicit_P_group', 'wet_code_zero_without_P']:
        weather[field] = weather[field].fillna(False).astype(bool)
    weather['precipitation_missing'] = ~weather.precipitation_observed
    weather['precipitation_source'] = 'IEM_LGA_routine'
    weather['precipitation_units'] = 'mm'
    weather['temperature_source'] = 'Meteostat_72503'
    weather['wind_speed_source'] = 'Meteostat_72503'
    require(weather.loc[weather.precipitation_trace, 'precipitation'].isna().all(),
            'Trace rainfall must remain numeric null')
    unusable = weather.precipitation_sensor_unavailable | weather.wet_code_zero_without_P
    require(weather.loc[unusable, 'precipitation'].isna().all()
            and weather.loc[unusable, 'precipitation_missing'].all(),
            'Unusable precipitation must remain missing')
    require((weather.precipitation.notna() | weather.precipitation_trace).equals(
        weather.precipitation_observed), 'Rainfall observation flags disagree')
    require(not weather.precipitation.dropna().lt(0).any(), 'Negative rainfall')
    return weather
