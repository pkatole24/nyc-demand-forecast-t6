"""Research SARIMAX models deployed for live forecasts.

Two specifications, both frozen on 2025 validation and scored on the 2026 test
(scripts/evaluate_sarimax.py, evaluate_test.py):

- sarimax_calendar: SARIMA(1,0,1)(1,1,1)24 per station and target, with weekly
  Fourier terms (K=4), an off-day indicator (weekend or US federal holiday) and
  off-day x daily Fourier terms (K=3). Seasonal differencing at 24 h absorbs the
  plain daily cycle, so no daily Fourier terms enter on their own.
- sarimax_weather_forecast: the same model plus Open-Meteo weather forecasts
  issued two days ahead (`_previous_day2`): temperature, temperature squared,
  wind speed and a rain indicator (forecast precipitation >= 0.1 mm).

Saved artifacts hold only what forecasting needs: estimated parameters and the
Kalman predicted state (and covariance) for the hour after the last observation.
A live forecast starts from that state and runs the filter forward with missing
observations, so it bridges the gap between the monthly trip archive and now.
"""

import json
import urllib.parse
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import ARTIFACTS, TZ, digest

MODELS = ('sarimax_calendar', 'sarimax_weather_forecast')
ORDER, SEASONAL_ORDER = (1, 0, 1), (1, 1, 1, 24)
WEEKLY_HARMONICS, OFFDAY_DAILY_HARMONICS = 4, 3
WEATHER_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
WEATHER_POINT = (40.7769, -73.874)  # LaGuardia, as in the evaluation
WEATHER_VARIABLES = ('temperature_2m_previous_day2', 'precipitation_previous_day2', 'wind_speed_10m_previous_day2')
WET_MM = 0.1
# Forecast hours missing from the archive may be carried from an adjacent hour,
# at most this many in a row; longer gaps make the forecast unavailable.
MAX_WEATHER_FILL_HOURS = 3


def folder(model):
    if model not in MODELS:
        raise ValueError(f'Unknown SARIMAX model {model}')
    return ARTIFACTS / model


def _fourier(phase, harmonics):
    return np.column_stack([f(2 * np.pi * k * phase) for k in range(1, harmonics + 1) for f in (np.sin, np.cos)])


def calendar_features(index):
    """Calendar regressors for UTC hourly instants; all are known in advance."""
    local = pd.DatetimeIndex(index).tz_convert(TZ)
    hour, weekday = local.hour.to_numpy(), local.dayofweek.to_numpy()
    days = local.tz_localize(None).normalize()
    # The panel's holiday flag also marks the actual date of a weekend holiday;
    # those days are weekends here anyway, so the off-day indicator is identical.
    holidays = USFederalHolidayCalendar().holidays(days.min() - pd.Timedelta(days=7), days.max() + pd.Timedelta(days=7))
    off = ((weekday >= 5) | days.isin(holidays)).astype(float)
    # The evaluated model computed hour-of-week from the panel's int8 weekday, so
    # values above 127 (Saturday 08:00 onward) wrapped to negative numbers. The
    # deployed models reproduce those exact regressors so their 2025/2026 scores
    # apply; a corrected encoding would need re-evaluation first.
    hour_of_week = (weekday * 24 + hour + 128) % 256 - 128
    return np.column_stack([_fourier(hour_of_week / 168, WEEKLY_HARMONICS), off,
                            off[:, None] * _fourier(hour / 24, OFFDAY_DAILY_HARMONICS)])


def weather_columns(temperature, precipitation, wind):
    """Evaluation order: temperature, temperature squared, wind speed, wet."""
    temperature, precipitation, wind = (np.asarray(v, dtype=float) for v in (temperature, precipitation, wind))
    wet = np.where(np.isnan(precipitation), np.nan, (precipitation >= WET_MM).astype(float))
    return np.column_stack([temperature, temperature**2, wind, wet])


@lru_cache(maxsize=8)
def _fetch_weather(start_date, end_date):
    params = {'latitude': WEATHER_POINT[0], 'longitude': WEATHER_POINT[1], 'start_date': start_date,
              'end_date': end_date, 'hourly': ','.join(WEATHER_VARIABLES), 'timezone': 'GMT'}
    with urllib.request.urlopen(f'{WEATHER_URL}?{urllib.parse.urlencode(params)}', timeout=60) as response:
        hourly = json.load(response)['hourly']
    frame = pd.DataFrame({v: pd.Series(hourly[v], dtype='float64').to_numpy() for v in WEATHER_VARIABLES},
                         index=pd.to_datetime(hourly['time']).tz_localize('UTC'))
    return frame


def live_weather(index):
    """Archived two-day-ahead forecasts for UTC hours `index`, fetched from Open-Meteo."""
    index = pd.DatetimeIndex(index)
    frame = _fetch_weather(f'{index.min():%Y-%m-%d}', f'{index.max():%Y-%m-%d}').reindex(index)
    filled = frame.ffill(limit=MAX_WEATHER_FILL_HOURS).bfill(limit=MAX_WEATHER_FILL_HOURS)
    if filled.isna().any().any():
        raise ValueError('Weather forecasts are unavailable for part of the forecast period')
    return weather_columns(filled[WEATHER_VARIABLES[0]], filled[WEATHER_VARIABLES[1]], filled[WEATHER_VARIABLES[2]])


def exog(model, index, weather=None):
    calendar = calendar_features(index)
    if model == 'sarimax_calendar':
        return calendar
    if weather is None:
        raise ValueError('The weather-forecast model needs weather regressors')
    return np.column_stack([calendar, weather])


def make_model(model, y, index, weather=None):
    return SARIMAX(np.asarray(y, dtype=float), exog=exog(model, index, weather), order=ORDER,
                   seasonal_order=SEASONAL_ORDER, trend='n')


def load(model, station, target):
    manifest = json.loads((folder(model) / 'manifest.json').read_text())
    record = manifest['models'][f'{station}/{target}']
    path = folder(model) / record['artifact']
    if digest(path) != record['sha256']:
        raise ValueError('SARIMAX model hash mismatch')
    return manifest, record, np.load(path)


def forecast(model, station, target, index):
    """Mean forecasts for future UTC hours `index`, from the saved end-of-history state."""
    manifest, record, saved = load(model, station, target)
    last = pd.Timestamp(record['last_observation'])
    index = pd.DatetimeIndex(index)
    if index.min() <= last:
        raise ValueError('Forecast hours must follow the last observation')
    grid = pd.date_range(last + pd.Timedelta(hours=1), index.max(), freq='h')
    weather = live_weather(grid) if model == 'sarimax_weather_forecast' else None
    state_space = make_model(model, np.full(len(grid), np.nan), grid, weather)
    state_space.ssm.initialize_known(saved['state'], saved['state_cov'])
    filtered = state_space.filter(saved['params'], cov_type='none', low_memory=False)
    mean = filtered.filter_results.forecasts[0, grid.get_indexer(index)]
    return np.maximum(0, mean), manifest, record
