from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm

from .data import TZ


@dataclass(frozen=True)
class Specification:
    order: tuple = (1, 0, 0)
    seasonal_order: tuple = (1, 0, 0, 24)
    harmonics: int = 0

    @property
    def label(self):
        return f'{self.order}x{self.seasonal_order}_K{self.harmonics}'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value['order']), tuple(value['seasonal_order']), value['harmonics'])


def weekly_terms(index, harmonics):
    if not harmonics:
        return None
    local = pd.DatetimeIndex(index).tz_convert(TZ)
    phase = (local.dayofweek.to_numpy() * 24 + local.hour.to_numpy()) / 168
    return np.column_stack([function(2 * np.pi * k * phase)
                            for k in range(1, harmonics + 1) for function in (np.sin, np.cos)])


def seasonal_naive(history, targets):
    """Repeat the last observed weekly template; ambiguous wall-hour lookups stay unknown."""
    if history.empty:
        return np.full(len(targets), np.nan)
    local = history.copy()
    local.index = local.index.tz_convert(TZ).tz_localize(None)
    # Both versions of a repeated local source hour are unavailable even if one is numeric.
    local = local[~local.index.duplicated(keep=False)]
    latest = history.index.max().tz_convert(TZ).tz_localize(None)
    values = []
    for timestamp in pd.DatetimeIndex(targets).tz_convert(TZ).tz_localize(None):
        source = timestamp - pd.Timedelta(days=7)
        if source > latest:
            source -= pd.Timedelta(days=7 * int(np.ceil((source - latest) / pd.Timedelta(days=7))))
        values.append(local.get(source, np.nan))
    return np.asarray(values, dtype=float)


def make_model(history, specification, exog=None):
    p = specification
    return SARIMAX(history, exog=exog if exog is not None else weekly_terms(history.index, p.harmonics),
                   order=p.order, seasonal_order=p.seasonal_order,
                   trend='c' if p.order[1] == p.seasonal_order[1] == 0 else 'n',
                   enforce_stationarity=True, enforce_invertibility=True)


def fit(history, specification, start_params=None, maxiter=100, exog=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = make_model(history, specification, exog=exog).fit(
            start_params=start_params, disp=False, maxiter=maxiter, low_memory=True, cov_type='none')
    return result, [str(item.message) for item in caught]


def update(result, actual, specification, exog=None):
    if actual.empty:
        return result
    # A regressor can be constant over a short update/forecast even when it varied in training.
    model = result.model.clone(actual, exog=exog if exog is not None else weekly_terms(actual.index, specification.harmonics),
                               validate_exog=False)
    state = result.filter_results
    # Low-memory results retain the final state internally but extend expects full arrays.
    model.ssm.initialize_known(state.predicted_state[:, -1], state.predicted_state_cov[:, :, -1])
    return model.filter(result.params, low_memory=True, cov_type='none')


def predict(result, specification, targets, exog=None):
    start = pd.Timestamp(result.model.data.dates[-1]) + pd.Timedelta(hours=1)
    grid = pd.date_range(start, pd.DatetimeIndex(targets).max(), freq='h')
    if not len(grid):
        raise ValueError('Forecast targets must follow the filtered history')
    future_model = result.model.clone(pd.Series(np.nan, index=grid),
                                    exog=exog if exog is not None else weekly_terms(grid, specification.harmonics),
                                    validate_exog=False)
    state = result.filter_results
    future_model.ssm.initialize_known(state.predicted_state[:, -1], state.predicted_state_cov[:, :, -1])
    future = future_model.filter(result.params, cov_type='none', low_memory=False)
    position = grid.get_indexer(targets)
    if (position < 0).any():
        raise ValueError('Forecast target is not an hourly instant')
    mean = future.filter_results.forecasts[0, position]
    deviation = np.sqrt(future.filter_results.forecasts_error_cov[0, 0, position])
    answer = {'prediction': np.maximum(0, mean), 'raw_prediction': mean}
    for level, alpha in ((80, .2), (95, .05)):
        width = norm.ppf(1 - alpha/2) * deviation
        interval = np.column_stack([mean-width, mean+width])
        answer[f'lower{level}'] = np.maximum(0, interval[:, 0])
        answer[f'upper{level}'] = np.maximum(0, interval[:, 1])
    return answer
