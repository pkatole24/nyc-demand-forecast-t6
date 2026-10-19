import numpy as np
import pandas as pd

from .data import TZ, ROOT, DATA, digest
import json
from .models import fit, predict, update


class PastWeather:
    lag_hours = 48

    def __init__(self, panel, incremental=None):
        columns = ['temperature', 'wind_speed', 'precipitation', 'precipitation_trace',
                   'precipitation_missing', 'zero_without_explicit_P_group',
                   'precipitation_report_timestamp_utc']
        weather = panel[['timestamp_utc', *columns]].drop_duplicates()
        if weather.timestamp_utc.duplicated().any():
            raise ValueError('Regional weather differs across stations at the same UTC hour')
        if incremental is not None:
            extra = incremental[['timestamp_utc', *columns]]
            boundary = weather.timestamp_utc.max()+pd.Timedelta(hours=1)
            expected = pd.date_range(boundary, extra.timestamp_utc.max(), freq='h')
            if not pd.DatetimeIndex(extra.timestamp_utc).equals(expected):
                raise ValueError('Incremental weather must be unique and consecutive after the base')
            weather = pd.concat([weather, extra], ignore_index=True)
        self.weather = weather.set_index('timestamp_utc').sort_index()

    @classmethod
    def live(cls, panel):
        folder = ROOT/'data/live/weather'
        path = folder/'weather_hourly.parquet'
        if not path.exists():
            return cls(panel)
        receipt = json.loads((folder/'receipt.json').read_text())
        if digest(path) != receipt['cache_sha256'] or digest(DATA/'modeling_dataset.parquet') != receipt['base_sha256']:
            raise ValueError('Live weather cache or base hash mismatch')
        return cls(panel, pd.read_parquet(path))

    def features(self, index, issuance=None):
        index = pd.DatetimeIndex(index)
        source = index-pd.Timedelta(hours=self.lag_hours)
        if issuance is not None:
            issuance = pd.Timestamp(issuance).tz_convert('UTC')
            if (source+pd.Timedelta(hours=6) >= issuance).any():
                raise ValueError('Weather sources violate the pre-issuance availability buffer')
        if (source > self.weather.index.max()).any():
            raise ValueError('ARIMAX needs newer observed weather; cached weather ends '
                             + str(self.weather.index.max()))
        values = self.weather.reindex(source)
        reports = values.precipitation_report_timestamp_utc
        limit = source+pd.Timedelta(hours=6)
        late = reports.notna() & (reports.to_numpy() >= limit)
        local = index.tz_convert(TZ)
        output = {}
        for period, phase, harmonics in [('daily', local.hour.to_numpy()/24, 3),
                                         ('weekly', (local.dayofweek.to_numpy()*24+local.hour.to_numpy())/168, 2)]:
            for k in range(1, harmonics+1):
                output[f'{period}_sin{k}'] = np.sin(2*np.pi*k*phase)
                output[f'{period}_cos{k}'] = np.cos(2*np.pi*k*phase)
        for name, divisor in [('temperature', 20), ('wind_speed', 20)]:
            numeric = values[name].to_numpy(dtype=float, na_value=np.nan)
            output[name+'_lag48'] = np.nan_to_num(numeric/divisor, nan=0.)
            output[name+'_missing'] = np.isnan(numeric).astype(float)
        trace = values.precipitation_trace.fillna(False).to_numpy(bool) & ~np.asarray(late)
        rain = values.precipitation.to_numpy(dtype=float, na_value=np.nan).copy()
        missing = values.precipitation_missing.fillna(True).to_numpy(bool) | np.asarray(late)
        rain[missing] = np.nan
        # Zero is a numerical encoding accompanied by trace/missing indicators, not a filled observation.
        output['rain_log1p_lag48'] = np.log1p(np.nan_to_num(rain, nan=0.))
        output['rain_trace_lag48'] = trace.astype(float)
        output['rain_missing_lag48'] = missing.astype(float)
        output['rain_normalized_zero_lag48'] = (
            values.zero_without_explicit_P_group.fillna(False).to_numpy(bool) & ~missing).astype(float)
        return pd.DataFrame(output, index=index, dtype=float)

    def fit(self, history, specification, start_params=None, maxiter=250):
        return fit(history, specification, start_params=start_params, maxiter=maxiter,
                   exog=self.features(history.index))

    def update(self, result, actual, specification):
        return update(result, actual, specification, exog=self.features(actual.index))

    def predict(self, result, specification, targets, issuance):
        start = pd.Timestamp(result.model.data.dates[-1])+pd.Timedelta(hours=1)
        grid = pd.date_range(start, pd.DatetimeIndex(targets).max(), freq='h')
        return predict(result, specification, targets, exog=self.features(grid, issuance))
