import numpy as np
import pandas as pd

from .data import DATA, TARGETS, TZ


def with_net_flow(forecasts):
    keys = ['station_id', 'origin', 'timestamp', 'family', 'partition', 'regime']
    pickup = forecasts[forecasts.target.eq('pickups')]
    dropoff = forecasts[forecasts.target.eq('dropoffs')]
    paired = pickup.merge(dropoff, on=keys, validate='one_to_one', suffixes=('_p', '_d'))
    net = paired[keys].copy()
    net['target'] = 'net_flow'
    for name in ['prediction', 'actual']:
        net[name] = paired[name+'_d'] - paired[name+'_p']
    for name in ['horizon', 'disruption', 'observation_cutoff', 'parameter_fit_cutoff']:
        net[name] = paired[name+'_p']
    net['high_demand'] = paired.high_demand_p | paired.high_demand_d
    net['status'] = np.where(paired.status_p.eq('ok') & paired.status_d.eq('ok'), 'ok',
                             'partial_or_failed_direction')
    return pd.concat([forecasts, net], ignore_index=True)


def summarize(forecasts):
    metadata = pd.read_csv(DATA/'selected_stations.csv', dtype={'station_id': str})
    frame = with_net_flow(forecasts).merge(metadata[['station_id', 'usage_label']],
                                          on='station_id', validate='many_to_one')
    local = frame.timestamp.dt.tz_convert(TZ)
    frame['local_hour'] = local.dt.hour
    frame['month'] = local.dt.strftime('%Y-%m')
    frame['day_type'] = np.where(local.dt.dayofweek < 5, 'weekday', 'weekend')
    frame['peak'] = (local.dt.dayofweek < 5) & local.dt.hour.isin([7, 8, 9, 16, 17, 18])
    common_keys = ['station_id', 'target', 'origin', 'timestamp', 'partition', 'regime']
    frame['_available'] = frame.prediction.notna() & frame.actual.notna()
    available_count = frame.groupby(common_keys)._available.transform('sum')
    expected_count = frame.groupby(['partition', 'regime', 'target']).family.transform('nunique')
    frame['_common'] = available_count.eq(expected_count)
    dimensions = ['station_id', 'usage_label', 'horizon', 'local_hour', 'month', 'day_type']
    base_keys = ['partition', 'regime', 'family', 'target']
    records = []
    for segment, selected in [('all', frame), ('peak', frame[frame.peak]),
                               ('high_demand', frame[frame.high_demand]),
                               ('excluding_disruption', frame[~frame.disruption])]:
        for dimension in ['overall'] + dimensions:
            keys = base_keys + ([] if dimension == 'overall' else [dimension])
            for values, group in selected.groupby(keys, dropna=False, observed=True):
                values = values if isinstance(values, tuple) else (values,)
                record = dict(zip(keys, values))
                valid = group._available & group._common
                scored = group[valid]
                errors = scored.prediction - scored.actual
                record.update(segment=segment, dimension=dimension,
                              value='all' if dimension == 'overall' else str(record.pop(dimension)),
                              eligible_rows=len(group), scored_rows=len(scored),
                              excluded_rows=int((~valid).sum()),
                              own_available_rows=int(group._available.sum()),
                              mae=float(errors.abs().mean()) if len(scored) else None,
                              rmse=float(np.sqrt((errors**2).mean())) if len(scored) else None,
                              bias=float(errors.mean()) if len(scored) else None)
                for level in [80, 95]:
                    interval_valid = valid & group[f'lower{level}'].notna() & group[f'upper{level}'].notna()
                    g = group[interval_valid]
                    record[f'interval_rows_{level}'] = len(g)
                    record[f'coverage_{level}'] = float((g.actual.ge(g[f'lower{level}']) &
                        g.actual.le(g[f'upper{level}'])).mean()) if len(g) else None
                    record[f'mean_width_{level}'] = float((g[f'upper{level}']-
                        g[f'lower{level}']).mean()) if len(g) else None
                records.append(record)
    result = pd.DataFrame(records)
    # Equal station weights prevent the busiest stations from defining aggregate ranking.
    station = result[(result.dimension == 'station_id') & (result.segment == 'all')]
    macro = station.groupby(base_keys)[['mae', 'rmse']].mean().reset_index()
    macro['dimension'] = 'equal_station_mean'
    macro['segment'] = 'all'
    macro['value'] = 'all'
    return pd.concat([result, macro], ignore_index=True)


def paired_bootstrap(forecasts, repetitions=1000):
    frame = with_net_flow(forecasts)
    frame['week'] = frame.timestamp.dt.tz_convert(TZ).dt.tz_localize(None).dt.to_period('W').astype(str)
    records = []
    for (partition, regime, target), group in frame.groupby(['partition', 'regime', 'target']):
        naive = group[group.family.eq('seasonal_naive')]
        for family in sorted(set(group.family) - {'seasonal_naive'}):
            paired = group[group.family.eq(family)].merge(naive,
                on=['station_id', 'origin', 'timestamp'], suffixes=('_m', '_n'), validate='one_to_one')
            paired = paired.dropna(subset=['actual_m', 'prediction_m', 'prediction_n'])
            if paired.empty:
                continue
            paired['difference'] = (paired.prediction_m - paired.actual_m).abs() - (
                paired.prediction_n - paired.actual_m).abs()
            blocks = paired.groupby('week_m').difference.agg(['sum', 'count'])
            rng = np.random.default_rng(2026)
            draws = rng.integers(len(blocks), size=(repetitions, len(blocks)))
            samples = blocks['sum'].to_numpy()[draws].sum(axis=1) / blocks['count'].to_numpy()[draws].sum(axis=1)
            records.append({'partition': partition, 'regime': regime, 'target': target,
                'family': family, 'paired_rows': len(paired), 'week_blocks': len(blocks),
                'mae_difference_vs_naive': float(paired.difference.mean()),
                'lower95': float(np.quantile(samples, .025)), 'upper95': float(np.quantile(samples, .975))})
    return pd.DataFrame(records)
