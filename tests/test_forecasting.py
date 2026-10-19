import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bikebalance.data import load_panel, local_day, publication_cutoff
from bikebalance.models import Specification, fit, make_model, predict, seasonal_naive, update, weekly_terms
from bikebalance.backtest import forecast_series
from bikebalance.gbfs import normalize
from bikebalance.live import current_forecasts
from bikebalance.simulation import simulate
from bikebalance.evaluation import summarize


class ForecastingBoundaries(unittest.TestCase):
    def test_weather_refresh_crosses_month_boundary_and_rejects_stale_sources(self):
        from scripts.refresh_live_weather import refresh_bounds, require_recent_observations
        panel = pd.DataFrame({'timestamp_utc': [pd.Timestamp('2026-10-01T03:00Z')]})
        now = pd.Timestamp('2026-11-02T12:30Z')
        start, stop = refresh_bounds(panel, now)
        self.assertEqual(start, pd.Timestamp('2026-10-01T04:00Z'))
        self.assertEqual(stop, pd.Timestamp('2026-11-02T12:00Z'))
        weather = pd.DataFrame({'timestamp_utc': pd.date_range(start, stop, freq='h', inclusive='left'),
                                'temperature': 10., 'wind_speed': 5., 'precipitation_observed': True})
        require_recent_observations(weather, now)
        weather.loc[weather.timestamp_utc >= now.floor('h')-pd.Timedelta(hours=24), 'temperature'] = np.nan
        with self.assertRaisesRegex(ValueError, 'temperature observations are too old'):
            require_recent_observations(weather, now)

    def test_past_weather_is_cutoff_safe_and_preserves_trace_and_missingness(self):
        from bikebalance.weather import PastWeather
        index = pd.date_range('2025-01-01', periods=120, freq='h', tz='UTC')
        panel = pd.DataFrame({'timestamp_utc': index, 'temperature': np.arange(120, dtype=float),
            'wind_speed': 10., 'precipitation': 0., 'precipitation_trace': False,
            'precipitation_missing': False, 'zero_without_explicit_P_group': False,
            'precipitation_report_timestamp_utc': index+pd.Timedelta(minutes=51)})
        panel.loc[0, ['precipitation', 'precipitation_trace']] = [np.nan, True]
        panel.loc[1, ['precipitation', 'precipitation_missing']] = [np.nan, True]
        issuance = index[48]
        targets = index[48:72]
        expected = PastWeather(panel).features(targets, issuance)
        changed = panel.copy()
        changed.loc[changed.timestamp_utc >= issuance, ['temperature', 'wind_speed', 'precipitation']] = 999.
        pd.testing.assert_frame_equal(expected, PastWeather(changed).features(targets, issuance))
        self.assertEqual(expected.rain_trace_lag48.iloc[0], 1.)
        self.assertEqual(expected.rain_missing_lag48.iloc[0], 0.)
        self.assertEqual(expected.rain_missing_lag48.iloc[1], 1.)
        with self.assertRaisesRegex(ValueError, 'availability buffer'):
            PastWeather(panel).features(index[48:100], issuance)
        with self.assertRaisesRegex(ValueError, 'newer observed weather'):
            PastWeather(panel).features(pd.date_range(index[-1]+pd.Timedelta(days=3), periods=24, freq='h'))
        incremental = panel.iloc[-3:].copy()
        incremental['timestamp_utc'] = index[-3:]+pd.Timedelta(hours=3)
        incremental['precipitation_report_timestamp_utc'] += pd.Timedelta(hours=3)
        engine = PastWeather(panel, incremental)
        future = pd.DatetimeIndex(incremental.timestamp_utc)+pd.Timedelta(hours=48)
        actual = engine.features(future, future.min())
        np.testing.assert_allclose(actual.temperature_lag48, incremental.temperature.to_numpy()/20)
        pd.testing.assert_frame_equal(engine.weather.loc[engine.weather.index <= index[-1]], PastWeather(panel).weather)
        for invalid in [incremental.iloc[1:], pd.concat([incremental, incremental.iloc[[0]]]), panel.iloc[-1:]]:
            with self.assertRaisesRegex(ValueError, 'unique and consecutive'):
                PastWeather(panel, invalid)

    def test_monthly_refits_and_daily_state_updates_respect_cutoffs(self):
        from types import SimpleNamespace
        index = pd.date_range('2024-01-01', '2025-02-03', freq='h', tz='America/New_York')
        y = pd.Series(5., index=index.tz_convert('UTC'))
        dates = pd.date_range('2025-01-30', periods=4, freq='D', tz='America/New_York')
        fitted = SimpleNamespace(params=np.array([1.]), mle_retvals={'converged': True})
        def predictions(result, spec, future):
            return {name: np.full(len(future), 5.) for name in
                    ['prediction', 'raw_prediction', 'lower80', 'upper80', 'lower95', 'upper95']}
        with patch('bikebalance.backtest.origins', return_value=dates), \
             patch('bikebalance.backtest.fit', return_value=(fitted, [])) as fitting, \
             patch('bikebalance.backtest.update', return_value=fitted) as updating, \
             patch('bikebalance.backtest.predict', side_effect=predictions):
            frame, logs = forecast_series(y, 'station', 'pickups', 'validation',
                                           'daily_sarima', Specification())
        self.assertEqual([r['refitted'] for r in logs], [True, False, True, False])
        self.assertEqual(fitting.call_count, 2)
        self.assertEqual(updating.call_count, 2)
        for call, origin in zip(fitting.call_args_list, dates[[0, 2]]):
            self.assertLess(call.args[0].index.max(), origin)
        self.assertTrue((frame.observation_cutoff < frame.origin).all())

    def test_hash_mismatch_fails_before_reading_parquet(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            folder = Path(directory)
            (folder/'modeling_dataset.parquet').write_bytes(b'corrupt')
            (folder/'manifest.json').write_text(json.dumps({'outputs_sha256': {
                'modeling_dataset.parquet': hashlib.sha256(b'expected').hexdigest()}}))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_panel(folder)

    def test_local_days_and_ambiguous_naive_source(self):
        spring = local_day(pd.Timestamp('2025-03-09', tz='America/New_York'))
        fall = local_day(pd.Timestamp('2025-11-02', tz='America/New_York'))
        self.assertEqual((len(spring), len(fall)), (23, 25))
        self.assertTrue(fall.is_unique)
        history = pd.Series(np.arange(len(fall), dtype=float), index=fall)
        target = pd.date_range('2025-11-09 01:00', periods=1, tz='America/New_York')
        self.assertTrue(np.isnan(seasonal_naive(history, target)[0]))

    def test_missing_lag_is_not_imputed_and_gap_uses_weekly_template(self):
        index = pd.date_range('2024-01-01', periods=168, freq='h', tz='UTC')
        history = pd.Series(np.arange(168, dtype=float), index=index)
        history.iloc[12] = np.nan
        prediction = seasonal_naive(history, index+pd.Timedelta(days=28))
        np.testing.assert_allclose(prediction, history.to_numpy(), equal_nan=True)

    def test_low_memory_update_matches_standard_filter_with_missing_observations(self):
        rng = np.random.default_rng(4)
        index = pd.date_range('2024-01-01', periods=160, freq='h', tz='UTC')
        y = pd.Series(20+rng.normal(size=160), index=index)
        y.iloc[130:132] = np.nan
        spec = Specification((1, 0, 0), (0, 0, 0, 24), 1)
        result, _ = fit(y.iloc[:120], spec)
        reference = make_model(y.iloc[:120], spec).filter(result.params)
        reference = reference.extend(y.iloc[120:144],
            exog=weekly_terms(index[120:144], 1))
        actual = update(result, y.iloc[120:144], spec)
        forecast = predict(actual, spec, index[144:])
        expected = reference.get_forecast(16,
            exog=weekly_terms(index[144:], 1))
        np.testing.assert_allclose(forecast['raw_prediction'], expected.predicted_mean, rtol=1e-7)
        np.testing.assert_allclose(forecast['upper95'], np.maximum(0, np.asarray(expected.conf_int())[:, 1]), rtol=1e-7)

    def test_regressors_may_be_constant_in_short_update_and_forecast_windows(self):
        index = pd.date_range('2024-01-01', periods=160, freq='h', tz='UTC')
        exog = pd.DataFrame({'weather': np.r_[np.linspace(0, 1, 120), np.ones(40)]}, index=index)
        y = pd.Series(10+2*exog.weather+np.random.default_rng(3).normal(size=160), index=index)
        spec = Specification((1, 0, 0), (0, 0, 0, 0))
        fitted, _ = fit(y.iloc[:120], spec, exog=exog.iloc[:120])
        state = update(fitted, y.iloc[120:144], spec, exog=exog.iloc[120:144])
        forecast = predict(state, spec, index[144:], exog=exog.iloc[144:])
        observed = y.copy()
        observed.iloc[144:] = np.nan
        reference = make_model(observed, spec, exog=exog).filter(fitted.params)
        np.testing.assert_allclose(forecast['raw_prediction'], reference.filter_results.forecasts[0, 144:], rtol=1e-7)

    def test_target_period_changes_do_not_change_earlier_forecast(self):
        y = pd.Series(5., index=pd.date_range('2024-01-01', '2025-01-03',
            inclusive='left', freq='h', tz='America/New_York').tz_convert('UTC'))
        before, _ = forecast_series(y, 'station', 'pickups', 'validation', 'seasonal_naive', days=1)
        changed = y.copy()
        changed.loc[changed.index >= pd.Timestamp('2025-01-01', tz='America/New_York')] = 999
        after, _ = forecast_series(changed, 'station', 'pickups', 'validation', 'seasonal_naive', days=1)
        np.testing.assert_array_equal(before.prediction, after.prediction)
        self.assertTrue((before.observation_cutoff < before.origin).all())

    def test_publication_cutoff_respects_revision_time(self):
        metadata = {'months': [{'month': '202501', 'available': True,
                               'last_modified': '2025-02-10T12:00:00Z'}]}
        self.assertIsNone(publication_cutoff(pd.Timestamp('2025-02-10T11:59:59Z'), metadata))
        self.assertEqual(publication_cutoff(pd.Timestamp('2025-02-10T12:00:00Z'), metadata),
                         pd.Timestamp('2025-02-01', tz='America/New_York').tz_convert('UTC'))

    def test_simulation_conserves_fulfilled_flow_and_exposes_ordering(self):
        index = pd.date_range('2026-01-01', periods=2, freq='h', tz='UTC')
        counts = pd.DataFrame({'pickups': [10., 3.], 'dropoffs': [10., 2.]}, index=index)
        output = simulate(counts, 0, 10, redistribution={1: 50})
        self.assertTrue(output.projected_bikes.between(0, 10).all())
        self.assertTrue(output.projected_docks.between(0, 10).all())
        pickup_first = output[output.ordering == 'pickups_first']
        return_first = output[output.ordering == 'returns_first']
        self.assertEqual(pickup_first.iloc[0].rejected_modeled_pickups, 10)
        self.assertEqual(return_first.iloc[0].rejected_modeled_pickups, 0)
        for _, group in output.groupby('ordering'):
            prior = 0.
            for position, row in enumerate(group.itertuples()):
                fulfilled_pickups = counts.iloc[position].pickups-row.rejected_modeled_pickups
                fulfilled_returns = counts.iloc[position].dropoffs-row.rejected_modeled_returns
                self.assertAlmostEqual(row.projected_bikes, prior+row.redistribution_applied+
                                       fulfilled_returns-fulfilled_pickups)
                prior = row.projected_bikes
        with self.assertRaises(ValueError):
            simulate(counts.assign(pickups=np.nan), 2, 10)

    def test_stale_closed_and_changed_station_status_blocks_simulation(self):
        now = pd.Timestamp('2026-10-18T12:00:00Z')
        epoch = int(now.timestamp())
        information = {'data': {'stations': [{'station_id': 'uuid', 'short_name': '123.01', 'capacity': 10}]}}
        state = {'station_id': 'uuid', 'num_bikes_available': 3, 'num_docks_available': 5,
                 'num_bikes_disabled': 2, 'num_docks_disabled': 0,
                 'is_installed': 1, 'is_renting': 1, 'is_returning': 1, 'last_reported': epoch}
        status = {'last_updated': epoch, 'ttl': 60, 'data': {'stations': [state]}}
        crosswalk = pd.DataFrame({'trip_station_id': ['123.01'], 'gbfs_station_id': ['uuid']})
        self.assertTrue(normalize(information, status, crosswalk, now).simulation_allowed.iloc[0])
        self.assertFalse(normalize(information, status, crosswalk, now+pd.Timedelta(minutes=6)).simulation_allowed.iloc[0])
        state['is_returning'] = 0
        self.assertFalse(normalize(information, status, crosswalk, now).simulation_allowed.iloc[0])
        crosswalk['gbfs_station_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'crosswalk changed'):
            normalize(information, status, crosswalk, now)

    def test_live_preview_covers_exactly_24_hours_without_observed_gap_updates(self):
        index = pd.date_range('2026-09-01', '2026-10-01', inclusive='left', freq='h', tz='UTC')
        panel = pd.DataFrame({'station_id': 'x', 'timestamp_utc': index, 'pickups': 2., 'dropoffs': 3.})
        output, info = current_forecasts(panel, 'x', pd.Timestamp('2026-10-18T12:30:00Z'), preview=True)
        self.assertEqual(output.duration_hours.sum(), 24)
        self.assertAlmostEqual(output.pickups.sum(), 48)
        self.assertGreater(info['observation_age_days'], 17)

    def test_model_comparison_uses_common_forecast_support(self):
        y = pd.Series(5., index=pd.date_range('2024-01-01', '2025-01-02',
            inclusive='left', freq='h', tz='America/New_York').tz_convert('UTC'))
        naive, _ = forecast_series(y, '5297.02', 'pickups', 'validation', 'seasonal_naive', days=1)
        other = naive.copy()
        other['family'] = 'daily_sarima'
        other.loc[0, 'prediction'] = np.nan
        # Add the matching direction so derived net-flow comparison uses the same boundary.
        pickup = pd.concat([naive, other], ignore_index=True)
        dropoff = pickup.assign(target='dropoffs')
        result = summarize(pd.concat([pickup, dropoff], ignore_index=True))
        overall = result[(result.dimension == 'overall') & (result.segment == 'all')]
        self.assertTrue(overall.scored_rows.eq(23).all())
        naive_pickups = overall[(overall.family == 'seasonal_naive') & (overall.target == 'pickups')].iloc[0]
        self.assertEqual(naive_pickups.own_available_rows, 24)


class DashboardBoundaries(unittest.TestCase):
    def test_dashboard_navigation_with_unavailable_live_feed_and_pending_models(self):
        from streamlit.testing.v1 import AppTest
        with patch('requests.get', side_effect=RuntimeError('Network access prohibited during startup')), \
             patch('dashboard_support.arimax_forecast', side_effect=ValueError('Model export pending')):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0, [str(x) for x in app.exception])
            self.assertTrue(any('BikeBalance' in element.value for element in app.title))
            self.assertEqual(app.metric[0].value, 'Unavailable')
            app.session_state['view'] = 'Next 24 hours'
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get('plotly_chart')), 3)
            app.checkbox[0].uncheck()
            app.checkbox[2].check().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('not ready' in element.value for element in app.info))
            self.assertEqual(len(app.get('plotly_chart')), 0)
            app.session_state['view'] = 'Test evaluation'
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get('plotly_chart')), 3)

    def test_shortfall_uses_usable_docks_and_rejects_stale_inventory(self):
        from dashboard_support import station_summary
        now = pd.Timestamp('2026-10-18T12:00:00Z')
        row = pd.Series({'reported_at': now, 'feed_at': now, 'capacity': 12,
                         'num_bikes_available': 1, 'num_docks_available': 7,
                         'operational': True, 'simulation_allowed': True})
        counts = pd.DataFrame({'pickups': [4., 0.], 'dropoffs': [0., 15.]})
        summary = station_summary(row, counts, now)
        self.assertEqual(summary['status'], 'Few bikes available')
        self.assertEqual(summary['bike_shortfall'], 3)
        self.assertEqual(summary['dock_shortfall'], 4)
        stale = station_summary(row, counts, now+pd.Timedelta(minutes=6))
        self.assertIsNone(stale['bike_shortfall'])

    def test_test_view_aligns_actuals_and_scores_full_common_test_support(self):
        from dashboard_support import test_frame, test_metrics, aggregate_test_metrics
        index = pd.date_range('2026-09-30T21:00:00Z', periods=3, freq='h')
        panel = pd.DataFrame({'station_id': 'x', 'partition': 'test', 'timestamp_utc': index,
                              'pickups': [2., 3., 5.], 'dropoffs': [1., 2., 3.], 'net_flow': [-1., -1., -2.]})
        rows = []
        for family in ['seasonal_naive', 'daily_sarima']:
            for target, values in [('pickups', [1., 4., 5.]), ('dropoffs', [1., 2., 3.])]:
                for t, value in zip(index, values):
                    rows.append({'station_id': 'x', 'partition': 'test', 'regime': 'daily_updated',
                                 'family': family, 'target': target, 'timestamp': t, 'prediction': value})
        actual, curves = test_frame(panel, pd.DataFrame(rows), 'x', 'daily_updated')
        curves['daily_sarima'].loc[index[0], 'net_flow'] = np.nan
        scores = test_metrics(actual, curves, 'net_flow')
        self.assertEqual(scores['seasonal_naive']['scored_rows'], 2)
        self.assertAlmostEqual(scores['seasonal_naive']['mae'], .5)
        predictions = pd.DataFrame(rows)
        second = predictions.assign(station_id='y', prediction=predictions.prediction+10)
        second.loc[(second.family == 'daily_sarima') & (second.target == 'pickups')
                   & second.timestamp.eq(index[0]), 'prediction'] = np.nan
        combined = pd.concat([predictions, second], ignore_index=True)
        both = pd.concat([panel, panel.assign(station_id='y')], ignore_index=True)
        aggregate = aggregate_test_metrics(both, combined, 'daily_updated')['pickups']
        self.assertEqual(aggregate['seasonal_naive']['scored_rows'], 5)
        self.assertAlmostEqual(aggregate['seasonal_naive']['mae'], 23/5)
        self.assertAlmostEqual(aggregate['seasonal_naive']['rmse'], np.sqrt(223/5))


if __name__ == '__main__':
    unittest.main()
