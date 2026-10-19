import json
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

from .backtest import forecast_series
from .data import ARTIFACTS, DATA, TARGETS, TZ, digest, experiment_id, series, write_json, write_parquet
from .models import Specification, fit


def diagnostics(y):
    observed = y.dropna()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = {'adf_pvalue': float(adfuller(observed, maxlag=24, autolag='AIC')[1]),
                 'kpss_pvalue': float(kpss(observed, regression='c', nlags='auto')[1]),
                 'acf': acf(y, nlags=336, missing='conservative').tolist(),
                 'pacf': pacf(observed, nlags=168, method='ywm').tolist(),
                 'pacf_note': 'Missing values removed only for PACF/ADF/KPSS diagnostics; fitting grid is intact.',
                 'missing': int(y.isna().sum())}
    value['warnings'] = [str(w.message) for w in caught]
    value['difference_diagnostics'] = {}
    for lag in [1, 24, 168]:
        different = y.diff(lag).dropna()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            value['difference_diagnostics'][str(lag)] = {
                'adf_pvalue': float(adfuller(different, maxlag=24, autolag='AIC')[1]),
                'kpss_pvalue': float(kpss(different, regression='c', nlags='auto')[1]),
                'variance_ratio': float(different.var()/observed.var())}
    return value


def candidate_specs(y, period):
    # Diagnose seasonal differencing on the training series; keep both when evidence is mixed.
    correlation = y.autocorr(period)
    differences = [0, 1] if correlation > .3 else [0]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        nonseasonal_difference = int(adfuller(y.dropna(), maxlag=24, autolag='AIC')[1] > .05
                                    and kpss(y.dropna(), nlags='auto')[1] < .05)
    return [Specification((p, nonseasonal_difference if d == 0 else 0, q), (1, d, 0, period))
            for d in differences for p, q in [(1, 0), (0, 1)]]


def diagnose_panel(panel):
    folder = ARTIFACTS/'diagnostics'
    folder.mkdir(parents=True, exist_ok=True)
    result = {'base_sha256': digest(DATA/'modeling_dataset.parquet'), 'training_year': 2024, 'series': {}}
    for station in sorted(panel.station_id.unique()):
        for target in TARGETS:
            y = series(panel, station, target)
            y = y[y.index.tz_convert(TZ).year == 2024]
            result['series'][f'{station}/{target}'] = diagnostics(y)
    write_json(folder/'diagnostics.json', result)
    return folder


def score(frame):
    valid = frame.prediction.notna() & frame.actual.notna()
    error = frame.loc[valid, 'prediction'] - frame.loc[valid, 'actual']
    local = frame.timestamp.dt.tz_convert(TZ)
    peak = valid & (local.dt.dayofweek < 5) & local.dt.hour.isin([7, 8, 9, 16, 17, 18])
    return {'mae': float(error.abs().mean()) if len(error) else None,
            'rmse': float(np.sqrt((error**2).mean())) if len(error) else None,
            'peak_mae': float((frame.loc[peak, 'prediction']-frame.loc[peak, 'actual']).abs().mean())
                        if peak.any() else None,
            'scored': int(valid.sum()), 'eligible': len(frame)}


def choose_weekly_family():
    path = ARTIFACTS/'benchmark.json'
    if not path.exists():
        raise ValueError('Run the full hardware benchmark before selecting Model 3')
    report = json.loads(path.read_text())
    if not report['full_gate'] or report['completed_cases'] != report['expected_cases']:
        raise ValueError('Full benchmark is incomplete')
    if report['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
        raise ValueError('Benchmark base hash changed')
    weekly = [r for r in report['cases'] if r['period'] == 168]
    bounds = [r.get('fit_seconds', r.get('lower_bound_seconds', float('inf'))) for r in weekly]
    # Two weekly validation finalists (53 fits each), one monthly comparison (12),
    # and the weekly test protocol (40), independently for each of 16 series.
    weekly_family_fits = 16 * (2*53 + 12 + 40)
    projected = max(bounds) * weekly_family_fits / 3600
    ram = report['hardware']['available_ram_bytes'] - 3 * 2**30
    fits = all(r['status'] == 'completed' and r.get('converged')
               and r['peak_working_set_bytes'] < ram for r in weekly)
    family = 'weekly_sarima' if fits and projected <= 48 else 'daily_weekly_fourier'
    return family, {'family': family, 'projected_refit_hours_at_pilot_limit': projected
                   if np.isfinite(projected) else None,
                   'censored_weekly_fit_count': sum(r['status'] == 'timeout' for r in weekly),
                   'projected_weekly_family_fit_count': weekly_family_fits,
                   'reason': 'All weekly pilots converged within runtime/memory gate' if family == 'weekly_sarima'
                   else 'At least one weekly pilot failed convergence, time or memory gate; approved alternative',
                   'benchmark_sha256': digest(path),
                   'estimate_limit': 'Timeouts are lower bounds, not completed fit timings. '
                                     'Representative fits extrapolated; candidate search extra.'}


def select(panel, days=None):
    families = ['daily_sarima']
    config = {'task': 'selection', 'fitted_families': families, 'validation_days': days,
              'refit_schedule': 'monthly',
              'candidate_policy': 'AR1/MA1, seasonal_AR1; D=0/1 when seasonal correlation>.3; d=1 only for ADF>.05 and KPSS<.05 without D'}
    key, provenance = experiment_id(config)
    folder = ARTIFACTS/'selection'/key
    folder.mkdir(parents=True, exist_ok=True)
    selection = {'status': 'partial' if days else 'complete', 'validation_days': days or 365,
                 'base_sha256': digest(DATA/'modeling_dataset.parquet'),
                 'fitted_families': families, 'refit_schedule': 'monthly',
                 'series': {}, 'provenance': provenance}
    write_json(folder/'progress.json', {'status': 'running', **provenance})
    for station in sorted(panel.station_id.unique()):
        for target in TARGETS:
            identity = f'{station}/{target}'
            receipt = folder/f'{station}_{target}.json'
            if receipt.exists():
                selection['series'][identity] = json.loads(receipt.read_text())
                continue
            y = series(panel, station, target)
            training = y[y.index.tz_convert(TZ).year == 2024]
            diagnostic = diagnostics(training)
            write_json(folder/f'{station}_{target}_diagnostics.json', diagnostic)
            chosen = {}
            for family in families:
                family_receipt = folder/f'{station}_{target}_{family}_selected.json'
                if family_receipt.exists():
                    cached = json.loads(family_receipt.read_text())
                    if digest(ARTIFACTS/cached['validation_artifact']) != cached['validation_sha256']:
                        raise ValueError(f'Validation artifact hash mismatch: {identity}/{family}')
                    chosen[family] = cached
                    continue
                candidates = candidate_specs(training, 168 if family == 'weekly_sarima' else 24)
                if family == 'daily_weekly_fourier':
                    candidates = [Specification(p.order, p.seasonal_order, k)
                                  for p in candidates for k in [1, 2, 3]]
                fits = []
                for spec in candidates:
                    clock = time.perf_counter()
                    try:
                        result, messages = fit(training, spec)
                        converged = bool(result.mle_retvals.get('converged', False))
                        record = {'specification': spec.to_dict(), 'aicc': float(result.aicc)
                                  if np.isfinite(result.aicc) else None,
                                  'converged': converged, 'seconds': time.perf_counter()-clock,
                                  'parameters': len(result.params), 'warnings': messages}
                        if np.isfinite(result.aicc) and converged:
                            fits.append(record)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        record = {'specification': spec.to_dict(), 'error': str(exc),
                                  'seconds': time.perf_counter()-clock}
                    write_json(folder/f'{station}_{target}_{family}_{spec.label}_fit.json', record)
                shortlist = sorted(fits, key=lambda r: r['aicc'])[:2]
                if not shortlist:
                    raise RuntimeError(f'No converged {family} candidate for {identity}')
                validated = []
                for candidate in shortlist:
                    spec = Specification.from_dict(candidate['specification'])
                    frame, logs = forecast_series(y, station, target, 'validation', family,
                                                  spec, 'monthly', days=days)
                    write_parquet(folder/f'{station}_{target}_{family}_{spec.label}_validation.parquet', frame)
                    validated.append({**candidate, **score(frame), 'schedule': 'monthly',
                                      'runtime': sum(r['seconds'] for r in logs),
                                      'failed_origins': sum(r['status'] != 'ok' for r in logs),
                                      'refit_warnings': [r for r in logs if r['warnings']]})
                valid = [v for v in validated if v['mae'] is not None and v['scored'] > .99*v['eligible']]
                if not valid:
                    raise RuntimeError(f'Insufficient validation forecast coverage: {identity}/{family}')
                best_mae = min(v['mae'] for v in valid)
                winner = min([v for v in valid if v['mae'] <= best_mae*1.01],
                             key=lambda r: (r['rmse'], r['parameters'], r['runtime']))
                chosen[family] = dict(winner)
                chosen[family]['candidate_validation'] = validated
                selected_spec = Specification.from_dict(winner['specification'])
                selected_frame = pd.read_parquet(
                    folder/f'{station}_{target}_{family}_{selected_spec.label}_validation.parquet')
                selected_path = folder/f'{station}_{target}_{family}_selected_validation.parquet'
                write_parquet(selected_path, selected_frame)
                chosen[family]['validation_artifact'] = selected_path.relative_to(ARTIFACTS).as_posix()
                chosen[family]['validation_sha256'] = digest(selected_path)
                # Standardized one-step residuals are retained by the final low-memory filter.
                residual_fit, _ = fit(training, Specification.from_dict(winner['specification']))
                residual = np.asarray(residual_fit.resid)[max(336, residual_fit.loglikelihood_burn):]
                residual = residual[np.isfinite(residual)]
                lagtest = acorr_ljungbox(residual, lags=[24, 168], model_df=2, return_df=True)
                chosen[family]['residual_diagnostics'] = {
                    'mean': float(residual.mean()), 'std': float(residual.std()),
                    'acf': acf(residual, nlags=168).tolist(),
                    'ljung_box': lagtest.reset_index().replace({np.nan: None}).to_dict('records')}
                write_json(family_receipt, chosen[family])
            naive, _ = forecast_series(y, station, target, 'validation', 'seasonal_naive', days=days)
            chosen['seasonal_naive'] = score(naive)
            eligible = {family: result['mae'] for family, result in chosen.items()
                        if isinstance(result, dict) and result.get('mae') is not None}
            chosen['deployment_family'] = min(eligible, key=eligible.get)
            selection['series'][identity] = chosen
            write_json(receipt, chosen)
            print(f'Selected {identity}: {chosen["deployment_family"]}', flush=True)
    write_json(folder/'selection.json', selection)
    write_json(folder/'progress.json', {'status': selection['status'], **provenance})
    if days is None:
        write_json(ARTIFACTS/'frozen_selection.json', selection)
    return folder
