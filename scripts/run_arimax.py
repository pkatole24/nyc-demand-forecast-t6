import os
for variable in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[variable] = '1'

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from bikebalance.data import (ARTIFACTS, DATA, TARGETS, TZ, digest, experiment_id,
                              load_panel, series, write_json, write_parquet)
from bikebalance.models import Specification
from bikebalance.selection import score
from bikebalance.backtest import forecast_series
from bikebalance.weather import PastWeather


def run():
    started = time.time()
    panel = load_panel()
    engine = PastWeather(panel)
    candidates = [Specification((1, 0, 0), (0, 0, 0, 0))]
    config = {'family': 'weather_arimax', 'weather_lag_hours': 48,
              'availability_buffer_hours': 6, 'schedule': 'monthly',
              'candidates': [s.to_dict() for s in candidates],
              'driver_sha256': digest(Path(__file__)),
              'weather_policy': 'Fixed zero encoding with distinct numeric-missing/trace/provider-zero flags; no future weather'}
    key, provenance = experiment_id(config)
    folder = ARTIFACTS/'arimax'/key
    folder.mkdir(parents=True, exist_ok=True)
    status_path = ARTIFACTS/'arimax_status.json'
    completed = 0
    def status(stage, **extra):
        write_json(status_path, {'status': 'running', 'stage': stage, 'run': str(folder),
                    'order': [1, 0, 0], 'refit_schedule': 'monthly',
                    'started_at': datetime.fromtimestamp(started, timezone.utc),
                    'updated_at': datetime.now(timezone.utc), 'elapsed_seconds': time.time()-started,
                    'completed_series': completed, 'total_series': 16, **extra})
    write_json(folder/'run.json', {'status': 'running', **provenance})
    chosen = {}
    pieces = []
    manifest = {'base_sha256': provenance['base'], 'models': {}, 'weather_lag_hours': 48}
    try:
        for station in sorted(panel.station_id.unique()):
            for target in TARGETS:
                identity = f'{station}/{target}'
                receipt = folder/f'{station}_{target}_selection.json'
                y = series(panel, station, target)
                status('validation', series=identity)
                if receipt.exists():
                    entry = json.loads(receipt.read_text())
                    for item in entry['candidates']:
                        if digest(folder/item['artifact']) != item['sha256']:
                            raise ValueError('ARIMAX checkpoint hash mismatch')
                else:
                    results = []
                    for spec in candidates:
                        tick = time.time()
                        status('validation', series=identity, candidate=spec.label)
                        frame, logs = forecast_series(y, station, target, 'validation',
                            'weather_arimax', spec, 'monthly', engine=engine)
                        path = folder/f'{station}_{target}_{spec.order[0]}_validation.parquet'
                        write_parquet(path, frame)
                        results.append({**score(frame), 'specification': spec.to_dict(), 'schedule': 'monthly',
                            'artifact': path.name, 'sha256': digest(path), 'seconds': time.time()-tick,
                            'failed_origins': sum(r['status'] != 'ok' for r in logs), 'origins': logs})
                        write_json(path.with_suffix('.json'), results[-1])
                        print(f'ARIMAX validation {identity} {spec.order}: {time.time()-tick:.1f}s MAE={results[-1]["mae"]}', flush=True)
                    eligible = [r for r in results if r['mae'] is not None and r['scored'] > .99*r['eligible']]
                    if not eligible:
                        raise ValueError(f'Insufficient ARIMAX validation coverage for {identity}')
                    best = min(r['mae'] for r in eligible)
                    winner = min([r for r in eligible if r['mae'] <= best*1.01],
                                 key=lambda r: (r['rmse'], r['specification']['order'][0], r['seconds']))
                    entry = {'winner': winner, 'candidates': results}
                    write_json(receipt, entry)
                chosen[identity] = entry['winner']
                completed += 1
        # All orders are frozen using 2025 before any 2026 scoring.
        write_json(folder/'selection.json', {'status': 'complete', 'selection_year': 2025,
                   'base_sha256': provenance['base'], 'series': chosen, **provenance})
        for station in sorted(panel.station_id.unique()):
            for target in TARGETS:
                identity = f'{station}/{target}'
                y = series(panel, station, target)
                selected = chosen[identity]
                spec = Specification.from_dict(selected['specification'])
                for partition in ['validation', 'test']:
                    for regime in ['daily_updated', 'publication_lag']:
                        status('backtest', series=identity, partition=partition, regime=regime)
                        path = folder/f'{partition}_{station}_{target}_{regime}.parquet'
                        if partition == 'validation' and regime == 'daily_updated':
                            frame = pd.read_parquet(folder/selected['artifact'])
                        elif path.exists() and path.with_suffix('.json').exists():
                            saved = json.loads(path.with_suffix('.json').read_text())
                            if saved['sha256'] != digest(path):
                                raise ValueError('ARIMAX backtest hash mismatch')
                            frame = pd.read_parquet(path)
                        else:
                            frame, logs = forecast_series(y, station, target, partition,
                                'weather_arimax', spec, 'monthly', regime, engine=engine)
                            write_parquet(path, frame)
                            write_json(path.with_suffix('.json'), {'sha256': digest(path), 'origins': logs})
                        pieces.append(frame)
                status('export', series=identity)
                result, messages = engine.fit(y, spec)
                if not result.mle_retvals.get('converged'):
                    result, retry = engine.fit(y, spec, maxiter=200)
                    messages.extend(retry)
                if not result.mle_retvals.get('converged'):
                    raise RuntimeError(f'ARIMAX export did not converge: {identity}')
                path = folder/f'{station}_{target}.pickle'
                temporary = path.with_suffix('.tmp')
                result.save(temporary)
                temporary.replace(path)
                manifest['models'][identity] = {'artifact': path.name, 'sha256': digest(path),
                    'specification': spec.to_dict(), 'last_observation': str(y.index.max()), 'warnings': messages}
        forecasts = pd.concat(pieces, ignore_index=True)
        write_parquet(folder/'forecasts.parquet', forecasts)
        from bikebalance.evaluation import summarize
        summarize(forecasts).to_csv(folder/'metrics.csv', index=False)
        write_json(folder/'manifest.json', manifest)
        write_json(folder/'run.json', {'status': 'complete', 'rows': len(forecasts),
                   'forecasts_sha256': digest(folder/'forecasts.parquet'), **provenance})
        write_json(ARTIFACTS/'latest_arimax_run.json', {'path': folder.relative_to(ARTIFACTS).as_posix()})
        write_json(status_path, {'status': 'complete', 'elapsed_seconds': time.time()-started,
                   'completed_series': 16, 'run': str(folder)})
    except Exception as exc:
        write_json(status_path, {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                   'elapsed_seconds': time.time()-started, 'completed_series': completed, 'run': str(folder)})
        raise


if __name__ == '__main__':
    run()
