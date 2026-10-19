import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(station, target, years, period, output):
    import psutil
    from bikebalance.data import load_panel, series
    from bikebalance.models import Specification, fit, predict, update
    import pandas as pd
    clock = time.perf_counter()
    try:
        y = series(load_panel(), station, target)
        cutoff = pd.Timestamp(f'{2024 + years}-01-01', tz='America/New_York').tz_convert('UTC')
        history = y[y.index < cutoff]
        spec = Specification((1, 0, 0), (1, 0, 0, period))
        result, messages = fit(history, spec, maxiter=50)
        converged = bool(result.mle_retvals.get('converged', False))
        fit_seconds = time.perf_counter() - clock
        update_start = time.perf_counter()
        # The two-year benchmark has available 2026 observations for this measurement.
        for day in range(30):
            start = cutoff + pd.Timedelta(days=day)
            index = pd.date_range(start, periods=24, freq='h')
            predict(result, spec, index)
            result = update(result, y.reindex(index), spec)
        output.put({'status': 'completed', 'fit_seconds': fit_seconds,
                    'update_forecast_seconds_per_day': (time.perf_counter()-update_start)/30,
                    'converged': converged,
                    'warnings': messages, 'peak_working_set_bytes':
                        psutil.Process().memory_info()._asdict().get('peak_wset',
                            psutil.Process().memory_info().rss)})
    except Exception as exc:
        output.put({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--quick', action='store_true', help='One series, one year; not the full gate')
    args = parser.parse_args()
    import psutil
    from bikebalance.data import ARTIFACTS, write_json, digest, DATA
    cases = [(s, t, y, m) for s in (['6197.08'] if args.quick else ['6197.08', '5696.03'])
             for t in (['pickups'] if args.quick else ['pickups', 'dropoffs'])
             for y in ([1] if args.quick else [1, 2]) for m in [24, 168]]
    hardware = {'platform': platform.platform(), 'processor': platform.processor(),
                'logical_cpus': os.cpu_count(), 'physical_cpus': psutil.cpu_count(logical=False),
                'ram_bytes': psutil.virtual_memory().total,
                'available_ram_bytes': psutil.virtual_memory().available,
                'workers': 1, 'blas_threads': 1, 'gpu_operations': []}
    try:
        hardware['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True,
                text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        hardware['nvidia_smi'] = 'Unavailable'
    rows = []
    path = ARTIFACTS / ('benchmark_quick.json' if args.quick else 'benchmark.json')
    for station, target, years, period in cases:
        channel = mp.Queue()
        process = mp.Process(target=worker, args=(station, target, years, period, channel))
        process.start()
        process.join(args.timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            value = {'status': 'timeout', 'lower_bound_seconds': args.timeout}
        else:
            try:
                value = channel.get(timeout=2)
            except queue.Empty:
                value = {'status': 'failed', 'error': f'Worker exited {process.exitcode}'}
        row = {'station_id': station, 'target': target, 'years': years, 'period': period, **value}
        rows.append(row)
        write_json(path, {'hardware': hardware, 'base_sha256': digest(DATA/'modeling_dataset.parquet'),
                         'timeout_seconds': args.timeout, 'full_gate': not args.quick,
                         'completed_cases': len(rows), 'expected_cases': len(cases), 'cases': rows})
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
