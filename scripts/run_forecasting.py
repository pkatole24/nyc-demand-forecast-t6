import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Offline chronological experiments; no data downloads')
    parser.add_argument('stage', choices=['baseline', 'diagnose', 'select', 'backtest', 'deploy',
                                        'all', 'check-releases'])
    parser.add_argument('--days', type=int, help='Smoke run only; never freezes specifications')
    args = parser.parse_args()
    if args.days is not None and args.days < 1:
        parser.error('--days must be positive')
    from bikebalance.data import load_panel, ARTIFACTS
    panel = load_panel()
    if args.stage == 'check-releases':
        from bikebalance.releases import check_releases
        print(json.dumps(check_releases(), indent=2))
    elif args.stage == 'all':
        if args.days is not None:
            parser.error('Use individual stages for smoke runs; all requires the full protocol')
        from bikebalance.backtest import run_backtest
        from bikebalance.selection import diagnose_panel, select
        from bikebalance.live import train_deployment
        from bikebalance.data import write_json
        from datetime import datetime, timezone
        status = ARTIFACTS/'pipeline_status.json'
        stages = [('baseline', lambda: run_backtest(panel, ['seasonal_naive'])),
                  ('diagnostics', lambda: diagnose_panel(panel)),
                  ('selection', lambda: select(panel)),
                  ('backtest', lambda: run_backtest(panel,
                      ['seasonal_naive'] +
                       json.loads((ARTIFACTS/'frozen_selection.json').read_text())['fitted_families'],
                      selection=json.loads((ARTIFACTS/'frozen_selection.json').read_text()))),
                  ('deployment', lambda: train_deployment(panel))]
        for name, action in stages:
            write_json(status, {'status': 'running', 'stage': name, 'updated_at': datetime.now(timezone.utc)})
            print(f'STAGE {name}', flush=True)
            try:
                action()
            except Exception as exc:
                write_json(status, {'status': 'failed', 'stage': name,
                                    'error': f'{type(exc).__name__}: {exc}'})
                raise
        write_json(status, {'status': 'complete', 'updated_at': datetime.now(timezone.utc)})
    elif args.stage == 'baseline':
        from bikebalance.backtest import run_backtest
        print(run_backtest(panel, ['seasonal_naive'], days=args.days))
    elif args.stage == 'diagnose':
        from bikebalance.selection import diagnose_panel
        print(diagnose_panel(panel))
    elif args.stage == 'select':
        from bikebalance.selection import select
        print(select(panel, days=args.days))
    elif args.stage == 'deploy':
        from bikebalance.live import train_deployment
        print(train_deployment(panel))
    else:
        from bikebalance.backtest import run_backtest
        frozen = json.loads((ARTIFACTS/'frozen_selection.json').read_text())
        if frozen['status'] != 'complete' or frozen['validation_days'] != 365:
            raise ValueError('Full 2025 selection must complete before fitted-model test evaluation')
        from bikebalance.data import DATA, digest
        if frozen['base_sha256'] != digest(DATA/'modeling_dataset.parquet'):
            raise ValueError('Frozen selection belongs to a different base')
        print(run_backtest(panel, ['seasonal_naive'] + frozen['fitted_families'],
                           selection=frozen, days=args.days))


if __name__ == '__main__':
    main()
