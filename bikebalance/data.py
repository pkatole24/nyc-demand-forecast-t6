from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/modeling'
ARTIFACTS = ROOT / 'artifacts'
TZ = 'America/New_York'
TARGETS = ('pickups', 'dropoffs')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_panel(folder=DATA):
    folder = Path(folder)
    path = folder / 'modeling_dataset.parquet'
    expected = json.loads((folder / 'manifest.json').read_text())['outputs_sha256'][path.name]
    if digest(path) != expected:
        raise ValueError('Base dataset SHA-256 mismatch; restore the committed base.')
    frame = pd.read_parquet(path)
    frame['station_id'] = frame.station_id.astype('string')
    frame['timestamp_utc'] = frame.timestamp.dt.tz_convert('UTC')
    if frame.duplicated(['station_id', 'timestamp_utc']).any():
        raise ValueError('Duplicate station/UTC keys')
    for _, group in frame.groupby('station_id'):
        index = pd.DatetimeIndex(group.timestamp_utc.sort_values())
        if not index.equals(pd.date_range(index.min(), index.max(), freq='h')):
            raise ValueError('The station grid is not consecutive hourly instants')
    if not frame.net_flow.equals(frame.dropoffs - frame.pickups):
        raise ValueError('Target net-flow arithmetic failed')
    expected_partition = frame.timestamp.dt.year.map(
        {2024: 'initial_train', 2025: 'validation', 2026: 'test'})
    if not frame.partition.eq(expected_partition).all():
        raise ValueError('Chronological partitions changed')
    return frame


def series(frame, station, target):
    selected = frame[frame.station_id.eq(station)].set_index('timestamp_utc')[target].sort_index()
    return selected.astype(float).asfreq('h')


def local_day(origin):
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError('Forecast origin must have a timezone')
    local = origin.tz_convert(TZ)
    if local != local.normalize():
        raise ValueError('Historical origins must be New York midnight')
    return pd.date_range(local, local + pd.DateOffset(days=1), freq='h',
                         inclusive='left').tz_convert('UTC')


def origins(partition):
    bounds = {'validation': ('2025-01-01', '2026-01-01'),
              'test': ('2026-01-01', '2026-10-01')}
    start, stop = bounds[partition]
    return pd.date_range(start, stop, freq='D', tz=TZ, inclusive='left')


def publication_cutoff(origin, availability):
    """Use current object revision times; these do not reconstruct original releases."""
    origin = pd.Timestamp(origin).tz_convert('UTC')
    released = []
    for item in availability['months']:
        if item['available'] and pd.Timestamp(item['last_modified']) <= origin:
            end = (pd.Timestamp(item['month'] + '01', tz=TZ) + pd.offsets.MonthBegin(1))
            if end.tz_convert('UTC') <= origin:
                released.append(end.tz_convert('UTC'))
    return max(released) if released else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def experiment_id(config):
    source = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'bikebalance').glob('*.py'))}
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in
                ('pandas', 'numpy', 'statsmodels', 'scipy')}
    value = {'config': config, 'source': source, 'versions': versions,
             'base': digest(DATA / 'modeling_dataset.parquet')}
    key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
    return key, value
