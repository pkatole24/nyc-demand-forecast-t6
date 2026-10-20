"""Shared paths, base loading and validated joins for predictor feature builds."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "data" / "modeling"
BASE_PATH = MODELING / "modeling_dataset.parquet"
RAW = ROOT / "data" / "raw" / "nyc_open_data"
FEATURES = ROOT / "data" / "processed" / "features"
TZ = "America/New_York"

# Every daily feature table covers the base panel's local dates.
PANEL_DATES = pd.date_range("2024-01-01", "2026-09-30", freq="D")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_base():
    manifest = json.loads((MODELING / "manifest.json").read_text())
    if sha256(BASE_PATH) != manifest["outputs_sha256"][BASE_PATH.name]:
        raise ValueError("Dataset hash mismatch: restore the committed base before using it.")
    base = pd.read_parquet(BASE_PATH)
    base["timestamp_utc"] = base["timestamp"].dt.tz_convert("UTC")
    # Explicit local calendar date for daily joins; fall-back days keep 25 rows.
    base["local_date"] = base["timestamp"].dt.tz_localize(None).dt.normalize()
    return base


def load_stations():
    return pd.read_csv(MODELING / "selected_stations.csv", dtype={"station_id": "string"})[
        ["station_id", "station_name", "lat", "lng"]
    ]


def read_raw(name):
    return pd.read_json(RAW / f"{name}.jsonl", lines=True, dtype=False)


def distance_m(lat1, lng1, lat2, lng2):
    """Equirectangular distance; error is negligible at sub-kilometre city scale."""
    mean_lat = np.radians((lat1 + lat2) / 2)
    dx = np.radians(lng2 - lng1) * np.cos(mean_lat) * 6_371_000
    dy = np.radians(lat2 - lat1) * 6_371_000
    return np.hypot(dx, dy)


def write_features(name, features, keys, sources, config):
    """Save a feature table with a provenance receipt after checking key uniqueness."""
    if features.duplicated(keys).any():
        raise ValueError(f"{name}: feature keys {keys} are not unique")
    FEATURES.mkdir(parents=True, exist_ok=True)
    path = FEATURES / f"{name}.parquet"
    features.to_parquet(path, index=False)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "keys": keys,
        "rows": len(features),
        "sources_sha256": {s: sha256(RAW / f"{s}.jsonl") for s in sources},
        "config": config,
        "missing_by_column": {c: int(features[c].isna().sum()) for c in features.columns if c not in keys},
        "output_sha256": sha256(path),
    }
    (FEATURES / f"{name}.receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    return path


def join_daily(panel, features, keys):
    """Left join a daily table (station-day or citywide day) with AGENTS.md checks."""
    if "local_date" not in keys:
        raise ValueError("daily features must join on local_date")
    if features.duplicated(keys).any():
        raise ValueError(f"feature keys {keys} are not unique")
    conflicts = (set(features.columns) - set(keys)) & set(panel.columns)
    if conflicts:
        raise ValueError(f"conflicting columns: {sorted(conflicts)}")

    joined = panel.merge(features, on=keys, how="left", validate="many_to_one", indicator=True)
    unmatched = int((joined["_merge"] != "both").sum())
    joined = joined.drop(columns="_merge")

    if len(joined) != len(panel):
        raise AssertionError("row count changed in join")
    if joined.duplicated(["station_id", "timestamp_utc"]).any():
        raise AssertionError("station/UTC keys are no longer unique")
    for target in ("pickups", "dropoffs", "net_flow"):
        if not joined[target].equals(panel[target]):
            raise AssertionError(f"{target} changed in join")
    return joined, unmatched
