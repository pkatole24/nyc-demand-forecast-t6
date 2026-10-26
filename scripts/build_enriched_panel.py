"""Join every predictor feature table to the hash-verified base panel.

Each join is a many-to-one left join on the explicit America/New_York local
date (plus station_id for station-level tables), or on timestamp_utc for the
citywide hourly weather forecasts, and must preserve row count,
station/UTC key uniqueness and target values. The base file is never modified.

Inputs:  data/modeling/modeling_dataset.parquet
         data/processed/features/*.parquet (build_*_features.py)
Outputs: data/processed/modeling_enriched.parquet
         data/processed/modeling_enriched.receipt.json

    ./.venv/bin/python scripts/build_enriched_panel.py
"""

import json
from datetime import datetime, timezone

import pandas as pd

from predictor_common import BASE_PATH, FEATURES, ROOT, join_features, load_base, sha256

TABLES = {
    "collisions_daily": ["station_id", "local_date"],
    "congestion_daily": ["local_date"],
    "subway_daily": ["station_id", "local_date"],
    "events_daily": ["station_id", "local_date"],
    "weather_forecast_hourly": ["timestamp_utc"],
}
OUT_PATH = ROOT / "data" / "processed" / "modeling_enriched.parquet"


def main():
    panel = load_base()
    joins = {}
    for name, keys in TABLES.items():
        path = FEATURES / f"{name}.parquet"
        receipt = json.loads((FEATURES / f"{name}.receipt.json").read_text())
        if sha256(path) != receipt["output_sha256"]:
            raise ValueError(f"{name} does not match its receipt; rebuild it")
        features = pd.read_parquet(path)
        panel, unmatched = join_features(panel, features, keys)
        added = [c for c in features.columns if c not in keys]
        joins[name] = {
            "keys": keys,
            "unmatched_rows": unmatched,
            "feature_sha256": receipt["output_sha256"],
            # A matched key with a null predictor is still missing.
            "available_by_partition": {
                col: {
                    part: {"rows": int(len(g)), "available": int(g.notna().sum())}
                    for part, g in panel.groupby("partition")[col]
                }
                for col in added
            },
        }

    panel.to_parquet(OUT_PATH, index=False)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_sha256": sha256(BASE_PATH),
        "rows": len(panel),
        "checks": "per join: many_to_one keys, row count, unique station/UTC keys, target equality",
        "joins": joins,
        "output_sha256": sha256(OUT_PATH),
    }
    OUT_PATH.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    rows = []
    for name, info in joins.items():
        for col, parts in info["available_by_partition"].items():
            rows.append({"table": name, "feature": col, "unmatched": info["unmatched_rows"],
                         **{p: f"{v['available'] / v['rows']:.1%}" for p, v in parts.items()}})
    print(f"rows {len(panel):,}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
