"""Zero-shot day-ahead forecasts from a pretrained AI time-series model (Chronos-Bolt).

Choice fixed before any scoring: amazon/chronos-bolt-small, no fine-tuning, the
last 2,048 hourly observations as context, mean forecast clipped at zero. One
series per station and target, like the per-station ARIMA models.

Regime matches every other model: at each New York midnight in 2025 and 2026
the context holds only hours before that origin (trips known through the
previous day), and the forecast covers that local day (23-25 hours). Missing
target hours stay NaN in the context; Chronos handles them natively. This
script only writes forecasts; scoring happens in build_story_tables.py (2025)
and evaluate_test.py (2026).

Run with the separate environment (PyTorch is not a dashboard dependency):
    python3 -m venv .venv-chronos && ./.venv-chronos/bin/pip install -r requirements-chronos.txt
    ./.venv-chronos/bin/python scripts/forecast_chronos.py

Output: data/processed/chronos/forecasts.parquet (+ receipt.json)
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from predictor_common import BASE_PATH, ROOT, TZ, load_base, sha256

MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT_HOURS = 2048
HORIZON = 25  # longest local day (fall-back)
BATCH = 256
OUT = ROOT / "data" / "processed" / "chronos"


def main():
    started = time.time()
    torch.manual_seed(0)
    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)
    panel = load_base().sort_values(["station_id", "timestamp_utc"])
    origins = pd.date_range("2025-01-01", "2026-09-30", freq="D", tz=TZ)

    jobs = []  # (station, target, origin index into series, day length)
    series = {}
    for (station, ), g in panel.groupby(["station_id"]):
        times = pd.DatetimeIndex(g["timestamp_utc"])
        for target in ("pickups", "dropoffs"):
            series[(station, target)] = (times, g[target].astype("float64").to_numpy(na_value=np.nan),
                                         g["partition"].to_numpy())
        for origin in origins:
            start = times.searchsorted(origin.tz_convert("UTC"))
            length = len(pd.date_range(origin, origin + pd.DateOffset(days=1), freq="h", inclusive="left"))
            for target in ("pickups", "dropoffs"):
                jobs.append((station, target, start, length))

    rows = []
    for i in range(0, len(jobs), BATCH):
        batch = jobs[i:i + BATCH]
        contexts = [torch.tensor(series[(s, t)][1][max(0, start - CONTEXT_HOURS):start], dtype=torch.float32)
                    for s, t, start, _ in batch]
        _, mean = pipeline.predict_quantiles(contexts, prediction_length=HORIZON, quantile_levels=[0.5])
        mean = np.clip(mean.numpy(), 0, None)
        for (s, t, start, length), forecast in zip(batch, mean):
            times, _, partition = series[(s, t)]
            rows.append(pd.DataFrame({"station_id": s, "timestamp": times[start:start + length], "target": t,
                                      "partition": partition[start:start + length], "chronos": forecast[:length]}))
        print(f"  {min(i + BATCH, len(jobs)):,}/{len(jobs):,} day-ahead forecasts", flush=True)

    forecasts = pd.concat(rows, ignore_index=True)
    if forecasts.duplicated(["station_id", "timestamp", "target"]).any():
        raise AssertionError("overlapping forecast days")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "forecasts.parquet"
    forecasts.to_parquet(path, index=False)
    (OUT / "receipt.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL_ID, "zero_shot": True,
        "context_hours": CONTEXT_HOURS, "point_forecast": "mean, clipped at zero", "origins": "New York midnight, 2025-01-01..2026-09-30",
        "base_sha256": sha256(BASE_PATH), "rows": len(forecasts), "output_sha256": sha256(path),
        "seconds": round(time.time() - started, 1),
    }, indent=2) + "\n")
    print(f"wrote {len(forecasts):,} hourly forecasts in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
