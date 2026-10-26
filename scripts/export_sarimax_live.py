"""Fit a live SARIMAX model per station and target and export its artifacts.

Parameters are estimated on 2024-02-01 through the last observed hour (the
specifications were frozen on 2025 validation; the February 2024 start matches
the research fits). The Kalman state after the last observation is saved so
the dashboard can forecast forward without refitting.

The weather-forecast model is fitted on the archived Open-Meteo `_previous_day2`
forecasts (data/processed/features/weather_forecast_hourly.parquet, from
scripts/collect_weather_forecasts.py and build_weather_forecast_features.py),
the same source the live dashboard queries.

Outputs: artifacts/<model>/{station}_{target}.npz and manifest.json

    ./.venv/bin/python scripts/export_sarimax_live.py --model sarimax_calendar --workers 7
    ./.venv/bin/python scripts/export_sarimax_live.py --model sarimax_weather_forecast --workers 7
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bikebalance import sarimax_calendar as sc  # noqa: E402
from bikebalance.data import DATA, digest, load_panel, series  # noqa: E402

FIT_START = pd.Timestamp("2024-02-01", tz="America/New_York")
WEATHER_FEATURES = ROOT / "data" / "processed" / "features" / "weather_forecast_hourly.parquet"
SELECTION = {
    "sarimax_calendar": "frozen on 2025 validation; 2026 test RMSE pickups 6.67 / drop-offs 6.59",
    "sarimax_weather_forecast": "frozen on 2025 validation; 2026 test RMSE pickups 6.37 / drop-offs 6.29",
}


def weather_for(index):
    frame = pd.read_parquet(WEATHER_FEATURES).set_index("timestamp_utc").reindex(index)
    if frame[["fc_temperature", "fc_wind_speed", "fc_wet"]].isna().any().any():
        raise ValueError("archived weather forecasts have gaps inside the fitting window")
    return sc.weather_columns(frame.fc_temperature, frame.fc_precipitation, frame.fc_wind_speed)


def fit_one(task):
    model, station, target = task
    y = series(load_panel(), station, target)
    y = y[y.index >= FIT_START.tz_convert("UTC")]
    weather = weather_for(y.index) if model == "sarimax_weather_forecast" else None
    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sc.make_model(model, y.to_numpy(), y.index, weather).fit(disp=False, maxiter=1000)
    state = result.filter_results
    path = sc.folder(model) / f"{station}_{target}.npz"
    np.savez(path, params=result.params, state=state.predicted_state[:, -1], state_cov=state.predicted_state_cov[:, :, -1],
             param_names=np.array(result.model.param_names))
    return task, {"artifact": path.name, "sha256": digest(path), "last_observation": str(y.index.max()),
                  "fit_start": str(y.index.min()), "observations": int(y.notna().sum()),
                  "converged": bool(result.mle_retvals.get("converged", False)), "aic": float(result.aic),
                  "fit_seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sc.MODELS, required=True)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    sc.folder(args.model).mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    tasks = [(args.model, s, t) for t in ("pickups", "dropoffs") for s in sorted(panel.station_id.unique())]
    models = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fit_one, task) for task in tasks]):
            (_, station, target), record = future.result()
            models[f"{station}/{target}"] = record
            print(f"{station} {target}: converged={record['converged']} in {record['fit_seconds']}s", flush=True)
    manifest = {
        "family": args.model, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_sha256": digest(DATA / "modeling_dataset.parquet"),
        "specification": {"order": sc.ORDER, "seasonal_order": sc.SEASONAL_ORDER,
                          "weekly_harmonics": sc.WEEKLY_HARMONICS, "offday_daily_harmonics": sc.OFFDAY_DAILY_HARMONICS,
                          "weather": list(sc.WEATHER_VARIABLES) if args.model == "sarimax_weather_forecast" else None},
        "selection": SELECTION[args.model] + " (data/story/test_leaderboard.csv)",
        "models": dict(sorted(models.items())),
    }
    (sc.folder(args.model) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("not converged:", [k for k, v in models.items() if not v["converged"]])


if __name__ == "__main__":
    main()
