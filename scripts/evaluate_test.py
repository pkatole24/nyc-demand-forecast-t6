"""Frozen 2026 test evaluation of the models chosen on 2025, beside the team's models.

Model choices were fixed on the 2025 validation year and are written to
data/story/frozen_choices.json before any test forecast is scored:

- Poisson regression baseline (evaluate_predictors.py, no added regressor):
  refit before each 2026 month on all earlier usable rows, as in validation.
- SARIMAX(1,0,1)(1,1,1)24 + weekly calendar (evaluate_sarimax.py): parameters
  re-estimated once on 2024-02-01..2025-12-31, then the Kalman state runs
  causally through 2026 with forecasts from each local midnight.

Regime: day-ahead from local midnight with trips known through the previous
day, the team's `daily_updated` regime. Every model, including the team's
last-week baseline, SARIMA and ARIMAX from artifacts/, is scored on the same
station-hours: actual observed and every model's forecast present.

Inputs:  data/processed/modeling_enriched.parquet, artifacts/ team forecasts
Outputs: data/story/frozen_choices.json, data/story/test_leaderboard.csv,
         data/processed/test_eval/ (forecasts, DM tests, SARIMAX cache)

    ./.venv/bin/python scripts/evaluate_test.py --workers 7
"""

import os

# One BLAS thread per process; SARIMAX parallelism is across station series.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm

from evaluate_predictors import PANEL_PATH, TARGETS, add_demand_lags, calendar_design, dm_test, fit_predict, verify_panel
from evaluate_sarimax import COLUMNS, FIT_START, SPECS, day_ahead, exog_matrix
from predictor_common import ROOT, TZ

ARTIFACTS = ROOT / "artifacts"
STORY = ROOT / "data" / "story"
OUT = ROOT / "data" / "processed" / "test_eval"
CACHE = OUT / "cache"
TEST_START = pd.Timestamp("2026-01-01")
SARIMAX_SPEC = "sarimax_calendar"
LABELS = {
    "seasonal_naive": "Last-week baseline",
    "daily_sarima": "Daily pattern (SARIMA)",
    "weather_arimax": "Weather + calendar (ARIMAX)",
    "sarimax_calendar": "SARIMAX + weekly calendar",
    "poisson": "Poisson regression",
}


def freeze(panel_hash):
    """Record the choices before scoring; an existing record must not change."""
    record = {
        "frozen_on_validation": "2025",
        "models": {
            "poisson": "Poisson GLM baseline: station x hour, off-day x hour, weekday, month, log1p lags 24h/168h/28-day same-hour mean; refit monthly",
            "sarimax_calendar": f"SARIMAX{SPECS[SARIMAX_SPEC]['order']}{SPECS[SARIMAX_SPEC]['seasonal']} + weekly Fourier (K=4) + off-day x daily Fourier (K=3); estimated 2024-02-01..2025-12-31",
        },
        "regime": "daily_updated: day-ahead from New York midnight, trips known through the previous day",
        "panel_sha256": panel_hash,
    }
    path = STORY / "frozen_choices.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if {k: saved[k] for k in record} != record:
            raise ValueError("frozen choices differ from the saved record; test results would no longer be a frozen evaluation")
        return
    STORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**record, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


def poisson_forecasts(panel):
    base_X = calendar_design(panel)
    month = panel["timestamp"].dt.tz_localize(None).dt.to_period("M")
    frames = []
    for target in TARGETS:
        lags = add_demand_lags(panel, target)
        y = panel[target].astype("float64").to_numpy(na_value=np.nan)
        usable = ~np.isnan(y) & lags.notna().all(axis=1).to_numpy()
        X = pd.concat([base_X, lags], axis=1).to_numpy(dtype="float64")
        # Forecast every test hour whose lags are known; the target need not be.
        known = lags.notna().all(axis=1).to_numpy()
        pred = np.full(len(y), np.nan)
        params = None
        for m in pd.period_range("2026-01", "2026-09", freq="M"):
            train = usable & (month < m).to_numpy()
            score = known & (month == m).to_numpy()
            pred[score], params = fit_predict(X, y, train, score, params)
            print(f"  poisson {target} {m}: refit on {train.sum():,} rows", flush=True)
        test = (panel["partition"] == "test").to_numpy()
        frames.append(pd.DataFrame({"station_id": panel["station_id"].to_numpy()[test],
                                    "timestamp": panel["timestamp_utc"].to_numpy()[test],
                                    "target": target, "poisson": pred[test]}))
    return pd.concat(frames, ignore_index=True)


def sarimax_one(task):
    station, target = task
    cache = CACHE / f"{station}_{target}.parquet"
    if cache.exists():
        return task, "cached"
    spec = SPECS[SARIMAX_SPEC]
    panel = pd.read_parquet(PANEL_PATH, columns=COLUMNS, filters=[("station_id", "==", station)])
    s = panel[panel["local_date"] >= FIT_START].sort_values("timestamp_utc").reset_index(drop=True)
    y = s[target].astype("float64").to_numpy(na_value=np.nan)
    X, _ = exog_matrix(s, spec)
    train = (s["local_date"] < TEST_START).to_numpy()

    def model(rows):
        return sm.tsa.SARIMAX(y[rows], exog=X[rows], order=spec["order"], seasonal_order=spec["seasonal"], trend="n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model(train).fit(disp=False, maxiter=1000)
        full = model(np.ones(len(y), dtype=bool)).filter(fit.params)
    first = s.groupby("local_date").head(1)
    origins = first.index[first["local_date"] >= TEST_START].to_numpy()
    pred = day_ahead(full, origins, np.append(origins[1:], len(s)), len(s))
    pd.DataFrame({"station_id": station, "timestamp": s["timestamp_utc"].to_numpy()[~train], "target": target,
                  "sarimax_calendar": pred[~train], "converged": bool(fit.mle_retvals.get("converged", False))}
                 ).to_parquet(cache, index=False)
    return task, "fitted"


def team_forecasts():
    frames = []
    for pointer in ("latest_run.json", "latest_arimax_run.json"):
        folder = ARTIFACTS / json.loads((ARTIFACTS / pointer).read_text())["path"]
        frames.append(pd.read_parquet(folder / "forecasts.parquet"))
    team = pd.concat(frames, ignore_index=True)
    team = team[(team["partition"] == "test") & (team["regime"] == "daily_updated") & (team["status"] == "ok")]
    wide = team.pivot_table(index=["station_id", "timestamp", "target"], columns="family", values="prediction").reset_index()
    actual = team.drop_duplicates(["station_id", "timestamp", "target"])[["station_id", "timestamp", "target", "actual", "disruption"]]
    return wide.merge(actual, on=["station_id", "timestamp", "target"], validate="one_to_one")


def score(joined):
    models = list(LABELS)
    joined = joined.dropna(subset=["actual", *models])
    rows, dm_rows = [], []
    for target, g in joined.groupby("target"):
        y = g["actual"].to_numpy()
        days = pd.to_datetime(g["timestamp"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize().to_numpy()
        for m in models:
            e = y - g[m].to_numpy()
            rows.append({"model": m, "label": LABELS[m], "target": target, "rows": len(y),
                         "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2))),
                         "wape_percent": float(100 * np.abs(e).sum() / y.sum())})
            for reference in ("seasonal_naive", "poisson"):
                if m == reference:
                    continue
                for loss, fn in (("squared", np.square), ("absolute", np.abs)):
                    mean_d, stat, p, n = dm_test(fn(y - g[reference].to_numpy()), fn(e), days)
                    dm_rows.append({"target": target, "model": m, "vs": reference, "loss": loss,
                                    "mean_daily_loss_reduction": mean_d, "dm_stat": stat, "p_value": p, "days": n})
    return pd.DataFrame(rows), pd.DataFrame(dm_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    panel_hash = verify_panel()
    freeze(panel_hash)
    CACHE.mkdir(parents=True, exist_ok=True)

    stations = sorted(pd.read_parquet(PANEL_PATH, columns=["station_id"])["station_id"].unique())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sarimax_one, (s, t)) for t in TARGETS for s in stations]
        panel = pd.read_parquet(PANEL_PATH).sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
        poisson = poisson_forecasts(panel)  # runs while the SARIMAX fits proceed
        for future in as_completed(futures):
            task, status = future.result()
            print(f"  sarimax {task}: {status}", flush=True)

    sarimax = pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob("*.parquet"))], ignore_index=True)
    if not sarimax["converged"].all():
        print("warning: some SARIMAX fits did not converge", flush=True)
    team = team_forecasts()
    for frame in (poisson, sarimax, team):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    keys = ["station_id", "timestamp", "target"]
    joined = team.merge(poisson, on=keys, how="left", validate="one_to_one").merge(
        sarimax.drop(columns="converged"), on=keys, how="left", validate="one_to_one")

    board, dm = score(joined)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUT / "test_forecasts.parquet", index=False)
    dm.to_csv(OUT / "dm_tests_test.csv", index=False)
    board.to_csv(STORY / "test_leaderboard.csv", index=False)
    dm[dm["loss"] == "squared"].to_csv(STORY / "test_dm_squared.csv", index=False)

    pd.set_option("display.width", 200)
    print(board.round(3).to_string(index=False))
    print(dm[dm["loss"] == "squared"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
