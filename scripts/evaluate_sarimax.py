"""ARIMAX / SARIMA / SARIMAX day-ahead forecasts, scored against the Poisson baseline.

One model per station and target on hourly counts (raw scale: a one-station
pilot gave 2025 RMSE 6.82 raw vs 7.04 for log1p with a lognormal back-transform).
Parameters are estimated once on 2024-02-01..2024-12-31 (January lacks the
28-day collision and 18-day subway look-backs) and then held fixed. The Kalman
filter runs causally through 2025, so at each local midnight the state uses
only hours already observed, and the forecast for that day's 23-25 hours is
Z T^k a + x'beta from the predicted state. Missing targets stay NaN; the
filter skips them. Forecasts are clipped at zero.

Specifications (orders fixed a priori, not searched on validation data):
    arimax_calendar       ARIMA(2,0,1)            + daily/weekly Fourier + off-day terms
    sarima                SARIMA(1,0,1)(1,1,1)24  no regressors
    sarimax_calendar      SARIMA(1,0,1)(1,1,1)24  + weekly Fourier + off-day terms
    sarimax_collisions    sarimax_calendar        + log1p crashes within 1,000 m, 28 days
    sarimax_subway        sarimax_calendar        + log subway ridership, days D-20..D-14
    sarimax_events        sarimax_calendar        + log1p street events in precinct
    oracle_sarimax_weather sarimax_calendar       + realized temperature, temp^2, wind, wet
Congestion pricing is not estimable here: its indicator is zero throughout
the 2024 estimation window.

Scoring uses the Poisson evaluation's 2025 rows (predictions_events.parquet),
so every model is compared on identical station-hours. The oracle model fills
the ~1% of hours with missing weather forward for filtering only and is scored
only where weather was observed.

Inputs:  data/processed/modeling_enriched.parquet, predictor_eval/predictions_events.parquet
Outputs: data/processed/predictor_eval/sarimax/*.csv, predictions_sarimax.parquet
         (per-fit cache in sarimax/cache/; delete it to refit)

    ./.venv/bin/python scripts/evaluate_sarimax.py            # 7 parallel workers
    ./.venv/bin/python scripts/evaluate_sarimax.py --workers 4
"""

import os

# One BLAS thread per worker; the parallelism is across fits.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

from evaluate_predictors import OUT_DIR as EVAL_DIR
from evaluate_predictors import PANEL_PATH, TARGETS, dm_test, poisson_deviance, verify_panel

OUT_DIR = EVAL_DIR / "sarimax"
CACHE = OUT_DIR / "cache"
FIT_START = pd.Timestamp("2024-02-01")
SCORE_START = pd.Timestamp("2025-01-01")

SPECS = {
    "arimax_calendar": {"order": (2, 0, 1), "seasonal": (0, 0, 0, 0), "calendar": "full", "extra": []},
    "sarima": {"order": (1, 0, 1), "seasonal": (1, 1, 1, 24), "calendar": None, "extra": []},
    "sarimax_calendar": {"order": (1, 0, 1), "seasonal": (1, 1, 1, 24), "calendar": "weekly", "extra": []},
    "sarimax_collisions": {"order": (1, 0, 1), "seasonal": (1, 1, 1, 24), "calendar": "weekly", "extra": ["coll_all_1000m_28d"]},
    "sarimax_subway": {"order": (1, 0, 1), "seasonal": (1, 1, 1, 24), "calendar": "weekly", "extra": ["subway_log_ridership_lag14_7d"]},
    "sarimax_events": {"order": (1, 0, 1), "seasonal": (1, 1, 1, 24), "calendar": "weekly", "extra": ["events_street_precinct"]},
    "oracle_sarimax_weather": {"order": (1, 0, 1), "seasonal": (1, 1, 1, 24), "calendar": "weekly", "extra": ["weather"]},
}
LOG1P = ("coll_all_1000m_28d", "events_street_precinct")
COLUMNS = [
    "station_id", "timestamp", "timestamp_utc", "local_date", "hour", "weekday", "holiday", "partition", *TARGETS,
    "coll_all_1000m_28d", "subway_log_ridership_lag14_7d", "events_street_precinct",
    "temperature", "wind_speed", "precipitation", "precipitation_trace", "precipitation_missing",
]


def fourier(x, period, terms):
    return np.column_stack([f(2 * np.pi * k * x / period) for k in range(1, terms + 1) for f in (np.sin, np.cos)])


def exog_matrix(s, spec):
    """Regressors known at the day's origin (calendar, daily predictors) or oracle weather."""
    blocks = []
    hour = s["hour"].to_numpy()
    hour_of_week = (s["weekday"] * 24 + s["hour"]).to_numpy()
    off = ((s["weekday"] >= 5) | s["holiday"]).to_numpy(dtype=float)
    if spec["calendar"] == "full":
        # No seasonal differencing, so the daily cycle must come from regressors.
        blocks += [fourier(hour, 24, 6), fourier(hour_of_week, 168, 4), off[:, None], off[:, None] * fourier(hour, 24, 3)]
    elif spec["calendar"] == "weekly":
        # Seasonal differencing at 24 absorbs the plain daily cycle; daily
        # Fourier terms would be differenced away and are left out.
        blocks += [fourier(hour_of_week, 168, 4), off[:, None], off[:, None] * fourier(hour, 24, 3)]
    weather_missing = np.zeros(len(s), dtype=bool)
    for col in spec["extra"]:
        if col == "weather":
            wet = ((s["precipitation"].astype("float64") > 0) | s["precipitation_trace"]).astype(float)
            wet = wet.where(~s["precipitation_missing"])
            temp = s["temperature"].astype("float64")
            w = pd.DataFrame({"t": temp, "t2": temp**2, "wind": s["wind_speed"].astype("float64"), "wet": wet})
            weather_missing = w.isna().any(axis=1).to_numpy()
            # Kalman filtering needs complete regressors; filled hours are never scored.
            blocks.append(w.ffill().bfill().to_numpy())
        else:
            v = s[col].to_numpy(dtype="float64")
            blocks.append((np.log1p(v) if col in LOG1P else v)[:, None])
    X = np.column_stack(blocks) if blocks else None
    if X is not None and np.isnan(X).any():
        raise ValueError("regressors have gaps inside the modelling window")
    return X, weather_missing


def day_ahead(results_filter, origins, day_end, n):
    """Mean forecast for each hour of each local day from the state predicted at its origin."""
    fr = results_filter.filter_results
    Z, T = fr.design[:, :, 0], fr.transition[:, :, 0]
    d = fr.obs_intercept[0] if fr.obs_intercept.shape[1] > 1 else np.full(n, fr.obs_intercept[0, 0])
    a = fr.predicted_state[:, origins]
    out = np.full(n, np.nan)
    v = Z.copy()
    for k in range(25):  # local days have at most 25 hours
        idx = origins + k
        ok = idx < day_end
        out[idx[ok]] = (v @ a[:, ok])[0] + d[idx[ok]]
        v = v @ T
    return np.clip(out, 0, None)


def run_one(task, force=False):
    station, target, name = task
    cache = CACHE / f"{station}_{target}_{name}.parquet"
    if cache.exists() and not force:
        return task, "cached"
    spec = SPECS[name]
    panel = pd.read_parquet(PANEL_PATH, columns=COLUMNS, filters=[("station_id", "==", station)])
    s = panel[(panel["local_date"] >= FIT_START) & (panel["partition"] != "test")].sort_values("timestamp_utc").reset_index(drop=True)
    y = s[target].astype("float64").to_numpy(na_value=np.nan)
    X, weather_missing = exog_matrix(s, spec)
    train = (s["local_date"] < SCORE_START).to_numpy()

    def model(rows):
        return sm.tsa.SARIMAX(y[rows], exog=None if X is None else X[rows], order=spec["order"],
                              seasonal_order=spec["seasonal"], trend="c" if spec["seasonal"][1] == 0 else "n")

    started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model(train).fit(disp=False, maxiter=1000)
        full = model(np.ones(len(y), dtype=bool)).filter(fit.params)
    first = s.groupby("local_date").head(1)
    origins = first.index[first["local_date"] >= SCORE_START].to_numpy()
    day_end = np.append(origins[1:], len(s))
    pred = day_ahead(full, origins, day_end, len(s))

    resid = fit.standardized_forecasts_error[0]
    resid = resid[np.isfinite(resid)][24 * 8:]  # skip the diffuse start-up
    lb = acorr_ljungbox(resid, lags=[24, 168])
    score = ~train
    out = pd.DataFrame({
        "station_id": station, "timestamp_utc": s["timestamp_utc"].to_numpy()[score], "target": target, "model": name,
        "forecast": pred[score], "weather_filled": weather_missing[score],
    })
    out.to_parquet(cache, index=False)
    diag = {"station_id": station, "target": target, "model": name, "order": str(spec["order"]),
            "seasonal_order": str(spec["seasonal"]), "n_exog": 0 if X is None else X.shape[1],
            "aic": float(fit.aic), "converged": bool(fit.mle_retvals.get("converged", False)),
            "ljung_box_p_lag24": float(lb["lb_pvalue"].iloc[0]), "ljung_box_p_lag168": float(lb["lb_pvalue"].iloc[1]),
            "fit_seconds": round(time.time() - started, 1)}
    if spec["extra"] and spec["extra"] != ["weather"]:
        # A single added regressor is the last exog column, named x<k> by statsmodels.
        names = list(fit.model.param_names)
        diag["regressor_coef"] = float(fit.params[names.index(f"x{X.shape[1]}")])
    (CACHE / f"{station}_{target}_{name}.json").write_text(json.dumps(diag) + "\n")
    return task, f"{diag['fit_seconds']}s"


def score(poisson):
    forecasts = pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob("*.parquet"))], ignore_index=True)
    wide = forecasts.pivot_table(index=["station_id", "timestamp_utc", "target"], columns="model", values="forecast").reset_index()
    filled = forecasts[forecasts["model"] == "oracle_sarimax_weather"][["station_id", "timestamp_utc", "target", "weather_filled"]]
    merged = poisson.merge(wide, on=["station_id", "timestamp_utc", "target"], how="left", validate="one_to_one")
    merged = merged.merge(filled, on=["station_id", "timestamp_utc", "target"], how="left", validate="one_to_one")
    if merged[list(SPECS)].isna().any().any():
        raise AssertionError("a scored Poisson row has no SARIMAX forecast")

    metric_rows, dm_rows = [], []
    comparisons = [(m, "baseline") for m in SPECS if not m.startswith("oracle")] + \
                  [(m, "sarimax_calendar") for m in ("sarimax_collisions", "sarimax_subway", "sarimax_events", "oracle_sarimax_weather")]
    for target in TARGETS:
        part = merged[merged["target"] == target]
        for regime, rows in (("strict", part), ("weather_observed", part[~part["weather_filled"]])):
            y = rows["actual"].to_numpy()
            models = ["seasonal_naive_168h", "same_hour_mean_28d", "baseline", *SPECS] if regime == "strict" else ["baseline", "sarimax_calendar", "oracle_sarimax_weather"]
            for m in models:
                mu = rows[m].to_numpy()
                metric_rows.append({"target": target, "regime": regime, "model": m, "scored_rows": len(y),
                                    "RMSE": np.sqrt(np.mean((y - mu) ** 2)), "MAE": np.mean(np.abs(y - mu)),
                                    "WAPE_percent": 100 * np.abs(y - mu).sum() / y.sum(),
                                    "mean_poisson_deviance": np.mean(poisson_deviance(y, mu))})
            days = rows["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize().to_numpy()
            for model_name, reference in comparisons:
                if (model_name.startswith("oracle")) != (regime == "weather_observed"):
                    continue
                for loss, fn in (("squared", np.square), ("absolute", np.abs)):
                    mean_d, stat, p, n_days = dm_test(fn(y - rows[reference].to_numpy()), fn(y - rows[model_name].to_numpy()), days)
                    dm_rows.append({"target": target, "regime": regime, "model": model_name, "vs": reference, "loss": loss,
                                    "mean_daily_loss_reduction": mean_d, "dm_stat": stat, "p_value": p, "days": n_days})
    return merged, pd.DataFrame(metric_rows), pd.DataFrame(dm_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    panel_hash = verify_panel()
    CACHE.mkdir(parents=True, exist_ok=True)
    stations = sorted(pd.read_parquet(PANEL_PATH, columns=["station_id"])["station_id"].unique())
    # Slowest specifications first so the pool drains evenly.
    tasks = [(s, t, m) for m in sorted(SPECS, key=lambda m: SPECS[m]["seasonal"][1] == 0) for t in TARGETS for s in stations]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            task, status = future.result()
            print(f"[{i}/{len(tasks)}] {task}: {status}", flush=True)

    poisson = pd.read_parquet(EVAL_DIR / "predictions_events.parquet")
    poisson["timestamp"] = pd.to_datetime(poisson["timestamp"], utc=True)
    poisson["timestamp_utc"] = poisson["timestamp"]
    poisson = poisson[["station_id", "timestamp", "timestamp_utc", "target", "actual",
                       "seasonal_naive_168h", "same_hour_mean_28d", "baseline"]]
    merged, metrics, dm = score(poisson)
    diagnostics = pd.DataFrame([json.loads(p.read_text()) for p in sorted(CACHE.glob("*.json"))])

    metrics.to_csv(OUT_DIR / "metrics_sarimax.csv", index=False)
    dm.to_csv(OUT_DIR / "dm_tests_sarimax.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "diagnostics_sarimax.csv", index=False)
    merged.to_parquet(OUT_DIR / "predictions_sarimax.parquet", index=False)
    (OUT_DIR / "receipt.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_sha256": panel_hash,
        "estimation_window": "2024-02-01..2024-12-31, parameters fixed thereafter",
        "regime": "day-ahead from local midnight; Kalman state updated hourly with observed data",
        "specs": SPECS,
        "scored_rows": "Poisson evaluation rows from predictions_events.parquet (2025)",
    }, indent=2, default=str) + "\n")

    pd.set_option("display.width", 200)
    print(metrics.round(3).to_string(index=False))
    print(dm.round(4).to_string(index=False))
    print(diagnostics.groupby("model")[["aic", "converged", "ljung_box_p_lag24", "ljung_box_p_lag168", "fit_seconds"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
