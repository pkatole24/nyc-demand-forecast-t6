"""Frozen 2026 test evaluation of the models chosen on 2025, beside the team's models.

Model choices were fixed on the 2025 validation year and written to
data/story/ before any of their test forecasts were scored:

- frozen_choices.json: Poisson regression baseline and SARIMAX + weekly calendar.
- frozen_choices_ai.json: Chronos-Bolt, a pretrained AI forecasting model used
  zero-shot (scripts/forecast_chronos.py); its settings were fixed a priori.
- frozen_choices_weather_forecast.json: the same two models plus Open-Meteo
  weather forecasts issued 48 h ahead (fc_temperature, fc_temperature_sq,
  fc_wind_speed, fc_wet), added after both improved 2025 RMSE (DM p < 0.002).

An existing record is never rewritten; a run fails if a recorded choice differs.
Records are compared on model specifications and regime only: the enriched
panel's hash changes whenever new feature columns are joined to the same base.

- Poisson models refit before each 2026 month on all earlier usable rows.
- SARIMAX models re-estimate parameters once on 2024-02-01..2025-12-31, then
  run the Kalman state causally through 2026 from each local midnight.

Regime: day-ahead from local midnight with trips known through the previous
day (the team's `daily_updated`). Every model, including the team's last-week
baseline, SARIMA and ARIMAX from artifacts/, is scored on the same station-hours.

Outputs: data/processed/test_eval/test_leaderboard_all.csv, dm_tests_all.csv,
         test_forecasts.parquet, cache/ (one parquet per SARIMAX station fit)
         data/story/test_leaderboard.csv and test_dm_squared.csv (dashboard
         inputs) are written only with --publish.

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
WEATHER_FORECAST = ["fc_temperature", "fc_temperature_sq", "fc_wind_speed", "fc_wet"]

# Poisson variants: name -> extra predictor columns (entered unchanged).
POISSON = {"poisson": [], "poisson_weather_forecast": WEATHER_FORECAST}
SARIMAX_MODELS = ("sarimax_calendar", "sarimax_weather_forecast")
LABELS = {
    "seasonal_naive": "Last-week baseline",
    "daily_sarima": "Daily pattern (SARIMA)",
    "weather_arimax": "Weather + calendar (ARIMAX)",
    "sarimax_calendar": "SARIMAX + weekly calendar",
    "sarimax_weather_forecast": "SARIMAX + calendar + weather forecast",
    "poisson": "Poisson regression",
    "poisson_weather_forecast": "Poisson + weather forecast",
    "chronos": "Chronos-Bolt (AI, zero-shot)",
}
# Each weather-forecast model is also tested against its no-weather counterpart.
COUNTERPART = {"poisson_weather_forecast": ("poisson",), "sarimax_weather_forecast": ("sarimax_calendar",),
               # The AI model against the best conventional model and the closest one without weather.
               "chronos": ("poisson_weather_forecast", "sarimax_calendar")}
CHRONOS = ROOT / "data" / "processed" / "chronos" / "forecasts.parquet"
REGIME = "daily_updated: day-ahead from New York midnight, trips known through the previous day"
FROZEN = {
    "frozen_choices.json": {
        "frozen_on_validation": "2025",
        "models": {
            "poisson": "Poisson GLM baseline: station x hour, off-day x hour, weekday, month, log1p lags 24h/168h/28-day same-hour mean; refit monthly",
            "sarimax_calendar": f"SARIMAX{SPECS['sarimax_calendar']['order']}{SPECS['sarimax_calendar']['seasonal']} + weekly Fourier (K=4) + off-day x daily Fourier (K=3); estimated 2024-02-01..2025-12-31",
        },
        "regime": REGIME,
    },
    "frozen_choices_weather_forecast.json": {
        "frozen_on_validation": "2025",
        "models": {
            "poisson_weather_forecast": "Poisson GLM baseline + Open-Meteo previous_day2 forecasts: temperature, temperature^2, wind speed, wet (precipitation >= 0.1 mm); refit monthly",
            "sarimax_weather_forecast": "SARIMAX + weekly calendar + the same four Open-Meteo previous_day2 forecast regressors; estimated 2024-02-01..2025-12-31",
        },
        "selection_evidence": "2025 validation: Poisson RMSE 6.12 -> 5.84 pickups, 5.99 -> 5.71 drop-offs (DM p 0.0008 / 0.0013); "
                              "SARIMAX 6.27 -> 6.05, 6.21 -> 5.99 (DM p < 0.0001)",
        "regime": REGIME,
    },
    "frozen_choices_ai.json": {
        "frozen_on_validation": "fixed a priori (zero-shot, no tuning on 2024 or 2025)",
        "models": {
            "chronos": "amazon/chronos-bolt-small, pretrained foundation model, no fine-tuning; last 2,048 hourly observations "
                       "per station and target as context; mean forecast clipped at zero (scripts/forecast_chronos.py)",
        },
        "regime": REGIME,
    },
}


def freeze(panel_hash):
    """Write each record once, before scoring; refuse to run if a recorded choice changed."""
    for name, record in FROZEN.items():
        path = STORY / name
        if path.exists():
            saved = json.loads(path.read_text())
            if {k: saved.get(k) for k in record} != record:
                raise ValueError(f"{name} differs from the recorded choices; this would not be a frozen evaluation")
            continue
        STORY.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**record, "panel_sha256": panel_hash,
                                    "frozen_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        print(f"recorded {name} before scoring", flush=True)


def poisson_forecasts(panel):
    base_X = calendar_design(panel)
    month = panel["timestamp"].dt.tz_localize(None).dt.to_period("M")
    test = (panel["partition"] == "test").to_numpy()
    frames = []
    for target in TARGETS:
        lags = add_demand_lags(panel, target)
        y = panel[target].astype("float64").to_numpy(na_value=np.nan)
        out = pd.DataFrame({"station_id": panel["station_id"].to_numpy()[test],
                            "timestamp": panel["timestamp_utc"].to_numpy()[test], "target": target})
        for name, extra in POISSON.items():
            X_frame = pd.concat([base_X, lags, panel[extra].astype("float64")], axis=1)
            known = X_frame.notna().all(axis=1).to_numpy()
            usable = known & ~np.isnan(y)
            X = X_frame.to_numpy(dtype="float64")
            pred = np.full(len(y), np.nan)
            params = None
            for m in pd.period_range("2026-01", "2026-09", freq="M"):
                train = usable & (month < m).to_numpy()
                score = known & (month == m).to_numpy()
                pred[score], params = fit_predict(X, y, train, score, params)
            out[name] = pred[test]
            print(f"  {name} {target}: 9 monthly refits done", flush=True)
            del X, X_frame
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def sarimax_one(task):
    station, target, name = task
    cache = CACHE / f"{station}_{target}_{name}.parquet"
    if cache.exists():
        return task, "cached"
    spec = SPECS[name]
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
                  "model": name, "forecast": pred[~train], "converged": bool(fit.mle_retvals.get("converged", False))}
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
            references = ({"seasonal_naive", "poisson"} | set(COUNTERPART.get(m, ()))) - {m}
            for reference in sorted(references):
                for loss, fn in (("squared", np.square), ("absolute", np.abs)):
                    mean_d, stat, p, n = dm_test(fn(y - g[reference].to_numpy()), fn(e), days)
                    dm_rows.append({"target": target, "model": m, "vs": reference, "loss": loss,
                                    "mean_daily_loss_reduction": mean_d, "dm_stat": stat, "p_value": p, "days": n})
    return pd.DataFrame(rows), pd.DataFrame(dm_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--publish", action="store_true", help="also write the dashboard's data/story test tables")
    args = parser.parse_args()

    panel_hash = verify_panel()
    freeze(panel_hash)
    CACHE.mkdir(parents=True, exist_ok=True)

    stations = sorted(pd.read_parquet(PANEL_PATH, columns=["station_id"])["station_id"].unique())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sarimax_one, (s, t, m)) for m in SARIMAX_MODELS for t in TARGETS for s in stations]
        panel = pd.read_parquet(PANEL_PATH).sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
        poisson = poisson_forecasts(panel)  # runs while the SARIMAX fits proceed
        for future in as_completed(futures):
            task, status = future.result()
            print(f"  sarimax {task}: {status}", flush=True)

    sarimax_long = pd.concat([pd.read_parquet(p) for p in sorted(CACHE.glob("*_*_*.parquet"))], ignore_index=True)
    if not sarimax_long["converged"].all():
        print("warning: some SARIMAX fits did not converge", flush=True)
    sarimax = sarimax_long.pivot_table(index=["station_id", "timestamp", "target"], columns="model", values="forecast").reset_index()
    team = team_forecasts()
    for frame in (poisson, sarimax, team):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    keys = ["station_id", "timestamp", "target"]
    chronos = pd.read_parquet(CHRONOS).query("partition == 'test'")[["station_id", "timestamp", "target", "chronos"]]
    chronos["timestamp"] = pd.to_datetime(chronos["timestamp"], utc=True)
    joined = (team.merge(poisson, on=keys, how="left", validate="one_to_one")
              .merge(sarimax, on=keys, how="left", validate="one_to_one")
              .merge(chronos, on=keys, how="left", validate="one_to_one"))

    board, dm = score(joined)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUT / "test_forecasts.parquet", index=False)
    board.to_csv(OUT / "test_leaderboard_all.csv", index=False)
    dm.to_csv(OUT / "dm_tests_all.csv", index=False)
    if args.publish:
        board.to_csv(STORY / "test_leaderboard.csv", index=False)
        dm[dm["loss"] == "squared"].to_csv(STORY / "test_dm_squared.csv", index=False)

    pd.set_option("display.width", 200)
    print(board.round(3).to_string(index=False))
    print(dm[dm["loss"] == "squared"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
