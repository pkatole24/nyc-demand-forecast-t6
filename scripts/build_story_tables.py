"""Small result tables for the dashboard's story tab (committed under data/story/).

The deployed dashboard cannot read data/processed/ (git-ignored), so this
writes the few numbers the story tab needs from the local evaluation outputs:

- validation_leaderboard.csv: every day-ahead model scored on the SAME 2025
  station-hours (team forecasts from artifacts/, regime daily_updated, joined
  with predictor_eval/sarimax/predictions_sarimax.parquet), with a
  Diebold-Mariano test against the last-week baseline.
- predictor_experiments.csv: the best variant of each external-data family
  against its own Poisson baseline (predictor_eval/metrics.csv, dm_tests.csv).

Inputs:  artifacts/{latest_run,latest_arimax_run}.json and their forecasts.parquet
         data/processed/predictor_eval/ (evaluate_predictors.py, evaluate_sarimax.py)
Outputs: data/story/validation_leaderboard.csv, data/story/predictor_experiments.csv

    ./.venv/bin/python scripts/build_story_tables.py
"""

import json

import numpy as np
import pandas as pd

from evaluate_predictors import dm_test
from predictor_common import ROOT, TZ

ARTIFACTS = ROOT / "artifacts"
EVAL = ROOT / "data" / "processed" / "predictor_eval"
OUT = ROOT / "data" / "story"

# key -> (label shown in the dashboard, who built it, specification)
MODELS = {
    "seasonal_naive": ("Last-week baseline", "team", "Same local hour one week earlier"),
    "daily_sarima": ("Daily pattern (SARIMA)", "team", "SARIMA per station, 24-hour seasonality, refit monthly"),
    "weather_arimax": ("Weather + calendar (ARIMAX)", "team", "ARIMAX per station, weather lagged 48 h, refit monthly"),
    "same_hour_mean_28d": ("Same-hour 28-day average", "predictor study", "Mean of the same local hour over the previous 28 days"),
    "sarima": ("SARIMA (1,0,1)(1,1,1)24", "predictor study", "Per station, parameters fixed on 2024, state updated hourly"),
    "sarimax_calendar": ("SARIMAX + weekly calendar", "predictor study", "SARIMA (1,0,1)(1,1,1)24 + weekly Fourier and off-day terms"),
    "baseline": ("Poisson regression", "predictor study", "Pooled count model: station x hour effects + recent-demand lags, refit monthly"),
}
FAMILY_LABELS = {"collisions": "Nearby traffic collisions", "subway": "Nearby subway ridership",
                 "events": "Permitted street events", "congestion": "Congestion pricing"}


def team_forecasts():
    frames = []
    for pointer in ("latest_run.json", "latest_arimax_run.json"):
        folder = ARTIFACTS / json.loads((ARTIFACTS / pointer).read_text())["path"]
        frames.append(pd.read_parquet(folder / "forecasts.parquet"))
    team = pd.concat(frames, ignore_index=True)
    team = team[(team["partition"] == "validation") & (team["regime"] == "daily_updated") & (team["status"] == "ok")]
    wide = team.pivot_table(index=["station_id", "timestamp", "target"], columns="family", values="prediction").reset_index()
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], utc=True)
    return wide


def leaderboard():
    mine = pd.read_parquet(EVAL / "sarimax" / "predictions_sarimax.parquet")
    mine["timestamp"] = pd.to_datetime(mine["timestamp_utc"], utc=True)
    joined = mine.merge(team_forecasts(), on=["station_id", "timestamp", "target"], how="inner", validate="one_to_one")
    joined = joined.dropna(subset=list(MODELS))
    rows = []
    for key, (label, source, spec) in MODELS.items():
        row = {"model": key, "label": label, "source": source, "specification": spec}
        for target, g in joined.groupby("target"):
            y = g["actual"].to_numpy()
            error = y - g[key].to_numpy()
            days = g["timestamp"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize().to_numpy()
            row[f"{target}_rmse"] = float(np.sqrt(np.mean(error**2)))
            row[f"{target}_mae"] = float(np.mean(np.abs(error)))
            row[f"{target}_rows"] = len(y)
            if key != "seasonal_naive":
                _, _, p, _ = dm_test((y - g["seasonal_naive"].to_numpy()) ** 2, error**2, days)
                row[f"{target}_dm_p_vs_last_week"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def experiments():
    metrics = pd.read_csv(EVAL / "metrics.csv")
    dm = pd.read_csv(EVAL / "dm_tests.csv")
    strict = metrics[metrics["regime"] == "strict"]
    rows = []
    for family, label in FAMILY_LABELS.items():
        part = strict[strict["family"] == family]
        base = part[part["model"] == "baseline"].set_index("target")["RMSE"]
        candidates = part[part["model"].str.startswith("+")]
        # Best variant by mean RMSE change across the two targets.
        change = candidates.assign(pct=lambda d: 100 * (d["RMSE"] / d["target"].map(base) - 1))
        best = change.groupby("model")["pct"].mean().idxmin()
        p = dm[(dm["family"] == family) & (dm["model"] == best) & (dm["loss"] == "squared") & (dm["regime"] == "strict")]
        rows.append({"family": family, "label": label, "best_variant": best,
                     "pickups_rmse_change_pct": float(change[(change["model"] == best) & (change["target"] == "pickups")]["pct"].iloc[0]),
                     "dropoffs_rmse_change_pct": float(change[(change["model"] == best) & (change["target"] == "dropoffs")]["pct"].iloc[0]),
                     "min_dm_p": float(p["p_value"].min())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    board = leaderboard()
    board.to_csv(OUT / "validation_leaderboard.csv", index=False)
    exp = experiments()
    exp.to_csv(OUT / "predictor_experiments.csv", index=False)
    pd.set_option("display.width", 200)
    print(board.round(4).to_string(index=False))
    print(exp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
