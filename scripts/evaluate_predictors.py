"""Do candidate predictor families improve hourly demand forecasts over a baseline?

Day-ahead regime: forecasts for every hour of local day D are issued at
D 00:00 America/New_York, so every predictor must be known by then.

Model: Poisson GLM with station x hour-of-day, weekend/holiday x hour-of-day,
weekday and month effects, plus log1p demand lags (24 h, 168 h, 28-day mean of
the same local hour). Each candidate adds predictors from one family. Because
station x hour effects absorb level differences between stations, station-level
coefficients are identified from within-station changes over time.

Evaluation: expanding-window rolling origin over the 2025 validation year,
refitting before each month. Within a family, the baseline and every candidate
train and score on identical rows (target, lags and all family predictors
known). Test (2026) rows are never fitted or scored here.

Realized weather is not known ahead; baselines exclude it. The
"oracle_weather" models use realized weather and are labelled as such.

Inputs:  data/processed/modeling_enriched.parquet (build_enriched_panel.py)
Outputs: data/processed/predictor_eval/*.csv and receipt.json

    ./.venv/bin/python scripts/evaluate_predictors.py         # every family, merged
    ./.venv/bin/python scripts/evaluate_predictors.py subway  # one family's *_subway.csv
"""

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from predictor_common import ROOT, TZ, sha256

PROCESSED = ROOT / "data" / "processed"
PANEL_PATH = PROCESSED / "modeling_enriched.parquet"
OUT_DIR = PROCESSED / "predictor_eval"

TARGETS = ("pickups", "dropoffs")
DM_HAC_LAGS = 7

# family -> {model name: predictor columns}. Count columns enter as log1p;
# columns already on a log scale or binary enter unchanged.
FAMILIES = {
    "collisions": {
        "+coll_all_500m_28d": ["coll_all_500m_28d"],
        "+coll_all_250m_28d": ["coll_all_250m_28d"],
        "+coll_all_1000m_28d": ["coll_all_1000m_28d"],
        "+coll_all_500m_7d": ["coll_all_500m_7d"],
        "+coll_cyclist_1000m_28d": ["coll_cyclist_1000m_28d"],
    },
    # crz_active is a 2025-onward step: it also absorbs any other 2024-to-2025
    # level change, so its coefficient is not a clean policy effect.
    "congestion": {
        "+crz_active": ["crz_active"],
        "+crz_active+entries": ["crz_active", "crz_log_entries_lag14_7d"],
    },
    "subway": {
        "+subway_ridership": ["subway_log_ridership_lag14_7d"],
    },
    "events": {
        "+events_all": ["events_all_precinct"],
        "+events_street": ["events_street_precinct"],
    },
}
LOG1P_PREFIXES = ("coll_", "events_")
# Realized-weather sensitivity, kept for the collisions primary feature.
ORACLE = {"collisions": "+coll_all_500m_28d"}
WEATHER = ("temperature", "temperature_sq", "wind_speed", "wet")


def verify_panel():
    receipt = json.loads(PANEL_PATH.with_suffix(".receipt.json").read_text())
    actual = sha256(PANEL_PATH)
    if actual != receipt["output_sha256"]:
        raise ValueError("enriched panel does not match its receipt; rerun build_enriched_panel.py")
    return actual


def add_demand_lags(panel, target):
    """Lags and rolling means using only target values observed before the origin."""
    df = panel[["station_id", "timestamp_utc", "local_date", "hour", target]].copy()
    origin_utc = df["local_date"].dt.tz_localize(TZ).dt.tz_convert("UTC")
    lookup = df.set_index(["station_id", "timestamp_utc"])[target]

    lags = {}
    for hours in (24, 168):
        source = df["timestamp_utc"] - pd.Timedelta(hours=hours)
        values = (
            lookup.reindex(pd.MultiIndex.from_arrays([df["station_id"], source]))
            .astype("Float64")
            .to_numpy(dtype="float64", na_value=np.nan)
        )
        # On 25-hour fall-back days the last hour minus 24 h lands at or after the origin.
        values[(source >= origin_utc).to_numpy()] = np.nan
        lags[f"log_lag{hours}"] = np.log1p(values)

    # Mean of the same local hour over the previous 28 local days. Fall-back
    # days have two rows for hour 1; collapse them first so no same-day value
    # feeds the rolling window.
    daily = (
        df.groupby(["station_id", "hour", "local_date"], observed=True)[target]
        .mean()
        .astype("float64")
        .reset_index()
        .sort_values(["station_id", "hour", "local_date"])
    )
    by_date = daily.set_index("local_date")
    daily["roll28"] = (
        by_date.groupby(["station_id", "hour"], observed=True)[target]
        .transform(lambda s: s.rolling("28D", closed="left", min_periods=20).mean())
        .to_numpy()
    )
    roll = df.merge(daily[["station_id", "hour", "local_date", "roll28"]], on=["station_id", "hour", "local_date"], how="left", validate="many_to_one")
    lags["log_roll28"] = np.log1p(roll["roll28"].to_numpy())
    return pd.DataFrame(lags, index=panel.index)


def calendar_design(panel):
    """Fixed-effect dummies shared by every model (no separate intercept)."""
    off_day = (panel["weekday"] >= 5) | panel["holiday"]
    station_hour = pd.get_dummies(panel["station_id"].astype(str) + "_h" + panel["hour"].astype(str), prefix="sh", dtype="float64")
    offday_hour = pd.get_dummies(panel["hour"].where(off_day).astype("Int64"), prefix="off_h", dtype="float64")
    weekday = pd.get_dummies(panel["weekday"], prefix="wd", drop_first=True, dtype="float64")
    month = pd.get_dummies(panel["timestamp"].dt.month, prefix="m", drop_first=True, dtype="float64")
    return pd.concat([station_hour, offday_hour, weekday, month], axis=1)


def weather_columns(panel):
    wet = (panel["precipitation"].astype("float64") > 0) | panel["precipitation_trace"]
    wet = wet.astype("float64").where(~panel["precipitation_missing"])
    temp = panel["temperature"].astype("float64")
    return pd.DataFrame(
        {"temperature": temp, "temperature_sq": temp**2, "wind_speed": panel["wind_speed"].astype("float64"), "wet": wet},
        index=panel.index,
    )


def predictor_columns(panel, columns):
    return pd.DataFrame(
        {c: np.log1p(panel[c].to_numpy(dtype="float64")) if c.startswith(LOG1P_PREFIXES) else panel[c].to_numpy(dtype="float64")
         for c in columns},
        index=panel.index,
    )


def fit_predict(X, y, train, score, start_params=None):
    # Return only coefficients: fitted results keep the full design matrix alive.
    # A predictor that is constant in training (crz_active before 2025) gets a
    # zero coefficient from the pseudo-inverse IRLS solve.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = sm.GLM(y[train], X[train], family=sm.families.Poisson()).fit(
            start_params=start_params, maxiter=100
        ).params
    return np.exp(X[score] @ params), params


def benchmarks(lags):
    """Rule-based forecasts known at the D 00:00 origin, on the count scale."""
    return {
        "seasonal_naive_168h": np.expm1(lags["log_lag168"].to_numpy()),
        "seasonal_naive_24h": np.expm1(lags["log_lag24"].to_numpy()),
        # Same local hour averaged over the previous 28 days; averaging across
        # all recent hours instead would flatten the daily cycle.
        "same_hour_mean_28d": np.expm1(lags["log_roll28"].to_numpy()),
    }


def poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-9, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu))


def dm_test(loss_a, loss_b, days):
    """Diebold-Mariano on daily mean loss differentials, HAC variance, HLN correction.

    Positive mean differential means model b has lower loss than a.
    """
    d = pd.Series(loss_a - loss_b).groupby(days).mean().to_numpy()
    T = len(d)
    centered = d - d.mean()
    var = centered @ centered / T
    for lag in range(1, DM_HAC_LAGS + 1):
        weight = 1 - lag / (DM_HAC_LAGS + 1)
        var += 2 * weight * (centered[lag:] @ centered[:-lag]) / T
    stat = d.mean() / np.sqrt(var / T)
    h = 1  # one-day loss blocks
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p_value = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(d.mean()), float(stat), float(p_value), T


def rolling_forecast(X, y, rows, month_start):
    pred = np.full(len(y), np.nan)
    params = None
    for month in pd.period_range("2025-01", "2025-12", freq="M"):
        train = rows & (month_start < month).to_numpy()
        score = rows & (month_start == month).to_numpy()
        # Warm start from the previous month's fit; the optimum barely moves.
        pred[score], params = fit_predict(X, y, train, score, params)
    return pred, params


def evaluate_family(family, specs, panel, target, context):
    base_X, lags, weather, y, month_start, base_usable = context
    columns = sorted({c for cols in specs.values() for c in cols})
    family_X = predictor_columns(panel, columns)
    rows = base_usable & family_X.notna().all(axis=1).to_numpy()

    groups = [("strict", "baseline", {name: cols for name, cols in specs.items()}, rows, False)]
    if family in ORACLE:
        oracle_rows = rows & weather.notna().all(axis=1).to_numpy()
        groups.append(("oracle", "oracle_weather_baseline",
                       {f"oracle_weather_{ORACLE[family]}": specs[ORACLE[family]]}, oracle_rows, True))

    metric_rows, dm_rows, station_rows, coef_rows = [], [], [], []
    rules = benchmarks(lags)
    days = panel["local_date"].to_numpy()
    saved = None
    for regime, reference, group_specs, group_rows, with_weather in groups:
        predictions = {}
        for name, cols in {reference: [], **group_specs}.items():
            parts = [base_X, lags] + ([weather] if with_weather else []) + ([family_X[cols]] if cols else [])
            X = pd.concat(parts, axis=1).to_numpy(dtype="float64")
            predictions[name], params = rolling_forecast(X, y, group_rows, month_start)
            del X
            for i, col in enumerate(cols):
                coef = float(params[len(params) - len(cols) + i])
                coef_rows.append({"family": family, "target": target, "model": name, "feature": col,
                                  "transform": "log1p" if col.startswith(LOG1P_PREFIXES) else "none",
                                  "coef_fit_through_2025_11": coef, "rate_ratio_per_unit": float(np.exp(coef))})
            print(f"  {family} {target} {name}: done", flush=True)

        scored = group_rows & (panel["partition"] == "validation").to_numpy()
        yt = y[scored]
        forecasts = {**(rules if regime == "strict" else {}), **predictions}
        for name, values in forecasts.items():
            mu = values[scored]
            metric_rows.append({
                "family": family, "target": target, "regime": regime, "model": name, "scored_rows": int(scored.sum()),
                "MAE": np.mean(np.abs(yt - mu)), "RMSE": np.sqrt(np.mean((yt - mu) ** 2)),
                # Total absolute error as a share of total demand; defined even
                # though many hours have zero demand (where MAPE is not).
                "WAPE_percent": 100 * np.abs(yt - mu).sum() / yt.sum(),
                "mean_poisson_deviance": np.mean(poisson_deviance(yt, mu)),
            })
            if name == reference or name in rules:
                continue
            ref_mu = predictions[reference][scored]
            for loss_name, fn in (("squared", np.square), ("absolute", np.abs)):
                mean_d, stat, p, T = dm_test(fn(yt - ref_mu), fn(yt - mu), days[scored])
                dm_rows.append({"family": family, "target": target, "regime": regime, "model": name, "vs": reference,
                                "loss": loss_name, "mean_daily_loss_reduction": mean_d, "dm_stat": stat,
                                "p_value": p, "days": T})
            if regime == "strict":
                frame = pd.DataFrame({"station_id": panel["station_id"].to_numpy()[scored],
                                      "ref": np.abs(yt - ref_mu), "new": np.abs(yt - mu)})
                for station, r in frame.groupby("station_id")[["ref", "new"]].mean().iterrows():
                    station_rows.append({"family": family, "target": target, "model": name, "station_id": station,
                                         "MAE_baseline": r.ref, "MAE_model": r.new, "pct_change": 100 * (r.new / r.ref - 1)})
        if regime == "strict":
            saved = pd.DataFrame({
                "station_id": panel["station_id"].to_numpy()[scored],
                "timestamp": panel["timestamp"].to_numpy()[scored],
                "target": target,
                "actual": yt,
                **{name: values[scored] for name, values in forecasts.items()},
            })
    return metric_rows, dm_rows, station_rows, coef_rows, saved


RESULT_KEYS = ("metrics", "dm_tests", "by_station", "coefficients")
PANEL_COLUMNS = [
    "station_id", "timestamp", "timestamp_utc", "local_date", "hour", "weekday", "holiday", "partition",
    *TARGETS, "temperature", "wind_speed", "precipitation", "precipitation_trace", "precipitation_missing",
]


def run_all():
    """One process per family: memory from repeated GLM fits is returned between
    families, and a crash keeps the families already finished."""
    for family in FAMILIES:
        subprocess.run([sys.executable, "-u", __file__, family], check=True)
    for key in RESULT_KEYS:
        merged = pd.concat([pd.read_csv(OUT_DIR / f"{key}_{f}.csv", dtype={"station_id": "string"}) for f in FAMILIES])
        merged.to_csv(OUT_DIR / f"{key}.csv", index=False)
    print(f"merged {len(FAMILIES)} families into {OUT_DIR.relative_to(ROOT)}")


def main(argv):
    if not argv:
        return run_all()
    families = argv
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        sys.exit(f"unknown families: {', '.join(unknown)}; choose from {', '.join(FAMILIES)}")

    panel_hash = verify_panel()
    columns = PANEL_COLUMNS + sorted({c for f in families for cols in FAMILIES[f].values() for c in cols})
    panel = pd.read_parquet(PANEL_PATH, columns=columns).sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    base_X = calendar_design(panel)
    weather = weather_columns(panel)
    month_start = panel["timestamp"].dt.tz_localize(None).dt.to_period("M")

    results = {k: [] for k in RESULT_KEYS}
    predictions = []
    for target in TARGETS:
        lags = add_demand_lags(panel, target)
        y = panel[target].astype("float64").to_numpy(na_value=np.nan)
        base_usable = ~np.isnan(y) & lags.notna().all(axis=1).to_numpy() & (panel["partition"] != "test").to_numpy()
        context = (base_X, lags, weather, y, month_start, base_usable)
        for family in families:
            *tables_out, saved = evaluate_family(family, FAMILIES[family], panel, target, context)
            for key, rows in zip(results, tables_out):
                results[key].extend(rows)
            predictions.append(saved.assign(family=family))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {k: pd.DataFrame(v) for k, v in results.items()}
    suffix = "_" + "_".join(families)
    for key, table in tables.items():
        table.to_csv(OUT_DIR / f"{key}{suffix}.csv", index=False)
    # Scored 2025 forecasts for plots and diagnostics (strict regime only).
    pd.concat(predictions, ignore_index=True).to_parquet(OUT_DIR / f"predictions{suffix}.parquet", index=False)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_sha256": panel_hash,
        "families": {f: FAMILIES[f] for f in families},
        "regime": "day-ahead, origin D 00:00 America/New_York; expanding window refit monthly over 2025",
        "scored_partition": "validation (2025); test not fitted or scored",
        "dm": f"daily mean loss differentials, Newey-West {DM_HAC_LAGS} lags, HLN correction, t(T-1)",
    }
    (OUT_DIR / f"receipt{suffix}.json").write_text(json.dumps(receipt, indent=2) + "\n")

    pd.set_option("display.width", 220)
    print(tables["metrics"].round(4).to_string(index=False))
    print(tables["dm_tests"].round(4).to_string(index=False))
    print(tables["coefficients"].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
