"""Hourly day-ahead weather-forecast features (citywide), keyed by timestamp_utc.

Source: Open-Meteo Previous Runs API, `_previous_day2` values at LaGuardia
(scripts/collect_weather_forecasts.py). Each value comes from a run initialized
48 hours before the valid hour, so it is available at the midnight origin of
the valid hour's local day. Missing archive hours stay missing.

Features: fc_temperature (degrees C), fc_temperature_sq, fc_wind_speed (km/h),
fc_precipitation (mm) and fc_wet (forecast precipitation of at least 0.1 mm;
smaller amounts are model drizzle noise rather than a rain signal).

Input:  data/raw/weather_forecasts/open_meteo_previous_day2.json
Output: data/processed/features/weather_forecast_hourly.parquet (+ .receipt.json)

    ./.venv/bin/python scripts/build_weather_forecast_features.py
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from predictor_common import FEATURES, ROOT, load_base, sha256

SOURCE = ROOT / "data" / "raw" / "weather_forecasts" / "open_meteo_previous_day2.json"
WET_MM = 0.1


def main():
    hourly = json.loads(SOURCE.read_text())["hourly"]
    times = pd.to_datetime(hourly["time"]).tz_localize("UTC").as_unit("us")
    temp = pd.Series(hourly["temperature_2m_previous_day2"], dtype="float64").to_numpy()
    rain = pd.Series(hourly["precipitation_previous_day2"], dtype="float64").to_numpy()
    features = pd.DataFrame({
        "timestamp_utc": times,
        "fc_temperature": temp,
        "fc_temperature_sq": temp**2,
        "fc_wind_speed": pd.Series(hourly["wind_speed_10m_previous_day2"], dtype="float64").to_numpy(),
        "fc_precipitation": rain,
        "fc_wet": np.where(np.isnan(rain), np.nan, (rain >= WET_MM).astype(float)),
    })
    if features.timestamp_utc.duplicated().any():
        raise ValueError("duplicate forecast hours")

    # Coverage over the panel's own hours, by partition (AGENTS.md: report denominators).
    base = load_base()[["timestamp_utc", "partition", "temperature"]].drop_duplicates("timestamp_utc")
    check = base.merge(features, on="timestamp_utc", how="left", validate="one_to_one")
    coverage = {part: {"hours": int(len(g)), "available": int(g.fc_temperature.notna().sum())}
                for part, g in check.groupby("partition")}
    both = check[["temperature", "fc_temperature"]].astype(float).dropna()
    agreement = {"hours_compared": len(both), "correlation_with_observed_temperature": round(float(both.corr().iloc[0, 1]), 4),
                 "mean_abs_error_vs_observed_c": round(float((both.temperature - both.fc_temperature).abs().mean()), 3)}

    FEATURES.mkdir(parents=True, exist_ok=True)
    path = FEATURES / "weather_forecast_hourly.parquet"
    features.to_parquet(path, index=False)
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "keys": ["timestamp_utc"], "rows": len(features),
               "source_sha256": sha256(SOURCE), "wet_threshold_mm": WET_MM,
               "information_regime": "Open-Meteo run initialized 48 h before each valid hour (published before the day's midnight origin)",
               "coverage_by_partition": coverage, "agreement": agreement, "output_sha256": sha256(path)}
    (FEATURES / "weather_forecast_hourly.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("rows", "coverage_by_partition", "agreement")}, indent=2))


if __name__ == "__main__":
    main()
