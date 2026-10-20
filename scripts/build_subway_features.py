"""Lagged subway ridership at station complexes near each Citi Bike station, per local day.

Spatial assignment: subway complexes within 750 m of the station; a station
with none (W 54 St & 11 Ave, nearest 857 m) uses its single nearest complex.
Ridership is realized data. Rows from 2026 show weekly portal loads about
7-14 days after the ride date; earlier rows were bulk-loaded, so their delay
is unmeasurable. A conservative fixed lag is used: day D sees mean daily
ridership over [D - 20, D - 14], requiring at least 5 complete days.

A day counts as complete only when every assigned complex reports; partial
sums would look like ridership drops.

Inputs: data/raw/nyc_open_data/subway_ridership_daily_2024.jsonl
        data/raw/nyc_open_data/subway_ridership_daily_2025_on.jsonl
Output: data/processed/features/subway_daily.parquet (station_id x local_date)

    ./.venv/bin/python scripts/build_subway_features.py
"""

import json

import numpy as np
import pandas as pd

from predictor_common import PANEL_DATES, distance_m, load_stations, read_raw, write_features

RADIUS_M = 750
LAG_DAYS = 14
WINDOW_DAYS = 7
MIN_DAYS = 5
SOURCES = ("subway_ridership_daily_2024", "subway_ridership_daily_2025_on")


def load_ridership():
    raw = pd.concat([read_raw(s) for s in SOURCES], ignore_index=True)
    rides = pd.DataFrame(
        {
            "local_date": pd.to_datetime(raw["transit_date"].str[:10]),
            "complex_id": raw["station_complex_id"].astype("string"),
            "complex_name": raw["station_complex"],
            "lat": pd.to_numeric(raw["latitude"]),
            "lng": pd.to_numeric(raw["longitude"]),
            "ridership": pd.to_numeric(raw["ridership"]).astype("float64"),
        }
    )
    if rides.duplicated(["local_date", "complex_id"]).any():
        raise ValueError("2024 and 2025+ subway sources overlap")
    return rides


def assign_complexes(stations, rides):
    complexes = rides.groupby("complex_id").agg(
        complex_name=("complex_name", "last"), lat=("lat", "median"), lng=("lng", "median"),
        lat_range=("lat", lambda s: s.max() - s.min()),
    )
    # A moved coordinate would mean the id does not denote one fixed place.
    # Times Sq (611) and Chambers St/WTC (624) were renamed between sources
    # and their centroids shift ~170 m, so allow ~330 m.
    if (complexes["lat_range"] > 0.003).any():
        raise ValueError("complex coordinates vary across rows")
    assignment = {}
    for station in stations.itertuples():
        dist = pd.Series(distance_m(station.lat, station.lng, complexes["lat"].to_numpy(), complexes["lng"].to_numpy()), index=complexes.index)
        near = dist[dist <= RADIUS_M]
        if near.empty:
            near = dist.nsmallest(1)
        assignment[station.station_id] = [
            {"complex_id": cid, "name": complexes.at[cid, "complex_name"], "distance_m": round(float(d))}
            for cid, d in near.sort_values().items()
        ]
    return assignment


def main():
    stations = load_stations()
    rides = load_ridership()
    assignment = assign_complexes(stations, rides)
    wide = rides.pivot(index="local_date", columns="complex_id", values="ridership").reindex(PANEL_DATES)

    frames = []
    for station_id, complexes in assignment.items():
        ids = [c["complex_id"] for c in complexes]
        # min_count=len(ids): any missing complex makes the day missing.
        daily = wide[ids].sum(axis=1, min_count=len(ids))
        lagged = daily.shift(LAG_DAYS).rolling(WINDOW_DAYS, min_periods=MIN_DAYS).mean()
        frames.append(pd.DataFrame({
            "station_id": station_id,
            "local_date": PANEL_DATES,
            "subway_log_ridership_lag14_7d": np.log(lagged.to_numpy()),
            "subway_complete_days_in_window": daily.shift(LAG_DAYS).rolling(WINDOW_DAYS, min_periods=1).count().to_numpy(),
        }))
    features = pd.concat(frames, ignore_index=True)
    features["station_id"] = features["station_id"].astype("string")

    write_features(
        "subway_daily",
        features,
        ["station_id", "local_date"],
        list(SOURCES),
        {
            "radius_m": RADIUS_M,
            "fallback": "nearest complex when none within radius",
            "lag_days": LAG_DAYS,
            "window_days": WINDOW_DAYS,
            "min_complete_days": MIN_DAYS,
            "assignment": assignment,
            "information_regime": "mean daily ridership over [D-20, D-14], fixed lag (2026 loads ~7-14 days)",
        },
    )
    print(json.dumps({s: [f"{c['name']} ({c['distance_m']} m)" for c in cs] for s, cs in assignment.items()}, indent=1))
    print(features.groupby("station_id")["subway_log_ridership_lag14_7d"].agg(["count", "mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
