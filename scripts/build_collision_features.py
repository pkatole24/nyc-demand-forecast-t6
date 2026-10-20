"""As-published motor-vehicle collision counts around each station, per local day.

Information regime: forecasts for local day D are issued at D 00:00
America/New_York. A crash counts toward day D only if its crash_date falls in
the look-back window [D - window, D - 1] and the portal had already published
the row (:created_at) before that origin. Rows deleted from the portal before
the download cannot be seen, and :created_at may reset if a row was re-created.

Input:  data/raw/nyc_open_data/motor_collisions.jsonl
Output: data/processed/features/collisions_daily.parquet (station_id x local_date)

    ./.venv/bin/python scripts/build_collision_features.py
"""

import numpy as np
import pandas as pd

from predictor_common import PANEL_DATES, TZ, distance_m, load_stations, read_raw, write_features

RADII_M = (250, 500, 1000)
WINDOWS_DAYS = (7, 28)
# Downloaded collisions start 2024-01-01, so windows reaching earlier are incomplete.
COLLISIONS_START = pd.Timestamp("2024-01-01")
# If no crash dated within this many days before the origin has been published,
# the feed is treated as stalled and features are missing rather than zero.
# 2024-2025 publication lag: median 4 days, 99th percentile 31 days.
STALE_AFTER_DAYS = 10


def load_collisions():
    raw = read_raw("motor_collisions")
    if raw["collision_id"].duplicated().any():
        raise ValueError("duplicate collision_id in raw collisions")
    cyclist = (
        pd.to_numeric(raw["number_of_cyclist_injured"]).fillna(0)
        + pd.to_numeric(raw["number_of_cyclist_killed"]).fillna(0)
    )
    return pd.DataFrame(
        {
            "collision_id": raw["collision_id"],
            # crash_date is a naive local calendar date; only the date is used,
            # which avoids assigning ambiguous fall-back clock times.
            "crash_date": pd.to_datetime(raw["crash_date"].str[:10]),
            "published_utc": pd.to_datetime(raw[":created_at"], utc=True),
            "lat": pd.to_numeric(raw["latitude"]),
            "lng": pd.to_numeric(raw["longitude"]),
            "cyclist_harm": cyclist > 0,
        }
    )


def build_daily_features(stations, collisions, dates):
    # Integer nanoseconds: naive local dates compare with naive local dates,
    # UTC instants with UTC instants.
    def ns(values):
        return pd.DatetimeIndex(values).as_unit("ns").asi8

    day_ns = 86_400 * 10**9
    origins_utc = ns(pd.DatetimeIndex(dates).tz_localize(TZ))
    date_ns = ns(dates)
    crash_ns = ns(collisions["crash_date"])
    pub_ns = ns(collisions["published_utc"])

    # Feed staleness is citywide (within the downloaded box), not per station.
    latest_published = np.full(len(dates), np.iinfo("int64").min)
    order = np.argsort(pub_ns)
    running_max = np.maximum.accumulate(crash_ns[order])
    idx = np.searchsorted(pub_ns[order], origins_utc, side="left") - 1
    has_any = idx >= 0
    latest_published[has_any] = running_max[idx[has_any]]
    stale_days = (date_ns - latest_published) / day_ns
    stale = stale_days > STALE_AFTER_DAYS

    frames = []
    for station in stations.itertuples():
        dist = distance_m(station.lat, station.lng, collisions["lat"].to_numpy(), collisions["lng"].to_numpy())
        out = pd.DataFrame({"station_id": station.station_id, "local_date": dates})
        for radius in RADII_M:
            near = dist <= radius
            for harm_name, mask in (("all", near), ("cyclist", near & collisions["cyclist_harm"].to_numpy())):
                c_day, c_pub = crash_ns[mask], pub_ns[mask]
                # counts[d, w] via broadcasting: crashes x days is small here.
                published = c_pub[None, :] < origins_utc[:, None]
                for window in WINDOWS_DAYS:
                    in_window = (c_day[None, :] >= date_ns[:, None] - window * day_ns) & (
                        c_day[None, :] < date_ns[:, None]
                    )
                    counts = (published & in_window).sum(axis=1).astype("float64")
                    incomplete = pd.DatetimeIndex(dates) - pd.Timedelta(days=window) < COLLISIONS_START
                    counts[incomplete | stale] = np.nan
                    out[f"coll_{harm_name}_{radius}m_{window}d"] = counts
        frames.append(out)

    features = pd.concat(frames, ignore_index=True)
    features["collision_feed_stale_days"] = np.tile(stale_days, len(stations))
    features["collision_feed_stale"] = np.tile(stale, len(stations))
    features["station_id"] = features["station_id"].astype("string")
    return features


def main():
    features = build_daily_features(load_stations(), load_collisions(), PANEL_DATES)
    write_features(
        "collisions_daily",
        features,
        ["station_id", "local_date"],
        ["motor_collisions"],
        {
            "radii_m": RADII_M,
            "windows_days": WINDOWS_DAYS,
            "stale_after_days": STALE_AFTER_DAYS,
            "information_regime": "origin D 00:00 America/New_York; crash_date in [D-w, D-1] and :created_at < origin",
        },
    )
    print(features.groupby("station_id")["coll_all_500m_28d"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
