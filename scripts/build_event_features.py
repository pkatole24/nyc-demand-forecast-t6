"""Permitted events active in each station's police precinct, per local day.

Spatial assignment: event rows carry police precinct lists but no
coordinates, so each station is placed in its NYPD precinct polygon and an
event counts when that precinct is listed.

Information regime (assumption): permits are issued before the event, so
events active on day D are treated as known at D 00:00. The portal records no
issue date, so this cannot be verified; late or cancelled permits would make
the feature look better than a prospective one.

Cleaning: Parks youth/adult sport permits are excluded at download. Rows whose
end precedes the start, and permits longer than MAX_DAYS (standing lawn
closures and similar), are dropped. One occurrence is (event_id, start, end);
an occurrence spanning several location rows is counted once per precinct.

Inputs: data/raw/nyc_open_data/permitted_events_{historical,current}.jsonl
        data/raw/nyc_open_data/police_precincts.jsonl
Output: data/processed/features/events_daily.parquet (station_id x local_date)

    ./.venv/bin/python scripts/build_event_features.py
"""

import numpy as np
import pandas as pd

from predictor_common import PANEL_DATES, load_stations, read_raw, write_features

MAX_DAYS = 7
SOURCES = ("permitted_events_historical", "permitted_events_current")
STREET_AGENCIES = {"Street Activity Permit Office", "Police Department", "Hudson River Park Trust"}
# Permits for logistics rather than gatherings.
NON_CROWD_TYPES = {
    "Production Event", "Theater Load in and Load Outs", "Shooting Permit",
    "Rigging Permit", "Filming/Photography", "Grid Request",
}


def point_in_multipolygon(lng, lat, multipolygon):
    """Even-odd ray casting; holes are inner rings of each polygon."""
    inside = False
    for polygon in multipolygon["coordinates"]:
        for ring in polygon:
            xs, ys = np.asarray(ring).T
            x1, y1, x2, y2 = xs[:-1], ys[:-1], xs[1:], ys[1:]
            crosses = ((y1 > lat) != (y2 > lat)) & (lng < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-300) + x1)
            inside ^= bool(crosses.sum() % 2)
    return inside


def station_precincts(stations):
    precincts = read_raw("police_precincts")
    out = {}
    for station in stations.itertuples():
        hits = [int(p.precinct) for p in precincts.itertuples() if point_in_multipolygon(station.lng, station.lat, p.the_geom)]
        if len(hits) != 1:
            raise ValueError(f"station {station.station_id} falls in {len(hits)} precincts")
        out[station.station_id] = hits[0]
    return out


def load_occurrences():
    raw = pd.concat([read_raw(s) for s in SOURCES], ignore_index=True)
    events = pd.DataFrame({
        "event_id": raw["event_id"].astype("string"),
        "start": pd.to_datetime(raw["start_date_time"]),
        "end": pd.to_datetime(raw["end_date_time"]),
        "agency": raw["event_agency"],
        "type": raw["event_type"],
        "precincts": raw["police_precinct"].fillna("").map(
            lambda s: {int(p) for p in s.split(",") if p.strip().isdigit()}
        ),
    })
    total = len(events)
    events = events[(events["end"] >= events["start"]) & (events["end"] - events["start"] <= pd.Timedelta(days=MAX_DAYS))]
    dropped = total - len(events)

    occ = events.explode("precincts").dropna(subset=["precincts"])
    occ = occ.drop_duplicates(["event_id", "start", "end", "precincts"])
    occ["street"] = occ["agency"].isin(STREET_AGENCIES) & ~occ["type"].isin(NON_CROWD_TYPES)
    # Naive local times. An end exactly at midnight does not occupy that day.
    occ["first_day"] = occ["start"].dt.normalize()
    occ["last_day"] = (occ["end"] - pd.Timedelta(microseconds=1)).clip(lower=occ["start"]).dt.normalize()
    return occ, total, dropped


def daily_counts(occ, precinct):
    """Occurrences active on each panel date, via a difference array over days."""
    here = occ[occ["precincts"] == precinct]
    index = {d: i for i, d in enumerate(PANEL_DATES)}
    out = {}
    for name, rows in (("events_all_precinct", here), ("events_street_precinct", here[here["street"]])):
        diff = np.zeros(len(PANEL_DATES) + 1)
        first = rows["first_day"].clip(lower=PANEL_DATES[0])
        last = rows["last_day"].clip(upper=PANEL_DATES[-1])
        keep = first <= last
        np.add.at(diff, first[keep].map(index).to_numpy(dtype=int), 1)
        np.add.at(diff, last[keep].map(index).to_numpy(dtype=int) + 1, -1)
        out[name] = np.cumsum(diff)[:-1]
    return out


def main():
    stations = load_stations()
    precinct_of = station_precincts(stations)
    occ, total, dropped = load_occurrences()

    frames = []
    for station_id, precinct in precinct_of.items():
        frames.append(pd.DataFrame({"station_id": station_id, "local_date": PANEL_DATES, **daily_counts(occ, precinct)}))
    features = pd.concat(frames, ignore_index=True)
    features["station_id"] = features["station_id"].astype("string")

    write_features(
        "events_daily",
        features,
        ["station_id", "local_date"],
        list(SOURCES) + ["police_precincts"],
        {
            "station_precinct": precinct_of,
            "max_event_days": MAX_DAYS,
            "rows_downloaded": total,
            "rows_dropped_bad_or_long_duration": dropped,
            "street_agencies": sorted(STREET_AGENCIES),
            "non_crowd_types": sorted(NON_CROWD_TYPES),
            "information_regime": "assumed known ahead: permits issued before the event (unverified)",
        },
    )
    print("station precincts:", precinct_of)
    print(f"rows {total:,}, dropped {dropped:,}, occurrence-precinct pairs {len(occ):,}")
    print(features.groupby("station_id")[["events_all_precinct", "events_street_precinct"]].agg(["mean", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
