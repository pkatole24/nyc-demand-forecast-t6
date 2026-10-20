"""Download predictor sources from Socrata portals through the SODA3 query API.

Covers NYC Open Data (data.cityofnewyork.us) and the MTA's datasets on
data.ny.gov. Writes raw or server-aggregated rows as JSON Lines plus a receipt
per dataset under data/raw/nyc_open_data/ (git-ignored). Uses only the
standard library so it runs before the project environment is installed.

Usage, from the repository root:
    python scripts/collect_nyc_open_data.py                    # all datasets
    python scripts/collect_nyc_open_data.py motor_collisions   # one or more by name

The app token comes from .env in the repository root (see .env.example):
    NYC_OPEN_DATA_APP_TOKEN   sent as X-App-Token (raises rate limits)
App tokens are platform-wide, so the same token is accepted by data.ny.gov.
Public datasets need no user authentication; on 2026-10-18 both portals
answered 403 to HTTP Basic API-key credentials, so none are sent.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "nyc_open_data"
NYC = "data.cityofnewyork.us"
NY_STATE = "data.ny.gov"
BASE_URL = "https://{domain}/api/v3/views/{view_id}/query.json"
PAGE_SIZE = 10_000

# Base panel window: January 2024 through the exclusive October 1, 2026 boundary.
START_DATE = "2024-01-01"
END_DATE = "2026-10-01"

# About 52k collisions in the window have a null borough but valid coordinates,
# so filtering on borough would silently drop them. This box covers Manhattan
# with margin around the eight stations; it also clips edges of Brooklyn,
# Queens and the Bronx, which a station-radius assignment discards later.
# Rows with missing or zero coordinates are excluded because they cannot be
# assigned to a station.
MANHATTAN_BBOX = {"lat_min": 40.68, "lat_max": 40.88, "lng_min": -74.03, "lng_max": -73.90}


def subway_daily(view_id, start, end):
    # Daily totals per Manhattan station complex. Hourly MTA timestamps are naive
    # local times that merge the two fall-back 01:00 hours, so only daily sums
    # are unambiguous. Selecting by complex happens later, in the feature build.
    return {
        "domain": NY_STATE,
        "view_id": view_id,
        "select": (
            "date_trunc_ymd(transit_timestamp) AS transit_date, station_complex_id,"
            " max(station_complex) AS station_complex, max(latitude) AS latitude,"
            " max(longitude) AS longitude, sum(ridership) AS ridership, count(*) AS source_rows"
        ),
        "where": (
            f"borough = 'Manhattan' AND transit_mode = 'subway'"
            f" AND transit_timestamp >= '{start}' AND transit_timestamp < '{end}'"
        ),
        "group_by": "transit_date, station_complex_id",
        "order_by": "transit_date, station_complex_id",
        "key": ["transit_date", "station_complex_id"],
        "max_date_field": "transit_date",
        "page_size": 50_000,
    }


def permitted_events(view_id):
    # Parks youth/adult sport permits (about 75% of Manhattan permits) are field
    # bookings rather than crowd events, so they are excluded at the source.
    # Events overlapping the window are kept; multi-day events start earlier.
    return {
        "domain": NYC,
        "view_id": view_id,
        "select": "*",
        "where": (
            "event_borough = 'Manhattan'"
            f" AND start_date_time < '{END_DATE}' AND end_date_time >= '2023-12-01'"
            " AND event_type NOT IN ('Sport - Youth', 'Sport - Adult')"
        ),
        "order_by": ":id",
        "key": None,
        "max_date_field": "start_date_time",
    }


DATASETS = {
    "motor_collisions": {
        "domain": NYC,
        "view_id": "h9gi-nx95",
        "select": "*",
        "where": (
            f"crash_date >= '{START_DATE}' AND crash_date < '{END_DATE}'"
            f" AND latitude BETWEEN {MANHATTAN_BBOX['lat_min']} AND {MANHATTAN_BBOX['lat_max']}"
            f" AND longitude BETWEEN {MANHATTAN_BBOX['lng_min']} AND {MANHATTAN_BBOX['lng_max']}"
        ),
        "order_by": ":id",
        "key": ["collision_id"],
        "max_date_field": "crash_date",
    },
    # Daily entries into the Congestion Relief Zone (tolling began 2025-01-05).
    # Fall-back 01:00 rows are merged in the source, so daily sums are used.
    "crz_entries_daily": {
        "domain": NY_STATE,
        "view_id": "t6yz-b64h",
        "select": (
            "toll_date, sum(crz_entries) AS crz_entries,"
            " sum(excluded_roadway_entries) AS excluded_roadway_entries, count(*) AS source_rows"
        ),
        "where": f"toll_date < '{END_DATE}'",
        "group_by": "toll_date",
        "order_by": "toll_date",
        "key": ["toll_date"],
        "max_date_field": "toll_date",
    },
    "subway_ridership_daily_2024": subway_daily("wujg-7c2s", START_DATE, "2025-01-01"),
    "subway_ridership_daily_2025_on": subway_daily("5wq4-mkjj", "2025-01-01", END_DATE),
    "permitted_events_historical": permitted_events("bkfu-528j"),
    "permitted_events_current": permitted_events("tvpp-9vvx"),
    "police_precincts": {
        "domain": NYC,
        "view_id": "y76i-bdw7",
        "select": "precinct, the_geom",
        "where": None,
        "order_by": "precinct",
        "key": ["precinct"],
        "max_date_field": None,
    },
}


def load_env(path):
    """Minimal KEY=VALUE .env reader; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def auth_headers():
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get("NYC_OPEN_DATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    else:
        print("warning: no app token in .env; requests may be throttled", file=sys.stderr)
    return headers


def post_query(url, query, headers, page_number=1, page_size=PAGE_SIZE, retries=5):
    body = json.dumps(
        {"query": query, "page": {"pageNumber": page_number, "pageSize": page_size}}
    ).encode()
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # 429 and 5xx are transient; auth and query errors are not.
            if error.code != 429 and error.code < 500 or attempt == retries - 1:
                detail = error.read().decode(errors="replace")[:500]
                raise RuntimeError(f"{url} HTTP {error.code}: {detail}") from error
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def build_queries(spec):
    clauses = f"SELECT {spec['select']}"
    if spec.get("where"):
        clauses += f" WHERE {spec['where']}"
    if spec.get("group_by"):
        clauses += f" GROUP BY {spec['group_by']}"
        count_query = f"{clauses} |> SELECT count(*)"
    else:
        count_query = "SELECT count(*)" + (f" WHERE {spec['where']}" if spec.get("where") else "")
    # A total ordering keeps pages stable and non-overlapping.
    return f"{clauses} ORDER BY {spec['order_by']}", count_query


def collect(name, spec, headers):
    url = BASE_URL.format(domain=spec["domain"], view_id=spec["view_id"])
    page_size = spec.get("page_size", PAGE_SIZE)
    query, count_query = build_queries(spec)

    expected = int(post_query(url, count_query, headers)[0]["count"])
    print(f"{name}: {expected:,} rows expected", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.partial")

    written, keys, page = 0, set(), 1
    max_date = None
    date_field = spec["max_date_field"]
    with tmp_path.open("w") as handle:
        while True:
            rows = post_query(url, query, headers, page_number=page, page_size=page_size)
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                if spec["key"]:
                    keys.add(tuple(row.get(k) for k in spec["key"]))
                value = row.get(date_field) if date_field else None
                if value and (max_date is None or value > max_date):
                    max_date = value
            written += len(rows)
            print(f"  page {page}: {written:,}/{expected:,}", flush=True)
            if len(rows) < page_size:
                break
            page += 1

    # The source can change mid-download; fail rather than keep a silently
    # inconsistent snapshot.
    if written != expected:
        tmp_path.unlink()
        raise RuntimeError(f"{name}: wrote {written:,} rows but count(*) was {expected:,}; rerun")
    if spec["key"] and len(keys) != written:
        tmp_path.unlink()
        raise RuntimeError(f"{name}: {written - len(keys):,} duplicate {spec['key']} values")
    tmp_path.replace(out_path)

    receipt = {
        "dataset": name,
        "domain": spec["domain"],
        "view_id": spec["view_id"],
        "endpoint": url,
        "query": query,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": written,
        "output": out_path.relative_to(ROOT).as_posix(),
        "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        # Fetch time is not publication time: these rows describe what the
        # portal held on this date, not what was known at each forecast origin.
        "note": "Snapshot of current portal contents; not an as-published history.",
    }
    if date_field:
        receipt[f"max_{date_field}"] = max_date
    receipt_path = OUT_DIR / f"{name}.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{name}: saved {out_path.relative_to(ROOT)}" + (f" (max {date_field} {max_date})" if date_field else ""))


def main(argv):
    load_env(ROOT / ".env")
    names = argv or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s): {', '.join(unknown)}; choose from {', '.join(DATASETS)}")
    headers = auth_headers()
    for name in names:
        collect(name, DATASETS[name], headers)


if __name__ == "__main__":
    main(sys.argv[1:])
