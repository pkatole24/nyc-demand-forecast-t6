"""Download archived day-ahead weather forecasts from the Open-Meteo Previous Runs API.

Each `_previous_day2` value is what the run initialized 48 hours before that
valid hour predicted, so it was published before the midnight forecast origin
of the valid hour's day. (`_previous_day1` runs start 24 hours before the valid
hour; for late-day hours they begin just before midnight and are published
after the origin.) The point is LaGuardia, matching the observed-weather
station in the base panel. Times are requested in GMT and stored as UTC.

Free for non-commercial use; check Open-Meteo's licence and attribution terms
before publishing results.

Output: data/raw/weather_forecasts/open_meteo_previous_day2.json (+ .receipt.json)

    python scripts/collect_weather_forecasts.py
"""

import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "weather_forecasts"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 40.7769, -73.874  # LaGuardia
START, END = "2024-01-01", "2026-10-01"  # panel ends 2026-10-01 03:00 UTC
VARIABLES = ["temperature_2m_previous_day2", "precipitation_previous_day2", "wind_speed_10m_previous_day2"]


def main():
    params = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": START, "end_date": END,
              "hourly": ",".join(VARIABLES), "timezone": "GMT"}
    url = f"{URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                body = response.read()
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    data = json.loads(body)
    hourly = data["hourly"]
    if any(len(hourly[v]) != len(hourly["time"]) for v in VARIABLES):
        raise ValueError("variable lengths differ from the time axis")
    if hourly.get("time") and data.get("utc_offset_seconds") != 0:
        raise ValueError("response is not in GMT/UTC")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "open_meteo_previous_day2.json"
    path.write_bytes(body)
    coverage = {v: round(sum(x is not None for x in hourly[v]) / len(hourly["time"]), 4) for v in VARIABLES}
    receipt = {"source": URL, "query": params, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
               "hours": len(hourly["time"]), "first_hour": hourly["time"][0], "last_hour": hourly["time"][-1],
               "coverage": coverage, "units": data.get("hourly_units"),
               "sha256": hashlib.sha256(body).hexdigest()}
    path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("hours", "first_hour", "last_hour", "coverage")}, indent=2))


if __name__ == "__main__":
    main()
