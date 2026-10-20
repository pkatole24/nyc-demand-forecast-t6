"""Congestion pricing policy indicator and lagged Congestion Relief Zone entries, per local day.

The CRZ toll started 2025-01-05 and was scheduled publicly in November 2024,
so the policy indicator is known ahead at every origin. Entry counts are
realized data: the portal's :created_at values all show a single 2026-10-16
reload, so the publication delay cannot be measured. The latest toll_date
on 2026-10-18 was 2026-09-30 (reloads appear weekly), so a conservative fixed
lag is used: day D sees mean daily entries over [D - 20, D - 14].

Entries are structurally zero before tolling rather than missing; the entries
feature is defined as 0 when the policy is inactive, and the indicator carries
the level shift. It is missing from 2025-01-05 until the first lagged window
contains a tolled day.

Input:  data/raw/nyc_open_data/crz_entries_daily.jsonl
Output: data/processed/features/congestion_daily.parquet (local_date, citywide)

    ./.venv/bin/python scripts/build_congestion_features.py
"""

import numpy as np
import pandas as pd

from predictor_common import PANEL_DATES, read_raw, write_features

POLICY_START = pd.Timestamp("2025-01-05")
LAG_DAYS = 14
WINDOW_DAYS = 7


def main():
    raw = read_raw("crz_entries_daily")
    daily = pd.DataFrame(
        {
            "local_date": pd.to_datetime(raw["toll_date"].str[:10]),
            "crz_entries": pd.to_numeric(raw["crz_entries"]).astype("float64"),
        }
    ).set_index("local_date")["crz_entries"]
    expected_days = pd.date_range(POLICY_START, daily.index.max(), freq="D")
    gaps = expected_days.difference(daily.index)
    if len(gaps):
        raise ValueError(f"CRZ source is missing {len(gaps)} days, first {gaps[0].date()}")

    entries = daily.reindex(PANEL_DATES)
    # Window [D-20, D-14]: shift by LAG_DAYS, then a trailing 7-day mean over
    # tolled days only (min_periods=1 lets the first tolled days count).
    lagged = entries.shift(LAG_DAYS).rolling(WINDOW_DAYS, min_periods=1).mean()

    active = PANEL_DATES >= POLICY_START
    lagged_log = np.log(lagged.to_numpy())
    lagged_log[~active] = 0.0
    features = pd.DataFrame(
        {
            "local_date": PANEL_DATES,
            "crz_active": active.astype("float64"),
            "crz_log_entries_lag14_7d": lagged_log,
        }
    )
    write_features(
        "congestion_daily",
        features,
        ["local_date"],
        ["crz_entries_daily"],
        {
            "policy_start": POLICY_START.date(),
            "lag_days": LAG_DAYS,
            "window_days": WINDOW_DAYS,
            "pre_policy_entries_feature": 0.0,
            "information_regime": "policy indicator known ahead; entries mean over [D-20, D-14], fixed lag (publication delay unmeasurable)",
        },
    )
    missing = features.loc[features["crz_log_entries_lag14_7d"].isna(), "local_date"]
    print(f"crz entries feature missing {len(missing)} days: {missing.min().date()} to {missing.max().date()}")
    print(features.loc[features["crz_active"] == 1, "crz_log_entries_lag14_7d"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
