# Dataset quality report

This report describes `data/modeling/modeling_dataset.parquet`, the processed historical base used by the dashboard and chronological forecasting experiments. This report documents dataset quality and selection limits. Model performance is reported separately through completed evaluation artifacts; saved preparation validation does not establish forecasting accuracy or operational benefit.

## Coverage and partitions

The panel has 192,760 rows across eight Manhattan stations and 24,095 unique hourly instants per station, from January 1, 2024 through September 30, 2026. Each station has a complete timezone-aware hourly grid. The exclusive end boundary is October 1, 2026 at midnight America/New_York.

| Partition | Period | Station-hour rows |
| --- | --- | ---: |
| initial_train | 2024 | 70,272 |
| validation | 2025 | 70,080 |
| test | January-September 2026 | 52,408 |

The cached official source check on October 13, 2026 found all 33 requested monthly archives. Original preparation processed 125,217,844 source records across 141 extracted CSVs. These are recorded preparation results; raw archive bytes were not rehashed during the modeling-base handoff. Cached archive LastModified values describe revisions, not verified original publication dates.

## Station details and geographic joins

Read `selected_stations.csv` with `dtype={"station_id": "string"}`. It contains station names, historical coordinates, usage labels, activity/continuity statistics and selection metrics. Read `station_crosswalk.csv` with string `trip_station_id` and `gbfs_station_id`; it connects historical trip IDs to captured GBFS UUIDs and capacity fields. Join the panel to selected stations on station_id with many-to-one validation. For the crosswalk, join station_id to trip_station_id.

| Station ID | Name | Latitude | Longitude | Usage proxy |
| --- | --- | ---: | ---: | --- |
| 5297.02 | Vesey Pl & River Terrace | 40.715338 | -74.016584 | commuter-leaning |
| 6955.05 | W 54 St & 11 Ave | 40.768333 | -73.992573 | commuter-leaning |
| 6197.08 | E 33 St & 1 Ave | 40.743227 | -73.974498 | commuter-leaning |
| 5184.08 | West St & Liberty St | 40.711444 | -74.014847 | commuter-leaning |
| 7059.08 | West End Ave & W 60 St | 40.772370 | -73.990050 | commuter-leaning |
| 6765.01 | 12 Ave & W 40 St | 40.760875 | -74.002777 | recreational-leaning |
| 5696.03 | Pier 40 - Hudson River Park | 40.727714 | -74.011296 | recreational-leaning |
| 5805.07 | 7 Ave S & Bleecker St | 40.732139 | -74.003638 | recreational-leaning |

Manhattan membership was checked against the NYC DCP borough polygon. Eligible stations appeared in all 33 months, had at least 85% active local days each month, at least 1,000 monthly pickups and dropoffs, and monthly station-center span no greater than 250 m. Usage selection used 2024-2025 activity, casual share, weekend/weekday ratios and weekday peak pickup share. Labels describe observed behavior, not verified trip purpose.

Full-window eligibility uses 2026 station continuity: disclose this retrospective selection when evaluating models. Selection statistics calculated over 2024-2025 also include the validation year; do not treat them as predictors known at the start of 2025 without recomputing them using prior history. Captured capacity values are not verified historical capacities. Validate GBFS identifiers and capacity against fresh information before live use. Names alone do not prove physical station identity.

## Targets, zeros and DST

Pickups and dropoffs count published trip endpoints. Net flow equals dropoffs minus pickups and remains null when either target is unresolved. It does not measure inventory, stockouts or unconstrained demand.

| Target condition | Station-hour bins |
| --- | ---: |
| Missing pickups | 30 |
| Missing dropoffs | 32 |
| Published zero pickups | 35,614 |
| Published zero dropoffs | 32,441 |

Original trip timestamps lack offsets. Repeated fall-back hours cannot be assigned uniquely; affected direction-specific bins remain missing. Original preparation retained unresolved combined totals separately: 85 selected-station pickups and 116 dropoffs. That separate Parquet is not distributed in the submitted base dataset. Do not allocate those totals or turn unresolved targets into zero.

Use UTC instants as hourly join keys. Local spring-forward dates contain 23 hours and fall-back dates contain 25. The two fall-back 01:00 hours have distinct offsets. The holiday field uses US federal holidays, including observed dates; it does not cover every school closure or local event.

February 23, 2026 has no published valid systemwide pickups and retains its audit flag. The original preparation associated this with storm disruption evidence, but exact reopening times are not established. Published zeros can reflect closures or source omissions and do not establish zero unconstrained demand. The flag is retrospective and unavailable as a known-ahead predictor.

## Weather in the shared base

Temperature and wind use Meteostat LaGuardia station 72503 observations. Rainfall uses audited IEM LGA routine METAR reports, replacing the original preparation's Meteostat rainfall. One airport series is replicated across eight stations; it is a regional covariate, not street-level weather at each station.

Coverage measured directly from the shared panel on October 14, 2026, before station replication:

| Field | Available hours / 24,095 | Coverage | Missing hours |
| --- | ---: | ---: | ---: |
| Temperature (degrees C) | 23,938 | 99.35% | 157 |
| Wind (km/h) | 23,917 | 99.26% | 178 |
| Observed rainfall, including traces | 24,038 | 99.76% | 57 |
| Numeric rainfall (mm) | 22,368 | 92.83% | 1,727 numeric nulls |
| Temperature, wind and observed rainfall together | 23,872 | 99.07% | 223 incomplete hours |
| All three numeric values together | 22,211 | 92.18% | 1,884 incomplete hours |

There are 1,670 observed traces with null numeric amounts. A trace is observed, not missing. Of usable rainfall hours, 20,689 are provider-normalized dry values, with a separate flag; these are not individually measured zeros. Unknown, unavailable-sensor and contradictory reports remain missing. No imputation is applied.

The nominal report-hour timestamp is joined to the trip grid, while the actual report time is retained separately. Rain amounts cover the interval since the previous hourly reset, which does not necessarily match the exact trip bin. Raw METAR text, amount tokens and interpretation flags remain in the panel. See the adjacent README for field semantics.

Realized historical weather may be revised and is not available ahead of the forecast origin. Use known-ahead weather forecasts, lagged observations, or an explicitly labeled oracle experiment. High observation coverage alone does not establish that a weather predictor is usable prospectively.

## Preparation checks and evidence limits

Saved preparation validation passed checks for complete grids, unique station/UTC keys, weather join cardinality, target equality/conservation, precipitation flags, station/crosswalk consistency and cached source integrity. During the team handoff, all eight local modeling outputs matched manifest hashes and the shared panel had zero duplicate station/timestamp keys. The original preparation bundle contains optional inventory and unresolved-total Parquets; only the main historical panel is distributed from that preparation bundle.

Original preprocessing handled 512 cross-month repeated-ID excess records, including 29 conflicting variants, with deterministic global deduplication. Missing source endpoint IDs excluded only the affected endpoint: 70,549 starts and 359,809 ends. These source-level counts are recorded preparation evidence, not a fresh scan of raw archives. Trips crossing the final requested boundary may be right-censored if they appear in a later archive; that loss is unquantified.

Before evaluating new predictors, report join cardinality, unmatched coverage, units, missingness by partition and publication timing. Preserve the base and save enriched results separately. Verify the panel SHA-256 against manifest.json; the manifest also lists local files intentionally omitted from Git.

Failure modes: retrospective continuity selection can make results look stronger than a prospective station universe, while airport weather can hide local effects. Duplicate or ambiguous-time joins can distort targets, and later-published predictors can leak future information. Validate each join and information cutoff before interpreting model gains.
