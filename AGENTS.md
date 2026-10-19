# BikeBalance NYC: methodology and contribution rules

## Scope and entry points

Develop hourly pickup/dropoff forecasts for eight Manhattan Citi Bike stations, evaluate additional predictors and explore station balancing scenarios. The project includes a prepared historical panel, a Streamlit dashboard, live GBFS integration and three forecasting implementations: seasonal naive, daily SARIMA and ARIMAX with lagged weather. Fitted-model and evaluation artifacts are complete and included in the submission; model comparisons require their saved evaluation receipts, and operational benefit remains unverified.

Use README.md for environment setup and loading. Before changing predictors, target handling or evaluation, read data/modeling/data_quality_report.md for measured evidence and selection limits. For rainfall transformations, read data/modeling/README.md for report-clock and observation semantics. Use the submitted repository and its documented artifacts as the reproducible project boundary.

## Base dataset and coverage

Use data/modeling/modeling_dataset.parquet as the immutable base: 192,760 rows, 35 columns, eight stations and 24,095 unique hourly instants per station, January 2024 through September 2026. Verify its SHA-256 against outputs_sha256 in data/modeling/manifest.json before deriving outputs. A hash mismatch must fail explicitly. The manifest and validation describe a larger local bundle; omitted Parquets do not establish readiness or inventory coverage in a clone. Historical work starts from the included panel and must not trigger automatic downloads or rebuilds.

| partition | Period | Rows | Role |
| --- | --- | ---: | --- |
| initial_train | 2024 | 70,272 | Initial fitting and feature development |
| validation | 2025 | 70,080 | Predictor/model selection |
| test | January-September 2026 | 52,408 | Frozen evaluation |

Preserve chronological partitions. Fit learned preprocessing using only history available in the evaluation regime; freeze predictors and model choices before inspecting test performance. Use origin cutoffs for every lag, rolling statistic and update. Random splits are inappropriate for this forecasting panel.

Weather coverage measured October 14, 2026 over unique hours: temperature 99.35%, wind 99.26%, observed rainfall including traces 99.76%, numeric rainfall 92.83%. Joint coverage is 99.07% when observed traces count as available, or 92.18% for three numeric values. For derived datasets, recompute coverage by partition and report denominators rather than repeating these base percentages.

Temperature is degrees C and wind is km/h from Meteostat LaGuardia 72503. Rainfall is mm from IEM LGA routine METARs. Weather is replicated across stations: it is a regional covariate, not eight independent local measurements. Preserve 1,670 observed rainfall traces with null numeric amounts and 57 genuinely missing rainfall hours. Use observation/trace/missing flags to distinguish them. Provider-normalized dry zeros have their own flag; no silent zero filling or imputation. Rain reporting intervals do not exactly equal trip bins.

## Predictor integration workflow

1. Establish each predictor's source/provenance, units, geographic scope, time grain, timezone and publication delay. Identify what is actually available at forecast issuance. Stop dependent feature work when these semantics are unresolved; progress on independent tasks. Realized future weather and retrospective disruption flags are excluded from known-ahead predictors; an oracle experiment must be labeled separately.
2. Read station IDs as strings at ingestion. Preserve timezone-aware base timestamps and derive timestamp_utc with dt.tz_convert("UTC"). Convert external timestamps according to their documented timezone, then to UTC. Never interpret naive local timestamps as UTC by convenience: unresolved fall-back hours require explicit handling.
3. Choose the join boundary. Station-hour predictors use station_id plus timestamp_utc and one-to-one validation. Citywide hourly predictors use timestamp_utc and many-to-one validation. Daily predictors use an explicitly derived America/New_York local date and many-to-one validation. Static station predictors join on station_id with many-to-one validation. Geographic predictors need a documented spatial assignment and unique resulting station keys.
4. Validate external key uniqueness before joining. Use a left join that preserves all base rows, with pandas merge(validate=...) and an indicator or equivalent diagnostics. Resolve conflicting column names explicitly; preserve target values and keys. Prove unchanged row count, unique station/UTC keys and target equality after the join.
5. Report matched/unmatched rows, missingness by station and partition, source coverage dates and information cutoffs. A matched key with a null predictor is still missing. Write enriched outputs separately from the base and record source/configuration provenance. Declare a predictor ready only after cardinality, target preservation, coverage and forecast-time availability are checked; an output file alone is not acceptance evidence.

## Station and target interpretation

Use data/modeling/selected_stations.csv for names, historical coordinates, descriptive usage labels and selection statistics. Join station_id many-to-one. Use station_crosswalk.csv by joining station_id to trip_station_id for captured GBFS UUID mappings. Validate against fresh metadata before live use. Captured capacities are not verified historical capacities; names alone do not establish station identity.

Station eligibility uses full-period continuity, including 2026; disclose this retrospective filter in evaluation. Usage and selection statistics use 2024-2025, including validation. Recompute any such predictor from prior history rather than using the saved statistic as if it were known at the start of 2025. Commuter/recreational labels are behavioral proxies, not verified trip purposes.

Preserve 30 missing pickup bins and 32 missing dropoff bins due to unresolved source timestamps in repeated fall-back hours. Use direction-specific uncertainty flags and preserve published zeros. Local days contain 23, 24 or 25 hours; UTC instants remain unique. The separately retained unresolved totals are not in the shared clone and cannot be allocated to specific hours.

net_flow = dropoffs - pickups describes endpoints, not observed inventory, shortage probability or unmet demand. February 23, 2026 has a retrospective systemwide no-published-pickups flag; it is not a verified hourly closure schedule or a known-ahead feature. Federal holidays do not cover all school closures/local events. Archive LastModified values are revision dates, not verified original publication times. The final archive boundary may censor trips represented in a later month; the loss is unquantified.

## Implementation and handoff

Challenge unsupported assumptions and surface meaningful design choices. Grow a working end-to-end implementation in small layers, separate concerns and use existing dependencies before introducing new ones. Remove obsolete paths instead of adding compatibility layers or speculative abstractions.

Add tests for concrete regressions, non-trivial invariants or observable boundaries. Prefer extending existing behavior-level coverage; avoid tests that mirror literals or merely assert removed code stays absent. Comments explain non-obvious reasoning, invariants and external quirks rather than restating code.

Report commands, measured results, scored/excluded rows, information regime and limitations. Distinguish saved preparation receipts, fresh runtime checks, model metrics and inferred operational implications. For each proposed feature or completed change, state likely failure modes in two or three lines and the checks that address them. Keep submission outputs reproducible from documented inputs and record any omitted source dependencies.
