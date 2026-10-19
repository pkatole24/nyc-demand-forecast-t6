# BikeBalance NYC

BikeBalance NYC forecasts hourly Citi Bike pickups and drop-offs at eight Manhattan stations. The Streamlit dashboard combines live station availability, forecasts for the next 24 hours, and historical model evaluation to explore station balancing needs.

The submission covers three models: seasonal naive, daily SARIMA, and ARIMAX with lagged weather. SARIMA and ARIMAX selection, evaluation and deployment artifacts are complete and included in the submission. The dashboard compares saved evaluation results; no operational benefit has been established through a field trial.

## Dashboard

| View | What it shows |
| --- | --- |
| Live station overview | Available bikes, empty docks and station capacity for eight stations, with live status refreshed every 60 seconds. A model selector controls estimated extra bikes or empty docks needed over the next 24 hours. |
| Next 24 hours | Station-level pickup, drop-off and net-flow forecasts, with selectable model comparisons and timestamps in New York time. |
| Test evaluation | January–September 2026 actuals and forecasts, pooled all-station MAE/RMSE and scored station-hours, plus station-level charts and metrics. Controls select the information regime and visible period. |

Live status uses official Citi Bike GBFS feeds. Reports must pass freshness, operational-status and identity checks. Low-availability labels use 10% of usable capacity, with a minimum threshold of two. Unavailable or invalid reports are shown explicitly.

Estimated balancing needs use cumulative forecast net flow against the current usable bike/dock count. They assume no operator redistribution and uniform activity within partial hours. They are scenario estimates, not observed stockouts, unmet demand or recommendations validated through an operational trial.

## Models and evaluation

| Model | Predictors and behavior | Submission status |
| --- | --- | --- |
| Seasonal naive | Same local hour of the previous week; repeats the latest observed week across publication gaps. | Implemented; available without fitted-model artifacts. |
| Daily SARIMA | Separate station pickup/drop-off histories, autoregressive and moving-average terms, and 24-hour seasonality. | Selection, evaluation and exported models complete; included in the submission. |
| ARIMAX + past weather | Trip history, daily/weekly calendar cycles, and temperature, wind and rainfall lagged 48 hours, with explicit missing/trace flags. Monthly refitting. | Selection, evaluation and exported models complete; included in the submission. Current forecasts also require recent observed weather. |

Models use 2024 for initial fitting and 2025 for selection. Model choices are frozen before evaluation on January–September 2026. Evaluation distinguishes prior-day trip availability from monthly published-trip availability. All-station metrics pool station-hour errors on shared valid forecast support; net-flow errors are evaluated separately from pickup/drop-off errors. Missing target bins are excluded rather than filled with zeros.

The dashboard uses saved model forecasts and evaluation results. Results remain pending when the corresponding artifacts are unavailable. Historical observations end September 30, 2026, so current forecasts bridge an unobserved publication gap. Current ARIMAX forecasts also depend on sufficiently recent weather observations.

## Dataset

The immutable base is `data/modeling/modeling_dataset.parquet`: 192,760 station-hour rows, 35 columns, eight stations and 24,095 hourly instants per station, January 2024 through September 2026. The dataset is protected by a SHA-256 integrity check.

| Partition | Period | Rows | Purpose |
| --- | --- | ---: | --- |
| initial_train | 2024 | 70,272 | Initial fitting and feature development |
| validation | 2025 | 70,080 | Predictor and model selection |
| test | January–September 2026 | 52,408 | Frozen evaluation |

The base contains published pickups, drop-offs, net flow, calendar fields, Meteostat LaGuardia temperature/wind and audited IEM LGA rainfall. Station IDs are strings. Timestamps retain America/New_York timezone information; UTC instants identify hourly keys uniquely.

Net flow equals drop-offs minus pickups. Historical endpoint counts do not establish station inventory or unconstrained demand. The base preserves 30 missing pickup bins and 32 missing drop-off bins from ambiguous fall-back timestamps. Rainfall traces remain distinct from missing observations and measured numeric amounts. Airport weather is a regional covariate shared across stations.

See [data quality and selection limits](data/modeling/data_quality_report.md), [weather field semantics](data/modeling/README.md), and [repository methodology and contribution rules](AGENTS.md).

## Repository contents

- `app.py` and `dashboard_support.py`: dashboard views, forecasts, model loading and evaluation summaries.
- `bikebalance/`: data validation, forecasting, chronological evaluation, GBFS integration, weather and balancing calculations.
- `data/modeling/`: historical panel, station metadata, source availability, manifest and preparation validation.
- `artifacts/`: generated model selection, fitted models and evaluation outputs; completed dashboard model bundles and saved evaluation results are included in the submission.
- `scripts/`: forecasting stages, model export, runtime benchmarking and weather refresh.
- `tests/test_forecasting.py`: forecasting and dashboard behavior checks.
- `requirements.txt`: pinned application and forecasting dependencies.

## Interpretation limits

Station eligibility uses continuity through 2026, creating a retrospective selection filter. Saved usage statistics include 2025 and must be recomputed from prior history before use as predictors. Archive revision dates do not establish original publication times, and the final archive boundary may censor trips. These limits apply when interpreting historical model comparisons.

Failure modes: missing or mismatched model bundles can leave comparisons unavailable; receipt/hash checks identify incomplete exports. Stale live counts or weather can invalidate current estimates; freshness checks and explicit unavailable states prevent silent reuse. Forecast error, unobserved redistribution and within-hour trip ordering can alter balancing needs, which remain exploratory estimates.
