# BikeBalance NYC

[Open the live dashboard](https://nyc-demand-forecast-t6.streamlit.app/)

## Automatic weather updates

The `Refresh observed weather` GitHub Actions workflow runs every six hours (00:23, 06:23, 12:23 and 18:23 UTC), and can be started from **Actions → Refresh observed weather → Run workflow**. Scheduling begins after the workflow is published on `master`; GitHub can delay scheduled jobs. The PC does not need to stay on.

Each successful run uses the established Meteostat LaGuardia and IEM LGA sources, validates UTC continuity, recent observations, existing-cache hashes and preservation of previously observed values, then commits only `data/live/weather/weather_hourly.parquet` and `receipt.json`. Streamlit Cloud's repository integration receives those updates. Raw provider responses and the receipt are retained as an Actions artifact for 30 days. No API keys are required for these weather sources; publishing uses GitHub's built-in token with repository contents permission. Branch rules must allow that bot to push; a rejected push fails visibly rather than forcing a change.

The refresh covers weather after the frozen trip-history endpoint through the last completed hour, including month changes. It re-queries this small weather-only period to capture provider corrections. Missing observations and rainfall traces remain explicit. Failed retrieval or validation leaves the published cache unchanged. Monitor the latest workflow run and its coverage summary; a successful refresh is not a guarantee of model accuracy.

Daily SARIMA forecasts move forward from saved model states without retraining on each dashboard visit. New forecast dates do not create new observed trips: the observation cutoff remains September 30, 2026. ARIMAX additionally uses refreshed weather lagged 48 hours. New monthly trip data and model retraining remain separate tasks.

BikeBalance NYC forecasts hourly Citi Bike pickups and drop-offs at eight Manhattan stations. The Streamlit dashboard combines live station availability, forecasts for the next 24 hours, and historical model evaluation to explore station balancing needs.

The live engine offers seasonal naive, daily SARIMA, and ARIMAX with lagged weather. The repository also includes the seasonal-naive/moving-average notebook, weather notebook and station comparison, calendar SARIMAX, pooled Poisson regression, and city-data predictor experiments. Research results retain their own horizons, periods and information regimes. No operational benefit has been established through a field trial.

## Dashboard

| View | What it shows |
| --- | --- |
| General overview | Original project narrative, weekday station patterns, 2025/2026 model comparisons and expandable city-data RMSE/significance results. |
| Live station outlook | Current conditions and a station attention table, refreshed every 60 seconds. Modeled bike/dock gaps are separate from observed alerts; select a station to open its forecast. |
| Forecast explorer | Pickups and returns together, signed net change, exact forecast window and trip-history cutoff, with a separate deployed-model comparison. |
| Research & validation | 2025 selection and 2026 test summaries, model specifications, source links and historical replay of deployed models. |

In the saved daily-updated 2026 test comparison, pooled Poisson reduces pickup RMSE from 8.82 to 6.63 and return RMSE from 8.63 to 6.49 versus the last-week baseline, on 52,400 shared reported station-hours per target. Calendar SARIMAX has similar observed RMSE. These research models are not deployed in the live engine; intermediate predictor-study panels and caches are not included in the clone.

Codex assisted the dashboard redesign, result presentation and local verification. Other team AI-tool use should be disclosed by the relevant contributors.

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
