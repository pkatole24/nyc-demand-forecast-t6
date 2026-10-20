# BikeBalance NYC

[Open the live dashboard](https://nyc-demand-forecast-t6.streamlit.app/)

BikeBalance NYC forecasts hourly Citi Bike pickups and drop-offs at eight Manhattan stations. The Streamlit dashboard combines live station availability, forecasts for the next 24 hours, and historical model evaluation to explore station balancing needs.

The live engine offers seasonal naive, daily SARIMA, and ARIMAX with lagged weather. The repository also includes the seasonal-naive/moving-average notebook, weather notebook and station comparison, calendar SARIMAX, pooled Poisson regression, and city-data predictor experiments. Research results retain their own horizons, periods and information regimes. No operational benefit has been established through a field trial.

## Dashboard

| View | What it shows |
| --- | --- |
| General overview | Project motivation, weekday station patterns, model comparisons, findings and conclusions. |
| Live station outlook | Current conditions and a station attention table, refreshed every 60 seconds. Modeled bike/dock gaps are separate from observed alerts; select a station to open its forecast. |
| Forecast explorer | Pickups and returns together, signed net change, exact forecast window and trip-history cutoff, with a separate deployed-model comparison. |
| Research & validation | 2025 selection and 2026 test summaries, model specifications, source links and historical replay of deployed models. |

In the saved daily-updated 2026 test comparison, pooled Poisson reduces pickup RMSE from 8.82 to 6.63 and return RMSE from 8.63 to 6.49 versus the last-week baseline, on 52,400 shared reported station-hours per target. Calendar SARIMAX has similar observed RMSE. These research models are not deployed in the live engine; intermediate predictor-study panels and caches are not included in the clone.

We used Codex and Claude as development assistants for this project.

Live status uses official Citi Bike GBFS feeds. Reports must pass freshness, operational-status and identity checks. Low-availability labels use 10% of usable capacity, with a minimum threshold of two. Unavailable or invalid reports are shown explicitly.

Estimated balancing needs use cumulative forecast net flow against the current usable bike/dock count. They assume no operator redistribution and uniform activity within partial hours. They are scenario estimates, not observed stockouts, unmet demand or recommendations validated through an operational trial.

## Models and evaluation

| Model | Predictors and behavior | Evaluation and availability |
| --- | --- | --- |
| Seasonal naive | Same local hour of the previous week; repeats the latest observed week across publication gaps. | Implemented; available without fitted-model artifacts. |
| Daily SARIMA | Separate station pickup/drop-off histories, autoregressive and moving-average terms, and 24-hour seasonality. | Selection, evaluation and exported models complete; included in the submission. |
| ARIMAX + past weather | Trip history, daily/weekly calendar cycles, and temperature, wind and rainfall lagged 48 hours, with explicit missing/trace flags. Monthly refitting. | Selection, evaluation and exported models complete; included in the submission. Current forecasts also require recent observed weather. |
| SARIMAX + weekly calendar | Station-level daily seasonal model with weekly Fourier and off-day terms. | Saved 2025 validation and 2026 daily-updated test scores; research evaluation only. |
| Pooled Poisson regression | Station-by-hour and calendar effects with recent-demand lags; parameters refit monthly. | Saved 2025 validation and 2026 daily-updated test scores; research evaluation only. |

For the day-ahead model comparison, 2024 supplies initial fitting data and 2025 supports model selection. Model choices are frozen before evaluation on January–September 2026. The three deployed approaches also have a separate monthly-publication replay; the two additional research models have daily-updated test results only. All-station metrics pool station-hour errors on shared valid forecast support within each comparison; net-flow errors are evaluated separately from pickup/drop-off errors. Missing target bins are excluded rather than filled with zeros.

The seasonal-naive/moving-average notebook explores one-hour-ahead pickup forecasts over a short final-period window; the ARIMAX notebook explores calendar, lagged-weather and annual-seasonality variants. Their horizons, scored support and preprocessing differ from the day-ahead comparison. The predictor study tests collision, subway, street-event and congestion-pricing features against matched baselines; realized-weather oracle specifications are identified separately from deployable predictors. See the [notebooks](notebooks/), [saved station weather comparison](results/arimax/station_model_comparison_2025.csv) and [research summaries](data/story/).

The dashboard uses saved model forecasts and evaluation results. It displays an unavailable state when required artifacts or inputs cannot be verified. Historical observations end September 30, 2026, so current forecasts bridge an unobserved publication gap. Current ARIMAX forecasts also depend on sufficiently recent weather observations.

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

- `app.py`, `dashboard_research.py` and `dashboard_support.py`: dashboard views, research presentation, forecasts, model loading and evaluation summaries.
- `bikebalance/`: data validation, forecasting, chronological evaluation, GBFS integration, weather and balancing calculations.
- `data/modeling/`: historical panel, station metadata, source availability, manifest and preparation validation.
- `data/story/`: saved validation/test comparisons, predictor summaries, model choices and daily-loss comparisons.
- `notebooks/`: seasonal-naive/moving-average and ARIMAX exploratory analyses.
- `results/arimax/`: saved station-level calendar-versus-weather comparison.
- `artifacts/`: generated model selection, fitted models and evaluation outputs; completed dashboard model bundles and saved evaluation results are included in the submission.
- `scripts/`: forecasting stages, predictor preparation/evaluation, model export, result summaries, runtime benchmarking and weather refresh.
- `tests/test_forecasting.py`: forecasting and dashboard behavior checks.
- `requirements.txt`: pinned application and forecasting dependencies.
- `requirements-analysis.txt`: additional dependencies for predictor evaluation and figures.

## Automatic weather updates

The [Refresh observed weather workflow](.github/workflows/refresh-weather.yml) is scheduled every six hours at 00:23, 06:23, 12:23 and 18:23 UTC. It retrieves Meteostat LaGuardia and IEM LGA observations after the historical trip endpoint, validates cache/base hashes and UTC continuity, and preserves rainfall trace and missing-value semantics. A successful run publishes the weather cache and receipt to the repository; failed retrieval or validation leaves the published cache unchanged. GitHub may delay scheduled runs.

Daily SARIMA advances from saved model states without retraining on each dashboard visit. ARIMAX also uses refreshed weather lagged 48 hours. Weather refreshes do not add observed trips or retrain models: trip history still ends September 30, 2026. New trip data and model retraining require separate updates.

## Interpretation limits

Station eligibility uses continuity through 2026, creating a retrospective selection filter. Saved usage statistics include 2025 and must be recomputed from prior history before use as predictors. Archive revision dates do not establish original publication times, and the final archive boundary may censor trips. These limits apply when interpreting historical model comparisons.

Failure modes: missing or mismatched model bundles can leave comparisons unavailable; receipt/hash checks identify incomplete exports. Stale live counts or weather can invalidate current estimates; freshness checks and explicit unavailable states prevent silent reuse. Forecast error, unobserved redistribution and within-hour trip ordering can alter balancing needs, which remain exploratory estimates.
