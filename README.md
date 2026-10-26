# BikeBalance NYC

[Open the live dashboard](https://nyc-demand-forecast-t6.streamlit.app/)

BikeBalance NYC forecasts hourly Citi Bike pickups and drop-offs at eight Manhattan stations. The Streamlit dashboard combines live station availability, forecasts for the next 24 hours, and historical model evaluation to explore station balancing needs.

The live engine offers seasonal naive, daily SARIMA, ARIMAX with lagged weather, SARIMAX with a weekly calendar, and the same SARIMAX with two-day-ahead weather forecasts. The repository also includes the seasonal-naive/moving-average notebook, weather notebook and station comparison, pooled Poisson regression, and city-data predictor experiments. Research results retain their own horizons, periods and information regimes. No operational benefit has been established through a field trial.

## Dashboard

| View | What it shows |
| --- | --- |
| General overview | Project motivation, weekday station patterns, model comparisons, findings and conclusions. |
| Live station outlook | Current conditions and a station attention table, refreshed every 60 seconds. Modeled bike/dock gaps are separate from observed alerts; select a station to open its forecast. Live views default to SARIMAX + calendar + weather forecast, or SARIMAX + weekly calendar when the weather forecasts cannot be retrieved. |
| Forecast explorer | Pickups and returns together, signed net change, exact forecast window and trip-history cutoff, with a separate deployed-model comparison. |
| Research & validation | 2025 selection and 2026 test summaries, model specifications, source links and historical replay of the last-week baseline, SARIMA and ARIMAX. |

In the saved daily-updated 2026 test comparison, on 52,400 shared reported station-hours per target, pooled Poisson reduces pickup RMSE from 8.82 to 6.63 and return RMSE from 8.63 to 6.49 versus the last-week baseline. Adding Open-Meteo weather forecasts issued 48 hours ahead lowers them further to 6.21 and 6.05, and lowers calendar SARIMAX from 6.67 and 6.59 to 6.37 and 6.29 (Diebold–Mariano p < 0.001 against the same model without weather). Both SARIMAX versions run in the live engine; the Poisson models remain research-only because they need the previous day's trips. Pooled Poisson remains research-only: it needs the previous day's trips at each station, which the monthly trip archive does not provide. Intermediate predictor-study panels and caches are not included in the clone.

We used Codex and Claude as development assistants for this project.

Live status uses official Citi Bike GBFS feeds. Reports must pass freshness, operational-status and identity checks. Low-availability labels use 10% of usable capacity, with a minimum threshold of two. Unavailable or invalid reports are shown explicitly.

Estimated balancing needs use cumulative forecast net flow against the current usable bike/dock count. They assume no operator redistribution and uniform activity within partial hours. They are scenario estimates, not observed stockouts, unmet demand or recommendations validated through an operational trial.

## Models and evaluation

| Model | Predictors and behavior | Evaluation and availability |
| --- | --- | --- |
| Seasonal naive | Same local hour of the previous week; repeats the latest observed week across publication gaps. | Implemented; available without fitted-model artifacts. |
| Daily SARIMA | Separate station pickup/drop-off histories, autoregressive and moving-average terms, and 24-hour seasonality. | Selection, evaluation and exported models complete; included in the submission. |
| ARIMAX + past weather | Trip history, daily/weekly calendar cycles, and temperature, wind and rainfall lagged 48 hours, with explicit missing/trace flags. Monthly refitting. | Selection, evaluation and exported models complete; included in the submission. Current forecasts also require recent observed weather. |
| SARIMAX + weekly calendar | SARIMA(1,0,1)(1,1,1)24 per station and target, with weekly Fourier terms and weekend/holiday (off-day) terms; no weather. | Saved 2025 validation and 2026 daily-updated test scores. Deployed live from `artifacts/sarimax_calendar/`: parameters estimated on February 2024–September 2026, forecasts run forward from the saved end-of-history state. |
| Pooled Poisson regression | Station-by-hour and calendar effects with recent-demand lags; parameters refit monthly. | Saved 2025 validation and 2026 daily-updated test scores; research evaluation only. |
| Poisson / SARIMAX + weather forecast | The models above plus Open-Meteo forecasts issued two days ahead: temperature, temperature squared, wind speed and a rain indicator (≥ 0.1 mm). | Chosen on 2025 and frozen in `data/story/frozen_choices_weather_forecast.json` before 2026 scoring. The SARIMAX version is deployed live from `artifacts/sarimax_weather_forecast/` and reads current Open-Meteo forecasts; the Poisson version is research-only. |
| Chronos-Bolt (AI, zero-shot) | Pretrained time-series foundation model `amazon/chronos-bolt-small`, no fine-tuning; the last 2,048 hourly observations per station and target as context; mean forecast. | Settings fixed a priori and recorded in `data/story/frozen_choices_ai.json` before 2026 scoring; research evaluation only (the dashboard does not install PyTorch). |

For the day-ahead model comparison, 2024 supplies initial fitting data and 2025 supports model selection. Model choices are frozen before evaluation on January–September 2026. The last-week baseline, SARIMA and ARIMAX also have a separate monthly-publication replay; SARIMAX + weekly calendar and Poisson have daily-updated test results only. All-station metrics pool station-hour errors on shared valid forecast support within each comparison; net-flow errors are evaluated separately from pickup/drop-off errors. Missing target bins are excluded rather than filled with zeros.

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
- `bikebalance/`: data validation, forecasting, chronological evaluation, GBFS integration, weather and balancing calculations; `sarimax_calendar.py` holds the live SARIMAX regressors, the Open-Meteo forecast retrieval and forecasting.
- `data/modeling/`: historical panel, station metadata, source availability, manifest and preparation validation.
- `data/story/`: saved validation/test comparisons, predictor summaries, model choices and daily-loss comparisons.
- `notebooks/`: seasonal-naive/moving-average and ARIMAX exploratory analyses.
- `results/arimax/`: saved station-level calendar-versus-weather comparison.
- `artifacts/`: generated model selection, fitted models and evaluation outputs; completed dashboard model bundles and saved evaluation results are included in the submission. `artifacts/sarimax_calendar/` and `artifacts/sarimax_weather_forecast/` store the live SARIMAX parameters and states (about 22 KB per station and target) with hashed manifests.
- `scripts/`: forecasting stages, predictor preparation/evaluation, model export (including `export_sarimax_live.py --model sarimax_calendar|sarimax_weather_forecast`, about 20–40 minutes each with seven workers), result summaries, runtime benchmarking and weather refresh.
- `tests/test_forecasting.py`: forecasting and dashboard behavior checks.
- `requirements.txt`: pinned application and forecasting dependencies.
- `requirements-analysis.txt`: additional dependencies for predictor evaluation and figures.
- `requirements-chronos.txt`: separate environment for the zero-shot AI forecasts (`scripts/forecast_chronos.py`, about one minute on a laptop CPU after the model download).

## Automatic weather updates

The [Refresh observed weather workflow](.github/workflows/refresh-weather.yml) is scheduled every six hours at 00:23, 06:23, 12:23 and 18:23 UTC. It retrieves Meteostat LaGuardia and IEM LGA observations after the historical trip endpoint, validates cache/base hashes and UTC continuity, and preserves rainfall trace and missing-value semantics. A successful run publishes the weather cache and receipt to the repository; failed retrieval or validation leaves the published cache unchanged. GitHub may delay scheduled runs.

Daily SARIMA and both SARIMAX models advance from saved model states without retraining on each dashboard visit. ARIMAX also uses refreshed weather lagged 48 hours; SARIMAX + weather forecast reads Open-Meteo forecasts directly and does not depend on this workflow. Weather refreshes do not add observed trips or retrain models: trip history still ends September 30, 2026. New trip data and model retraining require separate updates.

## Weather forecasts

Realized weather would cut day-ahead error by about 9% in the oracle experiment, but it is unknown at the midnight forecast origin, and weather lagged 48 hours (ARIMAX) did not beat the last-week baseline. The deployable middle ground is the weather **forecast** available at each origin. Adding it improved Poisson by about 6% and SARIMAX by about 4.5% on the 2026 test, roughly half of the oracle gain. Scripts: `collect_weather_forecasts.py`, `build_weather_forecast_features.py`, then `build_enriched_panel.py`, `evaluate_predictors.py weather_forecast`, `evaluate_sarimax.py` and `evaluate_test.py`.

- **Source:** [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), which archives what each model run predicted. Request LaGuardia (`latitude=40.7769&longitude=-73.874`, matching the observed-weather station) with `timezone=GMT`, for example: `https://previous-runs-api.open-meteo.com/v1/forecast?latitude=40.7769&longitude=-73.874&start_date=2024-01-01&end_date=2026-09-30&hourly=temperature_2m_previous_day2,precipitation_previous_day2,wind_speed_10m_previous_day2&timezone=GMT`.
- **Lead time:** use `_previous_day2` (the run initialized 48 hours before each valid hour). `_previous_day1` is initialized 24 hours before, so for late-day hours its run starts just before midnight and is published after the origin.
- **Coverage (checked October 2026):** temperature, precipitation and wind at both leads are 98.1% complete for January 2024–September 2026; only January 2024 is partial (first values January 19), before the February 2024 fitting start. Precipitation probability begins October 11, 2024, so it cannot enter 2024 fitting.
- **Join:** citywide hourly predictor on `timestamp_utc`, many-to-one, with the row-count, key and target checks in AGENTS.md; keep missing hours missing.
- **Evaluation:** the forecasts (temperature, temperature squared, wind, rain ≥ 0.1 mm) were added to the Poisson and SARIMAX models, chosen on 2025, and scored once on January–September 2026 with Diebold–Mariano tests.
- **Live use:** the deployed SARIMAX queries the same Previous Runs `_previous_day2` variables at forecast time. They cover the gap since the last published trips and about two days ahead, so live inputs match the evaluated ones; up to three missing hours in a row are carried from a neighbouring hour, and longer gaps make the model unavailable rather than substituting values.
- **Terms:** the free API is for non-commercial use; confirm the current licence and attribution requirements on Open-Meteo's terms page before publishing results.

## Interpretation limits

The SARIMAX weekly Fourier terms were computed from the panel's 8-bit weekday field, which wraps for hours from Saturday 08:00 onward; the live model reproduces those evaluated regressors exactly, and weekend behaviour is carried mainly by the off-day terms. A corrected encoding would need re-evaluation before deployment.

Station eligibility uses continuity through 2026, creating a retrospective selection filter. Saved usage statistics include 2025 and must be recomputed from prior history before use as predictors. Archive revision dates do not establish original publication times, and the final archive boundary may censor trips. These limits apply when interpreting historical model comparisons.

Failure modes: missing or mismatched model bundles can leave comparisons unavailable; receipt/hash checks identify incomplete exports. Stale live counts or weather can invalidate current estimates; freshness checks and explicit unavailable states prevent silent reuse. Forecast error, unobserved redistribution and within-hour trip ordering can alter balancing needs, which remain exploratory estimates.
