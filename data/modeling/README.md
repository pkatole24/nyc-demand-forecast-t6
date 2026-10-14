# Shared modeling base

Use `modeling_dataset.parquet` for predictor joins. See the root README for setup, sample loading, partitions and coverage. The panel contains eight stations and 192,760 hourly rows with original targets/calendar fields, Meteostat temperature/wind and audited IEM LGA rainfall.

`timestamp` retains America/New_York timezone information. UTC instants identify hours uniquely. `precipitation_report_timestamp_utc` records the actual routine METAR report time, usually within the nominal report hour. Rain amounts describe the interval since the previous hourly reset and do not necessarily cover the exact trip bin; `interval_semantics` retains that distinction. Reports are not shifted or summed over overlapping intervals.

Trace rainfall has a null numeric amount, precipitation_trace=True, precipitation_observed=True and precipitation_missing=False. Unknown, sensor-unavailable and contradictory reports remain missing. Provider-normalized dry zeros retain zero_without_explicit_P_group; they are not individually measured zero rainfall. Raw amount tokens, METAR text, units, observation types and report-specific flags preserve provenance. Absence of a sensor flag alone does not prove a valid observation.

Temperature uses degrees C, wind uses km/h, and precipitation uses mm. No weather or target imputation is performed. Realized weather is not a known-ahead forecast predictor.

selected_stations.csv documents historical IDs and usage proxies. station_crosswalk.csv provides captured GBFS mappings; validate against live metadata before operational use, and treat capacity fields as captured metadata rather than historical capacity. source_availability.json contains cached official bucket revision dates rather than verified original publication dates.

manifest.json retains input/output hashes for the complete maintainer bundle. Verify the shared panel against outputs_sha256; other Parquets and cached raw/research inputs are omitted from Git. validation.json records preparation results for that full local bundle, including optional inventory samples that are not distributed here. The shared clone supports predictor exploration but cannot reconstruct the IEM-enhanced panel from these files alone.
