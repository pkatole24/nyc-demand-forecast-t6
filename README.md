# BikeBalance NYC

BikeBalance NYC explores hourly Citi Bike pickup and dropoff forecasting at eight Manhattan stations. The shared base is ready for additional predictor joins and chronological model comparisons. Forecasts could later support station balancing scenarios; trip net flow alone does not measure bike inventory or unmet demand. No model accuracy or operational benefit has been established.

## Start with the shared dataset

Use `data/modeling/modeling_dataset.parquet`: 192,760 station-hour rows, 35 columns, January 1, 2024 through September 30, 2026. This is the only Parquet included in Git (approximately 9.0 MiB). Raw trips, intermediate datasets, research materials and personal implementation plans stay local.

The base includes pickups, dropoffs, net flow, calendar fields, Meteostat temperature/wind, audited IEM LGA precipitation, and missingness/provenance flags. Station IDs are strings. Timestamps retain America/New_York timezone information and uniquely identify UTC instants.

| Partition | Period | Rows |
| --- | --- | ---: |
| initial_train | 2024 | 70,272 |
| validation | 2025 | 70,080 |
| test | January-September 2026 | 52,408 |

See AGENTS.md for chronological model-selection and information-cutoff rules.

## Setup and loading

Python 3.12 was used for preparation. From the repository root:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
```

```python
import hashlib
import json
from pathlib import Path
import pandas as pd

folder = Path("data/modeling")
path = folder / "modeling_dataset.parquet"
manifest = json.loads((folder / "manifest.json").read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["outputs_sha256"][path.name]
base = pd.read_parquet(path)
print(base.head())
```

The manifest records the original complete local bundle. Only the main panel is distributed here; verify its hash independently. Other manifest-listed Parquets are intentionally omitted. A clone does not contain the cached inputs needed to rebuild this base.

## Working with predictors and models

Read [AGENTS.md](AGENTS.md) for the predictor integration workflow, coverage, data limits and evaluation rules. Read [data/modeling/data_quality_report.md](data/modeling/data_quality_report.md) for measured evidence and station details, and [data/modeling/README.md](data/modeling/README.md) for rainfall semantics.

## Repository contents

- `data/modeling/`: shared panel, schema notes, selected stations/crosswalk, source availability, manifest and saved validation.
- `data/modeling/data_quality_report.md`: shared-base coverage, station details, target/weather semantics and modeling limitations.
- `scripts/`: original trip/weather/geography download and preparation code; these produce the original preparation, not the shared IEM-enhanced base. The full pipeline downloads data and should not be run just to use the panel.
- `tests/`: focused original trip aggregation and DST grid checks.
- `AGENTS.md`: team working instructions for predictor joins, coverage, interpretation limits and evaluation.

Research-dependent forecasting preparation is retained locally and excluded from the shared repository. Cached availability dates describe archive revisions, not verified original release times. Historical station capacity and full-period inventory are not established by this dataset.

Run the shared preparation checks with `python -m unittest discover -s tests` from a clone. In the maintainer's full local checkout, research-dependent tests also exist; run `python -m unittest discover -s tests -p test_time_grid.py` and `python -m unittest discover -s tests -p test_trip_aggregation.py` for the shared checks alone.

Failure modes: duplicate join keys can multiply targets, and timezone mistakes can shift predictors around DST. Publication delays can leak future information despite matching timestamps. Validate join cardinality, unmatched coverage and availability at forecast issuance before evaluating added predictors.
