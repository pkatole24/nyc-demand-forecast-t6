# BikeBalance NYC

BikeBalance NYC explores hourly Citi Bike pickup and dropoff forecasting at eight Manhattan stations. The shared base is ready for additional predictor joins and chronological model comparisons. Forecasts could later support station balancing scenarios; trip net flow alone does not measure bike inventory or unmet demand. No model accuracy or operational benefit has been established.

## Shared dataset

Use `data/modeling/modeling_dataset.parquet`: 192,760 station-hour rows, 35 columns, January 1, 2024 through September 30, 2026. This is the only Parquet included in Git (approximately 9.0 MiB). Raw trips, intermediate datasets, research materials and personal implementation plans stay local.

The base includes pickups, dropoffs, net flow, calendar fields, Meteostat temperature/wind, audited IEM LGA precipitation, and missingness/provenance flags. Station IDs are strings. Timestamps retain America/New_York timezone information and uniquely identify UTC instants.

| Partition | Period | Rows |
| --- | --- | ---: |
| initial_train | 2024 | 70,272 |
| validation | 2025 | 70,080 |
| test | January-September 2026 | 52,408 |

See AGENTS.md for chronological model-selection and information-cutoff rules.

## Clone and install

Install Git and Python 3.12 first. Python 3.12 was used for preparation and local loading verification. Run these commands in a terminal:

```sh
git clone https://github.com/pkatole24/nyc-demand-forecast-t6.git
cd nyc-demand-forecast-t6
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe
```

macOS/Linux, with Python 3.12 installed:

```sh
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python
```

These commands use the environment's interpreter directly, so activation and PowerShell execution-policy changes are unnecessary. The last command opens a Python console showing `>>>`. Requirements contain only pandas and PyArrow for loading the base; pip installs their dependencies. Installation requires internet access, but loading the checked-in panel does not.

## Load and verify

Paste the following block into the Python console opened above. Keep the terminal in the repository root. The block also works in a notebook whose working directory is the repository root and whose kernel uses an environment with these dependencies.

```python
import hashlib
import json
from pathlib import Path
import pandas as pd

folder = Path("data/modeling")
path = folder / "modeling_dataset.parquet"
manifest = json.loads((folder / "manifest.json").read_text())
expected_hash = manifest["outputs_sha256"][path.name]
actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_hash != expected_hash:
    raise ValueError("Dataset hash mismatch: restore the committed base before using it.")

base = pd.read_parquet(path)
print(base.shape)
print(base[["station_id", "timestamp", "pickups", "dropoffs"]].head())
```

Expected shape: `(192760, 35)`. The first five rows are station `5184.08`, January 1, 2024, midnight through 04:00 New York time. Enter `exit()` to return to your terminal.

A missing-file error usually means the console/notebook is running outside the repository root; inspect `Path.cwd()`. An import error means the selected interpreter/kernel lacks the installed dependencies. A hash mismatch requires restoring the base, not skipping verification.

The manifest records the original complete local bundle. Only the main panel is distributed here; verify its hash independently. Other manifest-listed Parquets are intentionally omitted. A clone does not contain the preparation code or cached inputs needed to rebuild this base. Fresh dependency installation on other operating systems has not been verified.

## Working with predictors and models

Read [AGENTS.md](AGENTS.md) for the predictor integration workflow, coverage, data limits and evaluation rules. Read [data/modeling/data_quality_report.md](data/modeling/data_quality_report.md) for measured evidence and station details, and [data/modeling/README.md](data/modeling/README.md) for rainfall semantics.

## Repository contents

- `data/modeling/`: shared panel, schema notes, selected stations/crosswalk, source availability, manifest and saved validation.
- `data/modeling/data_quality_report.md`: shared-base coverage, station details, target/weather semantics and modeling limitations.
- `scripts/`: placeholder directory for future team scripts; currently contains only `.gitkeep`.
- `requirements.txt`: dependencies for reading the shared dataset.
- `AGENTS.md`: team working instructions for predictor joins, coverage, interpretation limits and evaluation.

This is a dataset handoff; download/preparation code and associated tests remain outside the shared repository. Cached availability dates describe archive revisions, not verified original release times. Historical station capacity and full-period inventory are not established by this dataset.

Failure modes: running from another directory or selecting a different notebook kernel can prevent loading; verify the working directory and interpreter. Predictor joins can duplicate rows or leak future information; follow the cardinality and availability checks in AGENTS.md.
