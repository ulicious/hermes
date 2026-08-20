# HERMES: Hydrogen Economy Routing Model for cost-efficient Supply

HERMES is a multi-commodity routing model for hydrogen and hydrogen-derived energy carriers. It combines production and conversion costs with ports, shipping routes, gas pipelines, liquid pipelines, roads, and potential new pipelines to search for cost-efficient supply paths to a user-defined destination.

The full user documentation is available on [Read the Docs](https://hermes-h2.readthedocs.io/en/main/index.html).

## Quickstart

HERMES keeps the source repository separate from a generated project folder. Install it with a virtual environment.

Linux and macOS (Bash):

```bash
git clone https://github.com/ulicious/hermes.git
cd hermes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/ulicious/hermes.git
Set-Location hermes
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installation normally takes approximately 10–20 minutes on a desktop computer
with a broadband connection. Download speed and the availability of binary
wheels for the geographic dependencies can change this substantially.

Then:

1. Open `_run_workflow.py` and set `PROJECT_FOLDER` to a separate working directory.
2. Set only `RUN_SETUP_PROJECT_FOLDER = True` and run `python _run_workflow.py`.
3. Edit the four YAML configuration files copied into `PROJECT_FOLDER`.
4. Disable setup and enable the required processing stages in `_run_workflow.py`.
5. Run `python _run_workflow.py` again.

All runner switches are disabled by default. Setup overwrites copied configuration files when it is deliberately run again. Input files that already match the published checksums are reused.

## Demo: minimal example

The HERMES demo is activated in the copied
`PROJECT_FOLDER/1_algorithm_configuration.yaml`:

```yaml
use_minimal_example: true
```

This restricts geographic preprocessing, start locations, and pipeline
infrastructure to the frame 35–71° N and 21° W–45° E. Start locations must also
lie on land classified as Europe by Natural Earth. It does not replace other
configuration values such as `number_locations`; the values supplied with the
demo configuration and data remain applicable.

After setup and configuration, select these runner stages:

```python
RUN_SETUP_PROJECT_FOLDER = False
RUN_PROCESS_RAW_DATA = True
RUN_CREATE_START_LOCATIONS = True
RUN_MAIN_ALGORITHM = True
```

Then run:

```bash
python _run_workflow.py
```

A successful demo run creates at least:

```text
PROJECT_FOLDER/processed_data/
PROJECT_FOLDER/start_destination_combinations.csv
PROJECT_FOLDER/results/location_results/
```

The location-results directory contains a final solution or a status file for
each processed start location.

Reference runtime for the minimal example:

- Raw-data and infrastructure processing: approximately 15 minutes
  (measured: 14.98 minutes).
- Start-location creation: approximately 2 minutes
  (measured: 1.71 minutes).
- Routing algorithm: approximately 8 minutes
  (measured: 8.38 minutes).
- Optional plot-data processing: approximately 23 seconds
  (measured: 22.66 seconds).
- Optional plot generation: approximately 31 seconds
  (measured: 30.83 seconds).
- Complete measured workflow including plot generation: approximately
  26 minutes (sum of measured stages: 25.96 minutes).

The measured times are reference values rather than guarantees. Runtime
depends on hardware, available CPU cores, storage performance, network speed for
the initial Zenodo data download, and the selected memory/storage settings.

## Inputs and outputs

Setup downloads the versioned input dataset from [Zenodo record 22031725](https://zenodo.org/records/22031725) to `PROJECT_FOLDER/raw_data/`. It verifies every file against the checksum published for this HERMES data release and extracts `natural_earth.zip`. The initial setup therefore requires Internet access; matching local files are reused on later setup runs.

Principal outputs are stored under:

- `results/location_results/` for per-location routing results
- `results/algorithm_tracking/` for optional tracking logs
- `results/processed_results/` for plotting-ready tables
- `results/plots/` for figures

See the documentation for the complete [workflow](https://hermes-h2.readthedocs.io/en/main/workflow.html), [configuration parameters](https://hermes-h2.readthedocs.io/en/main/parameters.html), [result files](https://hermes-h2.readthedocs.io/en/main/results.html), [custom data](https://hermes-h2.readthedocs.io/en/main/custom_data.html), [project structure](https://hermes-h2.readthedocs.io/en/main/project_overview.html), and [advanced usage](https://hermes-h2.readthedocs.io/en/main/advanced_usage.html).

## Data availability

The input data for this HERMES release is published separately as **HERMES raw data** via DOI [10.5281/zenodo.22031725](https://doi.org/10.5281/zenodo.22031725). Setup downloads this fixed record automatically so that all users work with the same input version.

## System requirements

HERMES currently supports Python 3.11 and the packages in `requirements.txt`.
The current code and dependency set have been used with:

- Python 3.11.9 on Microsoft Windows build 10.0.26200.8655
- Ubuntu 20.04.6 LTS for the original large case study

No non-standard hardware or GPU is required. Runtime and memory consumption
depend strongly on the geographic scope, input data, preprocessing options, and
number of parallel workers. The minimal example requires substantially fewer
resources than a full global run. Low-memory and low-storage modes are available
for constrained systems, with corresponding runtime trade-offs.

<details>
<summary>Software dependencies and supported versions</summary>

```text
numpy~=1.26.4
pandas~=2.2.0
scipy~=1.15.3
joblib~=1.3.2
networkx~=3.2.1
tables~=3.9.2
openpyxl~=3.1.5
shapely~=2.0.3
geopandas~=0.14.3
fiona~=1.9.6
pyproj~=3.7.1
rtree~=1.4.0
cartopy~=0.22.0
geopy~=2.4.1
geovoronoi==0.4.0
geojson~=3.1.0
vincenty==0.1.4
searoute~=1.3.1
matplotlib~=3.8.3
seaborn==0.13.2
plotly~=6.7.0
PyYAML~=6.0
tqdm~=4.66.2
psutil>=5.9,<8.0
gurobipy~=12.0.3
```

The authoritative install list is `requirements.txt`.

</details>

The optional MIP workflow additionally requires generated MIP data and a compatible Gurobi installation and licence. The Gurobi Python package is included in `requirements.txt`; the licence must be provided separately.

## Citation

Citation information is not yet available.

## Acknowledgements

- [SeaRoute](https://github.com/genthalili/searoute-py)
- NetworkX: Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function”, Proceedings of the 7th Python in Science Conference, 2008.
