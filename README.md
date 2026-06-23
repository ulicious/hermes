# HERMES: <ins>H</ins>ydrogen <ins>E</ins>conomy <ins>R</ins>outing <ins>M</ins>odel for cost-<ins>e</ins>fficient <ins>S</ins>upply

HERMES is a multi commodity multi mean of transport algorithm,
capable to find the most cost-effective transportation route from pre-defined starting points to a desired location.
The algorithm derives possible solutions for transportation from a starting point to the final destination.
Based on the underlying infrastructure data, the algorithm iteratively explores infrastructure nodes and calculates
costs for each reached node. It terminates as soon as it reaches the final destination and exploration
to other nodes is not possible anymore

# Documentation

A full documentary is provided here: https://hermes-h2.readthedocs.io/en/main/index.html. There you can find more information on the installation process, which should only take a few minutes, and the operation of the code.

# Data availability & Demo

Most of the data used are available in this repository, but can be found here as well: DOI: 10.5281/zenodo.15350282

Configuration templates and provided input data are stored in the repository `data` folder. Before processing data, use `_run_workflow.py` with `PROJECT_FOLDER` set and `RUN_SETUP_PROJECT_FOLDER = True` to create the working folder structure and copy the editable configuration files directly into `PROJECT_FOLDER/` and the raw input data into `PROJECT_FOLDER/raw_data/`. The setup step writes the given project folder path into the copied `1_algorithm_configuration.yaml`. If setup is run again, copied files are overwritten.

Users should adjust `1_algorithm_configuration.yaml`, `2_techno_economic_data_transportation.yaml`, `3_techno_economic_data_conversion.yaml`, and `4_plotting_configuration.yaml` in `PROJECT_FOLDER/`. Only the central `project_folder_path` is stored as a path in `1_algorithm_configuration.yaml`. All subfolders and raw-data file names are fixed by the code and are derived from the project folder. For example, `location_data.csv` and `country_data.csv` are expected in `PROJECT_FOLDER/raw_data/` and are not configured in the YAML file.

If an older project folder still contains numbered configuration files or a `config/` subfolder from a previous layout, the setup step removes the known obsolete configuration files and writes the current files directly into `PROJECT_FOLDER/`.

The workflow scripts are stored in `scripts/` and are normally started through `_run_workflow.py`. Advanced users can still run a single step from the repository root via module call, e.g. `python -m scripts._3_main PROJECT_FOLDER` or `python -m scripts._3_main --project-folder PROJECT_FOLDER`. To run the main algorithm with an alternative algorithm configuration file, use `python -m scripts._3_main PROJECT_FOLDER --algorithm-config path/to/config.yaml`. The `HERMES_PROJECT_FOLDER` environment variable can also be used.

For less experienced users, `_run_workflow.py` provides one central entry point. Set `PROJECT_FOLDER` and the `RUN_*` booleans at the top of the file, then run `python _run_workflow.py`. The script starts the selected workflow modules in order and passes the project folder automatically.

The central runner can start setup, raw-data processing, start-location creation, the main algorithm, MIP optimization, plot-data processing, plotting, and algorithm-tracking analysis. Keep only the desired `RUN_*` flags set to `True`.

For running multiple algorithm scenarios without repeatedly editing `1_algorithm_configuration.yaml`, set `RUN_ALGORITHM_CONFIG_BATCH = True` in `_run_workflow.py` and place alternative algorithm configuration YAML files in `PROJECT_FOLDER/algorithm_configurations/`. The runner executes `scripts._3_main` once per YAML file for all remaining locations. Results for each scenario are written to `PROJECT_FOLDER/results/<configuration filename>/location_results/` and tracking logs to `PROJECT_FOLDER/results/<configuration filename>/algorithm_tracking/`. The default `1_algorithm_configuration.yaml` still writes to the normal `PROJECT_FOLDER/results/location_results/` folder.

This data is the necessary input for the full model and the demo version. For the demo version, please indicate that in the general configuration at: use_minimal_example. This will only consider Europe.

Runtime of the data processing and calculation of the case study is around 1 hour on a normal desktop computer, but heavily depends on the system's hardware.

The demo will calculate the most cost-efficient transport routes and provides most cost-efficient supply costs.

# System Requirements

The model should be available on any operating system which allows the utilization of the applied python packages. It was tested with the following system:

The processor is an AMD Ryzen Threadripper 3990X 64-Core at 2.9 GHz, allowing us to utilize up to 128 CPUs. The system has 256 GB of RAM. The installed operating system is Ubuntu 20.04.6 LTS.

## Python dependencies

geopandas~=0.14.3
matplotlib~=3.8.3
pandas~=2.2.0
shapely~=2.0.3
joblib~=1.3.2
tqdm~=4.66.2
geopy~=2.4.1
requests~=2.31.0
searoute~=1.3.1
networkx~=3.2.1
numpy~=1.26.4
h5py~=3.7.0
pyyaml~=6.0.1
geojson~=3.1.0
cartopy~=0.22.0
vincenty~=0.1.4
openpyxl~=3.1.2
tables~=3.9.2

# Citation

Soon

# Big thanks to

- genthalili's SeaRoute package (https://github.com/genthalili/searoute-py/tree/main)
- NetworkX: Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

# todos:
- test again with single point
