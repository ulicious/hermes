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

Configuration templates and provided input data are stored in the repository `data` folder. Before processing data, run `_0_setup_project_folder.py PROJECT_FOLDER` to create the working folder structure and copy the editable configuration files directly into `PROJECT_FOLDER/` and the raw input data into `PROJECT_FOLDER/raw_data/`. The setup script writes the given project folder path into the copied `algorithm_configuration.yaml`. If the setup script is run again, copied files are overwritten.

Users should adjust `algorithm_configuration.yaml`, `plotting_configuration.yaml`, `techno_economic_data_conversion.yaml`, and `techno_economic_data_transportation.yaml` in `PROJECT_FOLDER/`. Only the central `project_folder_path` is stored as a path in `algorithm_configuration.yaml`. All subfolders and raw-data file names are fixed by the code and are derived from the project folder. For example, `location_data.csv` and `country_data.csv` are expected in `PROJECT_FOLDER/raw_data/` and are not configured in the YAML file.

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
