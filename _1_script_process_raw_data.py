import yaml
import multiprocessing
import os
import shutil
import logging
import shapely
import time
import json

import pandas as pd
import geopandas as gpd

from data_processing.get_landmass_polygons_and_coastlines import get_landmass_polygons_and_coastlines
from data_processing.group_linestrings import group_LineStrings
from data_processing.process_network_data_to_network_objects import \
    process_network_data_to_network_objects_with_additional_connection_points
from data_processing.process_ports import process_ports
from data_processing.calculate_inner_distances import get_distances_within_networks, get_distances_of_closest_infrastructure, calculate_searoute_distances
from data_processing.helpers_attach_costs import attach_conversion_costs_and_efficiency_to_infrastructure, calculate_conversion_costs_and_efficiencies_for_all_combinations
from data_processing.process_mip_data import prepare_global_mip_data
try:
    from data_processing.process_mip_data import prepare_minimal_mip_case
except ImportError:
    prepare_minimal_mip_case = None
from data_processing.helpers_geometry import get_destination, get_boundaries_from_config
from data_processing.helpers_continent_connections import build_continent_connectivity, save_continent_connectivity
from data_processing.natural_earth_data import download_natural_earth_data

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

time_start = time.time()

PIPELINE_GRAPH_COLUMNS = ['graph', 'node_start', 'node_end', 'distance', 'line']
PIPELINE_NODE_COLUMNS = ['longitude', 'latitude', 'graph']
PORT_COLUMNS = ['latitude', 'longitude', 'name', 'country', 'continent',
                'longitude_on_coastline', 'latitude_on_coastline']
MINIMAL_DISTANCE_COLUMNS = ['minimal_distance', 'closest_node']


def ensure_columns(data, columns):
    """Return a copy with all required columns available, even if the data is empty."""
    if data is None:
        data = pd.DataFrame()
    data = data.copy()
    for column in columns:
        if column not in data.columns:
            data[column] = pd.Series(dtype='object')
    return data[columns + [column for column in data.columns if column not in columns]]


def read_csv_or_empty(path_file, columns):
    if os.path.exists(path_file):
        return ensure_columns(pd.read_csv(path_file, index_col=0), columns)
    return pd.DataFrame(columns=columns)


def write_csv_with_schema(data, path_file, columns):
    data = ensure_columns(data, columns)
    data.to_csv(path_file, encoding='utf-8', index=True)
    return data


def ensure_processed_infrastructure_files(path_processed):
    """Create schema-correct empty infrastructure files for optional layers."""
    defaults = {
        'gas_pipeline_graphs.csv': PIPELINE_GRAPH_COLUMNS,
        'gas_pipeline_node_locations.csv': PIPELINE_NODE_COLUMNS,
        'oil_pipeline_graphs.csv': PIPELINE_GRAPH_COLUMNS,
        'oil_pipeline_node_locations.csv': PIPELINE_NODE_COLUMNS,
        'ports.csv': PORT_COLUMNS,
        'minimal_distances.csv': MINIMAL_DISTANCE_COLUMNS,
    }
    for filename, columns in defaults.items():
        path_file = path_processed + filename
        if not os.path.exists(path_file):
            pd.DataFrame(columns=columns).to_csv(path_file, encoding='utf-8', index=True)

    tolerance_file = path_processed + 'within_tolerance.json'
    if not os.path.exists(tolerance_file):
        with open(tolerance_file, 'w', encoding='utf-8') as file:
            json.dump({}, file, indent=2)


# load configuration file
path_config = os.getcwd() + '/_1_algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

use_minimal_example = config_file['use_minimal_example']
use_low_storage = config_file['use_low_storage']
use_low_memory = config_file['use_low_memory']
infrastructure_update_only_conversion_costs_and_efficiency = \
    config_file['infrastructure_update_only_conversion_costs_and_efficiency']

infrastructure_boundaries = get_boundaries_from_config(
    config_file, prefix='infrastructure_', use_minimal_example=use_minimal_example)
boundaries = [infrastructure_boundaries[0], infrastructure_boundaries[1],
              infrastructure_boundaries[2], infrastructure_boundaries[3]]

num_cores = config_file['number_cores']
if num_cores == 'max':
    num_cores = multiprocessing.cpu_count() - 1
else:
    num_cores = min(num_cores, multiprocessing.cpu_count() - 1)

path_overall_data = config_file['project_folder_path']
path_raw_data = path_overall_data + 'raw_data/'
path_processed_data = path_overall_data + 'processed_data/'

# check if project folders exists and create if not
if not os.path.exists(path_overall_data):
    os.mkdir(path_overall_data)

if not os.path.exists(path_raw_data):
    os.mkdir(path_raw_data)

if not os.path.exists(path_processed_data):
    os.mkdir(path_processed_data)

ensure_processed_infrastructure_files(path_processed_data)

if not os.path.exists(path_overall_data + 'results/'):
    os.mkdir(path_overall_data + 'results/')

if not os.path.exists(path_overall_data + 'results/location_results/'):
    os.mkdir(path_overall_data + 'results/location_results/')

if not os.path.exists(path_overall_data + 'results/plots/'):
    os.mkdir(path_overall_data + 'results/plots/')

# move raw data from this repository to project folder raw data if set in configuration
if config_file['use_provided_data']:
    file_directory = os.getcwd() + '/data/'
    files = os.listdir(file_directory)
    for f in files:
        shutil.copy(file_directory + f, path_raw_data)

# load techno-economic data transport
yaml_file = open(path_raw_data + 'techno_economic_data_transportation.yaml')
techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

# load techno economic data
yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

files_in_folder = os.listdir(path_processed_data)

infrastructure_enforce_update_of_data = config_file['infrastructure_enforce_update_of_data']
create_mip_data = config_file['create_mip_data']

download_natural_earth_data(path_raw_data, force_update=infrastructure_enforce_update_of_data)

destination = get_destination(config_file)  # todo: possible to load the natural earth data instead of using old packagaes

# todo: separate more clearly: Creation of basic infrastructure data and case sensitive data

# based on the input data, the data can further be processed to input data of a mixed-integer model to validate the heuristic
# This will be done while processing the data of the heuristic. However, this can significantly increase the processing time
files_in_mip_folder = []
if create_mip_data:
    name_folder = path_processed_data + 'mip_data/'
    if 'mip_data' not in os.listdir(path_processed_data):
        os.mkdir(name_folder)
    files_in_mip_folder = os.listdir(name_folder)

gap_distance = config_file['gap_distance']

if not infrastructure_update_only_conversion_costs_and_efficiency:

    # process coastlines
    logging.info('Processing coastlines and landmasses')
    if not (('landmasses.csv' in files_in_folder) & ('coastlines.csv' in files_in_folder) & ('water_availability.gpkg' in files_in_folder) & (not infrastructure_enforce_update_of_data)):
        landmasses, coastlines, water_availability = get_landmass_polygons_and_coastlines(path_raw_data, use_minimal_example=use_minimal_example)
        landmasses.to_csv(path_processed_data + 'landmasses.csv')
        coastlines.to_csv(path_processed_data + 'coastlines.csv')

        water_availability.to_file(
            path_processed_data + "water_availability.gpkg",
            layer="ptx_water_available",
            driver="GPKG"
        )

    else:
        coastlines = pd.read_csv(path_processed_data + 'coastlines.csv')
        coastlines = gpd.GeoDataFrame(geometry=coastlines['geometry'].apply(shapely.wkt.loads))

        landmasses = pd.read_csv(path_processed_data + 'landmasses.csv')
        landmasses = gpd.GeoDataFrame(geometry=landmasses['geometry'].apply(shapely.wkt.loads))

        water_availability = gpd.read_file(
            path_processed_data + "water_availability.gpkg",
            layer="ptx_water_available"
        )

    # process raw network data and place all connected lines into network folders
    logging.info('Processing raw pipeline data')
    if not (('gas_network_data' in files_in_folder) & (not infrastructure_enforce_update_of_data)):
        # process gas pipelines
        logging.info('Gas pipelines')
        path_gas_pipeline_data = path_raw_data + 'network_pipelines_gas.xlsx'
        group_LineStrings('gas', num_cores, path_gas_pipeline_data, path_processed_data, gap_distance,
                          boundaries, destination, use_minimal_example=use_minimal_example)

    if not (('oil_network_data' in files_in_folder) & (not infrastructure_enforce_update_of_data)):
        # process oil pipelines
        logging.info('Oil pipelines')
        path_oil_pipeline_data = path_raw_data + 'network_pipelines_oil.xlsx'
        group_LineStrings('oil', num_cores, path_oil_pipeline_data, path_processed_data, gap_distance,
                          boundaries, destination, use_minimal_example=use_minimal_example)

    # create network objects
    logging.info('Build pipeline networks')

    path_gas_pipeline_data = path_processed_data + 'gas_network_data/'
    path_oil_pipeline_data = path_processed_data + 'oil_network_data/'

    if not (('gas_pipeline_graphs.csv' in files_in_folder) & ('gas_pipeline_node_locations.csv' in files_in_folder)
            & (not infrastructure_enforce_update_of_data)):
        if use_minimal_example:
            gas_graph, gas_nodes \
                = process_network_data_to_network_objects_with_additional_connection_points('gas_pipeline', path_gas_pipeline_data,
                                                                                            minimal_distance_between_node=100000,
                                                                                            number_workers=num_cores)
        else:
            gas_graph, gas_nodes \
                = process_network_data_to_network_objects_with_additional_connection_points('gas_pipeline',
                                                                                            path_gas_pipeline_data,
                                                                                            number_workers=num_cores)

        gas_graph = write_csv_with_schema(
            gas_graph, path_processed_data + 'gas_pipeline_graphs.csv', PIPELINE_GRAPH_COLUMNS)
        gas_nodes = write_csv_with_schema(
            gas_nodes, path_processed_data + 'gas_pipeline_node_locations.csv', PIPELINE_NODE_COLUMNS)

    else:
        gas_graph = read_csv_or_empty(path_processed_data + 'gas_pipeline_graphs.csv', PIPELINE_GRAPH_COLUMNS)
        gas_nodes = read_csv_or_empty(path_processed_data + 'gas_pipeline_node_locations.csv', PIPELINE_NODE_COLUMNS)

    if not (('oil_pipeline_graphs.csv' in files_in_folder) & ('oil_pipeline_node_locations.csv' in files_in_folder)
            & (not infrastructure_enforce_update_of_data)):
        if use_minimal_example:
            oil_graph, oil_nodes \
                = process_network_data_to_network_objects_with_additional_connection_points('oil_pipeline', path_oil_pipeline_data,
                                                                                            minimal_distance_between_node=100000,
                                                                                            number_workers=num_cores)
        else:
            oil_graph, oil_nodes \
                = process_network_data_to_network_objects_with_additional_connection_points('oil_pipeline', path_oil_pipeline_data,
                                                                                            number_workers=num_cores)

        oil_graph = write_csv_with_schema(
            oil_graph, path_processed_data + 'oil_pipeline_graphs.csv', PIPELINE_GRAPH_COLUMNS)
        oil_nodes = write_csv_with_schema(
            oil_nodes, path_processed_data + 'oil_pipeline_node_locations.csv', PIPELINE_NODE_COLUMNS)

    else:
        oil_graph = read_csv_or_empty(path_processed_data + 'oil_pipeline_graphs.csv', PIPELINE_GRAPH_COLUMNS)
        oil_nodes = read_csv_or_empty(path_processed_data + 'oil_pipeline_node_locations.csv', PIPELINE_NODE_COLUMNS)

    # process ports
    logging.info('Processing ports')
    if not (('ports.csv' in files_in_folder) & (not infrastructure_enforce_update_of_data)):
        ports = process_ports(path_raw_data, coastlines, landmasses, boundaries, destination, use_minimal_example=use_minimal_example)
        ports = write_csv_with_schema(ports, path_processed_data + 'ports.csv', PORT_COLUMNS)
    else:
        ports = read_csv_or_empty(path_processed_data + 'ports.csv', PORT_COLUMNS)

    if not use_low_storage:
        # calculate distances within networks (shipping and pipeline network)
        logging.info('Calculate inner infrastructure distances')

        # create new folder containing distances and duration
        name_folder = path_processed_data + 'inner_infrastructure_distances/'
        if 'inner_infrastructure_distances' not in os.listdir(path_processed_data):
            os.mkdir(name_folder)

        if not (('inner_infrastructure_distances' in files_in_folder) & (not infrastructure_enforce_update_of_data)
                & (not (create_mip_data & (not 'port_distances.csv' in files_in_mip_folder)))):
            get_distances_within_networks(gas_graph, gas_nodes, path_processed_data, num_cores, use_low_memory=use_low_memory, create_mip_data=create_mip_data)
            get_distances_within_networks(oil_graph, oil_nodes, path_processed_data, num_cores, use_low_memory=use_low_memory, create_mip_data=create_mip_data)
            calculate_searoute_distances(ports, num_cores, path_processed_data, create_mip_data=create_mip_data)

    # calculate closest infrastructure for each node
    logging.info('Calculate closest infrastructure')
    options = pd.concat([gas_nodes, oil_nodes, ports])
    if not (('minimal_distances.csv' in files_in_folder) & (not infrastructure_enforce_update_of_data)):
        get_distances_of_closest_infrastructure(config_file, options, path_processed_data, num_cores)

    logging.info('Calculate continent connectivity')
    if not (('continent_connections.json' in files_in_folder) & (not infrastructure_enforce_update_of_data)):
        continent_connectivity = build_continent_connectivity(landmasses, gas_nodes, oil_nodes)
        save_continent_connectivity(path_processed_data + 'continent_connections.json', continent_connectivity)

else:
    ensure_processed_infrastructure_files(path_processed_data)
    ports = read_csv_or_empty(path_processed_data + 'ports.csv', PORT_COLUMNS)
    gas_nodes = read_csv_or_empty(path_processed_data + 'gas_pipeline_node_locations.csv', PIPELINE_NODE_COLUMNS)
    oil_nodes = read_csv_or_empty(path_processed_data + 'oil_pipeline_node_locations.csv', PIPELINE_NODE_COLUMNS)
    if os.path.exists(path_processed_data + 'landmasses.csv'):
        landmasses = pd.read_csv(path_processed_data + 'landmasses.csv')
        landmasses = gpd.GeoDataFrame(geometry=landmasses['geometry'].apply(shapely.wkt.loads))
    else:
        landmasses = gpd.GeoDataFrame(geometry=[])

    options = pd.concat([gas_nodes, oil_nodes, ports])

# calculate conversion costs at each location
logging.info('Calculate conversion costs and efficiency')
conversion_costs_and_efficiency \
    = attach_conversion_costs_and_efficiency_to_infrastructure(options, config_file, techno_economic_data_conversion)

conversion_costs_and_efficiency \
    = calculate_conversion_costs_and_efficiencies_for_all_combinations(config_file, conversion_costs_and_efficiency, techno_economic_data_conversion)

conversion_costs_and_efficiency.to_csv(path_processed_data + 'conversion_costs_and_efficiency.csv', encoding='utf-8', index=True)

# Build reusable MIP graph input once. Only the origin links and selected
# destination sink remain run-specific and are added later in `prepare_data`.
if create_mip_data:
    logging.info('Prepare origin-independent MIP infrastructure graph')
    prepare_global_mip_data(
        options, ports, config_file, techno_economic_data_conversion,
        techno_economic_data_transport, conversion_costs_and_efficiency,
        path_processed_data)
    if prepare_minimal_mip_case is not None:
        prepare_minimal_mip_case(
            config_file, techno_economic_data_conversion,
            techno_economic_data_transport, path_processed_data)
    else:
        logging.warning(
            'Skip minimal MIP case generation: prepare_minimal_mip_case is not '
            'available in data_processing.process_mip_data')
    logging.info('Finished origin-independent MIP infrastructure graph')


if time.time() - time_start < 60:
    print('total processing time [s]: ' + str(time.time() - time_start))
elif time.time() - time_start < 3600:
    print('total processing time [m]: ' + str((time.time() - time_start) / 60))
else:
    print('total processing time [h]: ' + str((time.time() - time_start) / 60 / 60))
