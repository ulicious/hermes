import yaml
import multiprocessing
import os

import pandas as pd
import geopandas as gpd

from shapely.wkt import loads

import warnings
warnings.filterwarnings('ignore')

from get_landmass_polygons_and_coastlines import get_landmass_polygons_and_coastlines
from group_linestrings import group_LineStrings
from process_network_data_to_network_objects import \
    process_network_data_to_network_objects_with_additional_connection_points
from process_ports import process_ports
from calculate_inner_distances import get_distances_within_networks, calculate_searoute_distances,\
    get_distances_of_closest_infrastructure

# load configuration file
path_config = 'C:/Users/mt5285/Transportmodell/configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

use_provided_data = True
use_minimal_example = config_file['use_minimal_example']

num_cores = config_file['number_cores']
if num_cores == 'max':
    num_cores = multiprocessing.cpu_count()
else:
    num_cores = min(num_cores, multiprocessing.cpu_count())

path_overall_data = config_file['paths']['project_folder']
path_processed_data = path_overall_data + config_file['paths']['processed_data']

if use_minimal_example:
    current_directory = os.path.dirname(os.getcwd()) + '/'
    path_raw_data = current_directory + 'raw_data/'
else:
    path_raw_data = path_overall_data + config_file['paths']['raw_data']

# process coastlines
polygons, coastlines = get_landmass_polygons_and_coastlines(use_minimal_example=use_minimal_example)
polygons.to_csv(path_processed_data + 'landmasses.csv')
coastlines.to_csv(path_processed_data + 'coastlines.csv')

# process raw network data and place all connected lines into network folders
path_gas_pipeline_data = path_raw_data + 'network_pipelines_gas.xlsx'
path_oil_pipeline_data = path_raw_data + 'network_pipelines_oil.xlsx'

group_LineStrings('gas', num_cores, path_gas_pipeline_data, path_processed_data,
                  use_minimal_example=use_minimal_example)
group_LineStrings('oil', num_cores, path_oil_pipeline_data, path_processed_data,
                  use_minimal_example=use_minimal_example)

# create network objects
path_gas_pipeline_data = path_processed_data + 'gas_network_data/'
path_oil_pipeline_data = path_processed_data + 'oil_network_data/'

gas_graph, gas_nodes, gas_lines \
    = process_network_data_to_network_objects_with_additional_connection_points('gas_pipeline', path_gas_pipeline_data,
                                                                                minimal_distance_between_node=50000)
gas_lines.to_csv(path_processed_data + 'gas_pipeline_graphs_object.csv')
gas_graph.to_csv(path_processed_data + 'gas_pipeline_graphs.csv')
gas_nodes.to_csv(path_processed_data + 'gas_pipeline_geodata.csv')

oil_graph, oil_nodes, oil_lines \
    = process_network_data_to_network_objects_with_additional_connection_points('oil_pipeline', path_oil_pipeline_data,
                                                                                minimal_distance_between_node=50000)

oil_lines.to_csv(path_processed_data + 'oil_pipeline_graphs_object.csv')
oil_graph.to_csv(path_processed_data + 'oil_pipeline_graphs.csv')
oil_nodes.to_csv(path_processed_data + 'oil_pipeline_geodata.csv')

# process ports
ports = process_ports(path_raw_data, coastlines, use_minimal_example=use_minimal_example)
ports.to_csv(path_processed_data + 'ports.csv')

ports = pd.read_csv(path_processed_data + 'ports.csv', index_col=0)

# calculate distances within networks (shipping and pipeline network)
get_distances_within_networks(gas_nodes, path_processed_data)
get_distances_within_networks(oil_nodes, path_processed_data)

calculate_searoute_distances(ports, num_cores, path_processed_data)

# calculate closest infrastructure for each node
options = pd.concat([gas_graph, oil_graph, ports])
get_distances_of_closest_infrastructure(options, path_processed_data)
