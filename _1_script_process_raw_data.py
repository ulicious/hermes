import yaml
import multiprocessing
import os
import shutil
import logging
import shapely
import time

import pandas as pd
import geopandas as gpd

from data_processing.get_landmass_polygons_and_coastlines import get_landmass_polygons_and_coastlines
from data_processing.group_linestrings import group_LineStrings
from data_processing.process_network_data_to_network_objects import \
    process_network_data_to_network_objects_with_additional_connection_points
from data_processing.process_ports import process_ports
from data_processing.calculate_inner_distances import get_distances_within_networks, get_distances_of_closest_infrastructure, calculate_searoute_distances
from data_processing.helpers_attach_costs import attach_conversion_costs_and_efficiency_to_infrastructure
from data_processing.process_mip_data import calculate_road_distances, calculate_efficiencies
from data_processing.helpers_geometry import get_destination


import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

time_start = time.time()

# load configuration file
path_config = os.getcwd() + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

use_minimal_example = config_file['use_minimal_example']
use_low_storage = config_file['use_low_storage']
use_low_memory = config_file['use_low_memory']
update_only_conversion_costs_and_efficiency = config_file['update_only_conversion_costs_and_efficiency']

boundaries = [config_file['minimal_latitude'], config_file['maximal_latitude'],
              config_file['minimal_longitude'], config_file['maximal_longitude']]

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

enforce_update_of_data = config_file['enforce_update_of_data']
create_mip_data = config_file['create_mip_data']

destination = get_destination(config_file)

# based on the input data, the data can further be processed to input data of a mixed-integer model to validate the heuristic
# This will be done while processing the data of the heuristic. However, this can significantly increase the processing time
if create_mip_data:
    name_folder = path_processed_data + 'mip_data/'
    if 'mip_data' not in os.listdir(path_processed_data):
        os.mkdir(name_folder)
    files_in_mip_folder = os.listdir(name_folder)

gap_distance = config_file['gap_distance']

if not update_only_conversion_costs_and_efficiency:

    # process coastlines
    logging.info('Processing coastlines and landmasses')
    if not (('landmasses.csv' in files_in_folder) & ('coastlines.csv' in files_in_folder) & (not enforce_update_of_data)):
        polygons, coastlines = get_landmass_polygons_and_coastlines(use_minimal_example=use_minimal_example)
        polygons.to_csv(path_processed_data + 'landmasses.csv')
        coastlines.to_csv(path_processed_data + 'coastlines.csv')

    else:
        coastlines = pd.read_csv(path_processed_data + 'coastlines.csv')
        coastlines = gpd.GeoDataFrame(geometry=coastlines['geometry'].apply(shapely.wkt.loads))

        landmasses = pd.read_csv(path_processed_data + 'landmasses.csv')
        landmasses = gpd.GeoDataFrame(geometry=landmasses['geometry'].apply(shapely.wkt.loads))

    # process raw network data and place all connected lines into network folders
    logging.info('Processing raw pipeline data')
    if not (('gas_network_data' in files_in_folder) & (not enforce_update_of_data)):
        # process gas pipelines
        logging.info('Gas pipelines')
        path_gas_pipeline_data = path_raw_data + 'network_pipelines_gas.xlsx'
        group_LineStrings('gas', num_cores, path_gas_pipeline_data, path_processed_data, gap_distance,
                          boundaries, destination, use_minimal_example=use_minimal_example)

    if not (('oil_network_data' in files_in_folder) & (not enforce_update_of_data)):
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
            & (not enforce_update_of_data)):
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

        gas_graph.to_csv(path_processed_data + 'gas_pipeline_graphs.csv')
        gas_nodes.to_csv(path_processed_data + 'gas_pipeline_node_locations.csv')

    else:
        gas_graph = pd.read_csv(path_processed_data + 'gas_pipeline_graphs.csv', index_col=0)
        gas_nodes = pd.read_csv(path_processed_data + 'gas_pipeline_node_locations.csv', index_col=0)

    if not (('oil_pipeline_graphs.csv' in files_in_folder) & ('oil_pipeline_node_locations.csv' in files_in_folder)
            & (not enforce_update_of_data)):
        if use_minimal_example:
            oil_graph, oil_nodes \
                = process_network_data_to_network_objects_with_additional_connection_points('oil_pipeline', path_oil_pipeline_data,
                                                                                            minimal_distance_between_node=100000,
                                                                                            number_workers=num_cores)
        else:
            oil_graph, oil_nodes \
                = process_network_data_to_network_objects_with_additional_connection_points('oil_pipeline', path_oil_pipeline_data,
                                                                                            number_workers=num_cores)

        oil_graph.to_csv(path_processed_data + 'oil_pipeline_graphs.csv')
        oil_nodes.to_csv(path_processed_data + 'oil_pipeline_node_locations.csv')

    else:
        oil_graph = pd.read_csv(path_processed_data + 'oil_pipeline_graphs.csv', index_col=0)
        oil_nodes = pd.read_csv(path_processed_data + 'oil_pipeline_node_locations.csv', index_col=0)

    # process ports
    logging.info('Processing ports')
    if not (('ports.csv' in files_in_folder) & (not enforce_update_of_data)):
        ports = process_ports(path_raw_data, coastlines, landmasses, boundaries, destination, use_minimal_example=use_minimal_example)
        ports.to_csv(path_processed_data + 'ports.csv')
    else:
        ports = pd.read_csv(path_processed_data + 'ports.csv', index_col=0)

    if not use_low_storage:
        # calculate distances within networks (shipping and pipeline network)
        logging.info('Calculate inner infrastructure distances')

        # create new folder containing distances and duration
        name_folder = path_processed_data + 'inner_infrastructure_distances/'
        if 'inner_infrastructure_distances' not in os.listdir(path_processed_data):
            os.mkdir(name_folder)

        if not (('inner_infrastructure_distances' in files_in_folder) & (not enforce_update_of_data) & ('port_distances.csv' in files_in_mip_folder)):
            get_distances_within_networks(gas_graph, gas_nodes, path_processed_data, num_cores, use_low_memory=use_low_memory, create_mip_data=create_mip_data)
            get_distances_within_networks(oil_graph, oil_nodes, path_processed_data, num_cores, use_low_memory=use_low_memory, create_mip_data=create_mip_data)
            calculate_searoute_distances(ports, techno_economic_data_transport['Shipping_Speed'], num_cores, path_processed_data, create_mip_data=create_mip_data)

    # calculate closest infrastructure for each node
    logging.info('Calculate closest infrastructure')
    options = pd.concat([gas_nodes, oil_nodes, ports])
    if not (('minimal_distances.csv' in files_in_folder) & (not enforce_update_of_data)):
        get_distances_of_closest_infrastructure(options, path_processed_data, num_cores)

else:
    ports = pd.read_csv(path_processed_data + 'ports.csv', index_col=0)
    gas_nodes = pd.read_csv(path_processed_data + 'gas_pipeline_node_locations.csv', index_col=0)
    oil_nodes = pd.read_csv(path_processed_data + 'oil_pipeline_node_locations.csv', index_col=0)

    options = pd.concat([gas_nodes, oil_nodes, ports])

# calculate conversion costs at each location
logging.info('Calculate conversion costs and efficiency')
conversion_costs_and_efficiency \
    = attach_conversion_costs_and_efficiency_to_infrastructure(options, config_file, techno_economic_data_conversion)
conversion_costs_and_efficiency.to_csv(path_processed_data + 'conversion_costs_and_efficiency.csv')

# missing data not yet processed: road and new pipeline distances; efficiencies; costs
if create_mip_data:  # todo: distances to the destination + conversion cost at destination

    # make uniform latitude and longitude
    options.loc[ports.index, 'longitude'] = options.loc[ports.index, 'longitude_on_coastline']
    options.loc[ports.index, 'latitude'] = options.loc[ports.index, 'latitude_on_coastline']

    # remove unnecessary columns
    options.drop(columns=['name', 'country', 'continent', 'longitude_on_coastline', 'latitude_on_coastline'], inplace=True)

    # save overall options data
    options.to_csv(path_processed_data + 'mip_data/' + 'options.csv')

    # distances
    road_distances = calculate_road_distances(config_file['tolerance_distance'], options)
    new_pipeline_distances = road_distances.copy()
    new_pipeline_distances = new_pipeline_distances[new_pipeline_distances['distance'] <= config_file['max_length_new_segment']]

    road_distances['distance'] *= config_file['no_road_multiplier']
    new_pipeline_distances['distance'] *= config_file['no_road_multiplier']

    road_distances.to_csv(path_processed_data + 'mip_data/' + 'road_distances.csv')
    new_pipeline_distances.to_csv(path_processed_data + 'mip_data/' + 'new_pipeline_distances.csv')

    # transport efficiencies: depend on distance and duration
    ports_distances = pd.read_csv(path_processed_data + 'mip_data/' + 'port_distances.csv', index_col=0)
    ports_durations = pd.read_csv(path_processed_data + 'mip_data/' + 'ports_durations.csv', index_col=0)

    for commodity in config_file['available_commodity']:
        if 'Shipping' in techno_economic_data_transport[commodity]['potential_transportation']:
            uses_commodity_as_shipping_fuel = techno_economic_data_transport[commodity]['Uses_Commodity_as_Shipping_Fuel']
            boil_off = techno_economic_data_transport[commodity]['Boil_Off']
            self_consumption = techno_economic_data_transport[commodity]['Self_Consumption']

            efficiency = calculate_efficiencies(ports_distances, ports_durations, boil_off, uses_commodity_as_shipping_fuel, self_consumption)
            efficiency.to_csv(path_processed_data + 'mip_data/' + commodity + '_efficiencies.csv')

    # conversion costs and efficiencies at nodes
    conversion_costs_and_efficiency.to_csv(path_processed_data + 'mip_data/' + 'conversion_costs_and_efficiency.csv')


if time.time() - time_start < 60:
    print('total processing time [s]: ' + str(time.time() - time_start))
elif time.time() - time_start < 3600:
    print('total processing time [m]: ' + str((time.time() - time_start) / 60))
else:
    print('total processing time [h]: ' + str((time.time() - time_start) / 60 / 60))
