import math
import os

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point

from process_input_data import process_network_data
from object_commodity import create_commodity_objects


def prepare_data_and_configuration_dictionary(config_file):

    path_project_folder = config_file['paths']['project_folder']
    path_processed_data = path_project_folder + config_file['paths']['processed_data']
    path_techno_economic_data = os.getcwd() + '/techno_economic_data/'

    # load input data
    location_data = pd.read_excel(path_project_folder + 'start_destination_combinations.xlsx', index_col=0)
    conversion_costs = pd.read_excel(path_techno_economic_data + 'commodities_conversions.xlsx', index_col=0)
    conversion_efficiencies = pd.read_excel(path_techno_economic_data + 'commodities_conversion_efficiencies.xlsx',
                                            index_col=0)
    transportation_costs = pd.read_excel(path_techno_economic_data + 'commodities_transportation.xlsx',
                                         index_col=0)

    pipeline_gas_geodata = pd.read_csv(path_processed_data + 'gas_pipeline_geodata.csv', index_col=0,
                                       dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_gas_graphs = pd.read_csv(path_processed_data + 'gas_pipeline_graphs.csv', index_col=0)
    pipeline_gas_graphs_objects = pd.read_csv(path_processed_data + 'gas_pipeline_graphs_object.csv',
                                              index_col=0)
    pipeline_liquid_geodata = pd.read_csv(path_processed_data + 'oil_pipeline_geodata.csv', index_col=0,
                                          dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_liquid_graphs = pd.read_csv(path_processed_data + 'oil_pipeline_graphs.csv',
                                         index_col=0)
    pipeline_liquid_graphs_objects = pd.read_csv(path_processed_data + 'oil_pipeline_graphs_object.csv', index_col=0)
    ports = pd.read_csv(path_processed_data + 'ports.csv', index_col=0)
    coastlines = pd.read_csv(path_processed_data + 'coastlines.csv', index_col=0)
    minimal_distances = pd.read_csv(path_processed_data + 'minimal_distances.csv', index_col=0)

    # get commodities and associated data
    commodities, commodity_names, commodity_names_to_commodity_object, means_of_transport \
        = create_commodity_objects(conversion_costs, conversion_efficiencies, transportation_costs)

    coastlines = gpd.GeoDataFrame(geometry=coastlines['geometry'].apply(loads))
    coastlines.set_geometry('geometry', inplace=True)

    final_commodities = config_file['target_commodity']
    destination_location = Point(config_file['destination_location'])
    destination_continent = config_file['destination_continent']

    # The data dictionary holds common information/data/parameter which apply for all solutions.
    # todo: alles klein schreiben + shipping anpassen
    data = {'Shipping': {'ports': ports},
            'minimal_distances': minimal_distances,
            'transport_means': means_of_transport,
            'commodities': {'final_commodities': final_commodities,
                            'commodity_objects': {}},
            'destination': {'location': destination_location,
                            'continent': destination_continent},
            'coastlines': coastlines}

    for c in commodities:
        data['commodities']['commodity_objects'][c.get_name()] = c

    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_geodata, pipeline_gas_graphs)

    data = process_network_data(data, 'Pipeline_Liquid', pipeline_liquid_geodata, pipeline_liquid_graphs)

    # get assumptions
    configuration = {'allow_first_iteration_conversion': config_file['allow_first_iteration_conversion'],
                     'tolerance_distance': config_file['tolerance_distance'],
                     'to_final_destination_tolerance': config_file['to_final_destination_tolerance'],
                     'no_road_multiplier': config_file['no_road_multiplier'],
                     'max_length_new_segment': config_file['max_length_new_segment'],
                     'max_length_road': config_file['max_length_road'],
                     'build_new_infrastructure': config_file['build_new_infrastructure'],
                     'H2_ready_infrastructure': config_file['H2_ready_infrastructure'],
                     'path_processed_data': path_processed_data,
                     'path_results': config_file['paths']['project_folder'] + config_file['paths']['results']}

    if isinstance(configuration['tolerance_distance'], str):
        configuration['tolerance_distance'] = math.inf

    if isinstance(configuration['to_final_destination_tolerance'], str):
        configuration['to_final_destination_tolerance'] = math.inf

    if isinstance(configuration['no_road_multiplier'], str):
        configuration['no_road_multiplier'] = math.inf

    if isinstance(configuration['max_length_new_segment'], str):
        configuration['max_length_new_segment'] = math.inf

    if isinstance(configuration['max_length_road'], str):
        configuration['max_length_road'] = math.inf

    return data, configuration, location_data
