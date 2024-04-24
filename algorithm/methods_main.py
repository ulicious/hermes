import math
import yaml
import shapely

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import cartopy.io.shapereader as shpreader

from shapely.wkt import loads
from shapely.geometry import Point, MultiLineString
from data_processing._8_attach_conversion_costs_and_efficiency_to_locations import attach_conversion_costs_and_efficiency_to_locations


def process_network_data(data, name, geo_data, graph_data):

    """
    Method creates dictionary with network data

    @param data: Existing dictionary
    @param name: name of network
    @param geo_data: geo data of network (locations of nodes)
    @param graph_data: information on lines of network

    @return: dictionary with network data
    """

    data[name] = {}

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

    print('Load ' + name + ' data')
    for g in geo_data['graph'].unique():
        graph = nx.Graph()
        edges_graph = graph_data[graph_data['graph'] == g].index
        lines = []
        for edge in edges_graph:
            node_start = graph_data.loc[edge, 'node_start']
            node_end = graph_data.loc[edge, 'node_end']
            distance = graph_data.loc[edge, 'distance']

            # graph.add_edge(node_start, node_end, distance)
            graph.add_edge(node_start, node_end, weight=distance)
            lines.append(graph_data.loc[edge, 'line'])

        nodes_graph_original = geo_data[geo_data['graph'] == g].index
        graph_object = MultiLineString(lines)

        data[name][g] = {'Graph': graph,
                         'GraphData': graph_data,
                         'GraphObject': graph_object,
                         'GeoData': geo_data.loc[nodes_graph_original]}

    return data


def prepare_data_and_configuration_dictionary(config_file):

    """
    Loads all data based on paths in configuration file and stores it in the data dictionary.
    Furthermore, takes some adjustments to the destination (e.g., add conversion costs)

    @param config_file: dictionary with configurations

    @return:
        - data dictionary
        - configuration dictionary
        - dataframe with location information
    """

    # paths
    path_project_folder = config_file['paths']['project_folder']
    path_raw_data = path_project_folder + config_file['paths']['raw_data']
    path_processed_data = path_project_folder + config_file['paths']['processed_data']

    # load input data
    location_data = pd.read_excel(path_project_folder + config_file['filenames']['location_data'], index_col=0)

    pipeline_gas_geodata = pd.read_csv(path_processed_data + config_file['filenames']['gas_pipeline_geodata'], index_col=0,
                                       dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_gas_graphs = pd.read_csv(path_processed_data + config_file['filenames']['gas_pipeline_graph'], index_col=0)
    pipeline_liquid_geodata = pd.read_csv(path_processed_data + config_file['filenames']['oil_pipeline_geodata'], index_col=0,
                                          dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_liquid_graphs = pd.read_csv(path_processed_data + config_file['filenames']['oil_pipeline_graph'],
                                         index_col=0)
    ports = pd.read_csv(path_processed_data + config_file['filenames']['ports'], index_col=0)
    coastlines = pd.read_csv(path_processed_data + config_file['filenames']['coastlines'], index_col=0)
    minimal_distances = pd.read_csv(path_processed_data + config_file['filenames']['minimal_distances'], index_col=0)
    conversion_costs_and_efficiencies = pd.read_csv(path_processed_data + config_file['filenames']['conversion_costs_and_efficiencies'], index_col=0)

    yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    coastlines = gpd.GeoDataFrame(geometry=coastlines['geometry'].apply(loads))
    coastlines.set_geometry('geometry', inplace=True)

    final_commodities = config_file['target_commodity']
    destination_location = Point(config_file['destination_location'])
    destination_continent = config_file['destination_continent']

    transport_means = config_file['available_transport_means']

    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)

    # attach conversion costs and efficiencies to destination
    destination = pd.DataFrame(config_file['destination_location'], index=['longitude', 'latitude'], columns=['Destination']).transpose()
    destination = attach_conversion_costs_and_efficiency_to_locations(destination, config_file, techno_economic_data_conversion)

    conversion_costs_and_efficiencies = pd.concat([conversion_costs_and_efficiencies, destination])

    # The data dictionary holds common information/data/parameter which apply for all following branches.
    data = {'Shipping': {'ports': ports},
            'minimal_distances': minimal_distances,
            'transport_means': transport_means,
            'commodities': {'final_commodities': final_commodities,
                            'commodity_objects': {}},
            'destination': {'location': destination_location,
                            'continent': destination_continent},
            'coastlines': coastlines,
            'conversion_costs_and_efficiencies': conversion_costs_and_efficiencies,
            'world': world}

    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_geodata, pipeline_gas_graphs)

    data = process_network_data(data, 'Pipeline_Liquid', pipeline_liquid_geodata, pipeline_liquid_graphs)

    # get assumptions
    configuration = {'tolerance_distance': config_file['tolerance_distance'],
                     'to_final_destination_tolerance': config_file['to_final_destination_tolerance'],
                     'no_road_multiplier': config_file['no_road_multiplier'],
                     'max_length_new_segment': config_file['max_length_new_segment'],
                     'max_length_road': config_file['max_length_road'],
                     'build_new_infrastructure': config_file['build_new_infrastructure'],
                     'H2_ready_infrastructure': config_file['H2_ready_infrastructure'],
                     'path_processed_data': path_processed_data,
                     'path_results': config_file['paths']['project_folder'] + config_file['paths']['results'],
                     'use_low_storage': config_file['use_low_storage'],
                     'use_low_memory': config_file['use_low_memory'],
                     'print_runtime_information': config_file['print_runtime_information'],
                     'print_benchmark_info': config_file['print_benchmark_info']}

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
