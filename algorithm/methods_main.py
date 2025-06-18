import math
import yaml
import shapely

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import cartopy.io.shapereader as shpreader

from shapely.wkt import loads
from shapely.geometry import Point, MultiLineString, MultiPolygon
from data_processing.helpers_attach_costs import attach_conversion_costs_and_efficiency_to_infrastructure


def process_network_data(data, name, node_locations, graph_data):

    """
    Method creates dictionary with network data

    @param dict data: Existing dictionary
    @param str name: name of network
    @param pandas.DataFrame node_locations: geo data of network (locations of nodes)
    @param pandas.DataFrame graph_data: information on lines of network

    @return: dict dictionary with network data
    """

    data[name] = {}

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

    print('Load ' + name + ' data')
    for g in node_locations['graph'].unique():
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

        nodes_graph_original = node_locations[node_locations['graph'] == g].index
        graph_object = MultiLineString(lines)

        data[name][g] = {'Graph': graph,
                         'GraphData': graph_data,
                         'GraphObject': graph_object,
                         'NodeLocations': node_locations.loc[nodes_graph_original]}

    return data


def prepare_data_and_configuration_dictionary(config_file):

    """
    Loads all data based on paths in configuration file and stores it in the data dictionary.
    Furthermore, takes some adjustments to the destination (e.g., add conversion costs)

    @param dict config_file: dictionary with configurations

    @return:
        - data dictionary
        - configuration dictionary
        - dataframe with location information
    """

    # paths
    path_project_folder = config_file['project_folder_path']
    path_raw_data = path_project_folder + 'raw_data/'
    path_processed_data = path_project_folder + 'processed_data/'

    # load input data
    location_data = pd.read_csv(path_project_folder + 'start_destination_combinations.csv', index_col=0)

    pipeline_gas_node_locations = pd.read_csv(path_processed_data + 'gas_pipeline_node_locations.csv', index_col=0,
                                       dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_gas_graphs = pd.read_csv(path_processed_data + 'gas_pipeline_graphs.csv', index_col=0)
    pipeline_liquid_node_locations = pd.read_csv(path_processed_data + 'oil_pipeline_node_locations.csv', index_col=0,
                                          dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_liquid_graphs = pd.read_csv(path_processed_data + 'oil_pipeline_graphs.csv', index_col=0)
    ports = pd.read_csv(path_processed_data + 'ports.csv', index_col=0)
    coastlines = pd.read_csv(path_processed_data + 'landmasses.csv', index_col=0)
    minimal_distances = pd.read_csv(path_processed_data + 'minimal_distances.csv', index_col=0)
    conversion_costs_and_efficiencies = pd.read_csv(path_processed_data + 'conversion_costs_and_efficiency.csv', index_col=0)

    yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    coastlines = gpd.GeoDataFrame(geometry=coastlines['geometry'].apply(loads))
    coastlines.set_geometry('geometry', inplace=True)

    final_commodities = config_file['target_commodity']

    # create shapely object of destination
    if config_file['destination_type'] == 'location':
        destination_location = Point(config_file['destination_location'])
        destination_continent = config_file['destination_continent']

        # attach conversion costs and efficiencies to destination
        destination = pd.DataFrame(config_file['destination_location'], index=['longitude', 'latitude'],
                                   columns=['Destination']).transpose()
        destination = attach_conversion_costs_and_efficiency_to_infrastructure(destination, config_file,
                                                                               techno_economic_data_conversion)

        conversion_costs_and_efficiencies = pd.concat([conversion_costs_and_efficiencies, destination])

        infrastructure_in_destination = None
    else:
        destination_continent = config_file['destination_continent']

        country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
        world = gpd.read_file(country_shapefile)

        state_shapefile = shpreader.natural_earth(resolution='10m', category='cultural',
                                                  name='admin_1_states_provinces')
        states = gpd.read_file(state_shapefile)

        country_states = config_file['destination_polygon']

        first = True
        destination_location = None
        for c in [*country_states.keys()]:
            if country_states[c]:
                for s in country_states[c]:
                    if first:
                        destination_location = states[states['name'] == s]['geometry'].values[0]
                        first = False
                    else:
                        destination_location.union(states[states['name'] == s]['geometry'].values[0])
            else:
                if first:
                    destination_location = world[world['NAME_EN'] == c]['geometry'].values[0]
                    first = False
                else:
                    destination_location.union(world[world['NAME_EN'] == c]['geometry'].values[0])

        if config_file['use_biggest_landmass']:  # use only biggest land area
            if len([*country_states.keys()]) == 1:
                if isinstance(destination_location, MultiPolygon):
                    largest_area = 0
                    chosen_geom = None
                    for geom in destination_location.geoms:
                        if geom.area > largest_area:
                            largest_area = geom.area
                            chosen_geom = geom

                    destination_location = chosen_geom

        infrastructure_in_destination = []
        for i in conversion_costs_and_efficiencies.index:
            if 'PG' in i:
                data_set_to_look_up = pipeline_gas_node_locations
            elif 'PL' in i:
                data_set_to_look_up = pipeline_liquid_node_locations
            elif 'H' in i:
                data_set_to_look_up = ports
            else:
                continue

            i_location = Point([data_set_to_look_up.loc[i, 'longitude'], data_set_to_look_up.loc[i, 'latitude']])
            if i_location.within(destination_location):
                infrastructure_in_destination.append(i)

        # one destination does not exist. However, destination efficiencies and costs are used later on in the code
        # when looking for the cheapest option --> take minimal values for costs and efficiencies
        # will not be applied to calculate actual costs
        costs_efficiencies_destinations = conversion_costs_and_efficiencies.loc[infrastructure_in_destination, :]
        min_costs_efficiencies_destinations = pd.DataFrame(costs_efficiencies_destinations.min(axis='index')).transpose()
        min_costs_efficiencies_destinations.index = ['Destination']

        conversion_costs_and_efficiencies = pd.concat([conversion_costs_and_efficiencies,
                                                       min_costs_efficiencies_destinations])

    transport_means = config_file['available_transport_means']

    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)

    # The data dictionary holds common information/data/parameter which apply for all following branches.
    data = {'Shipping': {'ports': ports},
            'minimal_distances': minimal_distances,
            'transport_means': transport_means,
            'commodities': {'final_commodities': final_commodities,
                            'commodity_objects': {}},
            'destination': {'location': destination_location,
                            'continent': destination_continent,
                            'infrastructure': infrastructure_in_destination},
            'coastlines': coastlines,
            'conversion_costs_and_efficiencies': conversion_costs_and_efficiencies,
            'world': world}

    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_node_locations, pipeline_gas_graphs)

    data = process_network_data(data, 'Pipeline_Liquid', pipeline_liquid_node_locations, pipeline_liquid_graphs)

    # get assumptions
    configuration = {'destination_type': config_file['destination_type'],
                     'tolerance_distance': config_file['tolerance_distance'],
                     'to_final_destination_tolerance': config_file['to_final_destination_tolerance'],
                     'no_road_multiplier': config_file['no_road_multiplier'],
                     'max_length_new_segment': config_file['max_length_new_segment'],
                     'max_length_road': config_file['max_length_road'],
                     'build_new_infrastructure': config_file['build_new_infrastructure'],
                     'H2_ready_infrastructure': config_file['H2_ready_infrastructure'],
                     'path_processed_data': path_processed_data,
                     'path_results': config_file['project_folder_path'] + 'results/',
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

    # if no new pipeline infrastructure can be built, then we set distance to 0
    if not configuration['build_new_infrastructure']:
        configuration['max_length_new_segment'] = 0

    return data, configuration, location_data
