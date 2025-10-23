import yaml
import os
import ast

from algorithm.methods_geographic import calc_distance_list_to_list_no_matrix

import pandas as pd
from shapely.ops import unary_union
import networkx as nx

import geopandas as gpd
import cartopy.io.shapereader as shpreader

from itertools import combinations


def prepare_data(start_location, end_node=None):

    path_config = os.getcwd()
    path_config = os.path.dirname(path_config) + '/algorithm_configuration.yaml'

    yaml_file = open(path_config)
    config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    path_overall_data = config_file['project_folder_path']
    path_raw_data = path_overall_data + 'raw_data/'
    path_mip_data = path_overall_data + 'processed_data/mip_data/'

    # load techno economic data
    yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    yaml_file = open(path_raw_data + 'techno_economic_data_transportation.yaml')
    techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # get all nodes
    nodes = pd.read_csv(path_mip_data + 'options.csv', index_col=0)

    all_nodes = nodes.index.tolist()
    all_commodities = config_file['available_commodity']

    # create edges: location identification + energy carrier
    all_nodes_adjusted = [n + '_' + commodity for n in all_nodes for commodity in all_commodities]

    start_commodities = config_file['available_commodity']
    start_nodes = ['start_' + c for c in all_commodities]

    target_commodities = config_file['target_commodity']
    target_nodes = ['end']

    all_nodes_adjusted += start_nodes + target_nodes

    transport_means = config_file['available_transport_means']

    # get information on start location
    start_data = pd.read_csv(path_overall_data + 'start_destination_combinations.csv', index_col=0)

    production_costs = {}
    for com in all_commodities:
        production_costs[com] = start_data.loc[start_location, com]

    edges = {}

    # conversions
    if True:
        conversion_costs_and_efficiencies = pd.read_csv(path_overall_data + 'processed_data/conversion_costs_and_efficiency.csv', index_col=0)
        for node_1 in all_nodes:

            if 'origin' in node_1:
                continue

            if not conversion_costs_and_efficiencies.loc[node_1, 'conversion_possible']:
                continue

            for node_2 in all_nodes:

                if ('PG' in node_1) | ('PG' in node_2) | ('PL' in node_1) | ('PL' in node_2):
                    continue

                if 'origin' in node_2:
                    continue

                if node_1 != node_2:
                    continue

                for com_1 in all_commodities:
                    for com_2 in all_commodities:
                        if com_2 in techno_economic_data_conversion[com_1]['potential_conversions']:
                            conversion_costs = conversion_costs_and_efficiencies.loc[node_1, com_1 + '_' + com_2 + '_conversion_costs']
                            conversion_efficiency = 1 - conversion_costs_and_efficiencies.loc[node_1, com_1 + '_' + com_2 + '_conversion_efficiency']

                            edges[node_1 + '_' + com_1 + '-' + node_2 + '_' + com_2] = \
                                ('conversion', node_1 + '_' + com_1, node_2 + '_' + com_2, conversion_costs,
                                 conversion_efficiency, com_2)

                            # if node_2 == end_node:
                            #     edges[node_1 + '_' + com_1 + '-end'] =\
                            #         ('conversion', node_1 + '_' + com_1, 'end', 0, 0, com_2)

    # load transport data
    road_distances = pd.read_csv(path_overall_data + 'processed_data/mip_data/road_distances.csv', index_col=0)
    start_road_distances = pd.read_csv(path_overall_data + 'processed_data/mip_data/' + str(start_location) + '_start_road_distances.csv', index_col=0)

    new_pipeline_distances = pd.read_csv(path_overall_data + 'processed_data/mip_data/new_pipeline_distances.csv', index_col=0)
    start_new_pipeline_distances = pd.read_csv(path_overall_data + 'processed_data/mip_data/' + str(start_location) + '_start_new_pipeline_distances.csv', index_col=0)

    port_distances = pd.read_csv(path_overall_data + 'processed_data/mip_data/port_distances.csv', index_col=0)
    port_distances = port_distances.stack().reset_index()
    port_distances.columns = ['pointA', 'pointB', 'distance']
    port_durations = pd.read_csv(path_overall_data + 'processed_data/mip_data/ports_durations.csv', index_col=0)
    port_durations = port_durations.stack().reset_index()
    port_durations.columns = ['pointA', 'pointB', 'duration']

    if False:

        road_distances = road_distances[~road_distances['pointA'].str.contains('PG', na=False)]
        road_distances = road_distances[~road_distances['pointB'].str.contains('PG', na=False)]

        road_distances = road_distances[~road_distances['pointA'].str.contains('PL', na=False)]
        road_distances = road_distances[~road_distances['pointB'].str.contains('PL', na=False)]

    start_road_distances = start_road_distances[~start_road_distances['pointA'].str.contains('PG', na=False)]
    start_road_distances = start_road_distances[~start_road_distances['pointB'].str.contains('PG', na=False)]

    start_road_distances = start_road_distances[~start_road_distances['pointA'].str.contains('PL', na=False)]
    start_road_distances = start_road_distances[~start_road_distances['pointB'].str.contains('PL', na=False)]

    options = {'Road': [road_distances, start_road_distances],
               'New_Pipeline': [new_pipeline_distances, start_new_pipeline_distances],
               'Shipping': [port_distances]}

    # todo: end distances berechnen
    # todo: kleinen case finden, den man nutzen kann
    # todo: sind hier beide Richtungen in den Distanzen drin --> checken
    max_costs = 0
    if True:
        for transport_mean in options.keys():

            distances = pd.concat(options[transport_mean], axis=0)
            distances.reset_index(inplace=True)

            for i in distances.index:

                start = distances.loc[i, 'pointA']
                end = distances.loc[i, 'pointB']
                distance = distances.loc[i, 'distance']

                if start == end:
                    continue

                if transport_mean == 'Shipping':
                    duration = port_durations.loc[i, 'duration']

                for com in all_commodities:

                    if ('start' == start) & (com not in start_commodities):
                        continue

                    if transport_mean in techno_economic_data_transport[com]['potential_transportation']:

                        transport_costs = distance / 1000 * techno_economic_data_transport[com][transport_mean] / 1000
                        transport_losses = 0

                        if transport_costs > max_costs:
                            max_costs = transport_costs

                        if transport_mean == 'Shipping':
                            boil_off = duration / 24 * techno_economic_data_transport[com]['Boil_Off']

                            self_consumption = 0
                            if techno_economic_data_transport[com]['Uses_Commodity_as_Shipping_Fuel']:
                                self_consumption = distance / 1000 * techno_economic_data_transport[com]['Self_Consumption']
                                transport_costs = 0

                            transport_losses = max(boil_off, self_consumption)

                        edges[start + '_' + com + '-' + end + '_' + com + '-' + transport_mean] = \
                            ('transport', start + '_' + com, end + '_' + com, transport_costs, transport_losses, com,
                             transport_mean)

                        # if the node is also an end node, one edge is added which leads to final sink
                        if (end == end_node) & (com in target_commodities):
                            edges[end + '_' + com + '-' + 'end'] = \
                                ('transport', end + '_' + com, 'end', 0, 0, com, transport_mean)

    # create warm-start solution from results
    result = pd.read_csv(path_overall_data + '/results/location_results/' + str(start_location) +'_final_solution.csv', index_col=0)
    result = result[result.columns[0]]
    route = ast.literal_eval(result.loc['taken_routes'])
    total_costs = result.loc['current_total_costs']

    cost_route = ast.literal_eval(result.loc['all_previous_total_costs'])
    cost_route = list(set(cost_route))

    commodity = None
    transport_mean = None
    start = None
    end = None
    solution_route = []  # same commodity conversion required?
    for n, segment in enumerate(route):
        if n > 0:
            if len(segment) == 5:  # transport

                start = segment[0]
                end = segment[3]

                if start == 'Start':
                    start = 'start'

                transport_mean = segment[1]

                solution_route.append(start + '_' + commodity + '-' + end + '_' + commodity + '-' + transport_mean)
                start = end
            elif len(segment) == 3:  # conversion

                if commodity == segment[1]:
                    continue

                solution_route.append(start + '_' + commodity + '-' + end + '_' + segment[1])
                commodity = segment[1]
        else:
            commodity = segment[0]
            start = 'start'

    solution_route += [end + '_' + commodity + '-end']

    print(total_costs)
    print(solution_route)

    return all_nodes_adjusted, target_nodes, edges, production_costs, transport_means, solution_route, cost_route, max_costs


def create_edges_from_distance_only(df_list, transport_means, techno_economic_data_transport,
                                    all_commodities, start_commodities):

    edges = {}

    distances = pd.concat(df_list, axis=0)
    distances.reset_index(inplace=True)

    for transport_mean in transport_means:

        for i in distances.index:

            start = distances.loc[i, 'pointA']
            end = distances.loc[i, 'pointB']
            distance = distances.loc[i, 'distance']

            for com in all_commodities:

                if ('start' == start) & (com not in start_commodities):
                    continue

                if transport_mean in techno_economic_data_transport[com]['potential_transportation']:

                    transport_costs = distance * techno_economic_data_transport[com][transport_mean] / 1000
                    transport_losses = 0

                    edges[start + '_' + com + '-' + end + '_' + com + '-' + transport_mean] = \
                        ('transport', start + '_' + com, end + '_' + com, transport_costs, transport_losses, com, transport_mean)

    return edges


def prepare_dummy_data():
    path_config = os.getcwd()
    path_config = os.path.dirname(path_config) + '/algorithm_configuration.yaml'

    yaml_file = open(path_config)
    config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    path_overall_data = config_file['project_folder_path']
    path_raw_data = path_overall_data + 'raw_data/'
    path_processed_data = path_overall_data + 'processed_data/'

    # load techno economic data
    yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    yaml_file = open(path_raw_data + 'techno_economic_data_transportation.yaml')
    techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # self.all_nodes = ['start', 'p_0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 's_0', 's_1', 's_2', 's_3', 's_4', 's_5', 'end']
    all_nodes = ['start', 's_0', 's_1', 's_2', 's_3', 's_4', 's_5', 'end']
    all_commodities = ['Ammonia', 'Hydrogen_Gas']

    # create edges: location identification + energy carrier
    all_nodes_adjusted = [n + '_' + commodity for n in all_nodes for commodity in all_commodities]

    start_node = 'start'
    start_commodities = ['Hydrogen_Gas']
    start_nodes = [start_node + '_' + commodity for commodity in start_commodities]

    target_nodes = ['end']
    target_commodities = ['Hydrogen_Gas']
    target_nodes = [t + '_' + commodity for t in target_nodes for commodity in target_commodities]

    transport_means = ['Pipeline_Gas', 'Shipping', 'Road']

    production_costs = {'Hydrogen_Gas': 25,
                             'Ammonia': 50}

    edges = {}
    for node_1 in all_nodes:
        if 'start' in node_1:
            continue

        for node_2 in all_nodes:

            if 'start' in node_2:
                continue

            if node_1 != node_2:
                continue

            for com_1 in all_commodities:
                for com_2 in all_commodities:
                    if com_2 in techno_economic_data_conversion[com_1]['potential_conversions']:
                        conversion_costs = 10
                        conversion_efficiency = 1 - 0.95

                        edges[node_1 + '_' + com_1 + '-' + node_2 + '_' + com_2] =\
                            ('conversion', node_1 + '_' + com_1, node_2 + '_' + com_2, conversion_costs, conversion_efficiency, com_2)

    transport_options = [('s_0', 's_4', 17, 2, 'Shipping'),
        ('s_0', 's_5', 4, 1, 'Shipping'),
        ('s_0', 's_2', 19, 3, 'Shipping'),
        ('s_1', 's_2', 6, 2, 'Shipping'),
        ('s_1', 's_5', 13, 2, 'Shipping'),
        ('s_2', 's_3', 1, 1, 'Shipping'),
        ('s_3', 's_5', 8, 1, 'Shipping'),
        ('start', 's_0', 2, 0, 'Road'),
        ('s_3', 'end', 2, 0, 'Road')]

    for t in transport_options:
        start = t[0]
        end = t[1]
        distance = t[2]
        duration = t[3]
        transport_mean = t[4]

        for com in all_commodities:

            if ('start' == start) & (com not in start_commodities):
                continue

            if transport_mean in techno_economic_data_transport[com]['potential_transportation']:

                transport_costs = distance * techno_economic_data_transport[com][transport_mean]
                transport_efficiency = 0

                if transport_mean == 'Shipping':
                    boil_off = duration * techno_economic_data_transport[com]['Boil_Off']

                    self_consumption = 0
                    if techno_economic_data_transport[com]['Uses_Commodity_as_Shipping_Fuel']:
                        self_consumption = distance * techno_economic_data_transport[com]['Self_Consumption']

                    transport_efficiency = max(boil_off, self_consumption)

                edges[start + '_' + com + '-' + end + '_' + com + '-' + transport_mean] = \
                    ('transport', start + '_' + com, end + '_' + com, transport_costs, transport_efficiency, com, transport_mean)

    return all_nodes_adjusted, target_nodes, edges, production_costs, transport_means


def create_graph(edges, nodes):

    graph = nx.Graph()

    for edge in edges.keys():

        data = edges[edge]

        if data[0] == 'conversion':  # conversion edge
            start = data[1]
            end = data[2]

            name = start + '_' + end + '_conversion'

        else:  # transport edge
            start = data[1]
            end = data[2]

            name = start + '_' + end + '_transport'

        graph.add_edge(start, end, name=name)

    max_length = 0
    for n_1 in nodes:
        for n_2 in nodes:

            if n_1 == n_2:
                continue

            paths = nx.all_simple_paths(graph, source=n_1, target=n_2)
            print(len(list(paths)))
            for p in paths:
                if len(p) > max_length:
                    max_length = len(p)

    print(max_length)