import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
import shapely

from data_processing.process_network_data_to_network_objects import get_geodata_and_graph_from_network_data_with_intermediate_points

import searoute as sr
from shapely.wkt import loads

from _helpers import calc_distance_single_to_single

import geojson

import itertools

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from shapely.ops import nearest_points

import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')





def get_graph(data_local, geo_data_local, graph_data_local, name, apply_shapely=False):

    data[name] = {}

    graph_data_local = gpd.GeoDataFrame(graph_data_local.copy(), index=graph_data_local.index)

    if apply_shapely:
        graph_data_local['line'] = graph_data_local['line'].apply(shapely.wkt.loads)

    for g in geo_data_local['graph'].unique():
        nodes_graph = graph_data_local[graph_data_local['graph'] == g].index
        lines = []
        for ind in nodes_graph:
            lines.append(graph_data_local.loc[ind, 'line'])

        nodes_graph = geo_data_local[geo_data_local['graph'] == g].index
        graph_object = MultiLineString(lines)

        data_local[name][g] = {'GraphData': graph_data_local,
                               'GraphObject': graph_object,
                               'GeoData': geo_data_local.loc[nodes_graph].copy()}

    return data_local


def adjust_network_to_ports(data_local):

    """
    This function checks if a new node should be added to a network based on closest distance to a port.
    The function iterates through the networks of a given network type (e.g. gas pipelines)
    and checks for each network and each port if a new node should be added based on closest distance.
    If so, the affected line segment is split into two parts at the location where the node with the closest distance
    is located.
    :param data_local: dictionary with network data of all network types
    :return: adjusted networks
    """

    added_nodes = 0

    for network_type in [*data_local.keys()]:

        # get data of specific network type
        if network_type == 'Pipeline_Gas':
            node_addition = 'PG'
            pipeline_geodata = gas_pipeline_geodata
            pipeline_graphs = gas_pipeline_graphs
        elif network_type == 'Pipeline_Liquid':
            node_addition = 'PL'
            pipeline_geodata = oil_pipeline_geodata
            pipeline_graphs = oil_pipeline_graphs
        else:  # Railroad
            continue
            node_addition = 'RR'

        # Not considered means that node is not closest to any other infrastructure. Therefore, it is not considered
        # when transportation changes from one infrastructure to another
        pipeline_geodata['considered'] = False

        # initialize number of new edges and nodes
        new_edge_number = len(pipeline_graphs.index)
        new_node_number = 0

        # iterate through all networks of network type
        i = 0
        for g_id in [*data_local[network_type].keys()]:

            # to show progress
            print(round(i / len([*data_local[network_type].keys()]) * 100, 2))
            i += 1

            # get data of network
            graph = data_local[network_type][g_id]['GraphObject']
            geo_data = data_local[network_type][g_id]['GeoData']

            considered_nodes = {}
            old_nodes = geo_data.copy().index

            # get for each port the closest node if distance is not too high # todo: lower distance?
            for p in ports.index:

                closest_node = nearest_points(graph, Point([ports.loc[p, 'longitude'], ports.loc[p, 'latitude']]))[0]
                distance = calc_distance_single_to_single(closest_node.y, closest_node.x,
                                                          ports.loc[p, 'latitude'], ports.loc[p, 'longitude'])
                if (distance < 50000) & (distance >= 1000):

                    # check if closest node is same as existing node
                    node_index_closest = geo_data[(geo_data['longitude'] == round(closest_node.x, 4))
                                                  & (geo_data['latitude'] == round(closest_node.y, 4))].index

                    if len(node_index_closest) > 0:  # closest node is existing node in graph
                        if node_index_closest[0] not in considered_nodes.keys():
                            considered_nodes[node_index_closest[0]] = closest_node
                    else:  # node does not exist
                        if closest_node not in considered_nodes.values():  # node has not been created in previous steps

                            # add new node to node and lines data
                            new_node_index = 'New_' + node_addition + '_Node_' + str(new_node_number)
                            new_node_number += 1
                            considered_nodes[new_node_index] = closest_node

                            geo_data.loc[new_node_index, 'longitude'] = round(closest_node.x, 4)
                            geo_data.loc[new_node_index, 'latitude'] = round(closest_node.y, 4)
                            geo_data.loc[new_node_index, 'graph'] = g_id

                            pipeline_geodata.loc[new_node_index, 'longitude'] = round(closest_node.x, 4)
                            pipeline_geodata.loc[new_node_index, 'latitude'] = round(closest_node.y, 4)
                            pipeline_geodata.loc[new_node_index, 'graph'] = g_id

            added_nodes += len(considered_nodes.keys())
            # add new nodes to graph data --> this will later be translated to graph objects
            if False:
                for node in [*considered_nodes.keys()]:
                    pipeline_geodata.loc[node, 'considered'] = True
                    if node in old_nodes:
                        continue

                    # Identify LineString where new node will be placed
                    node_object = considered_nodes[node]
                    left_df = gpd.GeoDataFrame(geometry=[node_object])
                    right_df = gpd.GeoDataFrame(geometry=[graph]).explode(ignore_index=True)
                    df_n = gpd.sjoin_nearest(left_df, right_df).merge(right_df, left_on="index_right",
                                                                      right_index=True)

                    affected_line = df_n['geometry_y'].values[0]  # this LineString will be divided
                    starting_node_x = round(affected_line.coords.xy[0][0], 4)
                    starting_node_y = round(affected_line.coords.xy[1][0], 4)

                    ending_node_x = round(affected_line.coords.xy[0][-1], 4)
                    ending_node_y = round(affected_line.coords.xy[1][-1], 4)

                    starting_node_index = geo_data[(geo_data['longitude'] == starting_node_x) &
                                                   (geo_data['latitude'] == starting_node_y)].index[0]
                    ending_node_index = geo_data[(geo_data['longitude'] == ending_node_x) &
                                                 (geo_data['latitude'] == ending_node_y)].index[0]

                    # Separate LineString into first and last part
                    first_part_line = []
                    last_part_line = []

                    c_before = None
                    affected_line_segment_reached = False
                    for c in affected_line.coords:

                        if c_before is not None:
                            c_now = Point([round(c[0], 4), round(c[1], 4)])
                            distance = round(LineString([c_before, c_now]).distance(node_object), 3)

                            if distance == 0:
                                affected_line_segment_reached = True

                            if not affected_line_segment_reached:
                                first_part_line.append((round(c[0], 4), round(c[1], 4)))
                            else:
                                last_part_line.append((round(c[0], 4), round(c[1], 4)))
                        else:
                            first_part_line.append((round(c[0], 4), round(c[1], 4)))

                        c_before = Point([round(c[0], 4), round(c[1], 4)])

                    first_part_line.append((round(node_object.x, 4), round(node_object.y, 4)))
                    last_part_line = [(round(node_object.x, 4), round(node_object.y, 4))] + last_part_line

                    # Calculate distances of both parts
                    c_before = None
                    distance_first_line = 0
                    for c in first_part_line:
                        if c_before is not None:
                            distance_first_line += calc_distance_single_to_single(c[1], c[0], c_before[1], c_before[0])
                        c_before = c

                    c_before = None
                    distance_last_line = 0
                    for c in last_part_line:
                        if c_before is not None:
                            distance_last_line += calc_distance_single_to_single(c[1], c[0], c_before[1], c_before[0])
                        c_before = c

                    # Add first part as new edge to graph & update graph data
                    if len(first_part_line) == 1:
                        print('')
                    first_part_linestring = LineString(first_part_line)

                    new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
                    new_edge_number += 1

                    while new_edge in pipeline_graphs.index:
                        new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
                        new_edge_number += 1

                    pipeline_graphs.loc[new_edge, 'graph'] = g_id
                    pipeline_graphs.loc[new_edge, 'node_start'] = starting_node_index
                    pipeline_graphs.loc[new_edge, 'node_end'] = node
                    pipeline_graphs.loc[new_edge, 'costs'] = distance_first_line
                    pipeline_graphs.loc[new_edge, 'line'] = first_part_linestring

                    last_part_linestring = LineString(last_part_line)

                    new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
                    new_edge_number += 1

                    while new_edge in pipeline_graphs.index:
                        new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
                        new_edge_number += 1

                    pipeline_graphs.loc[new_edge, 'graph'] = g_id
                    pipeline_graphs.loc[new_edge, 'node_start'] = node
                    pipeline_graphs.loc[new_edge, 'node_end'] = ending_node_index
                    pipeline_graphs.loc[new_edge, 'costs'] = distance_last_line
                    pipeline_graphs.loc[new_edge, 'line'] = last_part_linestring

                    # Add new line to graph
                    new_graph = []
                    for line in graph.geoms:
                        new_graph.append(line)

                    new_graph.append(first_part_line)
                    new_graph.append(last_part_line)

                    graph = MultiLineString(new_graph)

        if False:
            pipeline_graphs.sort_values(['graph'], inplace=True)
            pipeline_geodata.sort_values(['graph'], inplace=True)

            pipeline_graphs.to_csv(path_data + network_type + '_graphs_adjusted_to_ports.csv')
            pipeline_geodata.to_csv(path_data + network_type + '_geodata_adjusted_to_ports.csv')

    print('first: ' + str(added_nodes))


def test_adjustment(data_local):

    """
    This function checks if a new node should be added to a network based on closest distance to a port.
    The function iterates through the networks of a given network type (e.g. gas pipelines)
    and checks for each network and each port if a new node should be added based on closest distance.
    If so, the affected line segment is split into two parts at the location where the node with the closest distance
    is located.
    :param data_local: dictionary with network data of all network types
    :return: adjusted networks
    """

    def find_closest_points_lines_to_self(iteration):

        nep_test = nearest_points(left_lines[iteration], right_lines[iteration])

        # todo: hier könnte man noch ne maximale edge länge definieren
        # todo: man könnte über rundung eventuell auch noch ein paar Punkte raushauen

        return nep_test[0], nep_test[1] # (graph_1, combination[0], nep[0]), (graph_2, combination[1], nep[1])

    gas_lines = gas_pipeline_line_data['geometry'].apply(shapely.wkt.loads)
    # gas_lines = gas_lines.iloc[0:10]

    combinations = list(itertools.combinations(gas_lines.index, 2))
    left_combinations = [i[0] for i in combinations]
    right_combinations = [i[1] for i in combinations]

    left_lines = gas_lines.loc[left_combinations].tolist()
    left_graphs = gas_pipeline_graphs.loc[left_combinations, 'graph'].tolist()

    right_lines = gas_lines.loc[right_combinations].tolist()
    right_graphs = gas_pipeline_graphs.loc[right_combinations, 'graph'].tolist()

    print(len(combinations))

    inputs = tqdm(range(len(combinations)))
    results = Parallel(n_jobs=120)(
        delayed(find_closest_points_lines_to_self)(inp) for inp in inputs)

    results = [item for t in results for item in t]

    gas_df = pd.DataFrame(results, columns=['graph', 'edge', 'point'])
    gas_df.drop_duplicates(inplace=True)
    gas_df = gas_df.sort_values(by=['graph', 'edge'], ignore_index=True)

    gas_df.to_csv('/home/localadmin/Dokumente/Daten_Transportmodell/0_test_nodes_gas.csv')

    oil_lines = oil_pipeline_line_data['geometry'].apply(shapely.wkt.loads)
    # oil_lines = oil_lines.iloc[0:100]

    combinations = list(itertools.combinations(oil_lines.index, 2))
    print(len(combinations))

    inputs = tqdm(combinations)
    results = Parallel(n_jobs=120)(
        delayed(find_closest_points_lines_to_self)(inp, gas_pipeline_graphs) for inp in inputs)

    results = [item for t in results for item in t]

    oil_df = pd.DataFrame(results, columns=['graph', 'edge', 'point'])
    oil_df.drop_duplicates(inplace=True)
    oil_df = oil_df.sort_values(by=['graph', 'edge'], ignore_index=True)

    oil_df.to_csv('/home/localadmin/Dokumente/Daten_Transportmodell/0_test_nodes_oil.csv')

    for gas_line in gas_lines.index:
        for oil_line in oil_lines.index:

            line_1 = gas_lines.loc[gas_line]
            line_2 = oil_lines.loc[oil_line]

            nep = nearest_points(line_1, line_2)

            # todo: hier könnte man noch ne maximale edge länge definieren
            # todo: man könnte über rundung eventuell auch noch ein paar Punkte raushauen

            if (gas_pipeline_graphs.loc[gas_line, 'graph'], gas_line, nep[0]) not in new_points_tuples:
                new_points_tuples.append((gas_pipeline_graphs.loc[gas_line, 'graph'], gas_line, nep[0]))
            if (oil_pipeline_graphs.loc[oil_line, 'graph'], oil_line, nep[1]) not in new_points_tuples:
                new_points_tuples.append((oil_pipeline_graphs.loc[oil_line, 'graph'], oil_line, nep[1]))

    gas_oil_df = pd.DataFrame(new_points_tuples, columns=['graph', 'edge', 'point'])
    gas_oil_df = gas_oil_df.sort_values(by=['graph', 'edge'], ignore_index=True)

    gas_oil_df.to_csv('/home/localadmin/Dokumente/Daten_Transportmodell/0_test_nodes_gas_oil.csv')

    new_points_tuples = []
    for gas_line in gas_lines.index:
        gas_linestring = gas_lines.loc[gas_line]
        for p in ports.index:
            point = Point([ports.loc[p, 'longitude'], ports.loc[p, 'latitude']])

            nep = nearest_points(gas_linestring, point)
            if (gas_pipeline_graphs.loc[gas_line, 'graph'], gas_line, nep[0]) not in new_points_tuples:
                new_points_tuples.append((gas_pipeline_graphs.loc[gas_line, 'graph'], gas_line, nep[0]))

    for oil_line in oil_lines.index:
        oil_linestring = oil_lines.loc[oil_line]
        for p in ports.index:
            point = Point([ports.loc[p, 'longitude'], ports.loc[p, 'latitude']])

            nep = nearest_points(point, oil_linestring)
            if (oil_pipeline_graphs.loc[oil_line, 'graph'], oil_line, nep[1]) not in new_points_tuples:
                new_points_tuples.append((oil_pipeline_graphs.loc[oil_line, 'graph'], oil_line, nep[1]))

    gas_oil_ports_df = pd.DataFrame(new_points_tuples, columns=['graph', 'edge', 'point'])
    gas_oil_ports_df = gas_oil_ports_df.sort_values(by=['graph', 'edge'], ignore_index=True)

    gas_oil_ports_df.to_csv('/home/localadmin/Dokumente/Daten_Transportmodell/0_test_nodes_gas_oil_ports.csv')

    # drop duplicates
    network_df = pd.concat([gas_df, oil_df, gas_oil_df, gas_oil_ports_df], ignore_index=True)
    print(len(network_df.index))
    network_df.drop_duplicates(inplace=True, ignore_index=True)
    print(len(network_df.index))
    network_df.to_csv('/home/localadmin/Dokumente/Daten_Transportmodell/0_test_nodes_gas_oil_ports.csv')


def adjust_network_to_network(data_local, network_nodes, network_graphs, network_type, other_network):

    """
    This function checks if a new node should be added to a network based on closest distance to other network.
    The function iterates through the networks of a given network type (e.g. gas pipelines)
    and checks for each network and each port if a new node should be added based on closest distance.
    If so, the affected line segment is split into two parts at the location where the node with the closest distance
    is located.
    :param data_local: dictionary with all the data
    :param network_nodes: all existing nodes of the network type
    :param network_graphs: graph object of all networks of the network type
    :param network_type: name of the network
    :param other_network: name of the network to compare with
    :return: 
    """
    
    new_network_nodes = network_nodes.copy()
    new_network_graphs = network_graphs.copy()

    if network_type == 'Pipeline_Gas':
        node_addition = 'PG'
    elif network_type == 'Pipeline_Liquid':
        node_addition = 'PL'
    else:  # Railroad
        node_addition = 'RR'

    new_edge_number = len(network_nodes.index)
    new_node_number = 0

    i = 0
    for g_id in [*data_local[network_type].keys()]:

        print(round(i / len([*data_local[network_type].keys()]) * 100, 2))
        i += 1

        graph = data_local[network_type][g_id]['GraphObject']
        geo_data = data_local[network_type][g_id]['GeoData'].copy()

        considered_nodes = {}
        old_nodes = geo_data.copy().index

        # get shortest distance between each network

        for p in other_network.index:

            if other_network.loc[p, 'graph'] == g_id:
                continue

            closest_node = nearest_points(graph,
                                          Point([other_network.loc[p, 'longitude'],
                                                 other_network.loc[p, 'latitude']]))[0]
            distance = calc_distance_single_to_single(closest_node.y, closest_node.x,
                                                      other_network.loc[p, 'latitude'],
                                                      other_network.loc[p, 'longitude'])
            if (distance < 50000) & (distance >= 1000):

                node_index_closest = geo_data[(geo_data['longitude'] == round(closest_node.x, 4))
                                              & (geo_data['latitude'] == round(closest_node.y, 4))].index

                if len(node_index_closest) > 0:  # closest node is node in graph
                    if node_index_closest[0] not in considered_nodes.keys():
                        considered_nodes[node_index_closest[0]] = closest_node
                else:
                    if closest_node not in considered_nodes.values():

                        new_node_index = 'New_' + node_addition + '_Node_' + str(new_node_number)
                        new_node_number += 1

                        while new_node_index in new_network_nodes.index:
                            new_node_index = 'New_' + node_addition + '_Node_' + str(new_node_number)
                            new_node_number += 1

                        considered_nodes[new_node_index] = closest_node

                        geo_data.loc[new_node_index, 'longitude'] = round(closest_node.x, 4)
                        geo_data.loc[new_node_index, 'latitude'] = round(closest_node.y, 4)
                        geo_data.loc[new_node_index, 'graph'] = g_id

                        new_network_nodes.loc[new_node_index, 'longitude'] = round(closest_node.x, 4)
                        new_network_nodes.loc[new_node_index, 'latitude'] = round(closest_node.y, 4)
                        new_network_nodes.loc[new_node_index, 'graph'] = g_id

        # add new nodes to graph data --> this will later be translated to graph objects
        for node in [*considered_nodes.keys()]:
            new_network_nodes.loc[node, 'considered'] = True
            if node in old_nodes:
                continue

            # Identify LineString where new node will be placed
            node_object = considered_nodes[node]
            left_df = gpd.GeoDataFrame(geometry=[node_object])
            right_df = gpd.GeoDataFrame(geometry=[graph]).explode(ignore_index=True)
            df_n = gpd.sjoin_nearest(left_df, right_df).merge(right_df, left_on="index_right",
                                                              right_index=True)

            affected_line = df_n['geometry_y'].values[0]  # this LineString will be divided
            starting_node_x = round(affected_line.coords.xy[0][0], 4)
            starting_node_y = round(affected_line.coords.xy[1][0], 4)

            ending_node_x = round(affected_line.coords.xy[0][-1], 4)
            ending_node_y = round(affected_line.coords.xy[1][-1], 4)

            starting_node_index = geo_data[(geo_data['longitude'] == starting_node_x) &
                                           (geo_data['latitude'] == starting_node_y)].index[0]
            ending_node_index = geo_data[(geo_data['longitude'] == ending_node_x) &
                                         (geo_data['latitude'] == ending_node_y)].index[0]

            # Separate LineString into first and last part
            first_part_line = []
            last_part_line = []

            c_before = None
            affected_line_segment_reached = False
            for c in affected_line.coords:

                if c_before is not None:
                    c_now = Point([round(c[0], 4), round(c[1], 4)])
                    distance = round(LineString([c_before, c_now]).distance(node_object), 3)

                    if distance == 0:
                        affected_line_segment_reached = True

                    if not affected_line_segment_reached:
                        first_part_line.append((round(c[0], 4), round(c[1], 4)))
                    else:
                        last_part_line.append((round(c[0], 4), round(c[1], 4)))
                else:
                    first_part_line.append((round(c[0], 4), round(c[1], 4)))

                c_before = Point([round(c[0], 4), round(c[1], 4)])

            first_part_line.append((round(node_object.x, 4), round(node_object.y, 4)))
            last_part_line = [(round(node_object.x, 4), round(node_object.y, 4))] + last_part_line

            # Calculate distances of both parts
            c_before = None
            distance_first_line = 0
            for c in first_part_line:
                if c_before is not None:
                    distance_first_line += calc_distance_single_to_single(c[1], c[0], c_before[1], c_before[0])
                c_before = c

            c_before = None
            distance_last_line = 0
            for c in last_part_line:
                if c_before is not None:
                    distance_last_line += calc_distance_single_to_single(c[1], c[0], c_before[1], c_before[0])
                c_before = c

            # Add first part as new edge to graph & update graph data
            if len(first_part_line) == 1:
                print('')
            first_part_linestring = LineString(first_part_line)

            new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
            new_edge_number += 1

            while new_edge in new_network_graphs.index:
                new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
                new_edge_number += 1

            new_network_graphs.loc[new_edge, 'graph'] = g_id
            new_network_graphs.loc[new_edge, 'node_start'] = starting_node_index
            new_network_graphs.loc[new_edge, 'node_end'] = node
            new_network_graphs.loc[new_edge, 'costs'] = distance_first_line
            new_network_graphs.loc[new_edge, 'line'] = first_part_linestring

            # add second part as new edge
            last_part_linestring = LineString(last_part_line)

            new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
            new_edge_number += 1

            while new_edge in new_network_graphs.index:
                new_edge = 'New_' + node_addition + '_Edge_' + str(new_edge_number)
                new_edge_number += 1

            new_network_graphs.loc[new_edge, 'graph'] = g_id
            new_network_graphs.loc[new_edge, 'node_start'] = node
            new_network_graphs.loc[new_edge, 'node_end'] = ending_node_index
            new_network_graphs.loc[new_edge, 'costs'] = distance_last_line
            new_network_graphs.loc[new_edge, 'line'] = last_part_linestring

            # Add new line to graph
            new_graph = []
            for line in graph.geoms:
                new_graph.append(line)

            new_graph.append(first_part_line)
            new_graph.append(last_part_line)

            graph = MultiLineString(new_graph)

    new_network_graphs.to_csv(path_data + network_type + '_graphs_adjusted_to_ports_to_networks.csv')
    new_network_nodes.to_csv(path_data + network_type + '_geodata_adjusted_to_ports_to_networks.csv')


path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/'
path_railroad_data = path_data + 'railway_data/'
path_gas_pipeline_data = path_data + 'gas_pipeline_data/'
path_oil_pipeline_data = path_data + 'oil_pipeline_data/'
coastlines = pd.read_csv(path_data + 'coastlines.csv', index_col=0)
coastlines = gpd.GeoDataFrame(geometry=coastlines['geometry'].apply(loads))
coastlines.set_geometry('geometry', inplace=True)

if False:
    if False:
        ports, port_distances = process_seaport_data()
        ports.to_excel(path_data + 'ports_processed.xlsx')
        port_distances.to_csv(path_data + 'port_distances.csv')
    else:
        ports = pd.read_excel(path_data + 'ports_processed.xlsx', index_col=0)
        port_distances = pd.read_csv(path_data + 'port_distances.csv', index_col=0)

if False:
    if True:
        gas_pipeline_line_data, gas_pipeline_graphs, gas_pipeline_geodata \
            = get_geodata_and_graph_from_network_data_with_intermediate_points(path_gas_pipeline_data, 'gas_pipeline')
        # gas_pipeline_shortest_paths = get_shortest_paths_through_networks(gas_pipeline_graphs, gas_pipeline_geodata)

        gas_pipeline_line_data.to_csv(path_data + 'gas_pipeline_graphs_object.csv')
        gas_pipeline_graphs.to_csv(path_data + 'gas_pipeline_graphs.csv')
        gas_pipeline_geodata.to_csv(path_data + 'gas_pipeline_geodata.csv')
        # gas_pipeline_shortest_paths.to_csv(path_data + 'gas_pipeline_shortest_paths.csv')

    else:
        gas_pipeline_line_data = pd.read_csv(path_data + 'gas_pipeline_graphs_object.csv', index_col=0)

        if True:
            gas_pipeline_graphs = pd.read_csv(path_data + 'gas_pipeline_graphs.csv', index_col=0)
            gas_pipeline_geodata = pd.read_csv(path_data + 'gas_pipeline_geodata.csv', index_col=0)
        else:
            gas_pipeline_graphs = pd.read_csv(path_data + 'gas_pipeline_graphs_adjusted.csv', index_col=0)
            gas_pipeline_geodata = pd.read_csv(path_data + 'gas_pipeline_geodata_adjusted.csv', index_col=0)

if True:

    if True:
        oil_pipeline_line_data, oil_pipeline_graphs, oil_pipeline_geodata \
            = get_geodata_and_graph_from_network_data_with_intermediate_points(path_oil_pipeline_data, 'oil_pipeline')
        # oil_pipeline_shortest_paths = get_shortest_paths_through_networks(oil_pipeline_graphs, oil_pipeline_geodata)

        oil_pipeline_line_data.to_csv(path_data + 'oil_pipeline_graphs_object.csv')
        oil_pipeline_graphs.to_csv(path_data + 'oil_pipeline_graphs.csv')
        oil_pipeline_geodata.to_csv(path_data + 'oil_pipeline_geodata.csv')
        # oil_pipeline_shortest_paths.to_csv(path_data + 'oil_pipeline_shortest_paths.csv')
    else:
        oil_pipeline_line_data = pd.read_csv(path_data + 'oil_pipeline_graphs_object.csv', index_col=0)

        if True:
            oil_pipeline_graphs = pd.read_csv(path_data + 'oil_pipeline_graphs.csv', index_col=0)
            oil_pipeline_geodata = pd.read_csv(path_data + 'oil_pipeline_geodata.csv', index_col=0)
        else:
            oil_pipeline_graphs = pd.read_csv(path_data + 'gas_pipeline_graphs_adjusted.csv', index_col=0)
            oil_pipeline_geodata = pd.read_csv(path_data + 'gas_pipeline_geodata_adjusted.csv', index_col=0)

if False:
    data = {}
    data = get_graph(data, gas_pipeline_geodata, gas_pipeline_graphs, 'Pipeline_Gas', apply_shapely=True)
    data = get_graph(data, oil_pipeline_geodata, oil_pipeline_graphs, 'Pipeline_Liquid', apply_shapely=True)

    # adjust_network_to_ports(data)
    # test_adjustment(data)

if False:

    gas_pipeline_graphs = pd.read_csv(path_data + 'Pipeline_Gas_graphs_adjusted_to_ports.csv', index_col=0)
    gas_pipeline_geodata = pd.read_csv(path_data + 'Pipeline_Gas_geodata_adjusted_to_ports.csv', index_col=0)

    oil_pipeline_graphs = pd.read_csv(path_data + 'Pipeline_Liquid_graphs_adjusted_to_ports.csv', index_col=0)
    oil_pipeline_geodata = pd.read_csv(path_data + 'Pipeline_Liquid_geodata_adjusted_to_ports.csv', index_col=0)

    data = {}
    data = get_graph(data, gas_pipeline_geodata, gas_pipeline_graphs, 'Pipeline_Gas', apply_shapely=True)
    data = get_graph(data, oil_pipeline_geodata, oil_pipeline_graphs, 'Pipeline_Liquid', apply_shapely=True)

    adjust_network_to_network(data, gas_pipeline_geodata, gas_pipeline_graphs, 'Pipeline_Gas', oil_pipeline_geodata)

    print('')

if False:

    gas_pipeline_graphs = pd.read_csv(path_data + 'Pipeline_Gas_graphs_adjusted_to_ports_to_networks.csv', index_col=0)
    gas_pipeline_geodata = pd.read_csv(path_data + 'Pipeline_Gas_geodata_adjusted_to_ports_to_networks.csv', index_col=0)

    oil_pipeline_graphs = pd.read_csv(path_data + 'Pipeline_Liquid_graphs_adjusted_to_ports.csv', index_col=0)
    oil_pipeline_geodata = pd.read_csv(path_data + 'Pipeline_Liquid_geodata_adjusted_to_ports.csv', index_col=0)

    data = {}
    data = get_graph(data, gas_pipeline_geodata, gas_pipeline_graphs, 'Pipeline_Gas', apply_shapely=True)
    data = get_graph(data, oil_pipeline_geodata, oil_pipeline_graphs, 'Pipeline_Liquid', apply_shapely=True)

    adjust_network_to_network(data, oil_pipeline_geodata, oil_pipeline_graphs, 'Pipeline_Liquid', gas_pipeline_geodata)

    print('')

# test_adjustment(data)

if False:

    df_distance = calculate_road_distances([gas_pipeline_geodata, oil_pipeline_geodata],
                                           [gas_pipeline_graphs, oil_pipeline_graphs],
                                           ['Pipeline_Gas', 'Pipeline_Liquid'],
                                           ports)

    # df_distance.to_csv(path_data + 'test_distances.csv')

    get_distances_within_networks(gas_pipeline_graphs, path_data)
    get_distances_within_networks(oil_pipeline_graphs, path_data)

