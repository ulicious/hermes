import itertools
import math
import os

import pandas as pd
import searoute as sr
import networkx as nx

from joblib import Parallel, delayed
from tqdm import tqdm

from algorithm.methods_geographic import calc_distance_single_to_single, calc_distance_list_to_single, calc_distance_list_to_list

import warnings
warnings.filterwarnings('ignore')


def calculate_searoute_distances(ports, num_cores, path_processed_data):

    """
    Calculates sea route distances between pairs of ports and saves the distances in a CSV file.

    @param panda.DataFrame ports: A pandas DataFrame containing information about ports.
    @param int num_cores: Number of CPU cores to use for parallel processing.
    @param str path_processed_data: Path to the directory where the processed data will be saved.

    """

    def calculate_searoute(combination):

        """
        Calculates the sea route distance between two ports.

        @param tuple combination: A tuple containing the combination of two ports.

        @return: A tuple containing the combination and the calculated distance.
        """

        # todo: this could be done more efficiently by exporting the network and use dijkstra only once

        start_local = combination[0]
        end_local = combination[1]

        start_location = [ports.loc[start_local, 'longitude'], ports.loc[start_local, 'latitude']]
        end_location = [ports.loc[end_local, 'longitude'], ports.loc[end_local, 'latitude']]

        # apply searoute
        route = sr.searoute(start_location, end_location, append_orig_dest=True)
        distance_local = (round(float(format(route.properties['length'])), 2)) * 1000  # m

        # searoute is not always exact as locations are mapped to nodes. Therefore, it might be possible that distances
        # between ports is 0 --> use as-the-crow-flies distance to calculate distance if distance is 0
        if distance_local == 0:
            distance_local = calc_distance_single_to_single(ports.loc[start_local, 'latitude'],
                                                            ports.loc[start_local, 'longitude'],
                                                            ports.loc[end_local, 'latitude'],
                                                            ports.loc[end_local, 'longitude'])

        return combination, distance_local

    # get all combinations of ports
    combinations = list(itertools.combinations(ports.index, 2))

    # apply multiprocessing to calculate distances
    inputs = tqdm(combinations)
    results = Parallel(n_jobs=num_cores)(delayed(calculate_searoute)(inp) for inp in inputs)

    # process results
    ports_distances = pd.DataFrame(0, index=ports.index, columns=ports.index.tolist())
    for r in results:
        start = r[0][0]
        end = r[0][1]
        distance = r[1]

        ports_distances.loc[start, end] = distance
        ports_distances.loc[end, start] = distance

    ports_distances.to_csv(path_processed_data + 'inner_infrastructure_distances/' + 'port_distances.csv')


def get_distances_within_networks(network_graph_data, nodes, path_processed_data, num_workers, use_low_memory=False):

    """
    Calculates distances between nodes within each network graph using Dijkstra and saves the distances as HDF5 files.

    @param pandas.DataFrame network_graph_data: A pandas DataFrame containing information about network graph edges.
    @param str path_processed_data: Path to the directory where the processed data will be saved.
    @param int num_workers: numbers of threads to process distances
    @param bool use_low_memory: indicating of large matrices can be used or not to calculate distances

    """

    def save_distances_columnwise(inp):
        inp.fillna(math.inf, inplace=True)

        if False:  # todo: remove (was used to check if distances are reasonable)

            distances = calc_distance_list_to_single(nodes.loc[inp.columns, 'latitude'], nodes.loc[inp.columns, 'longitude'],
                                                     nodes.loc[inp.index[0], 'latitude'], nodes.loc[inp.index[0], 'longitude'])
            distances = pd.DataFrame(distances, index=inp.columns, columns=inp.index).transpose()

            difference = inp.div(distances)
            difference.fillna(1, inplace=True)

            if difference.min(axis=1).values[0] < 0.99:
                print(difference.min(axis=1))

        inp.to_hdf(path_processed_data + '/inner_infrastructure_distances/' + inp.columns[0] + '.h5', key=inp.columns[0],
                   mode='w', format='table')

    def save_distances_per_node(n):
        distances = nx.single_source_dijkstra_path_length(graph, n)
        distances = pd.DataFrame(distances.values(), index=[*distances.keys()], columns=[n])
        distances.fillna(math.inf, inplace=True)

        distances.to_hdf(path_processed_data + '/inner_infrastructure_distances/' + n + '.h5', key=n,
                         mode='w', format='table')

    def create_graphs(graph_id):
        # create networkx graph
        graph_local = nx.Graph()
        edges_graph_local = network_graph_data[network_graph_data['graph'] == graph_id].index
        processed_nodes = []

        for edge in edges_graph_local:
            node_start = network_graph_data.loc[edge, 'node_start']
            node_end = network_graph_data.loc[edge, 'node_end']
            distance = network_graph_data.loc[edge, 'distance']

            if node_start not in processed_nodes:
                graph_local.add_node(node_start, pos=(nodes.loc[node_start, 'longitude'], nodes.loc[node_start, 'latitude']))
                processed_nodes.append(node_start)

            if node_end not in processed_nodes:
                graph_local.add_node(node_end, pos=(nodes.loc[node_end, 'longitude'], nodes.loc[node_end, 'latitude']))
                processed_nodes.append(node_end)

            graph_local.add_edge(node_start, node_end, weight=distance)

        if not use_low_memory:
            all_distances_network = dict(nx.all_pairs_dijkstra_path_length(graph_local))
            all_distances_network = pd.DataFrame(all_distances_network)

            # save distances as hdf files
            all_distances_network = all_distances_network.transpose()

        else:
            all_distances_network = None

        return graph_id, graph_local, all_distances_network

    # save processed data in folder (each network own folder)
    name_folder = path_processed_data + 'inner_infrastructure_distances/'
    if 'inner_infrastructure_distances' not in os.listdir(path_processed_data):
        os.mkdir(name_folder)

    graphs = set(sorted(network_graph_data['graph'].tolist()))

    inputs = tqdm(graphs)
    results = Parallel(n_jobs=num_workers)(delayed(create_graphs)(i) for i in inputs)

    network_graph_data.sort_index(inplace=True)

    graph_objects = {}
    for r in results:

        # nodes_lon = nodes.loc[r[2].index, 'longitude']
        # nodes_lat = nodes.loc[r[2].index, 'latitude']
        #
        # all_distances = calc_distance_list_to_list(nodes_lat, nodes_lon, nodes_lat, nodes_lon)
        # all_distances = pd.DataFrame(all_distances, index=r[2].index, columns=r[2].index)
        #
        # affected_graph = False
        # for node_1 in all_distances.index:
        #     for node_2 in all_distances.index:
        #         if all_distances.loc[node_1, node_2] * 0.99 > r[2].loc[node_1, node_2]:
        #
        #             print(r[2].loc[node_1, node_2] / all_distances.loc[node_1, node_2])
        #
        #             # affected_graph = True
        #
        # if affected_graph:
        #     pos = nx.get_node_attributes(r[1], 'pos')
        #     nx.draw(r[1], pos)
        #     label = nx.get_edge_attributes(r[1], 'weight')
        #     nx.draw_networkx_edge_labels(r[1], pos, edge_labels=label)
        #
        #     import matplotlib.pyplot as plt
        #     plt.show()

        if not use_low_memory:
            graph_objects[r[0]] = {'graph': r[1],
                                   'distances': r[2]}

        else:
            graph_objects[r[0]] = {'graph': r[1],
                                   'distances': None}

    for g in graph_objects.keys():

        graph = graph_objects[g]['graph']
        if not nx.is_connected(graph):
            # only for debugging if nodes of a graph are not connected to graph

            print(g)

            sub_graphs = nx.connected_components(graph)
            subgraph_nodes = []
            i = 0
            for i, nodes in enumerate(sub_graphs):
                print(len(nodes))
                subgraph_nodes.append(nodes)

            print('number of subgraphs: ' + str(i + 1))

        # calculate distance and save in dataframe
        if not use_low_memory:

            all_distances_network_df = graph_objects[g]['distances']

            # all_distances_columns = all_distances_network_df.columns
            # all_distances_index = all_distances_network_df.index
            #
            # all_distances_network_np = all_distances_network_df.to_numpy()

            inputs = []
            for i in all_distances_network_df.index:
                inputs.append(all_distances_network_df.loc[[i], :].transpose())

            inputs = tqdm(inputs)

            # inputs = tqdm(all_distances_network_df.columns)
            Parallel(n_jobs=num_workers)(delayed(save_distances_columnwise)(i) for i in inputs)

        else:
            edges_graph = network_graph_data[network_graph_data['graph'] == g].index
            nodes = network_graph_data.loc[edges_graph, 'node_start'].tolist() + network_graph_data.loc[edges_graph, 'node_end'].tolist()
            nodes = list(set(nodes))

            inputs = tqdm(nodes)
            Parallel(n_jobs=num_workers)(delayed(save_distances_per_node)(i) for i in inputs)


def get_distances_of_closest_infrastructure(options, path_processed_data, number_workers):

    """
    Finds the distances to the closest infrastructure nodes for each option.

    @param pandas.DataFrame options: A DataFrame containing the options data, including latitude, longitude, and graph information.
    @param str path_processed_data: The path to save the processed data.
    @param int number_workers: Number of threads to process distances

    @return: A DataFrame containing the minimal distances and the corresponding closest nodes for each option.
    @rtype: pandas.DataFrame
    """

    def calculate_distance(i):

        latitude = options.at[i, 'latitude']
        longitude = options.at[i, 'longitude']

        graph_1 = options.at[i, 'graph']

        # if we keep i in options, distance will always be 0
        options_without_i = options.copy().drop(i)

        # if i belongs to an infrastructure, remove options of this infrastructure
        if graph_1 is not None:
            other_options = options_without_i[options_without_i['graph'] != graph_1]
        else:
            other_options = options_without_i.copy()

        direct_distances = calc_distance_list_to_single(other_options['latitude'], other_options['longitude'],
                                                        latitude, longitude)

        distances_df = pd.Series(direct_distances, index=other_options.index)

        return distances_df.min(), distances_df.idxmin()

    inputs = tqdm(options.index)
    results = Parallel(n_jobs=number_workers)(delayed(calculate_distance)(i) for i in inputs)

    min_distances = []
    min_nodes = []

    for r in results:
        min_distances.append(r[0])
        min_nodes.append(r[1])

    distances = pd.DataFrame({'minimal_distance': min_distances,
                              'closest_node': min_nodes},
                             index=options.index)
    distances.to_csv(path_processed_data + 'minimal_distances.csv')
