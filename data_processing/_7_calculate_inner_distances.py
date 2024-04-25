import itertools
import math
import os

import pandas as pd
import searoute as sr
import networkx as nx

from joblib import Parallel, delayed
from tqdm import tqdm

from algorithm.methods_geographic import calc_distance_single_to_single, calc_distance_list_to_single

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


def get_distances_within_networks(network_graph_data, path_processed_data, use_low_memory=False):

    """
    Calculates distances between nodes within each network graph using Dijkstra and saves the distances as HDF5 files.

    @param pandas.DataFrame network_graph_data: A pandas DataFrame containing information about network graph edges.
    @param str path_processed_data: Path to the directory where the processed data will be saved.
    @param bool use_low_memory: indicating of large matrixes can be used or not to calculate distances

    """

    # save processed data in folder (each network own folder)
    name_folder = path_processed_data + 'inner_infrastructure_distances/'
    if 'inner_infrastructure_distances' not in os.listdir(path_processed_data):
        os.mkdir(name_folder)

    graphs = set(network_graph_data['graph'])

    for g in tqdm(graphs):

        # create networkx graph
        graph = nx.Graph()
        edges_graph = network_graph_data[network_graph_data['graph'] == g].index
        for edge in edges_graph:
            node_start = network_graph_data.loc[edge, 'node_start']
            node_end = network_graph_data.loc[edge, 'node_end']
            distance = network_graph_data.loc[edge, 'distance']
            graph.add_edge(node_start, node_end, weight=distance)

        if not nx.is_connected(graph):
            # only for debugging if nodes of a graph are not connected to graph

            print(g)

            sub_graphs = nx.connected_components(graph)
            subgraph_nodes = []
            i = 0
            for i, nodes in enumerate(sub_graphs):
                subgraph_nodes.append(nodes)

            print('number of subgraphs: ' + str(i+1))

        # calculate distance and save in dataframe
        if not use_low_memory:
            all_distances_network = dict(nx.all_pairs_dijkstra_path_length(graph))
            all_distances_network_df = pd.DataFrame(all_distances_network)

            # save distances as hdf files
            all_distances_network_df = all_distances_network_df.transpose()
            for column in all_distances_network_df.columns:
                sub_df = all_distances_network_df[[column]]

                sub_df.to_hdf(path_processed_data + '/inner_infrastructure_distances/' + column + '.h5', column, mode='w',
                              format='table')

        else:
            nodes = network_graph_data.loc[edges_graph, 'node_start'].tolist() + network_graph_data.loc[edges_graph, 'node_end'].tolist()
            nodes = list(set(nodes))

            for n in nodes:

                distances = nx.single_source_dijkstra_path_length(graph, n)
                distances = pd.DataFrame(distances.values(), index=[*distances.keys()], columns=[n])

                if distances.isnull().values.any():
                    print('nan')

                import numpy as np
                if not distances[(distances == np.inf).any(axis=1)].empty:
                    print('inf')

                distances.to_hdf(path_processed_data + '/inner_infrastructure_distances/' + n + '.h5', n,
                                 mode='w', format='table')


def get_distances_of_closest_infrastructure(options, path_processed_data):

    """
    Finds the distances to the closest infrastructure nodes for each option.

    @param pandas.DataFrame options: A DataFrame containing the options data, including latitude, longitude, and graph information.
    @param str path_processed_data: The path to save the processed data.

    @return: A DataFrame containing the minimal distances and the corresponding closest nodes for each option.
    @rtype: pandas.DataFrame
    """

    minimal_values = {}
    minimal_value_nodes = {}
    for i in tqdm(options.index.tolist()):

        minimal_values[i] = math.inf

        latitude = options.at[i, 'latitude']
        longitude = options.at[i, 'longitude']

        graph_1 = options.at[i, 'graph']

        # if we keep i in options, distance will always be 0
        options_without_i = options.copy().drop(i)

        # if i belongs to a infrastructure, remove infrastructure options
        if graph_1 is not None:
            other_options = options_without_i[options_without_i['graph'] != graph_1]
        else:
            other_options = options_without_i.copy()

        distances = pd.DataFrame()
        distances.index = other_options.index
        distances['direct_distance'] = calc_distance_list_to_single(other_options['latitude'],
                                                                    other_options['longitude'],
                                                                    latitude, longitude)

        minimal_values[i] = distances['direct_distance'].min()
        minimal_value_nodes[i] = distances['direct_distance'].idxmin()

    distances = pd.DataFrame({'minimal_distance': minimal_values.values(),
                              'closest_node': minimal_value_nodes.values()},
                             index=list(minimal_values.keys()))
    distances.to_csv(path_processed_data + 'minimal_distances.csv')
