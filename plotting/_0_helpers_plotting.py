import shapely

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np

from shapely.geometry import MultiLineString, Point


def load_data(path_data, config_file):
    pipeline_gas_node_locations = pd.read_csv(path_data + 'gas_pipeline_node_locations.csv', index_col=0,
                                       dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_gas_graphs = pd.read_csv(path_data + 'gas_pipeline_graphs.csv', index_col=0)
    pipeline_liquid_node_locations = pd.read_csv(path_data + 'oil_pipeline_node_locations.csv', index_col=0,
                                          dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_liquid_graphs = pd.read_csv(path_data + 'oil_pipeline_graphs.csv', index_col=0)
    ports = pd.read_csv(path_data + 'ports.csv', index_col=0)

    data = {'Shipping': {'ports': ports}}

    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_node_locations, pipeline_gas_graphs)

    data = process_network_data(data, 'Pipeline_Liquid', pipeline_liquid_node_locations, pipeline_liquid_graphs)

    destination = Point(config_file['destination_location'])

    return data, destination


def process_network_data(data, name, geo_data, graph_data):

    """
    Function is used to create different data structures for the network data
    @param dict data: dictionary for all data
    @param str name: name of network
    @param pandas.DataFrame geo_data: geo data of network (locations of nodes)
    @param pandas.DataFrame graph_data: information on lines of network

    @return: different data structures
    """

    data[name] = {}

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

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


def get_complete_infrastructure(data, final_destination):

    options = pd.DataFrame()
    # Check final destination and add to option outside tolerance if applicable
    options.loc['Destination', 'latitude'] = final_destination.y
    options.loc['Destination', 'longitude'] = final_destination.x

    options_to_concat = []
    for m in ['Shipping', 'Pipeline_Gas', 'Pipeline_Liquid']:

        # get all options of current mean of transport
        if m == 'Shipping':

            # get all options of current mean of transport
            options_shipping = data[m]['ports']
            options_shipping['graph'] = None

            options_to_concat.append(options_shipping)

        else:
            networks = data[m].keys()
            for n in networks:
                options_network = data[m][n]['GeoData'].copy()
                options_to_concat.append(options_network)

    options = pd.concat([options] + options_to_concat)

    # create common infrastructure column
    options['infrastructure'] = options.index
    graph_df = options[options['graph'].apply(lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) if isinstance(x, (list, float)) else False)]
    options.loc[graph_df.index, 'infrastructure'] = options.loc[graph_df.index, 'infrastructure']

    return options
