import shapely
from shapely.geometry import LineString, MultiLineString
from dijkstar import Graph
import geopandas as gpd


def process_network_data(geo_data, graph_data):

    graphs_dict = {}

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

    for g in geo_data['graph'].unique():

        graphs_dict[g] = {}

        graph = Graph(undirected=True)
        nodes_graph = graph_data[graph_data['graph'] == g].index
        lines = []
        for ind in nodes_graph:

            node_start = graph_data.loc[ind, 'node_start']
            node_end = graph_data.loc[ind, 'node_end']
            distance = graph_data.loc[ind, 'costs']

            graph.add_edge(node_start, node_end, distance)
            lines.append(graph_data.loc[ind, 'line'])

        graphs_dict[g]['Graph'] = graph
        graphs_dict[g]['GraphData'] = graph_data

        nodes_graph = geo_data[geo_data['graph'] == g].index
        graphs_dict[g]['GeoData'] = geo_data.loc[nodes_graph]

        graph_object = MultiLineString(lines)
        graphs_dict[g]['GraphObject'] = graph_object

    return graphs_dict
