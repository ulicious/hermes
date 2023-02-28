import shapely
from shapely.geometry import LineString, MultiLineString
from dijkstar import Graph
import geopandas as gpd


def process_network_data(data, name, geo_data, graph_data):

    data[name] = {}

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

    for g in geo_data['graph'].unique():

        graph = Graph(undirected=True)
        nodes_graph = graph_data[graph_data['graph'] == g].index
        lines = []
        for ind in nodes_graph:

            node_start = graph_data.loc[ind, 'node_start']
            node_end = graph_data.loc[ind, 'node_end']
            distance = graph_data.loc[ind, 'costs']

            graph.add_edge(node_start, node_end, distance)
            lines.append(graph_data.loc[ind, 'line'])

        nodes_graph = geo_data[geo_data['graph'] == g].index
        graph_object = MultiLineString(lines)

        data[name][g] = {'Graph': graph,
                         'GraphData': graph_data,
                         'GraphObject': graph_object,
                         'GeoData': geo_data.loc[nodes_graph]}

    return data
