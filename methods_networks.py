from shapely.geometry import Point, LineString, MultiLineString
import shapely
from dijkstar import find_path
from copy import deepcopy
import geopandas as gpd
import math

from _helpers import calc_distance


def find_shortest_path_in_existing_graph(geodata, graph, graph_data, n_start, n_target,
                                         real_starting_point=None, existing_distance=0):

    # todo: vereinheitlichen der Toleranz-Distanzen: Wo und wie werden diese behandelt?

    """ Method uses dijkstra to find shortest path through graph """

    path = find_path(graph, n_start, n_target)
    points_list = []

    if real_starting_point is not None:
        points_list.append(real_starting_point)

    n_before = None
    for n in path.nodes:

        if n_before is not None:
            edge_index = graph_data[(graph_data['node_start'] == n_before) & (graph_data['node_end'] == n)].index

            if len(edge_index) == 0:
                edge_index = graph_data[(graph_data['node_start'] == n) & (graph_data['node_end'] == n_before)].index

            edge = graph_data.loc[edge_index, 'line'].values[0]
            if not isinstance(edge, LineString):
                edge = shapely.wkt.loads(edge)

            for i_x, x in enumerate(edge.coords.xy[0]):
                x = round(x, 5)
                y = round(edge.coords.xy[1][i_x], 5)

                points_list.append((x, y))

        n_before = n

    line = LineString(points_list)
    distance = path.total_cost + existing_distance
    target = Point(geodata.loc[n_target, 'longitude'], geodata.loc[n_target, 'latitude'])

    return [target, distance, line]


def attach_new_node_to_graph(geodata, graph, graph_data, graph_object, graph_name, new_node_in_network):

    """
    If new station of network is used, network (and data) is updated to include new station

    :param solution: Current solution
    :param target_system: E.g. gas pipeline, liquid pipeline, railroad
    :param graph_name
    :param new_node_in_network: Point which includes gps coordinates of new station to include
    :return: Solution with adjusted graph
    """

    if False:
        if target_system == 'Pipeline_Gas':
            graph_dict = solution.get_pipeline_gas_networks()[graph_name]
        elif target_system == 'Pipeline_Liquid':
            graph_dict = solution.get_pipeline_liquid_networks()[graph_name]
        else:
            graph_dict = solution.get_pipeline_railroad_networks()[graph_name]

        geodata = graph_dict['GeoData'].copy()
        graph = deepcopy(graph_dict['Graph'])
        graph_data = graph_dict['GraphData'].copy()
        graph_object = deepcopy(graph_dict['GraphObject'])

    # Find position of new node and add information to geodata
    new_node_in_network_index = max(geodata.index) + 1
    geodata.loc[new_node_in_network_index, 'latitude'] = round(new_node_in_network.y, 5)
    geodata.loc[new_node_in_network_index, 'longitude'] = round(new_node_in_network.x, 5)
    geodata.loc[new_node_in_network_index, 'graph'] = graph_name

    # Identify LineString where new node will be placed
    left_df = gpd.GeoDataFrame(geometry=[new_node_in_network])
    right_df = gpd.GeoDataFrame(geometry=[graph_object]).explode(ignore_index=True)
    df_n = gpd.sjoin_nearest(left_df, right_df).merge(right_df, left_on="index_right",
                                                      right_index=True)

    affected_line = df_n['geometry_y'].values[0]  # this LineString will be divided
    starting_node_x = round(affected_line.coords.xy[0][0], 5)
    starting_node_y = round(affected_line.coords.xy[1][0], 5)

    ending_node_x = round(affected_line.coords.xy[0][-1], 5)
    ending_node_y = round(affected_line.coords.xy[1][-1], 5)

    starting_node_index = geodata[(geodata['longitude'] == starting_node_x) &
                                  (geodata['latitude'] == starting_node_y)].index[0]
    ending_node_index = geodata[(geodata['longitude'] == ending_node_x) &
                                (geodata['latitude'] == ending_node_y)].index[0]

    # Remove old edges as a node between is inserted & adjust graph data
    graph.remove_edge(starting_node_index, ending_node_index)

    edge_1 = graph_data[(graph_data['node_start'] == starting_node_index) &
                        (graph_data['node_end'] == ending_node_index)].index[0]
    graph_data = graph_data.drop([edge_1])

    # Separate LineString into first and last part
    nodes_in_affected_line = []
    first_part_line = []
    last_part_line = []
    distance_to_line = math.inf
    for c in affected_line.coords:
        nodes_in_affected_line.append(c)

        if calc_distance(round(c[1], 5), round(c[0], 5), round(new_node_in_network.y, 5), round(new_node_in_network.x, 5)) < distance_to_line:
            distance_to_line = calc_distance(round(c[1], 5), round(c[0], 5), round(new_node_in_network.y, 5), round(new_node_in_network.x, 5))
            first_part_line.append((round(c[0], 5), round(c[1], 5)))
        else:
            last_part_line.append((round(c[0], 5), round(c[1], 5)))

    first_part_line.append((round(new_node_in_network.x, 5), round(new_node_in_network.y, 5)))
    last_part_line = [(round(new_node_in_network.x, 5), round(new_node_in_network.y, 5))] + last_part_line

    # Calculate distances of both parts
    c_before = None
    distance_first_line = 0
    for c in first_part_line:
        if c_before is not None:
            distance_first_line += calc_distance(c[1], c[0], c_before[1], c_before[0])
        c_before = c

    c_before = None
    distance_last_line = 0
    for c in last_part_line:
        if c_before is not None:
            distance_last_line += calc_distance(c[1], c[0], c_before[1], c_before[0])
        c_before = c

    # Add first part as new edge to graph & update graph data
    graph.add_edge(starting_node_index, new_node_in_network_index, distance_first_line)

    first_part_linestring = LineString(first_part_line)

    if len(graph_data.index) == 0:
        max_edge = -1
    else:
        max_edge = max(graph_data.index)

    new_edge = max_edge + 1
    graph_data.loc[new_edge, 'graph'] = graph_name
    graph_data.loc[new_edge, 'node_start'] = starting_node_index
    graph_data.loc[new_edge, 'node_end'] = new_node_in_network_index
    graph_data.loc[new_edge, 'costs'] = distance_first_line
    graph_data.loc[new_edge, 'line'] = first_part_linestring

    # Add last part as new edge to graph & update graph data
    graph.add_edge(ending_node_index, new_node_in_network_index, distance_last_line)

    last_part_linestring = LineString(last_part_line)

    if len(graph_data.index) == 0:
        max_edge = -1
    else:
        max_edge = max(graph_data.index)

    new_edge = max_edge + 1
    graph_data.loc[new_edge, 'graph'] = graph_name
    graph_data.loc[new_edge, 'node_start'] = new_node_in_network_index
    graph_data.loc[new_edge, 'node_end'] = ending_node_index
    graph_data.loc[new_edge, 'costs'] = distance_last_line
    graph_data.loc[new_edge, 'line'] = last_part_linestring

    # Update graph object todo: drop the update as it might take very long

    if False:
        line_list = [first_part_line, last_part_line]
        try:
            for line in graph.geoms:
                if line != affected_line:
                    line_list.append(line)

        except Exception:  # is LineString
            c_before = None
            for c in last_part_line:
                if c_before is not None:
                    if LineString([c, c_before]) != affected_line:
                        line_list.append(LineString([c, c_before]))
                c_before = c

        graph_object = MultiLineString(line_list)

    if False:

        # Update graph dict and solution with new data & graph
        graph_dict['GeoData'] = geodata
        graph_dict['Graph'] = graph
        graph_dict['GraphData'] = graph_data
        graph_dict['GraphObject'] = MultiLineString(line_list)

        # Overwrite solution with new graph dictionary
        if target_system == 'Pipeline_Gas':
            solution.get_pipeline_gas_networks()[graph_name] = graph_dict
        elif target_system == 'Pipeline_Liquid':
            solution.get_pipeline_liquid_networks()[graph_name] = graph_dict
        else:
            solution.get_pipeline_railroad_networks()[graph_name] = graph_dict

        return solution

    return geodata, graph, graph_data, graph_object
