import os
import math

import pandas as pd
import numpy as np

import shapely
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from shapely.affinity import scale
from shapely.ops import unary_union
from vincenty import vincenty
from geopy.distance import geodesic as calc_distance
from tqdm import tqdm


def process_network_data_to_network_objects_no_additional_connection_points(name_network, path_network_data):

    """
    This method connects LineStrings of networks to one common network. It does not add additional connection points.

    :param str name_network: Name of the network ('gas_pipeline', 'oil_pipeline', or 'railroad').
    :param str path_network_data: Path to the directory containing the network data files.

    :return: A tuple containing three pandas DataFrames:
             - line_data_local: DataFrame containing information about the processed lines.
             - graphs_local: DataFrame containing information about the graphs (edges and nodes).
             - geodata_local: DataFrame containing geospatial information about the nodes.
    """

    def process_line(node_number_local):

        """
        Processes a single line and updates node and edge information.

        :param int node_number_local: Current node number.

        :return: Updated node number.
        """

        coords = line.coords

        # get start node of line
        node_start = [round(coords.xy[0][0], 10), round(coords.xy[1][0], 10),
                      node_addition + '_Graph_' + str(graph_number)]
        if node_start not in existing_nodes:
            # if node does not exist, add new node information
            existing_nodes.append(node_start)
            node_start_name = node_addition + '_Node_' + str(node_number_local)
            existing_nodes_dict[node_start_name] = node_start
            node_number_local += 1

        else:
            # if node exists, use existing node information
            node_start_name = list(existing_nodes_dict.keys())[
                list(existing_nodes_dict.values()).index(node_start)]

        # get start node of line
        node_end = [round(coords.xy[0][-1], 10), round(coords.xy[1][-1], 10),
                    node_addition + '_Graph_' + str(graph_number)]
        if node_end not in existing_nodes:
            # if node does not exist, add new node information
            existing_nodes.append(node_end)
            node_end_name = node_addition + '_Node_' + str(node_number_local)
            existing_nodes_dict[node_end_name] = node_end
            node_number_local += 1

        else:
            # if node exists, use existing node information
            node_end_name = list(existing_nodes_dict.keys())[
                list(existing_nodes_dict.values()).index(node_end)]

        # if both are the same node, ignore
        if node_start == node_end:
            return node_number_local

        # else: calculate distance and add new edge information
        coords_before = None
        distance = 0
        for i_x, x in enumerate(coords.xy[0]):
            x = round(x, 10)
            y = round(coords.xy[1][i_x], 10)

            if coords_before is not None:
                distance += vincenty((x, y), (coords_before[0], coords_before[1])) * 1000

            coords_before = (x, y)

        existing_edges_dict[node_addition + '_Edge_' + str(edge_number)] \
            = [node_addition + '_Graph_' + str(graph_number), node_start_name,
               node_end_name, distance, line]
        existing_lines_dict[node_addition + '_Edge_' + str(edge_number)] = line

        return node_number_local

    graph_number = 0
    node_number = 0
    edge_number = 0

    existing_nodes_dict = {}
    existing_edges_dict = {}
    existing_lines_dict = {}
    existing_graphs_dict = {}

    existing_nodes = []

    if name_network == 'gas_pipeline':
        node_addition = 'PG'
    elif name_network == 'oil_pipeline':
        node_addition = 'PL'
    else:  # Railroad
        node_addition = 'RR'

    # iterate through all networks
    for file in os.listdir(path_network_data):

        network_data = pd.read_csv(path_network_data + file, index_col=0, sep=';')
        lines = [shapely.wkt.loads(line) for line in network_data['geometry']]

        # Split Multilinestring / LineString into separate LineStrings if intersections exist
        network = unary_union(lines)
        existing_graphs_dict[graph_number] = network

        if isinstance(network, MultiLineString):
            # network consists of several line strings. Process each individually
            for line in network.geoms:
                node_number = process_line(node_number)
                edge_number += 1

            graph_number += 1

        else:
            # network is just one line
            line = network
            node_number = process_line(node_number)

            edge_number += 1

        graph_number += 1

    # save files
    graphs = pd.DataFrame.from_dict(existing_edges_dict, orient='index', columns=['graph', 'node_start', 'node_end',
                                                                                  'distance', 'line'])
    geodata = pd.DataFrame.from_dict(existing_nodes_dict, orient='index', columns=['longitude', 'latitude', 'graph'])
    line_data = pd.DataFrame.from_dict(existing_lines_dict, orient='index', columns=['geometry'])

    return line_data, graphs, geodata


def process_network_data_to_network_objects_with_additional_connection_points(name_network, path_network_data,
                                                                              minimal_distance_between_node=50000):

    """
    This method connects LineStrings of networks to one common network. It does not add additional connection points.

    :param str name_network: Name of the network ('gas_pipeline', 'oil_pipeline', or 'railroad').
    :param str path_network_data: Path to the directory containing the network data files.
    :param int minimal_distance_between_node: Minimal distance between node

    :return: A tuple containing three pandas DataFrames:
             - line_data: DataFrame containing information about the processed lines.
             - graphs: DataFrame containing information about the graphs (edges and nodes).
             - geodata: DataFrame containing geospatial information about the nodes.
    """

    def process_line(node_number_local, edge_number_local, single_line=False):

        """
        Processes a single line and updates node and edge information.

        :param int node_number_local: Current node number.
        :param int edge_number_local: Current edge number.
        :param bool single_line: indicates if line is part of network

        :return: Updated node number.
        """

        coords = line.coords

        # Calculate distance of line string
        coords_before = None
        total_distance = 0
        for i_x, x in enumerate(coords.xy[0]):
            x = x
            y = coords.xy[1][i_x]

            if coords_before is not None:
                total_distance += calc_distance((y, x), (coords_before[1], coords_before[0])).meters

            coords_before = (x, y)

        # if line does not belong to a network and is very short (10km), it is removed
        if single_line:
            if total_distance < 10000:
                return node_number_local, edge_number_local

        # if distance of line is more than minimal distance, new lines are added
        if total_distance > minimal_distance_between_node:

            # distances between nodes not more than minimal_distance_between_node
            # Create new points based on minimal distance
            n = math.ceil(total_distance / minimal_distance_between_node) - 1 + 2
            distances = np.linspace(0, line.length, n)
            points = [line.interpolate(d) for d in distances]

            # sort points to ensure that they have same order
            line_used_to_sort = LineString(points)
            points = [Point([line_used_to_sort.coords.xy[0][i], line_used_to_sort.coords.xy[1][i]])
                      for i in range(len(line_used_to_sort.coords.xy[0]))]

            # todo: Kann man nicht einfach points nehmen um die Linie zu legen?

            # sort coords to ensure that they have same order
            points_in_line_list = []
            for i_x, x in enumerate(coords.xy[0]):
                y = coords.xy[1][i_x]
                points_in_line_list.append(Point((x, y)))

            line_used_to_sort = LineString(points_in_line_list)
            coords = [Point([line_used_to_sort.coords.xy[0][i], line_used_to_sort.coords.xy[1][i]])
                      for i in range(len(line_used_to_sort.coords.xy[0]))]

            points_in_line_list = []

            coords_before = None
            node_start_name = ''
            j = 0

            # now iterate over all coords
            distance = 0
            for c in coords:

                # process first coordinate as start of line
                if c == coords[0]:
                    # current point c is start of line

                    # add p as node to graph
                    node_start = [round(c.x, 10), round(c.y, 10),
                                  node_addition + '_Graph_' + str(graph_number)]

                    if node_start not in existing_nodes:
                        # node does not exist yet
                        existing_nodes.append(node_start)
                        node_start_name = node_addition + '_Node_' + str(node_number_local)
                        existing_nodes_dict[node_start_name] = node_start
                        node_number_local += 1
                    else:
                        # node already exists
                        node_start_name = list(existing_nodes_dict.keys())[
                            list(existing_nodes_dict.values()).index(node_start)]

                    points_in_line_list.append(c)

                    # remove c from points list. Due to floating point differences, it is done by distance
                    distance = math.inf
                    position_to_remove = 0
                    for pos, p in enumerate(points):
                        if p.distance(c) < distance:
                            distance = p.distance(c)
                            position_to_remove = pos

                    points.remove(points[position_to_remove])

                if coords_before is not None:

                    segment = LineString([Point(coords_before), c])

                    point_in_line = False

                    # remove all already considered points (j increases each time point was processed)
                    points = points[j:]
                    j = 0

                    # iterate through rest of points
                    for i, p in enumerate(points):

                        if segment.distance(p) < 0.0000001:
                            # point is on segment

                            if (p.distance(coords[0]) > 0.0000001) \
                                    & (p.distance(coords[-1]) > 0.0000001):
                                # point is neither start nor end of line

                                # point is in line
                                point_in_line = True

                                # create linestring
                                points_in_line_list.append(p)
                                sub_line = LineString(points_in_line_list)

                                # update distance of line
                                distance += calc_distance((coords_before.y, coords_before.x),
                                                          (p.y, p.x)).meters

                                # add p as node to graph
                                node_end = [round(p.x, 10), round(p.y, 10),
                                            node_addition + '_Graph_' + str(graph_number)]
                                if node_end not in existing_nodes:
                                    # node is new node
                                    existing_nodes.append(node_end)
                                    node_end_name = node_addition + '_Node_' + str(node_number_local)
                                    existing_nodes_dict[node_end_name] = node_end
                                    node_number_local += 1
                                else:
                                    # node is existing node
                                    node_end_name = list(existing_nodes_dict.keys())[
                                        list(existing_nodes_dict.values()).index(node_end)]

                                # add edge
                                existing_edges_dict[node_addition + '_Edge_' + str(edge_number_local)] \
                                    = [node_addition + '_Graph_' + str(graph_number), node_start_name,
                                       node_end_name, distance, sub_line]
                                # add line
                                existing_lines_dict[node_addition + '_Edge_' + str(edge_number_local)] = sub_line
                                edge_number_local += 1

                                # adjust node name as p is now starting point of a new line
                                node_start_name = node_end_name

                                # reset distance
                                if i + 1 < len(points):
                                    # there is a next point
                                    if segment.distance(points[i + 1]) < 0.0000001:  # point is on current segment
                                        distance = 0
                                    else:
                                        # next point is not on current segment

                                        # calculate distance from current point to end point of current segment
                                        # we use the distance from p to c as starting distance
                                        distance = calc_distance((p.y, p.x), (c.y, c.x)).meters
                                else:
                                    # there is no next point
                                    distance = 0

                                # reset points in line
                                points_in_line_list = [p]

                                # p is added as new line. New distance starts from p --> update coords before to p
                                coords_before = p

                                # point has been processed --> don't consider again
                                j += 1

                            elif p.distance(coords[-1]) < 0.0000001:
                                # p is end node

                                # point is in line
                                point_in_line = True

                                # create linestring
                                points_in_line_list.append(p)
                                sub_line = LineString(points_in_line_list)

                                # update distance of line
                                distance += calc_distance((coords_before.y, coords_before.x),
                                                          (p.y, p.x)).meters

                                # add p as node to graph
                                node_end = [round(p.x, 10), round(p.y, 10),
                                            node_addition + '_Graph_' + str(graph_number)]
                                if node_end not in existing_nodes:
                                    # node is new node
                                    existing_nodes.append(node_end)
                                    node_end_name = node_addition + '_Node_' + str(node_number_local)
                                    existing_nodes_dict[node_end_name] = node_end
                                    node_number_local += 1
                                else:
                                    # node is existing node
                                    node_end_name = list(existing_nodes_dict.keys())[
                                        list(existing_nodes_dict.values()).index(node_end)]

                                # add edge
                                existing_edges_dict[node_addition + '_Edge_' + str(edge_number_local)] \
                                    = [node_addition + '_Graph_' + str(graph_number), node_start_name,
                                       node_end_name, distance, sub_line]
                                existing_lines_dict[node_addition + '_Edge_' + str(edge_number_local)] = sub_line
                                edge_number_local += 1

                    # add
                    if not point_in_line:
                        points_in_line_list.append(c)
                        distance += calc_distance((c.y, c.x), (coords_before.y, coords_before.x)).meters

                coords_before = c

        else:
            # distance is lower than minimal distance between nodes.
            # Therefore, add no more intermediate points and only use boundaries

            node_start = [round(coords.xy[0][0], 10), round(coords.xy[1][0], 10),
                          node_addition + '_Graph_' + str(graph_number)]
            if node_start not in existing_nodes:
                # node is new node
                existing_nodes.append(node_start)
                node_start_name = node_addition + '_Node_' + str(node_number_local)
                existing_nodes_dict[node_start_name] = node_start
                node_number_local += 1
            else:
                # node is existing node
                node_start_name = list(existing_nodes_dict.keys())[
                    list(existing_nodes_dict.values()).index(node_start)]

            node_end = [round(coords.xy[0][-1], 10), round(coords.xy[1][-1], 10),
                        node_addition + '_Graph_' + str(graph_number)]

            if node_end not in existing_nodes:
                # node is new node
                existing_nodes.append(node_end)
                node_end_name = node_addition + '_Node_' + str(node_number_local)
                existing_nodes_dict[node_end_name] = node_end
                node_number_local += 1
            else:
                # node is existing node
                node_end_name = list(existing_nodes_dict.keys())[
                    list(existing_nodes_dict.values()).index(node_end)]

            # if start and end is same node, ignore line
            if node_start == node_end:
                return node_number_local, edge_number_local

            # add edge information
            existing_edges_dict[node_addition + '_Edge_' + str(edge_number_local)] \
                = [node_addition + '_Graph_' + str(graph_number), node_start_name, node_end_name, total_distance,
                   line]
            existing_lines_dict[node_addition + '_Edge_' + str(edge_number_local)] = line

            edge_number_local += 1

        return node_number_local, edge_number_local

    graph_number = 0
    node_number = 0
    edge_number = 0

    existing_nodes_dict = {}
    existing_edges_dict = {}
    existing_lines_dict = {}
    existing_graphs_dict = {}

    existing_nodes = []

    if name_network == 'gas_pipeline':
        node_addition = 'PG'
    elif name_network == 'oil_pipeline':
        node_addition = 'PL'
    else:  # Railroad
        node_addition = 'RR'

    for file in tqdm(os.listdir(path_network_data)):

        graph_number = int(file.split('_')[-1].split('.')[0])

        network_data = pd.read_csv(path_network_data + file, index_col=0, sep=';')
        lines = [shapely.wkt.loads(line) for line in network_data['geometry']]

        # Split Multilinestring / Linestring into separate LineStrings if intersections exist
        network = unary_union(lines)

        # It seems like that not all lines are 100% connected even though they are very close
        # To achieve this, we increase each line length by only 1 meter
        if isinstance(network, MultiLineString):
            scaled_lines = []
            for line in network.geoms:
                line_segments = list(map(LineString, zip(line.coords[:-1], line.coords[1:])))
                for segment in line_segments:
                    start = list(segment.boundary.geoms)[0]
                    end = list(segment.boundary.geoms)[1]
                    length = calc_distance((start.y, start.x), (end.y, end.x)).meters
                    if length != 0:
                        scaling_factor = 1 / length + 1
                        new_line = scale(segment, xfact=scaling_factor, yfact=scaling_factor)

                        new_line = LineString([Point(round(new_line.bounds[0], 10), round(new_line.bounds[1], 10)),
                                               Point(round(new_line.bounds[2], 10), round(new_line.bounds[3], 10))])

                        if not new_line.is_valid:
                            continue

                        if new_line not in scaled_lines:
                            scaled_lines.append(new_line)

            network = unary_union(scaled_lines)

        if isinstance(network, MultiLineString):

            for original_line in network.geoms:

                if original_line.is_empty:
                    continue

                # first check if line is a ring. Because then we have to consider that start is end
                if original_line.is_ring:
                    line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                    for line in line_segments:
                        node_number, edge_number = process_line(node_number, edge_number)

                else:
                    line = original_line
                    node_number, edge_number = process_line(node_number, edge_number)

            existing_graphs_dict[graph_number] = network

        else:

            if network.is_empty:
                continue

            node_number_before = node_number
            edge_number_before = edge_number

            # object is linestring --> single line instead of network --> single edge is added and nodes at end of edge
            original_line = network
            if original_line.is_ring:
                line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                for line in line_segments:
                    node_number, edge_number = process_line(node_number, edge_number)

            else:
                line = network
                node_number, edge_number = process_line(node_number, edge_number, single_line=True)

            if (node_number != node_number_before) & (edge_number != edge_number_before):
                # if node and edge numbers didn't change, line was too short --> not considered
                existing_graphs_dict[graph_number] = network

    graphs = pd.DataFrame.from_dict(existing_edges_dict, orient='index', columns=['graph', 'node_start', 'node_end',
                                                                                  'distance', 'line'])
    geodata = pd.DataFrame.from_dict(existing_nodes_dict, orient='index', columns=['longitude', 'latitude', 'graph'])
    line_data = pd.DataFrame.from_dict(existing_lines_dict, orient='index', columns=['geometry'])

    # finally, remove all lines which are not used to connect other lines and are very short (less than 10 km)
    edges_to_drop = []
    nodes_to_drop = []
    for node in geodata.index:
        if graphs['node_start'].values.tolist().count(node) + graphs['node_end'].values.tolist().count(node) <= 1:
            if node in graphs['node_start'].values.tolist():
                index_to_check = graphs[graphs['node_start'] == node].index.values.tolist()[0]
            else:
                index_to_check = graphs[graphs['node_end'] == node].index.values.tolist()[0]

            if graphs.at[index_to_check, 'distance'] < 10000:
                nodes_to_drop.append(node)
                edges_to_drop += [index_to_check]

    graphs.drop(edges_to_drop, inplace=True)
    line_data.drop(edges_to_drop, inplace=True)
    geodata.drop(nodes_to_drop, inplace=True)

    return line_data, graphs, geodata
