import itertools
import os
import math

import pandas as pd
import numpy as np
import geopandas as gpd

import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from vincenty import vincenty
from geopy.distance import geodesic as calc_distance
from joblib import Parallel, delayed
from tqdm import tqdm


def extend_line_in_one_direction(direction_coordinate, support_coordinate, extension_percentage):
    """
    extends a linestring in one direction without changing the direction

    @param direction_coordinate: coordinate where extensions takes place
    @param support_coordinate: support direction to create linestring
    @param extension_percentage: percentage by how much line should be extended
    @return: LineString of extended line
    """

    # Create a LineString from the two coordinates
    line = LineString([direction_coordinate, support_coordinate])

    # Calculate the direction vector of the LineString
    direction_vector = np.array([direction_coordinate.x, direction_coordinate.y]) \
                       - np.array([support_coordinate.x, support_coordinate.y])

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm != 0:
        direction_vector /= norm

    # Calculate the extension length based on the keyword (percentage)
    line_length = line.length
    extension_length = line_length * extension_percentage / 100

    # Calculate the new end point
    new_end_point = Point([direction_coordinate.x + direction_vector[0] * extension_length,
                           direction_coordinate.y + direction_vector[1] * extension_length])

    return new_end_point


def extend_line_in_both_directions(coord1, coord2, extension_percentage):
    """
    extends a linestring in both direction without changing the direction

    @param coord1: start coordinate
    @param coord2: end coordinate
    @param extension_percentage: percentage by how much line should be extended
    @return: LineString of extended line
    """

    # Create a LineString from the two coordinates
    line = LineString([coord1, coord2])

    # Calculate the direction vector of the LineString
    direction_vector = np.array([coord1.x, coord1.y]) - np.array([coord2.x, coord2.y])

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm != 0:
        direction_vector /= norm

    # Calculate the extension length based on the keyword (percentage)
    line_length = line.length
    extension_length = line_length * extension_percentage

    # Calculate the new end points in both directions
    new_end_point1 = Point([coord1.x + direction_vector[0] * extension_length,
                            coord1.y + direction_vector[1] * extension_length])
    new_end_point2 = Point([coord2.x - direction_vector[0] * extension_length,
                            coord2.y - direction_vector[1] * extension_length])

    # Create the extended LineString
    extended_linestring = LineString([new_end_point1, new_end_point2])

    return extended_linestring


def process_network_data_to_network_objects_no_additional_connection_points(name_network, path_network_data):

    """
    This method connects LineStrings of networks to one common network. It does not add additional connection points.

    @param str name_network: Name of the network ('gas_pipeline', 'oil_pipeline', or 'railroad').
    @param str path_network_data: Path to the directory containing the network data files.

    @return: A tuple containing three pandas DataFrames:
             - line_data_local: DataFrame containing information about the processed lines.
             - graphs_local: DataFrame containing information about the graphs (edges and nodes).
             - geodata_local: DataFrame containing geospatial information about the nodes.
    """

    def process_line(node_number_local):

        """
        Processes a single line and updates node and edge information.

        @param int node_number_local: Current node number.

        @return: Updated node number.
        """

        coords = line.coords

        # get start node of line
        node_start = [round(coords.xy[0][0], 5), round(coords.xy[1][0], 5),
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
        node_end = [round(coords.xy[0][-1], 5), round(coords.xy[1][-1], 5),
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
            x = round(x, 5)
            y = round(coords.xy[1][i_x], 5)

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


def process_line_super(original_line, minimal_distance_between_node=50000, single_line=False):
    def process_line():

        """
        This method adds the connection points to pipeline lines.

        if line is longer than minimal_distance_between_node:
        1.) it decides on how many connection points should exist --> at least every minimal_distance_between_node km
        2.) It iterates over all coordinatess of the line
            If the coordinate is also a point, node is added
            If the coordinate is not a point, we check if the point lies on the line between two coordinates --> add new node from point coordinate

        @param int node_number_local: Current node number.
        @param int edge_number_local: Current edge number.
        @param bool single_line: indicates if line is part of network

        @return: Updated node number.
        """

        node_number_local = 0
        edge_number_local = 0

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
                return

        # if distance of line is more than minimal distance, new lines are added
        if total_distance > minimal_distance_between_node:

            # distances between nodes more than minimal_distance_between_node
            # Create new points based on minimal distance
            n = math.ceil(total_distance / minimal_distance_between_node) - 1 + 2
            distances = np.linspace(0, line.length, n)
            points = [line.interpolate(d) for d in distances]

            # sort points to ensure that they have same order
            line_used_to_sort = LineString(points)
            points = [Point([line_used_to_sort.coords.xy[0][i], line_used_to_sort.coords.xy[1][i]])
                      for i in range(len(line_used_to_sort.coords.xy[0]))]

            # sort coords to ensure that they have same order
            points_in_line_list = []
            for i_x, x in enumerate(coords.xy[0]):
                y = coords.xy[1][i_x]
                points_in_line_list.append(Point((x, y)))

            line_used_to_sort = LineString(points_in_line_list)
            coords = [Point([line_used_to_sort.coords.xy[0][i], line_used_to_sort.coords.xy[1][i]])
                      for i in range(len(line_used_to_sort.coords.xy[0]))]

            # this list is used to create the linestrings between the points
            # each time an edge is created, we reset the list to start a new linestring
            points_in_line_list = []

            coords_before = None
            node_start = None
            j = 0

            # now iterate over all coords
            distance = 0
            for c in coords:

                # process first coordinate as start of line
                if c == coords[0]:
                    # current point c is start of line

                    # add p as node to graph
                    node_start = [round(c.x, 5), round(c.y, 5)]
                    nodes.append(node_start)
                    node_number_local += 1

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

                    # create line based on two coordinates and check if point is on line
                    # if yes --> create new node & calculate distance of edge
                    # if no --> calculate distance between coordinates and continue with next coordinate

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
                                distance += calc_distance((coords_before.y, coords_before.x), (p.y, p.x)).meters

                                # add p as node to graph
                                node_end = [round(p.x, 5), round(p.y, 5)]
                                nodes.append(node_end)
                                node_number_local += 1

                                # add edge
                                edges.append([node_start, node_end, distance, sub_line])
                                edge_number_local += 1

                                # adjust node name as p is now starting point of a new line
                                node_start = node_end

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

                                # reset points in line list
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
                                node_end = [round(p.x, 5), round(p.y, 5)]
                                nodes.append(node_end)
                                node_number_local += 1

                                # add edge
                                edges.append([node_start, node_end, distance, sub_line])
                                edge_number_local += 1

                                # adjust node name as p is now starting point of a new line
                                node_start = node_end

                    # to get all linestrings correctly, we add coordinate to points_in_line_list
                    if not point_in_line:
                        points_in_line_list.append(c)
                        distance += calc_distance((c.y, c.x), (coords_before.y, coords_before.x)).meters

                coords_before = c

        else:
            # distance is lower than minimal distance between nodes.
            # Therefore, add no more intermediate points and only use boundaries

            node_start = [round(coords.xy[0][0], 5), round(coords.xy[1][0], 5)]
            nodes.append(node_start)
            node_number_local += 1

            node_end = [round(coords.xy[0][-1], 5), round(coords.xy[1][-1], 5)]
            nodes.append(node_end)
            node_number_local += 1

            # if start and end is same node, ignore line
            if node_start == node_end:
                return

            # add edge information
            edges.append([node_start, node_end, total_distance, line])
            edge_number_local += 1

        return

    if original_line.is_empty:
        return None, None

    nodes, edges = [], []

    # first check if line is a ring. Because then we have to consider that start is end
    if original_line.is_ring:
        line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
        for line in line_segments:
            process_line()

    else:
        line = original_line
        process_line()

    return nodes, edges


def process_network_data_to_network_objects_with_additional_connection_points(name_network, path_network_data,
                                                                              minimal_distance_between_node=50000,
                                                                              number_workers=1):

    """
    This method connects LineStrings of networks to one common network.

    @param str name_network: Name of the network ('gas_pipeline', 'oil_pipeline', or 'railroad').
    @param str path_network_data: Path to the directory containing the network data files.
    @param int minimal_distance_between_node: Minimal distance between node
    @param int number_workers: number of works possible

    @return: A tuple containing three pandas DataFrames:
             - line_data: DataFrame containing information about the processed lines.
             - graphs: DataFrame containing information about the graphs (edges and nodes).
             - geodata: DataFrame containing geospatial information about the nodes.
    """

    existing_nodes_dict = {}
    existing_edges_dict = {}
    existing_lines_dict = {}
    existing_graphs_dict = {}

    if name_network == 'gas_pipeline':
        node_addition = 'PG'
    elif name_network == 'oil_pipeline':
        node_addition = 'PL'
    else:  # Railroad
        node_addition = 'RR'

    files = sorted(os.listdir(path_network_data))
    for file in tqdm(files):

        graph_number = int(file.split('_')[-1].split('.')[0])
        node_number = 0
        edge_number = 0
        edge_mapping = {}

        was_processed = False

        network_data = pd.read_csv(path_network_data + file, index_col=0, sep=';')
        lines = [shapely.wkt.loads(line) for line in network_data['geometry']]

        # Split Multilinestring / Linestring into separate LineStrings if intersections exist
        # merge split lines into single linestring if no intersection exists
        network = unary_union(lines)
        network = shapely.line_merge(network)

        if isinstance(network, MultiLineString):

            network_geoms = [l for l in network.geoms]
            line_gdf = gpd.GeoDataFrame(geometry=network_geoms)

            # start and end is same
            line_gdf["geom_start"] = shapely.get_point(line_gdf.geometry, 0)
            line_gdf["geom_end"] = shapely.get_point(line_gdf.geometry, -1)

            # start is end and end is start
            line_gdf_copy = line_gdf.copy()
            line_gdf_copy.rename(columns={'geom_start': 'geom_end',
                                          'geom_end': 'geom_start'}, inplace=True)

            line_gdf = pd.concat([line_gdf, line_gdf_copy], ignore_index=True)

            parallel_lines = line_gdf[line_gdf.duplicated(subset=['geom_start', 'geom_end'], keep=False)]

            def process_parallel_lines(line_index):

                duplicate_lines = []

                duplicates \
                    = parallel_lines[(parallel_lines['geom_start'] == parallel_lines.at[line_index, 'geom_start'])
                                     & (parallel_lines['geom_end'] == parallel_lines.at[line_index, 'geom_end'])].index

                line_1_local = parallel_lines.at[line_index, 'geometry']

                for d in duplicates:
                    if line_index == d:
                        continue

                    line_2_local = parallel_lines.at[d, 'geometry']

                    hausdorff_dist = line_1_local.hausdorff_distance(line_2_local)

                    if hausdorff_dist < 1:
                        duplicate_lines.append([line_index, d])

                if duplicate_lines:
                    return duplicate_lines

            inputs = tqdm(parallel_lines.index)
            results = Parallel(n_jobs=number_workers)(delayed(process_parallel_lines)(i) for i in inputs)

            lines_to_remove = []
            for r in results:
                if r is not None:

                    for i in r:
                        line_1 = i[0]
                        line_2 = i[1]

                        if line_1 not in lines_to_remove:
                            if line_2 not in lines_to_remove:
                                lines_to_remove.append(line_1)

            import random
            import matplotlib.pyplot as plt
            get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]

            # remove parallel lines and duplicates
            line_gdf.drop(lines_to_remove, inplace=True)
            line_gdf.drop_duplicates(subset=['geometry'], inplace=True)

            # merge line segments to whole lines
            lines = MultiLineString(line_gdf['geometry'].tolist())
            lines = shapely.line_merge(lines)

            lines = [l for l in lines.geoms]
            # print(len(lines))

            # df = gpd.GeoDataFrame(geometry=lines)
            # df.plot(color=get_colors(len(lines)))
            # plt.show()

            inputs = tqdm(lines)
            results = Parallel(n_jobs=number_workers)(delayed(process_line_super)(i) for i in inputs)

            for r in results:
                if r[0] is None:
                    continue

                edges = r[1]

                for e in edges:
                    start_node = e[0].copy()
                    start_node.append(node_addition + '_Graph_' + str(graph_number))
                    end_node = e[1].copy()
                    end_node.append(node_addition + '_Graph_' + str(graph_number))
                    distance = e[2]
                    line = e[3]

                    node_start_name = existing_nodes_dict.get(tuple(start_node))
                    if node_start_name is None:
                        node_start_name = node_addition + '_Graph_' + str(graph_number) + '_Node_' + str(node_number)
                        existing_nodes_dict[tuple(start_node)] = node_start_name
                        node_number += 1

                    node_end_name = existing_nodes_dict.get(tuple(end_node))
                    if node_end_name is None:
                        node_end_name = node_addition + '_Graph_' + str(graph_number) + '_Node_' + str(node_number)
                        existing_nodes_dict[tuple(end_node)] = node_end_name
                        node_number += 1

                    edge_key = node_addition + '_Graph_' + str(graph_number) + '_Edge_' + str(edge_number)
                    existing_edges_dict[edge_key] = [node_addition + '_Graph_' + str(graph_number),
                                                     node_start_name,
                                                     node_end_name, distance, line]
                    existing_lines_dict[edge_key] = line
                    edge_number += 1

        else:
            if network.is_empty:
                continue

            # object is linestring --> single line instead of network --> single edge is added and nodes at end of edge
            original_line = network
            if original_line.is_ring:
                line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                for line in line_segments:
                    nodes, edges = process_line_super(line)

                    for e in edges:
                        start_node = e[0].copy()
                        start_node.append(node_addition + '_Graph_' + str(graph_number))
                        end_node = e[1].copy()
                        end_node.append(node_addition + '_Graph_' + str(graph_number))
                        distance = e[2]
                        line = e[3]

                        node_start_name = existing_nodes_dict.get(tuple(start_node))
                        if node_start_name is None:
                            node_start_name = node_addition + '_Graph_' + str(graph_number) + '_Node_' + str(
                                node_number)
                            existing_nodes_dict[tuple(start_node)] = node_start_name
                            node_number += 1

                        node_end_name = existing_nodes_dict.get(tuple(end_node))
                        if node_end_name is None:
                            node_end_name = node_addition + '_Graph_' + str(graph_number) + '_Node_' + str(node_number)
                            existing_nodes_dict[tuple(end_node)] = node_end_name
                            node_number += 1

                        edge_key = node_addition + '_Graph_' + str(graph_number) + '_Edge_' + str(edge_number)
                        existing_edges_dict[edge_key] = [node_addition + '_Graph_' + str(graph_number),
                                                         node_start_name,
                                                         node_end_name, distance, line]
                        existing_lines_dict[edge_key] = line
                        edge_number += 1

                        was_processed = True

            else:
                line = network
                nodes, edges = process_line_super(line, single_line=True)

                for e in edges:
                    start_node = e[0].copy()
                    start_node.append(node_addition + '_Graph_' + str(graph_number))
                    end_node = e[1].copy()
                    end_node.append(node_addition + '_Graph_' + str(graph_number))
                    distance = e[2]
                    line = e[3]

                    node_start_name = existing_nodes_dict.get(tuple(start_node))
                    if node_start_name is None:
                        node_start_name = node_addition + '_Graph_' + str(graph_number) + '_Node_' + str(node_number)
                        existing_nodes_dict[tuple(start_node)] = node_start_name
                        node_number += 1

                    node_end_name = existing_nodes_dict.get(tuple(end_node))
                    if node_end_name is None:
                        node_end_name = node_addition + '_Graph_' + str(graph_number) + '_Node_' + str(node_number)
                        existing_nodes_dict[tuple(end_node)] = node_end_name
                        node_number += 1

                    edge_key = node_addition + '_Graph_' + str(graph_number) + '_Edge_' + str(edge_number)
                    existing_edges_dict[edge_key] = [node_addition + '_Graph_' + str(graph_number),
                                                     node_start_name,
                                                     node_end_name, distance, line]
                    existing_lines_dict[edge_key] = line
                    edge_number += 1

                    was_processed = True

            if was_processed:
                # if node and edge numbers didn't change, line was too short --> not considered
                existing_graphs_dict[graph_number] = network

    graphs = pd.DataFrame.from_dict(existing_edges_dict, orient='index', columns=['graph', 'node_start', 'node_end',
                                                                                  'distance', 'line'])
    existing_nodes_dict = dict((v, k) for k, v in existing_nodes_dict.items())
    nodes = pd.DataFrame.from_dict(existing_nodes_dict, orient='index', columns=['longitude', 'latitude', 'graph'])
    line_data = pd.DataFrame.from_dict(existing_lines_dict, orient='index', columns=['geometry'])

    # finally, remove lines which are not used to connect other lines and are less than 10 km long
    # Such have been added in the scaling process to ensure that lines are properly connected
    edges_to_drop = []
    nodes_to_drop = []
    sub_graph = graphs[graphs['distance'] < 500].copy()
    affected_nodes = set(sub_graph['node_start'].tolist() + sub_graph['node_end'].tolist())
    node_start_counts = graphs['node_start'].value_counts()
    node_end_counts = graphs['node_end'].value_counts()

    for node in affected_nodes:
        # Check if node is used only once --> edge has dead end
        if (node_start_counts.get(node, 0) + node_end_counts.get(node, 0)) == 1:
            if node in node_start_counts:
                index_to_check = graphs.loc[graphs['node_start'] == node].index[0]
            else:
                index_to_check = graphs.loc[graphs['node_end'] == node].index[0]

            # Check length of edge and remove if too low
            if graphs.at[index_to_check, 'distance'] < 500:
                nodes_to_drop.append(node)
                edges_to_drop.append(index_to_check)
                has_changed = True

        # If node is in nodes but not in graphs --> remove from nodes
        elif (node_start_counts.get(node, 0) + node_end_counts.get(node, 0)) == 0:
            nodes_to_drop.append(node)

    graphs.drop(edges_to_drop, inplace=True)
    line_data.drop(edges_to_drop, inplace=True)
    nodes.drop(nodes_to_drop, inplace=True)

    # remove duplicate edges and keep shortest
    graphs.sort_values(by=['distance'], inplace=True)
    # graphs.drop_duplicates(subset=['node_start', 'node_end'], keep='first')

    while True:

        short_edges = graphs[graphs['distance'] < 100]
        print(len(short_edges.index))

        affected_nodes = list(set(short_edges['node_start'].tolist() + short_edges['node_end'].tolist()))
        edges_of_affected_nodes = graphs[(graphs['node_start'].isin(affected_nodes)) | (graphs['node_end'].isin(affected_nodes))]
        edges_of_affected_nodes.sort_values(by=['distance'], ascending=False, inplace=True)

        edges_to_drop = []
        nodes_to_drop = []
        for i in tqdm(edges_of_affected_nodes.index):

            if edges_of_affected_nodes.at[i, 'node_start'] in affected_nodes:
                n = edges_of_affected_nodes.at[i, 'node_start']
            else:
                n = edges_of_affected_nodes.at[i, 'node_end']

            if n in nodes_to_drop:
                continue

            longest_edge = edges_of_affected_nodes[(edges_of_affected_nodes['node_start'] == n) | (edges_of_affected_nodes['node_end'] == n)].index[0]
            shorter_edges = edges_of_affected_nodes[(edges_of_affected_nodes['node_start'] == n) | (edges_of_affected_nodes['node_end'] == n)].index

            for s in shorter_edges:
                if s == longest_edge:
                    continue

                if edges_of_affected_nodes.at[s, 'distance'] >= 100:
                    continue

                # get node of other end of edge (second node)
                if graphs.at[s, 'node_start'] != n:
                    node_to_remove = graphs.at[s, 'node_start']
                else:
                    node_to_remove = graphs.at[s, 'node_end']

                # get all edges which are connected to node_to_remove
                other_edges = short_edges[(short_edges['node_start'] == node_to_remove) | (short_edges['node_end'] == node_to_remove)]

                for o in other_edges.index:
                    if o in edges_to_drop:
                        continue

                    if node_to_remove not in nodes_to_drop:
                        nodes_to_drop.append(node_to_remove)

                    if s not in edges_to_drop:
                        edges_to_drop.append(s)

                    # replace node_to_remove with n
                    if graphs.at[o, 'node_start'] == node_to_remove:
                        graphs.at[o, 'node_start'] = n
                    else:
                        graphs.at[o, 'node_end'] = n

        if len(edges_to_drop) == 0:
            break
        else:
            print(len(edges_to_drop))

        graphs.drop(edges_to_drop, inplace=True)
        line_data.drop(edges_to_drop, inplace=True)
        # nodes.drop(nodes_to_drop, inplace=True)

    all_nodes = list(set(graphs['node_start'].tolist() + graphs['node_end'].tolist()))
    nodes = nodes.loc[all_nodes]

    # todo: anhand der nodes könnte man schauen, welche nodes sehr nah beieinander liegen und dann
    #  diese zusammenfassen

    return line_data, graphs, nodes
