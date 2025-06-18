import itertools
import os
import math
import logging

import networkx as nx
import pandas as pd
import numpy as np
import random
import shapely
import geopandas as gpd
from shapely import geometry, ops
from shapely.geometry import Point, Polygon, MultiLineString, LineString
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

from algorithm.methods_geographic import calc_distance_single_to_single, calc_distance_list_to_list
from data_processing.helpers_misc import create_random_colors

import warnings
warnings.filterwarnings('ignore')


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


def close_gaps(line_combinations, existing_lines, gap_distance, apply_duplicate_removing=False,
               min_distance=0, extend_lines=False):

    """
    checks all combinations of lines if the distance between the two lines is less than 20,000. If so, a new line is
    added to close the gap

    @param list line_combinations: combination of two lines / networks to compare
    @param list existing_lines: list of existing lines for comparison with newly created lines
    @param float gap_distance: If two lines / networks are closer to each other than gap distance, connect both
    @param bool apply_duplicate_removing: if two lines are parallel and close to each other, remove one
    @param float min_distance: only connect lines / networks where distance is higher than min distance
    @param bool extend_lines: sometimes line are not connected due to floating point errors. Increasing them slightly solves problem

    @return: list of old lines including gap closing lines
    """

    new_lines = {}
    lines_to_remove = []
    for lines in tqdm(line_combinations):
        l1 = lines[0]
        l2 = lines[1]

        closest_points = ops.nearest_points(l1, l2)
        distance = calc_distance_single_to_single(closest_points[0].y, closest_points[0].x,
                                                  closest_points[1].y, closest_points[1].x)
        if min_distance <= distance <= gap_distance:
            new_line_points = ops.nearest_points(l1, l2)
            if extend_lines:

                if distance < 1:
                    # sometimes distances are very short. Here it is possible to get issues with floating
                    # point errors. Therefore, very short distances are addressed by building a rectangle
                    # around the connection point and add all 4 edges to the graph
                    center = Point(closest_points[0])

                    p1 = Point([center.x - 0.01, center.y + 0.01])
                    p2 = Point([center.x + 0.01, center.y + 0.01])
                    p3 = Point([center.x + 0.01, center.y - 0.01])
                    p4 = Point([center.x - 0.01, center.y - 0.01])

                    polygon_around_center = Polygon([p1, p2, p3, p4])
                    exterior = polygon_around_center.boundary.coords
                    exterior = [LineString(exterior[k:k + 2]) for k in range(len(exterior) - 1)]

                    # get intersection points with polygon exterior
                    points = []
                    l1_intersects = False
                    l2_intersects = False
                    for e in exterior:
                        if e.intersects(l1):

                            if isinstance(e.intersection(l1), LineString):
                                continue

                            l1_intersects = True

                            if isinstance(e.intersection(l1), Point):
                                points.append(e.intersection(l1))
                            else:
                                for p in e.intersection(l1).geoms:
                                    points.append(p)

                        if e.intersects(l2):

                            if isinstance(e.intersection(l2), LineString):
                                continue

                            l2_intersects = True

                            if isinstance(e.intersection(l2), Point):
                                points.append(e.intersection(l2))
                            else:
                                for p in e.intersection(l2).geoms:
                                    points.append(p)

                    if not l1_intersects:
                        lines_to_remove.append(l1)

                    if not l2_intersects:
                        lines_to_remove.append(l2)

                    added_lines = []
                    for combination in itertools.combinations(points, 2):
                        int_1 = combination[0]
                        int_2 = combination[1]

                        new_line = LineString([int_1, int_2])
                        new_lines[new_line] = 0

                        added_lines.append(new_line)

                    # if added_lines:
                    #
                    #     fig, ax = plt.subplots()
                    #
                    #     old = gpd.GeoDataFrame(geometry=[l1, l2])
                    #     new = gpd.GeoDataFrame(geometry=added_lines)
                    #
                    #     old.plot(ax=ax, color=['blue', 'green'])
                    #     new.plot(ax=ax, color=create_random_colors(len(added_lines)), linestyle='dashed')
                    #
                    #     plt.show()

                    continue
                else:
                    l3 = extend_line_in_both_directions(new_line_points[0], new_line_points[1], 0.1)

            else:
                l3 = geometry.LineString(new_line_points)

            if l3 not in existing_lines:
                new_lines[l3] = distance

                existing_lines.append(l3)

                # if return_combinations:
                #
                #     if l1 not in [*line_combinations_dict.keys()]:
                #         if isinstance(l1, shapely.MultiLineString):
                #             l1_segments = [s for s in l1.geoms]
                #         else:
                #             l1_segments = [l1]
                #
                #         if isinstance(l3, shapely.MultiLineString):
                #             l3_segments = [s for s in l3.geoms]
                #         else:
                #             l3_segments = [l3]
                #
                #         line_combinations_dict[l1] = l1_segments + l3_segments
                #     else:
                #
                #         if isinstance(l3, shapely.MultiLineString):
                #             l3_segments = [s for s in l1.geoms]
                #         else:
                #             l3_segments = [l3]
                #
                #         line_combinations_dict[l1] += l3_segments

    if apply_duplicate_removing:

        # sort new lines by their length. If new lines were created close to each other, remove the shorter ones
        # This approach helps to reduce the number of lines added
        sorted_lines = dict(sorted(new_lines.items(), key=lambda item: item[1]))
        sorted_lines = set(sorted_lines)

        lines_to_remove = set()
        combinations = list(itertools.combinations(sorted_lines, 2))
        for c in tqdm(combinations):
            l1 = c[0]
            l2 = c[1]

            if (l1 in lines_to_remove) | (l2 in lines_to_remove):
                continue

            if l1 == l2:
                lines_to_remove.update([l2])

            closest_points = ops.nearest_points(l1, l2)
            distance = calc_distance_single_to_single(closest_points[0].y, closest_points[0].x,
                                                      closest_points[1].y, closest_points[1].x)

            if distance <= gap_distance:
                lines_to_remove.update([l2])

        new_lines = sorted_lines - lines_to_remove

        print(len(new_lines))
        return new_lines, lines_to_remove

    else:
        new_lines = [*new_lines.keys()]
        return new_lines, lines_to_remove

    # if not return_combinations:
    #     return new_lines
    # else:
    #     return line_combinations_dict


def bruteforce_fill_gaps(network, single_line, gap_distance, min_distance=0):

    residual_geometries = []
    new_single_line = single_line
    if isinstance(single_line, MultiLineString):
        # single line is multilinestring
        # --> connect the closest part of geometry_2 to geometry_1 and add all other lines of geometry_2 to geometry_1
        min_distance_to_network = math.inf
        for line in single_line.geoms:
            closest_distance = ops.nearest_points(line, network)
            distance = calc_distance_single_to_single(closest_distance[0].y, closest_distance[0].x,
                                                      closest_distance[1].y, closest_distance[1].x)

            if distance < min_distance_to_network:
                min_distance_to_network = distance
                new_single_line = line

        for line in single_line.geoms:
            if line != new_single_line:
                residual_geometries.append(line)

    points = [Point(coord) for coord in new_single_line.coords]

    start_point = points[0]
    end_point = points[-1]

    if isinstance(network, LineString):
        network_geoms = [network]
    else:
        network_geoms = network.geoms

    # first, attach network line to start if existing
    min_distance_to_network = math.inf
    closest_distance_points = None
    n_line_points = None
    chosen_line = None
    for n_line in network_geoms:
        closest_distance = ops.nearest_points(start_point, n_line)
        distance = calc_distance_single_to_single(closest_distance[0].y, closest_distance[0].x,
                                                  closest_distance[1].y, closest_distance[1].x)

        if distance < min_distance_to_network:
            min_distance_to_network = distance
            n_line_points = [Point(coord) for coord in n_line.coords]
            closest_distance_points = closest_distance
            chosen_line = n_line

    if min_distance <= min_distance_to_network <= gap_distance:

        # connect both lines
        if closest_distance_points[-1] == n_line_points[0]:
            new_line_points = points + n_line_points
        else:
            new_line_points = n_line_points + points

        new_network_geoms = [LineString(new_line_points)]

    else:
        new_network_geoms = []

    for n_line in network_geoms:
        if n_line != chosen_line:
            new_network_geoms.append(n_line)

    network_geoms = new_network_geoms

    # do the same for the end point
    min_distance_to_network = math.inf
    closest_distance_points = None
    n_line_points = None
    chosen_line = None
    for n_line in network_geoms:
        closest_distance = ops.nearest_points(end_point, n_line)
        distance = calc_distance_single_to_single(closest_distance[0].y, closest_distance[0].x,
                                                  closest_distance[1].y, closest_distance[1].x)

        if distance < min_distance_to_network:
            min_distance_to_network = distance
            n_line_points = [Point(coord) for coord in n_line.coords]
            closest_distance_points = closest_distance
            chosen_line = n_line

    if min_distance <= min_distance_to_network <= gap_distance:

        # connect both lines
        if closest_distance_points[-1] == n_line_points[0]:
            new_line_points = points + n_line_points
        else:
            new_line_points = n_line_points + points

        new_network_geoms = [LineString(new_line_points)]

    else:
        new_network_geoms = []

    for n_line in network_geoms:
        if n_line != chosen_line:
            new_network_geoms.append(n_line)

    # now run over all points and add point to line if close
    new_lines = []
    min_distance_to_network = math.inf
    line_to_points = {}
    for n_line in network_geoms:
        for p in points:

            closest_distance = ops.nearest_points(p, n_line)
            distance = calc_distance_single_to_single(closest_distance[0].y, closest_distance[0].x,
                                                      closest_distance[1].y, closest_distance[1].x)

            if distance < min_distance_to_network:
                min_distance_to_network = distance

                if min_distance <= distance <= gap_distance:
                    line_to_points[n_line] = p

    new_network_geoms = new_lines
    for n_line in network_geoms:

        if n_line in [*line_to_points.keys()]:
            n_line_points = [Point(coord) for coord in n_line.coords]
            p = line_to_points[n_line]

            closest_distance_points = ops.nearest_points(p, n_line)

            # check which side to connect to and create order of coordinates
            if closest_distance_points[-1] == n_line_points[0]:
                new_line_points = [p] + n_line_points
            else:
                new_line_points = n_line_points + [p]

            new_network_geoms.append(LineString(new_line_points))

        else:
            new_network_geoms.append(n_line)

    return MultiLineString(new_network_geoms + residual_geometries)


def find_group_in_data(data):

    """
    Finds connected groups of lines within the given list of geometric line data.

    @param list data: A list containing geometric line data. The first two elements of the list represent
                      the initial groups of lines (group_one and group_two, respectively). Subsequent elements
                      are ignored.
    @return: A tuple containing two lists:
             - group_one: A list representing the first group of connected lines found in the data.
             - group_two: A list representing the residual lines.

    @rtype (list, list)
    """

    # input list with lines
    group_one = data[0]
    group_two = data[1]

    old_group = None
    while group_one != old_group:  # while loop stops as soon as group one has not changed since last loop

        old_group = group_one.copy()

        # each time an element has been added to a group, we restart the for loop as new element might be connected
        # to not connected elements
        broken = False

        # there might be a third line which connects l1 and l2
        l1 = None
        l2 = None
        for l1 in group_one:
            for l2 in group_two:

                if l1 == l2:
                    # if same line --> ignore
                    continue

                if (l1.intersects(l2)) | (l1.equals(l2)):
                    # The different elements do not exist in both groups but at least two intersect
                    broken = True
                    break

            if broken:
                break

        if broken:
            # remove lines from groups as merged line replaces them
            if l1 in group_one:
                group_one.remove(l1)

            if l2 in group_one:
                group_one.remove(l2)

            if l1 in group_two:
                group_two.remove(l1)

            if l2 in group_two:
                group_two.remove(l2)

            if not l1.equals(l2):

                if isinstance(l1, MultiLineString):
                    l1 = [i for i in l1.geoms]
                else:
                    l1 = [l1]

                if isinstance(l2, MultiLineString):
                    l2 = [i for i in l2.geoms]
                else:
                    l2 = [l2]

                data_new = gpd.GeoDataFrame(geometry=l1 + l2)
                data_new = data_new.explode(ignore_index=True)
                lines = data_new['geometry'].tolist()

            else:
                if isinstance(l1, MultiLineString):
                    l1 = [i for i in l1.geoms]
                else:
                    l1 = [l1]

                data_new = gpd.GeoDataFrame(geometry=l1)
                data_new = data_new.explode(ignore_index=True)
                lines = data_new['geometry'].tolist()

            multi_line = geometry.MultiLineString(lines)
            merged_line = ops.linemerge(multi_line)

            # try:
            #     multi_line = geometry.MultiLineString([l1, l2])
            #     merged_line = ops.linemerge(multi_line)
            #
            # except:
            #     # l1 or l2 are several LineStrings and need to be split first
            #     def segments(curve):
            #         return list(map(geometry.LineString, zip(curve.coords[:-1],
            #                                                  curve.coords[1:])))
            #
            #     try:
            #         line_segments_1 = segments(l1)
            #         line_segments_2 = segments(l2)
            #
            #     except:
            #         # checks for undetected MultiLineStrings
            #         data_new = gpd.GeoDataFrame([l1], columns=['geometry'])
            #         data_new.set_geometry('geometry')
            #         data_new_exploded = data_new.explode(ignore_index=True)
            #
            #         line_segments_1 = []
            #         for line in data_new_exploded['geometry'].tolist():
            #             new_line = segments(line)
            #             for nl in new_line:
            #                 line_segments_1.append(nl)
            #
            #         data_new = gpd.GeoDataFrame([l2], columns=['geometry'])
            #         data_new.set_geometry('geometry')
            #         data_new_exploded = data_new.explode(ignore_index=True)
            #
            #         line_segments_2 = []
            #         for line in data_new_exploded['geometry'].tolist():
            #             new_line = segments(line)
            #             for nl in new_line:
            #                 line_segments_2.append(nl)
            #
            #     lines_list = line_segments_1 + line_segments_2
            #
            #     multi_line = geometry.MultiLineString(lines_list)
            #     merged_line = ops.linemerge(multi_line)

            # add merged line to group one
            group_one.append(merged_line)

    return group_one, group_two


def divide_data(lines_local, chunk_size=100):

    """
    shuffles all lines and puts them into equal size lists

    @param list lines_local: list with all lines
    @param int chunk_size: number of lists which will be returned

    @return: nested list with all lists containing the lines
    @rtype: list
    """

    # Adjust chunk size s.t. multiprocessing is optimally used
    if len(lines_local) / chunk_size < 120:
        chunk_size = max(math.ceil(len(lines_local) / 120), 50)

    divided_data = []
    random.shuffle(lines_local)
    for i in range(0, len(lines_local), chunk_size):
        divided_data.append(lines_local[i:i + chunk_size])

    return divided_data


def process_line_strings(lines_local, num_cores, gap_distance, with_adding_lines=False, extend_lines=False):

    if with_adding_lines:

        combinations = list(itertools.combinations(lines_local, 2))
        new_lines, lines_to_remove = close_gaps(combinations, lines_local, gap_distance, extend_lines=extend_lines)

        # remove lines to remove
        considered_lines = []
        for line in lines_local:
            if line not in lines_to_remove:
                considered_lines.append(line)
            else:
                print('line was removed')

        lines_local = considered_lines + new_lines

    # process for fast accumulation of common networks
    runs = 0
    while runs < 10:
        # the while loop is started as long as no changes have occurred within 10 runs
        old_sl = lines_local.copy()

        # if a large amount of geometries exist, we shuffle them so we don't have to combine all with each other
        lines_local = divide_data(lines_local)

        # process data
        inputs = []
        for s_local in lines_local:
            inputs.append([s_local.copy(), s_local.copy()])

        inputs = tqdm(inputs)

        results = Parallel(n_jobs=num_cores)(delayed(find_group_in_data)(i) for i in inputs)

        lines_local = []
        for result in results:
            lines_local = lines_local + result[0]

        if len(lines_local) == len(old_sl):
            # if no changes have occurred, increase run
            runs += 1
        else:
            # if lines have been connected, run is reset
            runs = 0

    return list(lines_local)


def group_LineStrings(name, num_cores, path_to_file, path_processed_data, gap_distance, use_minimal_example=False):

    """
    Reads global energy monitor data, filters pipeline data based on status, removes rows with no geodata information,
    constructs a GeoDataFrame, sorts lines into groups if they intersect, and saves processed data in a folder.

    @param str path_to_file: Path to the file containing network.
    @param str path_processed_data: Path to the directory containing the raw data files.
    @param str name: Name of the file containing the energy monitor data (without file extension).
    @param int num_cores: Number of CPU cores to use for parallel processing (default is 4).
    @param bool use_minimal_example: Indicates if only subset of data should be used
    @param float gap_distance: connects two linestrings automatically if their distance is below gap_distance
    """

    # read global energy monitor data
    data = pd.read_excel(path_to_file)

    # filter pipeline data based on status
    data = data.loc[data['Status'].isin(['Operating', 'Construction'])]

    # remove rows which have no geodata information
    data = data[data['WKTFormat'].notna()]
    empty_rows = data[data['WKTFormat'] == '--'].index.tolist()
    data.drop(empty_rows, inplace=True)

    # drop duplicates
    data.drop_duplicates(subset=['WKTFormat'], inplace=True)

    # construct geodataframe
    data_new = gpd.GeoDataFrame(pd.Series(data['WKTFormat'].tolist()).apply(shapely.wkt.loads), columns=['geometry'])
    data_new.set_geometry('geometry')

    fig, ax = plt.subplots()

    data_old = data_new.copy()

    single_lines = data_new['geometry'].tolist()

    if use_minimal_example:
        # If minimal example is applied, we set a frame on top of Europe and only consider pipelines within this frame

        x_split_point_left = -21
        x_split_point_right = 45

        y_split_point_top = 71
        y_split_point_bottom = 35

        frame_polygon = Polygon([Point(x_split_point_left, y_split_point_top),
                                 Point(x_split_point_right, y_split_point_top),
                                 Point(x_split_point_right, y_split_point_bottom),
                                 Point(x_split_point_left, y_split_point_bottom)])

        new_single_lines = []
        for line in single_lines:

            if line.intersects(frame_polygon):
                new_single_lines.append(line.intersection(frame_polygon))

        single_lines = new_single_lines

    logging.info('Group single lines to networks')
    single_lines = process_line_strings(single_lines, num_cores, gap_distance, with_adding_lines=True)
    single_lines = process_line_strings(single_lines, num_cores, gap_distance, with_adding_lines=True, extend_lines=True)

    logging.info('Check created networks')
    checked_single_lines = set()
    for s in single_lines:  # each s represents network geometry objects
        if isinstance(s, shapely.MultiLineString):

            # Add edges representing connections between LineString endpoints
            s_adjusted = shapely.unary_union(s)
            if isinstance(s_adjusted, shapely.MultiLineString):
                s_geoms = [i for i in s_adjusted.geoms]
            else:
                s_geoms = [s_adjusted]

            graph = nx.Graph()
            nodes = {}
            node_number = 0
            edge_number = 0
            edges = []
            edges_to_linestring = {}
            for line_string in s_geoms:
                start_point = (round(line_string.coords[0][0], 10), round(line_string.coords[0][1], 10))
                if start_point not in [*nodes.keys()]:
                    nodes[start_point] = node_number
                    node_number += 1

                end_point = (round(line_string.coords[-1][0], 10), round(line_string.coords[-1][1], 10))
                if end_point not in [*nodes.keys()]:
                    nodes[end_point] = node_number
                    node_number += 1

                graph.add_edge(nodes[start_point], nodes[end_point], name=edge_number)
                edges.append((nodes[start_point], nodes[end_point]))
                edges_to_linestring[(nodes[start_point], nodes[end_point])] = line_string
                edges_to_linestring[(nodes[end_point], nodes[start_point])] = line_string
                edge_number += 1

            # Check if all nodes are reachable from each other
            if not nx.is_connected(graph):

                # not all elements of network are connected to each other (probably because of floating point error)

                print('not connected')
                # try to connect geometries again, considering only geometries of network
                s = process_line_strings(s_geoms, num_cores, gap_distance)
                s = process_line_strings(s, num_cores, gap_distance, with_adding_lines=True, extend_lines=True)

                if len(s) > 1:
                    # s should be [geometry] (length = 1) and not a nested list with individual geometries

                    print('deep processing')

                    for i in range(5):
                        s = process_line_strings(s, num_cores, gap_distance, with_adding_lines=True,
                                                 extend_lines=True)
                    #
                    #     # fig, ax = plt.subplots()
                    #     # before = gpd.GeoDataFrame(geometry=s_before)
                    #     # before.plot(ax=ax, colors='blue')
                    #     #
                    #     # after = gpd.GeoDataFrame(geometry=s)
                    #     # after.plot(ax=ax, colors=create_random_colors(len(s)))
                    #     # plt.show()

                    if len(s) > 1:
                        print('still disconnected')

                        fig, ax = plt.subplots()
                        network = gpd.GeoDataFrame(geometry=s)
                        network.plot(ax=ax, colors=create_random_colors(len(s)))
                        plt.show()

                        # choose network based on total distance to other geometries
                        # geometry with the smallest distance is network
                        min_total_distance_overall = math.inf
                        network = None
                        for geom_1 in s:
                            min_total_distance_geom_1 = 0
                            for geom_2 in s:
                                if geom_1 == geom_2:
                                    continue

                                min_total_distance_geom_1 += geom_1.distance(geom_2)

                            if min_total_distance_geom_1 < min_total_distance_overall:
                                min_total_distance_overall = min_total_distance_geom_1
                                network = geom_1

                            elif min_total_distance_geom_1 == min_total_distance_overall:
                                # in the case where several geometries have same distances,
                                # choose the one which is multilinestring
                                if isinstance(geom_1, MultiLineString):
                                    network = geom_1

                        lines = []
                        for geom in s:
                            if geom != network:
                                lines.append(geom)

                        for line in lines:
                            not_connected_line = line

                            old_network = network

                            network = bruteforce_fill_gaps(network, line, gap_distance)

                            # fig, ax = plt.subplots()
                            #
                            # after = gpd.GeoDataFrame(geometry=[old_network])
                            # after.plot(ax=ax, colors='black')
                            #
                            # before = gpd.GeoDataFrame(geometry=[not_connected_line])
                            # before.plot(ax=ax, colors='yellow', linestyle='dashed')
                            #
                            # plt.show()

                        fig, ax = plt.subplots()
                        network_gdf = gpd.GeoDataFrame(geometry=[network])
                        network_gdf.plot(ax=ax, colors=create_random_colors(len([network])))
                        plt.show()

                        if isinstance(network, MultiLineString):
                            network_geoms = []
                            for n_geom in network.geoms:
                                network_geoms.append(n_geom)

                            network = network_geoms
                        else:
                            network = [network]

                        s = process_line_strings(network, num_cores, gap_distance)
                        s = process_line_strings(s, num_cores, gap_distance, with_adding_lines=True, extend_lines=True)

                        if len(s) > 1:
                            print('still not connected --> add parts individually')
                            for s_element in s:
                                checked_single_lines.update([s_element])
                        else:
                            checked_single_lines.update(s)

                    else:
                        checked_single_lines.update(s)

                else:
                    checked_single_lines.update(s)

            else:
                checked_single_lines.update([s])

        else:
            checked_single_lines.update([s])

    fig, ax = plt.subplots()

    data_new = gpd.GeoDataFrame(checked_single_lines, columns=['geometry'])
    data_new.set_geometry('geometry')

    data_new.plot(ax=ax, color='red')
    data_old.plot(ax=ax)

    plt.show()

    fig, ax = plt.subplots()

    number_of_colors = len(data_new.index)
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    data_new.plot(ax=ax, color=color)
    plt.show()

    # save processed data in folder (each network own folder)
    name_folder = path_processed_data + name + '_network_data/'
    if name + '_network_data' not in os.listdir(path_processed_data):
        os.mkdir(name_folder)

    networks = []
    network_number = 0
    for line_1 in checked_single_lines:

        if not line_1:
            continue

        new_gpd = gpd.GeoDataFrame()

        if isinstance(line_1, MultiLineString):
            i = 0
            for line_2 in line_1.geoms:
                new_gpd.loc[i, 'geometry'] = line_2
                i += 1
        else:
            new_gpd.loc[0, 'geometry'] = line_1

        # add geometries to geopandas dataframe and save them
        new_gpd.set_geometry('geometry')
        networks.append(new_gpd)

        new_gpd.to_csv(name_folder + 'sorted_' + name + '_networks_' + str(network_number) + '.csv', sep=';')
        network_number += 1

