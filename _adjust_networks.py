import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from shapely import wkt
from shapely.affinity import scale
import shapely
import os

from vincenty import vincenty
from shapely.ops import unary_union
import searoute as sr
import numpy as np

from geopy.distance import geodesic as calc_distance

from _helpers import calc_distance_list_to_single, calc_distance_single_to_single, calc_distance_list_to_list

import geojson

import itertools

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

import requests
import json  # call the OSMR API
import time

from shapely.ops import nearest_points

import reverse_geocode

from geopy.geocoders import Nominatim

import geopandas as gpd

import networkx as nx

import matplotlib.pyplot as plt

import math


def get_geodata_and_graph_from_network_data_only_boundaries(path_network_data, name_network):

    def process_line(node_number_local):
        distance = 0

        coords = line.coords

        node_start = [round(coords.xy[0][0], 10), round(coords.xy[1][0], 10),
                      node_addition + '_Graph_' + str(graph_number)]
        if node_start not in existing_nodes:
            existing_nodes.append(node_start)
            node_start_name = node_addition + '_Node_' + str(node_number_local)
            existing_nodes_dict[node_start_name] = node_start
            node_number_local += 1

        else:
            node_start_name = list(existing_nodes_dict.keys())[
                list(existing_nodes_dict.values()).index(node_start)]

        node_end = [round(coords.xy[0][-1], 10), round(coords.xy[1][-1], 10),
                    node_addition + '_Graph_' + str(graph_number)]
        if node_end not in existing_nodes:
            existing_nodes.append(node_end)
            node_end_name = node_addition + '_Node_' + str(node_number_local)
            existing_nodes_dict[node_end_name] = node_end
            node_number_local += 1

        else:
            node_end_name = list(existing_nodes_dict.keys())[
                list(existing_nodes_dict.values()).index(node_end)]

        if node_start == node_end:
            return node_number_local

        coords_before = None
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

    graphs_local = pd.DataFrame.from_dict(existing_edges_dict, orient='index', columns=['graph',
                                                                                        'node_start', 'node_end',
                                                                                        'distance', 'line'])
    geodata_local = pd.DataFrame.from_dict(existing_nodes_dict, orient='index', columns=['longitude',
                                                                                         'latitude',
                                                                                         'graph'])
    line_data_local = pd.DataFrame.from_dict(existing_lines_dict, orient='index', columns=['geometry'])

    return line_data_local, graphs_local, geodata_local


def get_geodata_and_graph_from_network_data_with_intermediate_points(path_network_data, name_network,
                                                                     minimal_distance_between_node=50000):

    def process_line(node_number_local, edge_number_local, single_line=False):

        total_distance = 0
        coords = line.coords

        # Calculate distance of line string
        coords_before = None
        for i_x, x in enumerate(coords.xy[0]):
            x = x
            y = coords.xy[1][i_x]

            if coords_before is not None:
                total_distance += calc_distance((y, x), (coords_before[1], coords_before[0])).meters

            coords_before = (x, y)

        total_distance_from_single = 0  # todo: remove as only needed for checking

        if single_line:
            if total_distance < 10000:
                return node_number_local, edge_number_local

        if total_distance > minimal_distance_between_node:
            distance = 0

            # distances between nodes not more than 50 km
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

            points_in_line_list = []

            coords_before = None
            j = 0
            node_start_name = ''

            for c in coords:

                if c == coords[0]:
                    # current point c is start of line

                    # add p as node to graph
                    node_start = [round(c.x, 10), round(c.y, 10),
                                  node_addition + '_Graph_' + str(graph_number)]

                    if node_start not in existing_nodes:

                        existing_nodes.append(node_start)
                        node_start_name = node_addition + '_Node_' + str(node_number_local)
                        existing_nodes_dict[node_start_name] = node_start
                        node_number_local += 1
                    else:
                        node_start_name = list(existing_nodes_dict.keys())[
                            list(existing_nodes_dict.values()).index(node_start)]

                    points_in_line_list.append(c)

                    # remove point of points which is the closest to c. Due to different numbers of
                    # decimal places, this needs to be done via distances
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
                            if (p.distance(coords[0]) > 0.0000001) \
                                    & (p.distance(coords[-1]) > 0.0000001):
                                # point is neither start or end of line

                                # point is in line
                                point_in_line = True

                                # create linestring
                                points_in_line_list.append(p)
                                sub_line = LineString(points_in_line_list)

                                # update distance of line
                                distance += calc_distance((coords_before.y, coords_before.x),
                                                          (p.y, p.x)).meters
                                total_distance_from_single += distance

                                # add p as node to graph
                                node_end = [round(p.x, 10), round(p.y, 10),
                                            node_addition + '_Graph_' + str(graph_number)]
                                if node_end not in existing_nodes:

                                    existing_nodes.append(node_end)
                                    node_end_name = node_addition + '_Node_' + str(node_number_local)
                                    existing_nodes_dict[node_end_name] = node_end
                                    node_number_local += 1
                                else:
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
                                    if segment.distance(
                                            points[i + 1]) < 0.0000001:  # point is on current segment
                                        if points[i + 1].distance(coords[0]) < 0.0000001:
                                            # next point is end point

                                            # This point is calculated when end point is processed
                                            distance = 0
                                        else:
                                            # This point is calculated when next point is processed
                                            distance = 0
                                    else:
                                        # next point is not on current segment

                                        # calculate distance from current point to end point of current segment
                                        distance = calc_distance((p.y, p.x), (c.y, c.x)).meters
                                else:
                                    # there is no next point
                                    distance = 0

                                # reset points in line
                                points_in_line_list = [p]

                                # update coords before to resemble point
                                coords_before = p
                                # segment = LineString([Point(coords_before), Point([x, y])])

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
                                total_distance_from_single += distance

                                # add p as node to graph
                                node_end = [round(p.x, 10), round(p.y, 10),
                                            node_addition + '_Graph_' + str(graph_number)]
                                if node_end not in existing_nodes:

                                    existing_nodes.append(node_end)
                                    node_end_name = node_addition + '_Node_' + str(node_number_local)
                                    existing_nodes_dict[node_end_name] = node_end
                                    node_number_local += 1
                                else:
                                    node_end_name = list(existing_nodes_dict.keys())[
                                        list(existing_nodes_dict.values()).index(node_end)]

                                # add edge
                                existing_edges_dict[node_addition + '_Edge_' + str(edge_number_local)] \
                                    = [node_addition + '_Graph_' + str(graph_number), node_start_name,
                                       node_end_name, distance, sub_line]
                                existing_lines_dict[node_addition + '_Edge_' + str(edge_number_local)] = sub_line
                                edge_number_local += 1

                    if not point_in_line:
                        points_in_line_list.append(c)
                        distance += calc_distance((c.y, c.x), (coords_before.y, coords_before.x)).meters

                coords_before = c

        else:  # distance is lower than 50 km. Therefore, add no more intermediate points and only use boundaries

            # todo: it seems that distances are so short that different nodes are seen as the same
            #  edges are not added as same start and end
            node_start = [round(coords.xy[0][0], 10), round(coords.xy[1][0], 10),
                          node_addition + '_Graph_' + str(graph_number)]
            if node_start not in existing_nodes:

                existing_nodes.append(node_start)
                node_start_name = node_addition + '_Node_' + str(node_number_local)
                existing_nodes_dict[node_start_name] = node_start
                node_number_local += 1
            else:
                node_start_name = list(existing_nodes_dict.keys())[
                    list(existing_nodes_dict.values()).index(node_start)]

            node_end = [round(coords.xy[0][-1], 10), round(coords.xy[1][-1], 10),
                        node_addition + '_Graph_' + str(graph_number)]

            if node_end not in existing_nodes:

                existing_nodes.append(node_end)
                node_end_name = node_addition + '_Node_' + str(node_number_local)
                existing_nodes_dict[node_end_name] = node_end
                node_number_local += 1
            else:
                node_end_name = list(existing_nodes_dict.keys())[
                    list(existing_nodes_dict.values()).index(node_end)]

            if node_start == node_end:
                return node_number_local, edge_number_local

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

        # Split Multilinestring / Linestring into separate Linestrings if intersections exist
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
                        scaled_lines.append(scale(segment, xfact=scaling_factor, yfact=scaling_factor))
            network = unary_union(scaled_lines)

        if False: #graph_number == 126: # todo: 126 still not working
            if isinstance(network, MultiLineString):
                fig, ax = plt.subplots()
                line_gdf = gpd.GeoDataFrame(geometry=list(network.geoms))
                line_gdf.plot(ax=ax)

                plt.show()

        plot_lines = []
        if isinstance(network, MultiLineString):

            for original_line in network.geoms:

                if original_line.is_empty:
                    continue

                if True:

                    # first check if line is a ring. Because then we have to consider that start is end
                    if original_line.is_ring:
                        line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                        for line in line_segments:
                            node_number, edge_number = process_line(node_number, edge_number)

                    else:
                        line = original_line
                        node_number, edge_number = process_line(node_number, edge_number)

                else:

                    line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                    for line in line_segments:

                        #  line = scale(line, xfact=1.01, yfact=1.01)

                        #  plot_lines.append(scale(line, xfact=1.01, yfact=1.01))

                        node_number, edge_number = process_line(node_number, edge_number)

            if False: #graph_number == 129:
                fig, ax = plt.subplots()
                line_gdf = gpd.GeoDataFrame(geometry=list(network.geoms))
                line_gdf.plot(ax=ax)

                line_ex_gdf = gpd.GeoDataFrame(geometry=plot_lines)
                line_ex_gdf.plot(ax=ax, alpha=0.5, color='r')
                # plt.show()

            existing_graphs_dict[graph_number] = network

        else:

            if network.is_empty:
                continue

            node_number_before = node_number
            edge_number_before = edge_number

            # object is linestring --> single line instead of network --> single edge is added and nodes at end of edge
            original_line = network
            if True:
                if original_line.is_ring:
                    line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                    for line in line_segments:
                        node_number, edge_number = process_line(node_number, edge_number)

                else:
                    line = network
                    node_number, edge_number = process_line(node_number, edge_number, single_line=True)
            else:
                line_segments = list(map(LineString, zip(original_line.coords[:-1], original_line.coords[1:])))
                for line in line_segments:
                    node_number, edge_number = process_line(node_number, edge_number)

            if (node_number != node_number_before) & (edge_number != edge_number_before):
                # if node and edge numbers didn't change, line was too short --> not considered
                existing_graphs_dict[graph_number] = network

        if False: #graph_number == 126: # todo: 126 still not working
            fig, ax = plt.subplots()
            line_gdf = gpd.GeoDataFrame(geometry=lines_to_plot)
            line_gdf.plot(ax=ax)

            point_gdf = gpd.GeoDataFrame(geometry=points_to_plot)
            point_gdf.plot(ax=ax, color='r')

            for x, y in zip(point_gdf.geometry.x, point_gdf.geometry.y):
                ax.annotate(str(x) + ' ' + str(y), xy=(x, y), xytext=(3, 3), textcoords="offset points")

            plt.show()

    graphs_local = pd.DataFrame.from_dict(existing_edges_dict, orient='index', columns=['graph',
                                                                                        'node_start', 'node_end',
                                                                                        'distance', 'line'])
    geodata_local = pd.DataFrame.from_dict(existing_nodes_dict, orient='index', columns=['longitude',
                                                                                         'latitude',
                                                                                         'graph'])
    line_data_local = pd.DataFrame.from_dict(existing_lines_dict, orient='index', columns=['geometry'])

    # finally, remove all lines which are not used to connect other lines and are very short
    # todo: mehrmals machen weil ja jedes mal eine rausfallen könnte?
    edges_to_drop = []
    nodes_to_drop = []
    for node in geodata_local.index:
        if graphs_local['node_start'].values.tolist().count(node) + graphs_local['node_end'].values.tolist().count(node) <= 1:
            if node in graphs_local['node_start'].values.tolist():
                index_to_check = graphs_local[graphs_local['node_start'] == node].index.values.tolist()[0]
            else:
                index_to_check = graphs_local[graphs_local['node_end'] == node].index.values.tolist()[0]

            if graphs_local.at[index_to_check, 'distance'] < 10000:
                nodes_to_drop.append(node)
                edges_to_drop += [index_to_check]

    graphs_local.drop(edges_to_drop, inplace=True)
    line_data_local.drop(edges_to_drop, inplace=True)
    geodata_local.drop(nodes_to_drop, inplace=True)

    return line_data_local, graphs_local, geodata_local
