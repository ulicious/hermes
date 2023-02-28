import math

from shapely.geometry import LineString, Point, MultiLineString
import osmnx as ox
from pyproj import Geod
from shapely.ops import nearest_points
import pandas as pd

from _helpers import calc_distance, calc_distance_lists
from copy import deepcopy

import requests
import json  # call the OSMR API
import searoute as sr

from shapely.ops import unary_union
import shapely

import geopandas as gpd


def find_benchmark_solution(s, ports):

    # todo: check if this approach should include other paths (direct roads etc.)

    s_new = deepcopy(s)
    s_location = s_new.get_current_location()
    s_destination = s_new.get_destination()
    s_commodity = s_new.get_current_commodity_object()

    s_new.set_name('benchmark_solution')
    s_new.add_previous_solution(s)

    # Get closest port to start and destination
    considered_port_start = None
    lowest_distance = 100000000
    for p in ports.index:

        distance = calc_distance(s_location.y, s_location.x,
                                 ports.loc[p, 'latitude'], ports.loc[p, 'longitude'])

        if distance < lowest_distance:
            lowest_distance = distance
            considered_port_start = p

    considered_port_end = None
    lowest_distance = 100000000
    for p in ports.index:

        distance = calc_distance(s_destination.y, s_destination.x,
                                 ports.loc[p, 'latitude'], ports.loc[p, 'longitude'])

        if distance < lowest_distance:
            lowest_distance = distance
            considered_port_end = p

    # Second, calculate road transportation to start port
    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{s_location.x},{s_location.y};"
        f"{ports.loc[considered_port_start, 'longitude']},{ports.loc[considered_port_start, 'latitude']}?steps=true&geometries=geojson""")
    routes = json.loads(r.content)

    if routes['code'] != 'NoRoute':
        route = routes.get("routes")[0]
        point_list = []
        for point in route['geometry']['coordinates']:
            point_list.append(Point(point))

        line = LineString(point_list)
        distance = route['distance']

        s_new.add_used_transport_mean('Road')
        s_new.add_result_line(line)

        route_costs = (s_commodity.get_transportation_costs_specific_mean_of_transport('Road') / 1000) * distance
        s_new.increase_total_costs(route_costs)

    # Now add shipping from start port to destination port
    route = sr.searoute((ports.loc[considered_port_start, 'longitude'], ports.loc[considered_port_start, 'latitude']),
                        (ports.loc[considered_port_end, 'longitude'], ports.loc[considered_port_end, 'latitude']))
    coordinates = []

    for coordinate in route.geometry['coordinates']:
        coordinates.append((coordinate[0], coordinate[1]))

    line = LineString(coordinates)
    distance = (round(float(format(route.properties['length'])), 2)) * 1000  # m

    s_new.add_used_transport_mean('Shipping')
    s_new.add_result_line(line)

    route_costs = (s_commodity.get_transportation_costs_specific_mean_of_transport('Shipping') / 1000) * distance
    s_new.increase_total_costs(route_costs)

    # Finally, add road transport to destination if necessary
    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{s_destination.x},{s_destination.y};"
        f"{ports.loc[considered_port_end, 'longitude']},{ports.loc[considered_port_end, 'latitude']}?steps=true&geometries=geojson""")
    routes = json.loads(r.content)

    if routes['code'] != 'NoRoute':
        route = routes.get("routes")[0]
        point_list = []
        for point in route['geometry']['coordinates']:
            point_list.append(Point(point))

        line = LineString(point_list)
        distance = route['distance']

        s_new.set_name('benchmark_solution')

        s_new.set_current_location(s_destination)
        s_new.add_used_transport_mean('Road')
        s_new.add_result_line(line)

        route_costs = (s_commodity.get_transportation_costs_specific_mean_of_transport('Road') / 1000) * distance
        s_new.increase_total_costs(route_costs)

    return s_new


def find_locations_within_tolerance(data, solution, tolerance_distance):

    """
    Method checks if infrastructure like ports, pipelines or railroads are reachable within tolerance
    :param solution: Current solution
    :param tolerance_distance: Distance within the infrastructure has to be available
    :return: All ports, feed-in and railroad stations within the tolerance
    """

    def find_possible_options_based_on_location(mean_of_transport):

        # todo: don't consider options which are not reachable --> not same continent

        if mean_of_transport == 'Shipping':  # Finds ports

            options = data['Shipping'].copy()

            # Don't use already used infrastructure again
            used_ports = solution.get_used_ports()

            # Sort ports by distance to location
            options['distance'] = calc_distance_lists(options['latitude'], options['longitude'], location.y, location.x)
            options.sort_values(['distance'], inplace=True)

            possible_options = []
            for option in options.index:

                if used_ports:
                    if option in used_ports:
                        continue

                option_location_lon = options.loc[option, 'longitude']
                option_location_lat = options.loc[option, 'latitude']

                distance = calc_distance(location.y, location.x, option_location_lat, option_location_lon)
                if distance < tolerance_distance:
                    possible_options.append(option)
                else:
                    # If distance is higher than tolerance, this will be the case for all other ports as they are sorted
                    break

            if possible_options:
                return options.loc[possible_options].copy()
            else:
                return None

        else:  # Case networks (pipelines and railway)

            options = data[mean_of_transport].copy()

            # Don't use already used infrastructure again
            if mean_of_transport == 'Railroad':
                used_stations = solution.get_used_railroad_networks()
            elif mean_of_transport == 'Pipeline_Liquid':
                used_stations = solution.get_used_railroad_networks()
            else:
                used_stations = solution.get_used_railroad_networks()

            # Finds feed-in and railroad stations
            overall_geodata_df = pd.DataFrame(columns=['latitude', 'longitude', 'graph'])
            for g in [*options.keys()]:

                if used_stations:
                    if g in used_stations:
                        continue

                geo_data = options[g]['GeoData']
                graph_object = options[g]['GraphObject']
                possible_options = []

                # First check if shortest path to graph is already too far away.
                # All stations of graph will be to too far away as well
                closest_node = nearest_points(graph_object, location)[0]
                distance_to_closest = calc_distance(location.y, location.x, closest_node.y, closest_node.x)
                if distance_to_closest < tolerance_distance:

                    # Sort ports by distance to location
                    geo_data['distance'] = calc_distance_lists(geo_data['latitude'], geo_data['longitude'], location.y,
                                                               location.x)
                    geo_data.sort_values(['distance'], inplace=True)

                    # Second, check all stations of graph
                    for g_location in geo_data.index:

                        g_location_lon = geo_data.loc[g_location, 'longitude']
                        g_location_lat = geo_data.loc[g_location, 'latitude']

                        distance = calc_distance(location.y, location.x, g_location_lat, g_location_lon)
                        if distance < tolerance_distance:
                            possible_options.append(g_location)
                        else:
                            # break as soon as one station is further away than distance
                            break

                    overall_geodata_df = pd.concat([overall_geodata_df, geo_data.loc[possible_options].copy()])

                    if True:  #todo: define boolean which is set in config
                        # Option to use direct path to graph (new feed-in / train station)
                        # Check if existing node or new node

                        left_df = gpd.GeoDataFrame(geometry=[closest_node])
                        right_df = gpd.GeoDataFrame(geometry=[graph_object]).explode(ignore_index=True)
                        df_n = gpd.sjoin_nearest(left_df, right_df).merge(right_df, left_on="index_right",
                                                                          right_index=True)

                        affected_line = df_n['geometry_y'].values[0]

                        distance_to_line = math.inf
                        is_new_node = True
                        for c in affected_line.coords:

                            if calc_distance(round(c[1], 5), round(c[0], 5),
                                             round(closest_node.y, 5), round(closest_node.x, 5)) < distance_to_line:
                                if calc_distance(round(c[1], 5), round(c[0], 5),
                                                 round(closest_node.y, 5), round(closest_node.x, 5)) <= 100:
                                    is_new_node = False
                                    break

                                distance_to_line = calc_distance(round(c[1], 5), round(c[0], 5),
                                                                 round(closest_node.y, 5), round(closest_node.x, 5))
                            else:
                                if calc_distance(round(c[1], 5), round(c[0], 5),
                                                 round(closest_node.y, 5), round(closest_node.x, 5)) <= 100:
                                    is_new_node = False
                                    break

                        if is_new_node:

                            closest_node_index = max(geo_data.index) + 1

                            overall_geodata_df.loc[closest_node_index, 'latitude'] = round(closest_node.y, 5)
                            overall_geodata_df.loc[closest_node_index, 'longitude'] = round(closest_node.x, 5)
                            overall_geodata_df.loc[closest_node_index, 'graph'] = g

            if len(overall_geodata_df.index) > 0:
                return overall_geodata_df
            else:
                return None

    # Check where current location is placed and which mean of transport exist at location
    commodity = solution.get_current_commodity_object()
    location = solution.get_current_location()

    means_of_transport = {}

    if 'Pipeline_Liquid' in commodity.get_transportation_options():
        means_of_transport['Pipeline_Liquid'] = find_possible_options_based_on_location('Pipeline_Liquid')

    if 'Pipeline_Gas' in commodity.get_transportation_options():
        means_of_transport['Pipeline_Gas'] = find_possible_options_based_on_location('Pipeline_Gas')

    if 'Shipping' in commodity.get_transportation_options():
        means_of_transport['Shipping'] = find_possible_options_based_on_location('Shipping')

    if 'Railroad' in commodity.get_transportation_options():
        means_of_transport['Railroad'] = find_possible_options_based_on_location('Railroad')

    if False:

        # Find closest street
        lon = location.x
        lat = location.y

        if lat < 0:
            north = lat + 0.25
            south = lat - 0.25
        else:
            north = lat - 0.25
            south = lat + 0.25

        G = ox.graph_from_bbox(north, south, lon - 0.25, lon + 0.25)
        G_proj = ox.project_graph(G)
        nearest_edge = ox.nearest_edges(G_proj, X=lon, Y=lat, return_dist=True)

        # todo: the location is missing
        means_of_transport['Road'] = nearest_edge

    return means_of_transport


def find_shortest_path_to_existing_networks(location, networks, max_length_new_segment_option):

    # todo: not possible if shortest distance is over water

    geod = Geod(ellps="WGS84")

    possible_destinations_and_routes = []
    for network in networks:
        distance = geod.geometry_length(LineString(nearest_points(network, location)))

        if distance < max_length_new_segment_option:
            location_closest_point = nearest_points(network, location)[0]
            line = LineString([location, location_closest_point])

            possible_destinations_and_routes.append((location_closest_point, distance, line))

    return possible_destinations_and_routes

