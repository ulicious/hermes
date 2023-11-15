import pandas as pd
import shapely
from shapely.ops import nearest_points
from shapely.geometry import LineString, MultiLineString, Point
import geopandas as gpd
import networkx as nx

import searoute as sr
from _helpers import calc_distance_list_to_single, check_if_reachable_by_road


def process_network_data(data, name, geo_data, graph_data, path_data):

    """
    Function is used to create different data structures for the network data
    :param data: Existing dictionary
    :param name: name of network
    :param geo_data: geo data of network (locations of nodes)
    :param graph_data: information on lines of network
    :return: different data structures
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

        # distance dataframe
        distances = pd.read_csv(path_data + '/inner_infrastructure_distances/' + g + '.csv', index_col=0)
        distances_index = distances.index.tolist()

        if True:
            data[name][g] = {'Graph': graph,
                             'GraphData': graph_data,
                             'GraphObject': graph_object,
                             'GeoData': geo_data.loc[nodes_graph_original],
                             'Distances': {'value': distances.to_numpy(),
                                           'index': distances_index,
                                           'columns': distances.columns}}
        else:
            data[name][g] = {'Graph': graph,
                             'GraphData': graph_data,
                             'GraphObject': graph_object,
                             'GeoData': geo_data.loc[nodes_graph_original],
                             'Distances': distances}

    return data


def attach_new_ports(data, configuration, continent_start, location, continent_destination, final_destination,
                     coastline):

    """
    Method allows the implementation of new ports at the coastline. New ports are implemented based on the shortest
    distance to coastline. If new port has same location as existing port, no new port is added
    :param data: data to provide existing ports
    :param configuration: used to check if new ports are allowed
    :param continent_start: continent of starting location
    :param location: starting location
    :param continent_destination: continent of final destination
    :param final_destination: location of final destination
    :param coastline: shape of global coastlines
    :return: returns update ports within the data dictionary
    """

    options_shipping = data['Shipping']['ports']

    # closest harbor to destination only if no other port is in tolerance distance to destination
    options_shipping['distance_to_final_destination'] \
        = calc_distance_list_to_single(options_shipping['latitude'], options_shipping['longitude'],
                                       final_destination.y, final_destination.x)

    if configuration['Shipping']['build_new_infrastructure']:

        all_distances = data['all_distances_inner_infrastructure']

        start_availability, start_polygon = check_if_reachable_by_road(Point([location.x,
                                                                              location.y]),
                                                                       options_shipping['longitude'],
                                                                       options_shipping['latitude'],
                                                                       coastline,
                                                                       get_only_poly=True)
        # options_shipping['reachable_by_road_start'] = availability

        if start_polygon is not None:

            polygon = coastline.loc[start_polygon].values[0]
            new_port_location = nearest_points(polygon.exterior, location)[0]

            # Check if new port is already in ports
            index_new_port = options_shipping[(options_shipping['latitude'] == new_port_location.y) &
                                              (options_shipping['longitude'] == new_port_location.x)].index
            if len(index_new_port) == 0:

                new_port = 'S_New_' + str(len(options_shipping.index) + 1)

                options_shipping.loc[new_port, 'longitude'] = new_port_location.x
                options_shipping.loc[new_port, 'latitude'] = new_port_location.y
                options_shipping.loc[new_port, 'continent'] = continent_start
                options_shipping.loc[new_port, 'reachable_by_road_start'] = True

                start_location = [new_port_location.x, new_port_location.y]

                distances = []
                new_port_is_removed = False
                for p in options_shipping.index:
                    if p == new_port:
                        distances.append(0)
                    else:

                        # todo: this takes quite long as this has to be done for each new port

                        end_location = [options_shipping.loc[p, 'longitude'],
                                        options_shipping.loc[p, 'latitude']]
                        try:
                            route = sr.searoute(start_location, end_location)  # m
                            distances.append((round(float(format(route.properties['length'])), 2)) * 1000)
                        except ValueError:
                            if p == options_shipping.index[0]:
                                # new port can not be processed as location is not reachable by searoute algorithm.
                                # Therefore, remove this port again
                                new_port_is_removed = True
                                break

                if not new_port_is_removed:
                    all_distances.loc[new_port, options_shipping.index.tolist()] = distances
                    all_distances.loc[options_shipping.index.tolist(), new_port] = distances
                else:
                    options_shipping.drop(new_port, inplace=True)

        index_in_tolerance = options_shipping[options_shipping['distance_to_final_destination']
                                              <= configuration['to_final_destination_tolerance']].index
        if len(index_in_tolerance) == 0:
            # only new port is added if none of the existing is in tolerance distance to final destination

            availability, destination_polygon = check_if_reachable_by_road(
                Point([final_destination.x, final_destination.y]),
                options_shipping['longitude'], options_shipping['latitude'],
                coastline)

            if destination_polygon is not None:

                options_shipping['reachable_by_road_destination'] = availability

                polygon = coastline.loc[destination_polygon].values[0]
                new_port_location = nearest_points(polygon.exterior, final_destination)[0]

                # Check if new port is already in ports
                index_new_port = options_shipping[(options_shipping['latitude'] == new_port_location.y) &
                                                  (options_shipping['longitude'] == new_port_location.x)].index
                if len(index_new_port) == 0:
                    # only new port is added if none of the existing is in tolerance distance to final destination

                    new_port = 'S_New_' + str(len(options_shipping.index) + 1)

                    options_shipping.loc[new_port, 'longitude'] = new_port_location.x
                    options_shipping.loc[new_port, 'latitude'] = new_port_location.y
                    options_shipping.loc[new_port, 'continent'] = continent_destination
                    options_shipping.loc[new_port, 'reachable_by_road_start'] = False
                    options_shipping.loc[new_port, 'reachable_by_road_destination'] = False

                    start_location = [new_port_location.x, new_port_location.y]

                    distances = []
                    new_port_is_removed = False
                    for p in options_shipping.index:
                        if p == new_port:
                            distances.append(0)
                        else:

                            # todo: this takes quite long as this has to be done for each new port
                            #  could be improve when searoute is only applied for ports which are
                            #  on the same continent as the destination --> only these will be considered in
                            #  the search for potential destinations

                            end_location = [options_shipping.loc[p, 'longitude'],
                                            options_shipping.loc[p, 'latitude']]
                            try:
                                route = sr.searoute(start_location, end_location)  # m
                                distances.append((round(float(format(route.properties['length'])), 2)) * 1000)
                            except:
                                if p == options_shipping.index[0]:
                                    # new port can not be processed as location is not reachable by searoute algorithm.
                                    # Therefore, remove this port again
                                    new_port_is_removed = True
                                    break

                    if not new_port_is_removed:
                        all_distances.loc[new_port, options_shipping.index.tolist()] = distances
                        all_distances.loc[options_shipping.index.tolist(), new_port] = distances
                    else:
                        options_shipping.drop(new_port, inplace=True)

            data['all_distances_inner_infrastructure'] = all_distances

    # closest harbor to destination only if no other port is in tolerance distance to destination
    # recalculate because new ports might exist
    options_shipping['distance_to_final_destination'] \
        = calc_distance_list_to_single(options_shipping['latitude'], options_shipping['longitude'],
                                       final_destination.y, final_destination.x)

    data['Shipping']['ports'] = options_shipping  # todo: these are not necessary if not copy

    return data
