import requests
import json
import time
import reverse_geocode

import pandas as pd
import geopandas as gpd

import shapely
from shapely import wkt
from shapely.geometry import MultiLineString, Point
from geopy.geocoders import Nominatim

from _helpers import calc_distance_list_to_single


def get_graph(data, geo_data, graph_data, name):

    """
    Process geographical data and create a structured data dictionary.

    :param data: Dictionary with already processed data
    :param graph_data: A DataFrame or similar data structure containing graph data.
                       This data typically includes information about lines and their relationship to a graph.
    :param geo_data: A DataFrame or similar data structure containing geographical data.
                     This data typically includes information about nodes and their relationship to a graph.
    :param name: A string representing the name or identifier for the data being processed.

    :return: A structured data dictionary containing processed information.
             The dictionary is organized by graph and includes the following:
             - 'GraphData': A GeoDataFrame with graph-related data.
             - 'GraphObject': A MultiLineString object representing the graph's lines.
             - 'GeoData': A subset of geo_data containing data related to the specific graph.
    """

    data[name] = {}

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

    # iterate through networks in the geo_data dataframe and create lines
    for g in geo_data['graph'].unique():
        nodes_graph = graph_data[graph_data['graph'] == g].index
        lines = []
        for ind in nodes_graph:
            lines.append(graph_data.loc[ind, 'line'])

        nodes_graph = geo_data[geo_data['graph'] == g].index
        graph_object = MultiLineString(lines)

        data[name][g] = {'GraphData': graph_data,
                         'GraphObject': graph_object,
                         'GeoData': geo_data.loc[nodes_graph]}

    return data


def get_road_distance_to_options(location_local, options_local, step_size=400):

    """
    Calculate road distances from a given location to a list of options.

    :param location_local: A Point object or similar representing the source location for distance calculation.
    :param options_local: A DataFrame or similar data structure containing options with latitude and longitude information.
    :param step_size: (default: 400) The number of options to process in each step.

    :return: A Series containing road distances from the source location to the options.
    """

    # check for more information: http://project-osrm.org/docs/v5.24.0/api/?language=Python#services
    # todo: try googly polyline then it might be possible to process more at once than just step size

    # create first part of string
    start_string = 'http://router.project-osrm.org/table/v1/driving/' + str(round(location_local.x, 4)) \
                   + ',' + str(round(location_local.y, 4))

    # create last part of string
    end_string = '?sources=0&annotations=distance'

    # create middle part of string. Middle part consists of all locations which need to be included in a certain way
    all_options = options_local.index.tolist()
    all_distances_local = pd.DataFrame(index=all_options, columns=[0])
    while all_options:
        options_to_check = all_options[0:step_size]
        destination_string = ''
        locations = [(location_local.y, location_local.x)]
        for option in options_to_check:

            locations.append((round(options_local.loc[option, 'latitude'], 4),
                              round(options_local.loc[option, 'longitude'], 4)))

            if True:
                destination_string += ';' + str(round(options_local.loc[option, 'longitude'], 4)) \
                                      + ',' + str(round(options_local.loc[option, 'latitude'], 4))

        # request string consists of start part, middle part and end part
        request_string = start_string + destination_string + end_string

        adjust_source = False  # problem with project osrm exist and string might need to be adjusted
        r = None
        while True:
            try:
                # request road distances of given string
                # if request does not work, this code will stop and go into exception
                r = requests.get(request_string)
                routes = json.loads(r.content)

                # check the distance from the source --> if too high than snapping of source didn't work
                # it's too high when project osrm does not work correctly
                if routes['distances'] > 10000:
                    adjust_source = True
                    continue

                # if request worked, save distances and stop while loop
                all_distances_local.loc[options_to_check, 'road_distance'] = routes['distances'][0][1:]
                break

            except json.JSONDecodeError as json_err:
                print(f"JSON Decode Error: {json_err}")

                print('Check for HTTP status with code: ' + str(r))
                print(len(request_string))

                # sometimes access to OSRM data is limited. Sleep to avoid breaking the limit
                time.sleep(1)

            if adjust_source:
                # request was successful but results are useless (distances are way too high). Adjust string
                # with a city which definitely exists and has roads
                coordinates = (round(location_local.x, 4), round(location_local.y, 4)),
                closest_city = reverse_geocode.search(coordinates)

                geolocator = Nominatim(user_agent="MyApp")
                city_location = geolocator.geocode(closest_city)

                start_string = 'http://router.project-osrm.org/table/v1/driving/' \
                               + str(round(city_location.x, 4)) \
                               + ',' + str(round(city_location.y, 4))

                request_string = start_string + destination_string + end_string

        # remove options which have been processed
        all_options = all_options[step_size:]

    return all_distances_local['road_distance']


def calculate_road_distances(network_geo_data=None, network_graph_data=None, network_data_name=None,
                             ports=None, approach='simplified'):

    """
    Calculate road distances between geographical points in a network, between networks, and between networks and ports.

    :param network_geo_data: (optional) A list of geographical data for different network components.
                             Each element in the list represents the geographical data for a specific network.
                             This data typically includes latitude and longitude information for network nodes or points.
                             If not provided or set to None, network-related distance calculations are skipped.
    :param network_graph_data: (optional) A list of network graph data corresponding to the geographical data provided
                               in network_geo_data. Each element in the list represents the graph structure of a specific network.
                               This graph structure is used for network-related calculations.
                               If network_geo_data is provided, this argument should also be provided; otherwise,
                               it can be set to None.
    :param network_data_name: (optional) A list of names or identifiers for the network data provided in network_geo_data.
                             It helps associate the network data with their respective names.
                             If network_geo_data is provided, this argument should also be provided; otherwise,
                             it can be set to None.
    :param ports: (optional) A DataFrame or similar data structure containing information about ports or locations for which
                 road distances need to be calculated. It typically includes latitude and longitude information for these ports.
                 If not provided or set to None, port-related distance calculations are skipped.
    :param approach: (default: 'simplified') Specifies the approach used for distance calculation.
                    It can take one of two values: 'simplified' or another value (e.g., 'advanced').
                    When set to 'simplified', the function calculates distances using a simplified method,
                    considering beeline distances and multiplying them by a factor of 1.5 to account for obstacles.
                    If set to another value, it uses an alternative method, possibly involving road network data
                    (e.g., OSRM data) for more accurate road distance calculations.

    :return: A DataFrame (all_distances) filled with the calculated road distances.
             Distances are calculated between ports, between ports and network nodes,
             and between network nodes in various combinations, depending on the provided data and network structures.
    """

    # if no data is input then return nothing
    if (network_geo_data is None) & (ports is None):
        return pd.DataFrame()

    data = {}
    columns = []

    # iterate through network data to get graph and index
    # todo: might not be needed as we could use network_geo_data directly
    if network_geo_data is not None:
        for i, network_geo in enumerate(network_geo_data):
            network_name = network_data_name[i]
            network_graph = network_graph_data[i]

            data = get_graph(data, network_geo, network_graph, network_name)

            columns += network_geo.index.tolist()

    if ports is not None:
        columns += ports.index.tolist()

    # create dataframe which will be filled with distances
    all_distances = pd.DataFrame(index=columns, columns=columns)

    processed_network_combinations = []
    if ports is not None:
        # calculate distance between each port
        i = 0
        for p in ports.index:

            print('Ports processing at ' + str(round(i / len(ports.index) * 100, 2)) + '%')
            i += 1

            # calculate road distance between ports
            if approach == 'simplified':
                # calculate beeline distance
                distances = calc_distance_list_to_single(ports['longitude'], ports['latitude'],
                                                         ports.loc[p, 'latitude'], ports.loc[p, 'longitude'])

            else:
                distances = \
                    get_road_distance_to_options(Point([ports.loc[p, 'longitude'], ports.loc[p, 'latitude']]),
                                                 ports['latitude', 'longitude'])

                distances.dropna(inplace=True)

            all_distances.loc[p, ports.index] = distances
            all_distances.loc[ports.index, p] = distances

            if network_geo_data is not None:
                # calculate distances between ports and networks
                for network_type in [*data.keys()]:
                    for network_id in [*data[network_type].keys()]:
                        geo_data = data[network_type][network_id]['GeoData']
                        if approach == 'simplified':
                            # calculate beeline distance
                            distances = calc_distance_list_to_single(geo_data['longitude'], geo_data['latitude'],
                                                                     ports.loc[p, 'latitude'],
                                                                     ports.loc[p, 'longitude'])

                        else:
                            # calculate road distances based on OSRM data
                            distances = \
                                get_road_distance_to_options(Point([ports.loc[p, 'longitude'],
                                                                    ports.loc[p, 'latitude']]),
                                                             geo_data['latitude', 'longitude'])

                            distances.dropna(inplace=True)

                        all_distances.loc[p, geo_data.index] = distances
                        all_distances.loc[geo_data.index, p] = distances

    if network_geo_data is not None:
        # calculate distances between networks
        for network_type_1 in [*data.keys()]:
            i = 0
            for network_id_1 in [*data[network_type_1].keys()]:

                print('Network processing at ' + str(round(i / len([*data[network_type_1].keys()]) * 100, 2)) + '%')
                i += 1

                geo_data_1 = data[network_type_1][network_id_1]['GeoData']

                for network_type_2 in [*data.keys()]:

                    for network_id_2 in [*data[network_type_2].keys()]:

                        if (network_id_1, network_id_2) in processed_network_combinations:
                            # combination was already processed
                            continue

                        geo_data_2 = data[network_type_2][network_id_2]['GeoData']

                        # iterate through all nodes of first network and calculate distances to second network
                        for ind in geo_data_1.index:
                            if approach == 'simplified':
                                # calculate beeline distance
                                distances = calc_distance_list_to_single(geo_data_2['latitude'],
                                                                         geo_data_2['longitude'],
                                                                         geo_data_1.loc[ind, 'latitude'],
                                                                         geo_data_1.loc[ind, 'longitude'])

                            else:
                                # calculate road distances based on OSRM data
                                distances = \
                                    get_road_distance_to_options(Point([geo_data_1.loc[ind, 'longitude'],
                                                                        geo_data_1.loc[ind, 'latitude']]),
                                                                 geo_data_2['latitude', 'longitude'])

                                distances.dropna(inplace=True)

                            all_distances.loc[ind, geo_data_2.index] = distances
                            all_distances.loc[geo_data_2.index, ind] = distances

                    # update processed network combinations to avoid repeated calculations of same networks
                    processed_network_combinations.append((network_id_1, network_id_2))
                    processed_network_combinations.append((network_id_2, network_id_1))

    return all_distances


