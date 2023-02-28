import openrouteservice
from openrouteservice import convert
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import pandas as pd
import requests
import json  # call the OSMR API

from _helpers import calc_distance, calc_distance_lists


def find_routes_road_transportation(data, solution, benchmark, configuration):

    """

    :param solution: Currently processed solution
    :param tolerance_distance: Distance, within road transport is excluded as it is assumed that
    solution is at location. Used to avoid unnecessary road transport for very short distances
    :param benchmark: Current benchmark of algorithm. Used to exclude road transport
    if using the direct path is already more expensive than benchmark
    :param configuration
    :return: unique solutions
    """

    # todo 2: Direct path to network not just stations
    # todo 1: implement the option that not every possible target (e.g., all ports) but only closest
    # todo: could be further filtered that only solutions are processed which are, for example, on the same continent
    #  as no road transportation exists of the landmass is not connected

    def calculate_route(option_lat, option_lon, network):

        if True:

            # todo: Compare with openrouteservice
            # todo: The tool also uses direct transportation somehow. Needs to be avoided

            r = requests.get(
                f"http://router.project-osrm.org/route/v1/car/{location.x},{location.y};{option_lon},{option_lat}?steps=true&geometries=geojson""")
            routes = json.loads(r.content)

            if routes['code'] != 'NoRoute':
                route = routes.get("routes")[0]
                point_list = []
                for point in route['geometry']['coordinates']:
                    point_list.append(Point(point))

                additional_distance = 0
                if point_list[0] != location:
                    # Case, that initial starting point is not on street:
                    # todo, hier gibt es anscheinend auch einen service
                    #  http://project-osrm.org/docs/v5.5.1/api/?language=Python#nearest-service
                    #  Wäre hilfreich für die Distanz zu Häfen, wenn kein direkter Weg hinführt
                    #  (aber z.B. nur 500 m entfernt)
                    point_list = [location] + point_list
                    additional_distance = calc_distance(location.y, location.x, point_list[0].y, point_list[0].x)

                line = LineString(point_list)
                distance = route['distance'] + additional_distance

                possible_destinations_and_routes.append((network, point_list[-1], distance, line))

        if False:

            coords = ((location.x, location.y), (option_lon, option_lat))
            client = openrouteservice.Client(key='5b3ce3597851110001cf624899e56e0020834bdab4da20c23edad68f')

            try:

                # todo: Important: utilization of openrouteservice is limited. As the application is wrapped in try,
                #  problems with access might not be stated

                routes = client.directions(coords, profile='driving-hgv')  # Route für LKW
                geometry = client.directions(coords)['routes'][0]['geometry']
                decoded = convert.decode_polyline(geometry)
                point_list = []
                for point in decoded['coordinates']:
                    point_list.append(Point(point))

                additional_distance = 0
                if point_list[0] != location:
                    # Case, that initial starting point is not on street:
                    point_list = [location] + point_list
                    additional_distance = calc_distance(location.y, location.x, point_list[0].y, point_list[0].x)

                line = LineString(point_list)
                distance = routes['routes'][0]['summary']['distance'] + additional_distance

                possible_destinations_and_routes.append((network, name_network, name_option,
                                                         point_list[-1], distance, line))

            except Exception:
                pass

    def find_possible_destinations_and_routes(target_network):

        """
        Finds all possible ports and feed-in / railroad stations which are reachable by road transport
        Finds road to destination of possible
        """

        if target_network == 'Shipping':  # ports

            options = data['Shipping'].copy()

            # Don't use already used infrastructure again
            used_ports = solution.get_used_ports()

            options['distance'] = calc_distance_lists(options['latitude'], options['longitude'], location.y, location.x)
            options.sort_values(['distance'], inplace=True)

            for option in options.index:

                if used_ports:
                    if option in used_ports:
                        continue

                option_location_lon = float(options.loc[option, 'longitude'])
                option_location_lat = float(options.loc[option, 'latitude'])

                # check if direct route already more expensive than benchmark
                direct_path = options.loc[option, 'distance']
                if (total_costs + direct_path / 1000 * transportation_costs) > benchmark:
                    break  # as ports are sorted by distance, following ports have automatically higher costs

                # Check if direct route is higher than tolerance
                if direct_path > configuration['tolerance_distance']:
                    calculate_route(option_location_lat, option_location_lon, target_network)

                    # if system is set to find only the closest port
                    if configuration['find_only_closest']:
                        break

        else:

            # todo: get as close as possible with road transport

            options = data[target_network].copy()

            # Don't use already used infrastructure again
            if target_network == 'Railroad':
                used_stations = solution.get_used_railroad_networks()
            elif target_network == 'Pipeline_Liquid':
                used_stations = solution.get_used_railroad_networks()
            else:
                used_stations = solution.get_used_railroad_networks()

            # Iterate through networks
            for g in [*options.keys()]:

                if used_stations:
                    if g in used_stations:
                        continue

                geo_data = options[g]['GeoData'].copy()
                graph_object = options[g]['GraphObject']

                closest_node = nearest_points(graph_object, location)[0]
                direct_path = calc_distance(location.y, location.x, closest_node.y, closest_node.x)
                # check if direct route already more expensive than benchmark
                if (total_costs + direct_path / 1000 * transportation_costs) > benchmark:
                    # Check if direct route is higher than tolerance
                    continue

                if direct_path < configuration['tolerance_distance']:
                    continue

                geo_data['distance'] = calc_distance_lists(geo_data['latitude'], geo_data['longitude'],
                                                           location.y, location.x)
                geo_data.sort_values(['distance'], inplace=True)

                for g_location in geo_data.index:

                    g_location_lon = float(geo_data.loc[g_location, 'longitude'])
                    g_location_lat = float(geo_data.loc[g_location, 'latitude'])

                    # check if direct route already more expensive than benchmark
                    direct_path = geo_data.loc[g_location, 'distance']
                    if (total_costs + direct_path / 1000 * transportation_costs) > benchmark:
                        break  # as stations are sorted by distance, following ports have automatically higher costs

                    # Check if direct route is higher than tolerance
                    if direct_path > configuration['tolerance_distance']:
                        calculate_route(g_location_lat, g_location_lon, target_network)

                        # if system is set to find only the closest node
                        if configuration['find_only_closest']:
                            break

    location = solution.get_current_location()
    destination = solution.get_destination()

    total_costs = solution.get_total_costs()
    commodity = solution.get_current_commodity_object()
    transportation_costs = commodity.get_transportation_costs_specific_mean_of_transport('Road')

    possible_destinations_and_routes = []

    direct_path_destination = calc_distance(location.y, location.x, destination.y, destination.x)
    if (total_costs + direct_path_destination / 1000 * transportation_costs) < benchmark:
        calculate_route(destination.y, destination.x, 'Destination')

    find_possible_destinations_and_routes('Shipping')
    find_possible_destinations_and_routes('Pipeline_Gas')
    find_possible_destinations_and_routes('Pipeline_Liquid')
    find_possible_destinations_and_routes('Railroad')

    return possible_destinations_and_routes



