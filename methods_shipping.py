import pandas as pd
import searoute as sr
from shapely.geometry import LineString, Point
from _helpers import calc_distance


def find_searoute(start_coordinates, destination_coordinates, real_starting_point=None, existing_distance=0):

    route = sr.searoute(start_coordinates, destination_coordinates)
    coordinates = []

    if real_starting_point is not None:
        coordinates.append((real_starting_point.x, real_starting_point.y))

    for coordinate in route.geometry['coordinates']:
        coordinates.append((coordinate[0], coordinate[1]))

    line = LineString(coordinates)
    distance = (round(float(format(route.properties['length'])), 2) + existing_distance) * 1000  # m

    return (Point(destination_coordinates), distance, line)


def remove_and_sort_ports(ports, destination_continent, destination_point,
                          transportation_costs, current_costs, benchmark):

    """
    Method removes all ports where the direct transport is already more expensive than benchmark
    :param ports:
    :param destination_continent:
    :return:
    """

    if False:
        if destination_continent in ['Europe', 'Asia']:
            destination_continent = ['Europe', 'Asia']
        else:
            destination_continent = [destination_continent]

    considered_ports = []
    for p in ports.index:
        if ports.loc[p, 'continent'] in destination_continent:

            distance = calc_distance(destination_point.y, destination_point.x,
                                     ports.loc[p, 'latitude'], ports.loc[p, 'longitude'])

            if current_costs + distance * transportation_costs < benchmark:
                considered_ports.append(p)
                ports.loc[p, 'distance'] = distance

    if considered_ports:
        ports = ports.loc[considered_ports]
        ports = ports.sort_values(by=['distance'])
    else:
        ports = pd.DataFrame()

    return ports
