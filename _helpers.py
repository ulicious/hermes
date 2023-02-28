from shapely.geometry import LineString
from math import cos, sin, asin, sqrt, radians
import numpy as np


def calc_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 6371 * c * 1000
    return m


def calc_distance_lists(lat1_list, lon1_list, lat2, lon2):
    """
    Calculates distance based in latitudes and longitude but allows further lists of latitudes and longitudes
    """
    lon1_list = np.radians(lon1_list)
    lat1_list = np.radians(lat1_list)
    lon2, lat2 = map(radians, [lon2, lat2])

    dlon = lon2 - lon1_list
    dlat = lat2 - lat1_list

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_list) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000
    return m


def get_direct_line_and_distance_between_two_points(latitude_1, longitude_1, latitude_2, longitude_2):
    line = LineString([(latitude_1, longitude_1), (latitude_2, longitude_2)])
    distance = calc_distance(latitude_1, longitude_1, latitude_2, longitude_2)

    return distance, line