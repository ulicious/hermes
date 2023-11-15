import itertools

import pandas as pd
import pycountry_convert as pc
import shapely.geometry.polygon
from shapely.geometry import LineString
from math import cos, sin, asin, sqrt, radians
import numpy as np
import reverse_geocode
import math

import geopandas as gp
from geopandas.tools import sjoin


def calc_distance_single_to_single(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    This methods calculates direct distance between two locations
    
    :param latitude_1: 
    :param longitude_1: 
    :param latitude_2: 
    :param longitude_2: 
    :return: single direct distance values in meter
    """
    # convert decimal degrees to radians
    longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    # haversine formula
    dlon = longitude_2 - longitude_1
    dlat = latitude_2 - latitude_1
    a = sin(dlat / 2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 6371 * c * 1000
    return m


def calc_distance_list_to_single(latitude_list_1, longitude_list_1, latitude_2, longitude_2):
    """
    This method calculates the direct distance between a single location and an array of locations

    :param latitude_list_1:
    :param longitude_list_1:
    :param latitude_2:
    :param longitude_2:
    :return: array of direct distances in meter
    """

    # convert decimal degrees to radians
    longitude_list_1 = np.radians(longitude_list_1.values.astype(float))
    latitude_list_1 = np.radians(latitude_list_1.values.astype(float))
    longitude_2, latitude_2 = map(radians, [longitude_2, latitude_2])

    # haversine formula
    dlon = longitude_2 - longitude_list_1
    dlat = latitude_2 - latitude_list_1
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude_list_1) * np.cos(latitude_2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000
    return m


def calc_distance_list_to_list(latitude_list_1, longitude_list_1, latitude_list_2, longitude_list_2):
    """
    This methods calculates the direct distance between two lists of coordinates.

    :param latitude_list_1:
    :param longitude_list_1:
    :param latitude_list_2:
    :param longitude_list_2:
    :return: Matrix with direct distances in meter
    """

    # convert decimal degrees to radians
    longitude_list_1 = np.array(np.radians(longitude_list_1.values.astype(float)))
    latitude_list_1 = np.array(np.radians(latitude_list_1.values.astype(float)))
    longitude_list_2 = np.array(np.radians(longitude_list_2.values.astype(float)))
    latitude_list_2 = np.array(np.radians(latitude_list_2.values.astype(float)))

    # haversine formula
    dlon = np.subtract(longitude_list_2, longitude_list_1[:, np.newaxis])
    dlat = np.subtract(latitude_list_2, latitude_list_1[:, np.newaxis])

    latitude_list_1 = pd.DataFrame(np.cos(latitude_list_1))  # to allow transpose operations, convert to dataframe #todo: probably be possible with np arrays as well
    latitude_list_2 = pd.DataFrame(np.cos(latitude_list_2))  # to allow transpose operations, convert to dataframe #todo: probably be possible with np arrays as well

    matrix_lat1_lat2 = latitude_list_1.dot(latitude_list_2.T)
    a = np.sin(dlat / 2) ** 2 + matrix_lat1_lat2 * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000

    return m


def get_country_and_continent_from_location(location_longitude, location_latitude):

    """
    This method derives continent and country from location coordinates

    :param location_longitude:
    :param location_latitude:
    :return:
    """

    coordinates = (location_latitude, location_longitude),
    closest_city = reverse_geocode.search(coordinates)
    country_destination = closest_city[0]['country']
    country_code = closest_city[0]['country_code']

    # Important: reverse geocode does not necessarily give the right country. It gives the closest city.
    # Other packages might be more suitable

    # The dataset seems not to be complete. Therefore, we have to catch some special cases
    # todo: all prints can be deleted as soon as no new exceptions pop up. prints are necessary to recognize exceptions
    if country_destination in ['Sint Maarten', 'Bonaire, Saint Eustatius and Saba', 'Curacao', 'Aruba']:
        continent_name = 'South America'
    elif country_destination in ['Libyan Arab Jamahiriya', 'Western Sahara', "Cote d'Ivoire"]:
        continent_name = 'Africa'
    elif country_destination in ['Aland Islands']:
        continent_name = 'Europe'
    elif country_destination in ['Palestinian Territory', 'Timor-Leste']:
        continent_name = 'Asia'
    elif country_destination == '':
        if country_code in ['XK']:
            continent_name = 'Europe'
        else:
            print(closest_city)
    else:
        try:
            country_alpha2 = pc.country_name_to_country_alpha2(country_destination)
            continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        except:
            print(coordinates)

            print(closest_city)

            print(country_destination)

            country_alpha2 = pc.country_name_to_country_alpha2(country_destination)
            print(country_alpha2)

            continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
            print(continent_code)

    return country_destination, continent_name


def get_direct_line_and_distance_between_two_points(latitude_1, longitude_1, latitude_2, longitude_2):
    line = LineString([(latitude_1, longitude_1), (latitude_2, longitude_2)])
    distance = calc_distance_single_to_single(latitude_1, longitude_1, latitude_2, longitude_2)

    return distance, line


def calculate_cheapest_option_to_final_destination(data, solution, targets):

    """
    This methods calculates the lowest transportation costs from different targets to final destination.
    The lowest costs include conversion and transportation

    :param data: dictionary with general data
    :param solution: current solution which includes current commodity and location
    :param targets: targets considered
    :return: targets with 'costs to final destination' column
    """

    means_of_transport = data['Means_of_Transport']
    current_commodity = solution.get_current_commodity_object()

    conversion_options = current_commodity.get_conversion_options()

    # Iterate through each commodity and calculate conversion costs + transportation costs for each option
    # option = combination of commodity & mean of transport
    cheapest_transport = pd.DataFrame(index=targets.index)
    for c in [*data['Commodities'].keys()]:
        commodity = data['Commodities'][c]
        if c != current_commodity.get_name():
            if not conversion_options[c]:
                # current commodity cannot be converted into c
                continue

            else:
                conversion_costs = current_commodity.get_conversion_costs_specific_commodity(c)
        else:
            conversion_costs = 0

        transportation_options = commodity.get_transportation_options()
        for m in means_of_transport:
            if not transportation_options[m]:
                # commodity not transportable via this option
                continue

            options_costs = conversion_costs \
                + commodity.get_transportation_costs_specific_mean_of_transport(m) / 1000 \
                * targets['distance_to_final_destination']
            cheapest_transport.loc[:, c + '_' + m] = options_costs

    # Choose the cheapest option for each target and set them as new column in target dataframe
    min_values = cheapest_transport.min(axis=1)
    targets['costs_to_final_destination'] = min_values

    return targets


def check_if_reachable_by_road(start_location, list_longitude, list_latitude, coastline, get_only_availability=False,
                               get_only_poly=False):

    """
    This method checks if a location can reach different locations by checking if they are in the same polygon.
    The polygon are based on the coastlines
    :param start_location: starting location
    :param list_longitude: longitude of target locations
    :param list_latitude: latitude of target locations
    :param coastline: polygons based on coastline
    :param get_only_poly: used to get only the polygon of the starting location
    :return: returns tuple with the boolean if reachable by road (within the same polygon) and the index of the polygon
    """

    # todo: ports and not necessarily on the polygons as they are in the water --> cannot be found
    # todo: when processing ports --> closest point to coastline add to information

    df_start = gp.GeoSeries.from_wkt(['Point(' + str(start_location.x) + ' ' + str(start_location.y) + ')'])
    gdf_start = gp.GeoDataFrame(df_start, geometry=0)

    points = []
    for i in list_latitude.index:
        points.append('Point(' + str(list_longitude.loc[i]) + ' ' + str(list_latitude.loc[i]) + ')')

    df_target = gp.GeoSeries.from_wkt(points)
    gdf_target = gp.GeoDataFrame(df_target, geometry=0)

    polygons = sjoin(gdf_start, coastline, predicate='within', how='right').dropna(subset=['index_left'])

    if len(polygons.index) > 0:

        index_poly_start = polygons.index[0]
        index_poly_target = sjoin(gdf_target, coastline, predicate='within', how='right')# .dropna(subset=['index_left']

        if not get_only_poly:
            availability = []
            for i in gdf_target.index:

                if i in index_poly_target['index_left'].values.tolist():
                    poly = index_poly_target[index_poly_target['index_left'] == i].index[0]

                    if index_poly_start == poly:
                        availability.append(True)
                    else:
                        availability.append(False)

                else:
                    availability.append(False)

            if not get_only_availability:
                return availability, index_poly_start
            else:
                return availability
        else:
            return index_poly_start

    else:
        return None, None # todo: it seems that not all coastlines are in the data set (some might be too small)


def get_polygon_of_location(location, coastlines):
    # df_start = gp.GeoSeries.from_wkt([location])
    gdf_start = gp.GeoDataFrame(index=['0'], geometry=[location])

    polygons = sjoin(gdf_start, coastlines, predicate='within', how='right').dropna(subset=['index_left'])

    if len(polygons.index) > 0:
        index_poly_start = polygons.index[0]
        return polygons.loc[index_poly_start, 'geometry']


def check_if_last_was_new_segment(solution, target_system):

    check_if_iteration_is_high_enough = False
    applies = False

    # check if iteration is high enough to check
    last_transport_means = solution.get_used_transport_means()

    last_iterations = list(last_transport_means.keys())
    last_transport_iteration_number = None
    if last_iterations[-1] != 0:
        last_transport_iteration_number = last_iterations[-2]
        check_if_iteration_is_high_enough = True

    if check_if_iteration_is_high_enough:
        # check if it applies due to past infrastructure and transport means
        applies = (not last_infrastructure[last_transport_iteration_number]) \
                  & (last_transport_means[last_transport_iteration_number] == target_system)

    return check_if_iteration_is_high_enough, applies
