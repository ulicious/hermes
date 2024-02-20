import itertools

import pandas as pd
import pycountry_convert as pc
import shapely.geometry.polygon
from shapely.geometry import LineString
from math import cos, sin, asin, sqrt, radians
import numpy as np
import reverse_geocode
import math

from joblib import Parallel, delayed

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


def calc_distance_list_to_list_no_matrix(latitude_list_1, longitude_list_1, latitude_list_2, longitude_list_2):
    """
    This method calculates the direct distance between a single location and an array of locations

    :param latitude_list_1:
    :param longitude_list_1:
    :param latitude_list_2:
    :param longitude_list_2:
    :return: array of direct distances in meter
    """

    # convert decimal degrees to radians
    longitude_list_1 = np.radians(longitude_list_1.values.astype(float))
    latitude_list_1 = np.radians(latitude_list_1.values.astype(float))
    longitude_list_2 = np.radians(longitude_list_2.values.astype(float))
    latitude_list_2 = np.radians(latitude_list_2.values.astype(float))

    # haversine formula
    dlon = longitude_list_2 - longitude_list_1
    dlat = latitude_list_2 - latitude_list_1
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude_list_1) * np.cos(latitude_list_2) * np.sin(dlon / 2) ** 2
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

    # Convert decimal degrees to radians using Numpy arrays directly
    longitude_list_1 = np.radians(longitude_list_1.values.astype(float))
    latitude_list_1 = np.radians(latitude_list_1.values.astype(float))
    longitude_list_2 = np.radians(longitude_list_2.values.astype(float))
    latitude_list_2 = np.radians(latitude_list_2.values.astype(float))

    # Haversine formula
    dlon = np.subtract.outer(longitude_list_2, longitude_list_1)
    dlat = np.subtract.outer(latitude_list_2, latitude_list_1)

    # Use Numpy arrays directly, avoid unnecessary DataFrame conversion
    matrix_lat1_lat2 = np.outer(np.cos(latitude_list_1), np.cos(latitude_list_2))
    a = np.sin(dlat / 2) ** 2 + matrix_lat1_lat2.T * np.sin(dlon / 2) ** 2
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


def get_continent_from_location(location):

    """
    This method derives continent and country from location coordinates

    :param location_longitude:
    :param location_latitude:
    :return:
    """

    location_longitude = location[0]
    location_latitude = location[1]

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

    return continent_name


def get_direct_line_and_distance_between_two_points(latitude_1, longitude_1, latitude_2, longitude_2):
    line = LineString([(latitude_1, longitude_1), (latitude_2, longitude_2)])
    distance = calc_distance_single_to_single(latitude_1, longitude_1, latitude_2, longitude_2)

    return distance, line


def calculate_cheapest_option_to_final_destination(data, options, configuration, benchmark, cost_column_name=None,
                                                   solution=None, check_minimal_distance=False):

    """
    This methods calculates the lowest transportation costs from different targets to final destination.
    The lowest costs include conversion and transportation

    :param data: dictionary with general data
    :param options: current solution which includes current commodity and location
    :param benchmark: targets considered
    :return: targets with 'costs to final destination' column
    """
    means_of_transport = data['transport_means']
    final_commodities = data['commodities']['final_commodities']
    max_length_new_segment = configuration['max_length_new_segment']
    max_length_road = configuration['max_length_road']
    no_road_multiplier = configuration['no_road_multiplier']

    # load minimal distances and add Destination with 0
    minimal_distances = data['minimal_distances']
    minimal_distances.loc['Destination', 'minimal_distances'] = 0

    if not check_minimal_distance:

        columns = ['current_commodity', cost_column_name, 'distance_to_final_destination',
                   'current_transport_mean']
        cheapest_options = pd.DataFrame(options[columns], columns=columns) # todo: nach 5626 schauen

        cheapest_options.index = range(len(options.index))

        considered_commodities = cheapest_options['current_commodity'].unique()

    if check_minimal_distance:
        # after using infrastructure, the approach uses new pipelines or road to get to the next infrastructure
        # based on the, the minimal distance to the closest infrastructure is calculated. If the distance is
        # above max length of new segments, we will definitely use road. As road is quite expensive, this approach
        # will throw out several some options

        # add minimal distance and check if road is necessary
        options['minimal_distance'] = minimal_distances.loc[
            options['current_node'].tolist(), 'minimal_distance'].tolist()

        columns = ['current_commodity', cost_column_name, 'distance_to_final_destination', 'minimal_distance',
                   'current_transport_mean']
        cheapest_options = pd.DataFrame(options[columns], columns=columns)  # todo: nach 5626 schauen

        cheapest_options.index = range(len(options.index))

        considered_commodities = cheapest_options['current_commodity'].unique()

        # first adjustment: if minimal distance is below tolerance distance, no further costs occur
        in_tolerance = cheapest_options[cheapest_options['minimal_distance']
                                        <= configuration['tolerance_distance']].index
        cheapest_options.loc[in_tolerance, 'minimal_distance'] = 0

        # second adjustment: all distances to final destination below tolerance to final destination are 0 as well
        # such cases are only missing final conversion to final commodity
        in_destination_tolerance \
            = cheapest_options[cheapest_options['distance_to_final_destination']
                               <= configuration['to_final_destination_tolerance']].index
        cheapest_options.loc[in_destination_tolerance, 'minimal_distance'] = 0
        cheapest_options.loc[in_destination_tolerance, 'road_only'] = False

        # check what kind of transport means are applicable
        road_possible = cheapest_options[cheapest_options['minimal_distance'] <= max_length_road / no_road_multiplier].index.tolist()
        new_possible = cheapest_options[cheapest_options['minimal_distance'] <= max_length_new_segment / no_road_multiplier].index.tolist()

        cheapest_options['road_distance'] = max_length_road / no_road_multiplier
        cheapest_options['road_possible'] = False
        cheapest_options.loc[road_possible, 'road_possible'] = True

        cheapest_options['new_distance'] = max_length_new_segment / no_road_multiplier
        cheapest_options['new_possible'] = False
        cheapest_options.loc[new_possible, 'new_possible'] = True

        cheapest_options['residual_road_distance'] = cheapest_options['road_distance'] - cheapest_options['new_distance']

        # all non possible will have infinity as distance
        max_length = max(max_length_road / no_road_multiplier, max_length_new_segment / no_road_multiplier)
        non_possible = cheapest_options[cheapest_options['minimal_distance'] <= max_length].index.tolist()
        cheapest_options.loc[non_possible, 'road_distance'] = math.inf
        cheapest_options.loc[non_possible, 'new_distance'] = math.inf

    # approach uses 'continue'. Therefore, it might be possible that no columns are generated for some options
    # because they are too expensive. These will use basic costs and will be remove due to infinity costs
    created_columns = ['basic_costs']
    cheapest_options['basic_costs'] = math.inf

    for c_start in considered_commodities:

        c_start_df = cheapest_options[cheapest_options['current_commodity'] == c_start]

        # get index of all options using road
        if check_minimal_distance:
            road_possible = c_start_df[c_start_df['road_possible']].index
            new_possible = c_start_df[c_start_df['new_possible']].index

        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()

        # c_start is converted into c_transported
        for c_transported in [*data['commodities']['commodity_objects'].keys()]:
            if c_start != c_transported:
                # calculate conversion costs from c_start to c_transported
                if c_start_conversion_options[c_transported]:
                    c_start_df[c_transported + '_conversion_costs'] = \
                        (c_start_df[cost_column_name]
                         + c_start_object.get_conversion_costs_specific_commodity(c_transported)) \
                        / c_start_object.get_conversion_loss_of_educt_specific_commodity(c_transported)
                else:
                    continue
            else:
                # also no conversion is possible and c_start = c_transported is transported
                c_start_df[c_transported + '_conversion_costs'] = c_start_df[cost_column_name]

            if c_start_df[c_transported + '_conversion_costs'].min() > benchmark:
                # if all conversion costs are already higher than benchmark no further calculations will be made
                # as benchmark is already violated
                continue

            c_transported_object = data['commodities']['commodity_objects'][c_transported]
            transportation_options = c_transported_object.get_transportation_options()
            c_transported_conversion_options = c_transported_object.get_conversion_options()
            c_transported_transportation_costs = c_transported_object.get_transportation_costs()

            for m in means_of_transport:
                if not transportation_options[m]:
                    # commodity not transportable via this option
                    continue

                else:
                    if check_minimal_distance:
                        c_start_df[c_transported + '_transportation_costs_' + m] = math.inf

                        if (not transportation_options['Road']) & (transportation_options['Pipeline_Gas'] | transportation_options['Pipeline_Liquid']):
                            # if road transportation is not possible but new is

                            # minimal distance will be covered with new infrastructure if road is not applicable
                            if transportation_options['Pipeline_Gas']:
                                c_start_df.loc[new_possible, c_transported + '_transportation_costs_' + m] \
                                    = c_transported_transportation_costs[m] / 1000 \
                                    * (c_start_df.loc[new_possible, 'distance_to_final_destination']
                                      - c_start_df.loc[new_possible, 'minimal_distance']) \
                                    + c_start_df.loc[new_possible, 'minimal_distance'] / 1000 \
                                    * c_transported_transportation_costs['New_Pipeline_Gas'] * no_road_multiplier

                            elif transportation_options['Pipeline_Liquid']:
                                c_start_df.loc[new_possible, c_transported + '_transportation_costs_' + m] \
                                    = c_transported_transportation_costs[m] / 1000 \
                                    * (c_start_df.loc[new_possible, 'distance_to_final_destination']
                                       - c_start_df.loc[new_possible, 'minimal_distance']) \
                                    + c_start_df.loc[new_possible, 'minimal_distance'] / 1000 \
                                    * c_transported_transportation_costs['New_Pipeline_Liquid'] * no_road_multiplier

                        elif transportation_options['Road'] & ((not transportation_options['Pipeline_Gas']) | (not transportation_options['Pipeline_Liquid'])):
                            # if road transportation is possible but new is not

                            c_start_df.loc[road_possible, c_transported + '_transportation_costs_' + m] \
                                = c_transported_transportation_costs[m] / 1000 \
                                  * (c_start_df.loc[road_possible, 'distance_to_final_destination']
                                     - c_start_df.loc[road_possible, 'minimal_distance']) \
                                  + c_start_df.loc[road_possible, 'minimal_distance'] / 1000 \
                                  * c_transported_transportation_costs['Road'] * no_road_multiplier

                        elif transportation_options['Road'] & (transportation_options['Pipeline_Gas'] | transportation_options['Pipeline_Liquid']):
                            # if road and new pipeline is applicable, we try to cover as much as possible with
                            # new infrastructure and the rest with road
                            if max_length_new_segment >= max_length_road:
                                # new segments cover longer distances -> only use new
                                if transportation_options['Pipeline_Gas']:
                                    c_start_df.loc[new_possible, c_transported + '_transportation_costs_' + m] \
                                        = c_transported_transportation_costs[m] / 1000 \
                                          * (c_start_df.loc[new_possible, 'distance_to_final_destination']
                                             - c_start_df.loc[new_possible, 'minimal_distance']) \
                                          + c_start_df.loc[new_possible, 'minimal_distance'] / 1000 \
                                          * c_transported_transportation_costs['New_Pipeline_Gas'] * no_road_multiplier

                                elif transportation_options['Pipeline_Liquid']:
                                    c_start_df.loc[new_possible, c_transported + '_transportation_costs_' + m] \
                                        = c_transported_transportation_costs[m] / 1000 \
                                          * (c_start_df.loc[new_possible, 'distance_to_final_destination']
                                             - c_start_df.loc[new_possible, 'minimal_distance']) \
                                          + c_start_df.loc[new_possible, 'minimal_distance'] / 1000 \
                                          * c_transported_transportation_costs['New_Pipeline_Liquid'] * no_road_multiplier
                            else:
                                # new segments cover less distance than road --> use new as far as possible, then road
                                if transportation_options['Pipeline_Gas']:
                                    c_start_df.loc[new_possible, c_transported + '_transportation_costs_' + m] \
                                        = c_transported_transportation_costs[m] / 1000 \
                                          * (c_start_df.loc[new_possible, 'distance_to_final_destination']
                                             - c_start_df.loc[new_possible, 'minimal_distance']) \
                                          + (c_start_df.loc[new_possible, 'new_distance'] / 1000
                                             * c_transported_transportation_costs['New_Pipeline_Gas'] * no_road_multiplier) \
                                        + (c_start_df.loc[new_possible, 'residual_road_distance'] / 1000
                                           * c_transported_transportation_costs['Road'] * no_road_multiplier)

                                elif transportation_options['Pipeline_Liquid']:
                                    c_start_df.loc[new_possible, c_transported + '_transportation_costs_' + m] \
                                        = c_transported_transportation_costs[m] / 1000 \
                                          * (c_start_df.loc[new_possible, 'distance_to_final_destination']
                                             - c_start_df.loc[new_possible, 'minimal_distance']) \
                                          + (c_start_df.loc[new_possible, 'new_distance'] / 1000
                                             * c_transported_transportation_costs['New_Pipeline_Liquid'] * no_road_multiplier) \
                                          + (c_start_df.loc[new_possible, 'residual_road_distance'] / 1000
                                             * c_transported_transportation_costs['Road'] * no_road_multiplier)

                    else:
                        c_start_df[c_transported + '_transportation_costs_' + m] \
                            = c_transported_object.get_transportation_costs_specific_mean_of_transport(m) / 1000 \
                            * c_start_df['distance_to_final_destination']

                    # shipping is only applicable once. Therefore, shipping costs are set to infinity for all
                    # options which have used shipping before (see below)
                    if m == 'Shipping':
                        options_m = c_start_df[c_start_df['current_transport_mean'] == m].index
                        c_start_df.loc[options_m, c_transported + '_transportation_costs_' + m] = math.inf

                    # print(c_transported)
                    # print(c_start_df[c_transported + '_conversion_costs'])
                    # print(c_start_df[c_transported + '_transportation_costs_' + m])

                    # after transportation, conversion to final commodity if necessary
                    for c_end in [*data['commodities']['commodity_objects'].keys()]:

                        name_column = 'costs_' + c_start + '_' + c_transported + '_' + m + '_' + c_end

                        if c_end in final_commodities:
                            if c_transported != c_end:
                                if c_transported_conversion_options[c_end]:
                                    cheapest_options.loc[c_start_df.index, name_column] = \
                                        (c_start_df[c_transported + '_conversion_costs']
                                         + c_start_df[c_transported + '_transportation_costs_' + m]
                                         + c_transported_object.get_conversion_costs_specific_commodity(c_end)) \
                                        / c_transported_object.get_conversion_loss_of_educt_specific_commodity(c_end)
                                    created_columns.append(name_column)
                                else:
                                    continue
                            else:
                                cheapest_options.loc[c_start_df.index, name_column] \
                                    = c_start_df[c_transported + '_conversion_costs'] \
                                    + c_start_df[c_transported + '_transportation_costs_' + m]
                                created_columns.append(name_column)
                        else:
                            continue

    return cheapest_options[created_columns].min(axis=1).tolist()


def calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure(data, solutions, cost_column_name=None):

    """
    This methods calculates the lowest transportation costs from different targets to final destination.
    The lowest costs include conversion and transportation

    :param data: dictionary with general data
    :param solutions: current solution which includes current commodity and location
    :param benchmark: targets considered
    :return: targets with 'costs to final destination' column
    """

    columns = ['current_commodity', cost_column_name, 'distance_to_final_destination',
               'current_transport_mean']
    cheapest_options = pd.DataFrame(solutions[columns], columns=columns)

    cheapest_options.index = range(len(solutions.index))

    considered_commodities = cheapest_options['current_commodity'].unique()

    # approach uses 'continue'. Therefore, it might be possible that no columns are generated for some options
    # because they are too expensive. These will use basic costs and will be remove due to infinity costs
    created_columns = ['basic_costs_Pipeline_Gas', 'basic_costs_Pipeline_Liquid']
    cheapest_options[created_columns] = math.inf

    for c_start in considered_commodities:
        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()
        c_start_df = cheapest_options[cheapest_options['current_commodity'] == c_start]

        # if commodity is already transportable via pipelines, no additional conversion
        c_start_transportation_options = c_start_object.get_transportation_options()
        for m in ['Pipeline_Gas', 'Pipeline_Liquid']:
            if c_start_transportation_options[m]:
                cheapest_options.loc[c_start_df.index, c_start + '_' + m] = c_start_df[cost_column_name]

        for c_conversion in [*data['commodities']['commodity_objects'].keys()]:
            if c_start != c_conversion:
                if c_start_conversion_options[c_conversion]:

                    c_conversion_object = data['commodities']['commodity_objects'][c_conversion]
                    transportation_options = c_conversion_object.get_transportation_options()

                    for m in ['Pipeline_Gas', 'Pipeline_Liquid']:
                        if transportation_options[m]:
                            cheapest_options.loc[c_start_df.index, c_conversion + '_' + m] = \
                                (c_start_df[cost_column_name]
                                 + c_start_object.get_conversion_costs_specific_commodity(c_conversion)) \
                                / c_start_object.get_conversion_loss_of_educt_specific_commodity(c_conversion)
                            created_columns.append(c_conversion + '_' + m)

    pipeline_gas_columns = [c for c in cheapest_options.columns if 'Pipeline_Gas' in c]
    pipeline_gas_cheapest_options = cheapest_options[pipeline_gas_columns].min(axis=1).tolist()

    pipeline_liquid_columns = [c for c in cheapest_options.columns if 'Pipeline_Liquid' in c]
    pipeline_liquid_cheapest_options = cheapest_options[pipeline_liquid_columns].min(axis=1).tolist()

    return pipeline_gas_cheapest_options, pipeline_liquid_cheapest_options


def check_if_reachable_on_land(start_location, list_longitude, list_latitude, coastline, get_only_availability=False,
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

            def get_availability(n):

                if n in index_poly_target['index_left'].values.tolist():
                    poly = index_poly_target[index_poly_target['index_left'] == n].index[0]

                    if index_poly_start == poly:
                        return True
                    else:
                        return False

                else:
                    return False

            inputs = gdf_target.index.tolist()
            result = Parallel(n_jobs=100)(delayed(get_availability)(inp) for inp in inputs)

            if not get_only_availability:
                return result, index_poly_start
            else:
                return result
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
