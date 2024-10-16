import tqdm
import math

import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from shapely.geometry import LineString, Point

from algorithm.methods_geographic import calc_distance_list_to_single
from data_processing.helpers_raw_data_processing import calculate_conversion_costs


def extend_line_in_one_direction(direction_coordinate, support_coordinate, extension_percentage):
    """
    extends a linestring in one direction without changing the direction

    @param direction_coordinate: coordinate where extensions takes place
    @param support_coordinate: support direction to create linestring
    @param extension_percentage: percentage by how much line should be extended
    @return: LineString of extended line
    """

    # Create a LineString from the two coordinates
    line = LineString([direction_coordinate, support_coordinate])

    # Calculate the direction vector of the LineString
    direction_vector = np.array([direction_coordinate.x, direction_coordinate.y]) \
        - np.array([support_coordinate.x, support_coordinate.y])

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm != 0:
        direction_vector /= norm

    # Calculate the extension length based on the keyword (percentage)
    line_length = line.length
    extension_length = line_length * extension_percentage / 100

    # Calculate the new end point
    new_end_point = Point([direction_coordinate.x + direction_vector[0] * extension_length,
                           direction_coordinate.y + direction_vector[1] * extension_length])

    return new_end_point


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


def attach_conversion_costs_and_efficiency_to_locations(locations, config_file, techno_economic_data_conversion,
                                                        with_tqdm=True):

    """
    Iterates over all locations and calculates conversion costs and efficiency at specific location

    @param pandas.DataFrame locations: dataframe with the different locations and levelized costs
    @param dict config_file: dictionary with configuration
    @param dict techno_economic_data_conversion: dictionary with techno economic parameters
    @param bool with_tqdm: boolean to show progress bar (default = True)

    @return: updated dataframe with added columns showing conversion costs and efficiency at each location
    """

    def round_to_quarter(number):

        """
        rounds number to the closest quarter step

        @param float number: coordinate
        @return: quarter step
        """

        # Find the fractional part
        fraction = number % 1

        # Round the fractional part to the nearest .25, .5, .75, or 0
        if fraction <= 0.125:
            return int(number) + 0.00
        elif fraction <= 0.375:
            return int(number) + 0.25
        elif fraction <= 0.625:
            return int(number) + 0.50
        elif fraction <= 0.875:
            return int(number) + 0.75
        else:
            return int(number) + 1.00

    def apply_conversion(locations_to_process, location):

        """
        uses techno economic data and levelized costs to calculate conversion costs and efficiency at each location

        @param pandas.DataFrame locations_to_process: locations of infrastructure
        @param str location: information where location is located: node, port, start or destination

        @return: updated dataframe including conversion costs and efficiency
        """

        # some support materials are not necessarily needed -> 0 costs
        if 'Electricity' in [*techno_economic_data_conversion['cost_type'].keys()]:
            electricity_costs = locations_to_process['Electricity']
        else:
            electricity_costs = 0

        if 'Nitrogen' in [*techno_economic_data_conversion['cost_type'].keys()]:
            nitrogen_costs = locations_to_process['Nitrogen']
        else:
            nitrogen_costs = 0

        if 'CO2' in [*techno_economic_data_conversion['cost_type'].keys()]:
            co2_costs = locations_to_process['CO2']
        else:
            co2_costs = 0

        for c1_local in config_file['available_commodity']:
            for c2_local in techno_economic_data_conversion[c1_local]['potential_conversions']:
                electricity_demand = techno_economic_data_conversion[c1_local][c2_local]['electricity_demand']
                co2_demand = techno_economic_data_conversion[c1_local][c2_local]['co2_demand']
                nitrogen_demand = techno_economic_data_conversion[c1_local][c2_local]['nitrogen_demand']
                heat_demand = techno_economic_data_conversion[c1_local][c2_local]['heat_demand']

                specific_investment = techno_economic_data_conversion[c1_local][c2_local]['specific_investment']
                fixed_maintenance = techno_economic_data_conversion[c1_local][c2_local]['fixed_maintenance']
                lifetime = techno_economic_data_conversion[c1_local][c2_local]['lifetime']
                operating_hours = techno_economic_data_conversion[c1_local][c2_local]['operating_hours']

                interest_rate = locations_to_process['interest_rate']

                if heat_demand == 0:
                    conversion_costs \
                        = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                     operating_hours, interest_rate,
                                                     electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                     nitrogen_costs, nitrogen_demand)

                    conversion_efficiency = techno_economic_data_conversion[c1_local][c2_local]['efficiency_autothermal']

                else:

                    heat_demand_niveau = techno_economic_data_conversion[c1_local][c2_local]['heat_demand_niveau']

                    if (heat_demand_niveau == 'low_temperature') & config_file['low_temp_heat_available_at_' + location]:
                        heat_costs = locations_to_process['Low_Temperature_Heat']

                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)

                        conversion_efficiency = techno_economic_data_conversion[c1_local][c2_local]['efficiency_external_heat']

                    elif (heat_demand_niveau == 'mid_temperature') & config_file['mid_temp_heat_available_at_' + location]:
                        heat_costs = locations_to_process['Mid_Temperature_Heat']

                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)
                        conversion_efficiency = techno_economic_data_conversion[c1_local][c2_local]['efficiency_external_heat']

                    elif (heat_demand_niveau == 'high_temperature') & config_file['high_temp_heat_available_at_' + location]:
                        heat_costs = locations_to_process['High_Temperature_Heat']

                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)

                        conversion_efficiency = techno_economic_data_conversion[c1_local][c2_local]['efficiency_external_heat']

                    else:
                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand)

                        conversion_efficiency = techno_economic_data_conversion[c1_local][c2_local]['efficiency_autothermal']

                locations_to_process[c1_local + '_' + c2_local + '_conversion_costs'] = conversion_costs
                locations_to_process[c1_local + '_' + c2_local + '_conversion_efficiency'] = conversion_efficiency

        return locations_to_process

    path_raw_data = config_file['project_folder_path'] + 'raw_data/'

    levelized_costs_location = pd.read_csv(path_raw_data + config_file['location_data_name'], index_col=0)
    levelized_costs_country = pd.read_csv(path_raw_data + config_file['country_data_name'], index_col=0)

    # add country information to options
    not_shipping_options = [i for i in locations.index if 'H' not in i]
    not_shipping_options = locations.loc[not_shipping_options, :]

    shipping_options = [i for i in locations.index if 'H' in i]  # harbours have information on country already

    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)
    gdf = gpd.GeoDataFrame(not_shipping_options, geometry=gpd.points_from_xy(not_shipping_options.longitude, not_shipping_options.latitude))
    result = gpd.sjoin(gdf, world, how='left')
    not_shipping_options['country'] = result['NAME_EN']

    locations = pd.concat([locations.loc[shipping_options, :], not_shipping_options])

    no_country_options = locations[locations['country'].isna()]
    country_options = locations[~locations['country'].isna()]

    # get cost of location
    new_nan_values = []
    for key in [*techno_economic_data_conversion['cost_type'].keys()]:

        if key == 'Hydrogen_Gas':
            # hydrogen is not needed since commodity is transported to i
            continue

        cost_type = techno_economic_data_conversion['cost_type'][key]

        if key == 'interest_rate':
            if cost_type == 'location':
                raise TypeError('Interest rate cannot be location specific')

        if cost_type == 'uniform':
            country_options[key] = techno_economic_data_conversion['uniform_costs'][key]

        elif cost_type == 'all_countries':
            for country in country_options['country'].unique():

                sub_options = country_options[country_options['country'] == country].index

                if country in levelized_costs_country.index.tolist():
                    try:
                        country_options.loc[sub_options, key] = levelized_costs_country.loc[country, key]
                    except IndexError:
                        print(country + ' is not in country file')
                else:
                    print(country)
                    raise IndexError(country + ' is not in country file')

        elif isinstance(cost_type, list):
            # cost type is list of countries. Countries not in list will be treated as locations
            for country in cost_type:

                sub_options = country_options[country_options['country'] == country].index

                if len(sub_options) > 0:
                    if country in levelized_costs_country.index:
                        try:
                            country_options.loc[sub_options, key] = levelized_costs_country.loc[country, key]
                        except IndexError:
                            raise IndexError(country + ' is not in country file')

            # process locations in countries which are not in list
            not_processed_countries = set(country_options['country'].unique()) - set(cost_type)
            affected_options = country_options[country_options['country'].isin(not_processed_countries)].index

            if key == 'interest_rate':
                country_options.loc[affected_options, key] = techno_economic_data_conversion['uniform_costs'][key]
            else:
                for i in affected_options.index:
                    if 'H' in i:
                        adjusted_latitude = round_to_quarter(country_options.loc[i, 'latitude_on_coastline'])
                        adjusted_longitude = round_to_quarter(country_options.loc[i, 'longitude_on_coastline'])
                    else:
                        adjusted_latitude = round_to_quarter(country_options.loc[i, 'latitude'])
                        adjusted_longitude = round_to_quarter(country_options.loc[i, 'longitude'])

                    # apply small grid search to get the closest location
                    sub_locations = levelized_costs_location[
                        (levelized_costs_location['latitude'] <= adjusted_latitude + 0.5) &
                        (levelized_costs_location['latitude'] >= adjusted_latitude - 0.5) &
                        (levelized_costs_location['longitude'] <= adjusted_longitude + 0.5) &
                        (levelized_costs_location['longitude'] >= adjusted_longitude - 0.5)]

                    if not sub_locations.empty:
                        distances = calc_distance_list_to_single(sub_locations['latitude'], sub_locations['longitude'],
                                                                 adjusted_latitude, adjusted_longitude)
                        distances = pd.DataFrame(distances, index=sub_locations.index)

                        idxmin = distances.idxmin().values[0]
                        country_options.loc[i, key] = levelized_costs_location.loc[idxmin, key]
                    else:
                        new_nan_values.append((i, key))

        else:  # search costs for location
            if with_tqdm:
                options_index = tqdm.tqdm(country_options.index)
            else:
                options_index = country_options.index

            for i in options_index:
                if 'H' in i:
                    adjusted_latitude = round_to_quarter(country_options.loc[i, 'latitude_on_coastline'])
                    adjusted_longitude = round_to_quarter(country_options.loc[i, 'longitude_on_coastline'])
                else:
                    adjusted_latitude = round_to_quarter(country_options.loc[i, 'latitude'])
                    adjusted_longitude = round_to_quarter(country_options.loc[i, 'longitude'])

                # apply small grid search to get the closest location
                sub_locations = levelized_costs_location[
                    (levelized_costs_location['latitude'] <= adjusted_latitude + 0.5) &
                    (levelized_costs_location['latitude'] >= adjusted_latitude - 0.5) &
                    (levelized_costs_location['longitude'] <= adjusted_longitude + 0.5) &
                    (levelized_costs_location['longitude'] >= adjusted_longitude - 0.5)]

                if not sub_locations.empty:
                    distances = calc_distance_list_to_single(sub_locations['latitude'], sub_locations['longitude'],
                                                             adjusted_latitude, adjusted_longitude)
                    distances = pd.DataFrame(distances, index=sub_locations.index)

                    idxmin = distances.idxmin().values[0]
                    country_options.loc[i, key] = levelized_costs_location.loc[idxmin, key]
                else:
                    new_nan_values.append((i, key))

    if with_tqdm:
        options_index = tqdm.tqdm(no_country_options.index)
    else:
        options_index = no_country_options.index

    for i in options_index:

        if 'H' in i:
            adjusted_latitude = round_to_quarter(no_country_options.loc[i, 'latitude_on_coastline'])
            adjusted_longitude = round_to_quarter(no_country_options.loc[i, 'longitude_on_coastline'])
        else:
            adjusted_latitude = round_to_quarter(no_country_options.loc[i, 'latitude'])
            adjusted_longitude = round_to_quarter(no_country_options.loc[i, 'longitude'])

        # get cost of location
        for key in [*techno_economic_data_conversion['cost_type'].keys()]:

            if key == 'Hydrogen_Gas':
                # hydrogen is not needed since commodity is transported to i
                continue

            if techno_economic_data_conversion['cost_type'][key] != 'location':
                continue

            # apply small grid search to get the closest location
            sub_locations = levelized_costs_location[(levelized_costs_location['latitude'] <= adjusted_latitude + 0.5) &
                                                     (levelized_costs_location['latitude'] >= adjusted_latitude - 0.5) &
                                                     (levelized_costs_location['longitude'] <= adjusted_longitude + 0.5) &
                                                     (levelized_costs_location['longitude'] >= adjusted_longitude - 0.5)]

            if not sub_locations.empty:
                distances = calc_distance_list_to_single(sub_locations['latitude'], sub_locations['longitude'],
                                                         adjusted_latitude, adjusted_longitude)
                distances = pd.DataFrame(distances, index=sub_locations.index)

                idxmin = distances.idxmin().values[0]
                no_country_options.loc[i, key] = levelized_costs_location.loc[idxmin, key]

            else:
                new_nan_values.append((i, key))

    conversion_solutions = pd.concat([country_options, no_country_options])

    # if data for index - commodity combination does not exist, we set costs to infinity
    for combination in new_nan_values:
        i = combination[0]
        key = combination[1]

        conversion_solutions.at[i, key] = math.inf

    port_options = [i for i in conversion_solutions.index if 'H' in i]
    port_options = conversion_solutions.loc[port_options, :]

    port_options = apply_conversion(port_options, 'ports')
    port_options['conversion_possible'] = True

    pipeline_options = [i for i in conversion_solutions.index if 'P' in i]
    pipeline_options = conversion_solutions.loc[pipeline_options, :]

    pipeline_options = apply_conversion(pipeline_options, 'pipelines')
    pipeline_options['conversion_possible'] = True

    start_options = [i for i in conversion_solutions.index if 'Start' in i]
    start_options = conversion_solutions.loc[start_options, :]

    start_options = apply_conversion(start_options, 'start')
    start_options['conversion_possible'] = True

    destination_options = [i for i in conversion_solutions.index if 'Destination' in i]
    destination_options = conversion_solutions.loc[destination_options, :]

    destination_options = apply_conversion(destination_options, 'destination')
    destination_options['conversion_possible'] = True

    columns_to_keep = ['conversion_possible']

    # if offshore, no conversion possible
    for c1 in config_file['available_commodity']:
        for c2 in techno_economic_data_conversion[c1]['potential_conversions']:
            columns_to_keep.append(c1 + '_' + c2 + '_conversion_costs')
            columns_to_keep.append(c1 + '_' + c2 + '_conversion_efficiency')

    locations = pd.concat([port_options, pipeline_options, start_options, destination_options])

    locations = locations[columns_to_keep]

    return locations
