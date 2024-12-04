import random
import math

import matplotlib.pyplot as plt
import shapely

import geopandas as gpd
import pandas as pd

from shapely.geometry import Polygon, Point

import tqdm
import math

import numpy as np
import cartopy.io.shapereader as shpreader

from shapely.geometry import LineString, Point

from algorithm.methods_geographic import calc_distance_list_to_single
from data_processing.helpers_geometry import round_to_quarter


def check_if_location_is_valid(techno_economic_data_conversion, country_start, adjusted_latitude,
                               adjusted_longitude, levelized_costs_location, levelized_costs_country):

    """
    Checks if location is valid based on used data and configuration

    @return: boolean
    """

    # get cost of location
    for key in [*techno_economic_data_conversion['cost_type'].keys()]:

        cost_type = techno_economic_data_conversion['cost_type'][key]

        if key == 'interest_rate':
            if cost_type == 'location':
                raise TypeError('Interest rate cannot be location specific')

        if cost_type == 'uniform':
            return True

        elif cost_type == 'all_countries':
            if country_start in levelized_costs_country.index.tolist():
                return True
            else:
                raise IndexError(country_start + ' is not in levelized costs country file')

        elif isinstance(cost_type, list):
            # cost type is list of countries. Countries not in list will be treated as locations
            if country_start in cost_type:
                if country_start in levelized_costs_country.index:
                    return True
                else:
                    raise IndexError(country_start + ' is not in country file')

            elif key == 'interest_rate':  # interest rate is never location specific
                return True

            else:

                # apply small grid search to get the closest location
                sub_locations = levelized_costs_location[
                    (levelized_costs_location['latitude'] <= adjusted_latitude + 0.5) &
                    (levelized_costs_location['latitude'] >= adjusted_latitude - 0.5) &
                    (levelized_costs_location['longitude'] <= adjusted_longitude + 0.5) &
                    (levelized_costs_location['longitude'] >= adjusted_longitude - 0.5)]

                if not sub_locations.empty:
                    return True
                else:
                    return False

        else:  # search costs for location

            # apply small grid search to get the closest location
            sub_locations = levelized_costs_location[
                (levelized_costs_location['latitude'] <= adjusted_latitude + 0.5) &
                (levelized_costs_location['latitude'] >= adjusted_latitude - 0.5) &
                (levelized_costs_location['longitude'] <= adjusted_longitude + 0.5) &
                (levelized_costs_location['longitude'] >= adjusted_longitude - 0.5)]

            if not sub_locations.empty:
                return True
            else:
                return False


def attach_feedstock_costs_and_interest_rate(i, locations, techno_economic_data_conversion, country_start,
                                             adjusted_latitude, adjusted_longitude, levelized_costs_location,
                                             levelized_costs_country, config_file, spatial_index=None):

    """
    attaches feedstock costs and interest rate to locations

    @return: returns location dataframe with updated information
    """

    # get cost of location
    for key in [*techno_economic_data_conversion['cost_type'].keys()]:

        cost_type = techno_economic_data_conversion['cost_type'][key]

        if key == 'interest_rate':
            if cost_type == 'location':
                raise TypeError('Interest rate cannot be location specific')

        if cost_type == 'uniform':
            locations.loc[i, key] = techno_economic_data_conversion['uniform_costs'][key]

        elif cost_type == 'all_countries':
            if country_start in levelized_costs_country.index.tolist():
                try:
                    locations.loc[i, key] = levelized_costs_country.loc[country_start, key]
                except IndexError:
                    print(country_start + ' is not in levelized costs country file')
            else:
                raise IndexError(country_start + ' is not in levelized costs country file')

        elif isinstance(cost_type, list):
            # cost type is list of countries. Countries not in list will be treated as locations
            if country_start in cost_type:
                if country_start in levelized_costs_country.index:
                    try:
                        locations.loc[i, key] = levelized_costs_country.loc[country_start, key]
                    except IndexError:
                        raise IndexError(country_start + ' is not in country file')

            elif key == 'interest_rate':  # interest rate is never location specific
                locations.loc[i, key] = techno_economic_data_conversion['uniform_costs'][key]

            else:

                if (key != 'Hydrogen_Gas') | (not config_file['create_voronoi_cells']):
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
                        locations.loc[i, key] = levelized_costs_location.loc[idxmin, key]
                else:
                    if not config_file['weight_hydrogen_costs_by_quantity']:
                        locations = attach_unweighted_costs(i, locations, levelized_costs_location, spatial_index,
                                                            config_file)
                    else:
                        locations = attach_weighted_costs(i, locations, levelized_costs_location, spatial_index,
                                                          config_file)

        else:  # search costs for location

            if (key != 'Hydrogen_Gas') | (not config_file['create_voronoi_cells']):
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
                    locations.loc[i, key] = levelized_costs_location.loc[idxmin, key]
            else:
                if not config_file['weight_hydrogen_costs_by_quantity']:
                    locations = attach_unweighted_costs(i, locations, levelized_costs_location, spatial_index,
                                                        config_file)
                else:
                    locations = attach_weighted_costs(i, locations, levelized_costs_location, spatial_index,
                                                      config_file)

    return locations


def attach_unweighted_costs(i, locations, hydrogen_costs_and_quantities, spatial_index, config_file):
    # get polygons from era data

    polygons = hydrogen_costs_and_quantities['polygon'].tolist()

    poly = locations.at[i, 'geometry']

    if isinstance(poly, float):  # todo: why are there some floats in the data? All locations should have geometry
        return locations

    # Query the spatial index with the bounding box of the location  # todo: there are some missing
    possible_matches = list(spatial_index.intersection(poly.bounds))

    # Filter to find polygons strictly within the location
    result = [polygons[i] for i in possible_matches if polygons[i].intersects(poly)]

    total_potential = 0
    total_costs = 0
    for n, r in enumerate(result):
        # Calculate the intersection
        intersection = poly.intersection(r)

        # Calculate the overlap percentage
        overlap_percentage = (intersection.area / r.area)

        # weighted only by the area not the potential
        total_potential += overlap_percentage
        total_costs += total_potential * hydrogen_costs_and_quantities.at[n, 'Hydrogen_Gas']

    if total_potential == 0:
        locations.at[i, 'Hydrogen_Gas'] = math.inf
        locations.at[i, 'Hydrogen_Gas_Quantity'] = math.nan

    else:
        average_costs = total_costs / total_potential

        locations.at[i, 'Hydrogen_Gas'] = average_costs
        locations.at[i, 'Hydrogen_Gas_Quantity'] = math.nan

    return locations


def attach_weighted_costs(i, locations, levelized_costs_location, spatial_index, config_file):

    polygons = levelized_costs_location['polygon'].tolist()

    poly = locations.at[i, 'geometry']

    if not poly.is_valid:
        poly = shapely.make_valid(poly)

    if isinstance(poly, float):  # todo: why are there some floats in the data? All locations should have geometry
        return locations

    # Query the spatial index with the bounding box of the location  # todo: there are some missing
    possible_matches = list(spatial_index.intersection(poly.bounds))

    # Filter to find polygons strictly within the location
    result = [polygons[i] for i in possible_matches if polygons[i].intersects(poly)]

    if False:
        affected_polygons = gpd.GeoDataFrame(geometry=result)
        voronoi = gpd.GeoDataFrame(geometry=[poly])

        fig, ax = plt.subplots()

        affected_polygons.plot(ax=ax)
        voronoi.plot(ax=ax, fc='none', ec='red')

        plt.show()

    total_potential = 0
    total_costs = 0
    for n, r in enumerate(result):
        # Calculate the intersection
        intersection = poly.intersection(r)

        # Calculate the overlap percentage
        overlap_percentage = (intersection.area / r.area)

        # weighted by the potential
        quantity = random.random() * 100 * overlap_percentage  # hydrogen_costs_and_quantities.at[n, 'Hydrogen_Gas_Quantity']
        total_potential += quantity
        total_costs += quantity * levelized_costs_location.at[n, 'Hydrogen_Gas']

    if total_potential == 0:
        locations.at[i, 'Hydrogen_Gas'] = math.inf
        locations.at[i, 'Hydrogen_Gas_Quantity'] = 0

    else:
        average_costs = total_costs / total_potential

        locations.at[i, 'Hydrogen_Gas'] = average_costs
        locations.at[i, 'Hydrogen_Gas_Quantity'] = total_potential

    return locations


def calculate_conversion_costs(specific_investment, depreciation_period, fixed_maintenance,
                               operating_hours, interest_rate,
                               electricity_costs, electricity_demand, co2_costs, co2_demand,
                               nitrogen_costs, nitrogen_demand, heat_demand=0, heat_costs=0):

    """
    Uses annuity method, investment, operation and maintenance parameters, and costs of commodity to calculate the
    conversion costs of one commodity to another

    @param float specific_investment:
    @param float depreciation_period:
    @param float fixed_maintenance:
    @param float operating_hours:
    @param float interest_rate:
    @param float electricity_costs:
    @param float electricity_demand:
    @param float co2_costs:
    @param float co2_demand:
    @param float nitrogen_costs:
    @param float nitrogen_demand:
    @param float heat_demand:
    @param float heat_costs:
    @return: conversion costs
    """

    annuity_factor \
        = (interest_rate * (1 + interest_rate) ** depreciation_period) / ((1 + interest_rate) ** depreciation_period - 1)

    conversion_costs \
        = specific_investment * (annuity_factor + fixed_maintenance) / operating_hours \
        + electricity_costs * electricity_demand + co2_costs * co2_demand \
        + nitrogen_costs * nitrogen_demand + heat_demand * heat_costs

    return conversion_costs


def attach_conversion_costs_and_efficiency_to_start_locations(locations, techno_economic_data_conversion, config_file):

    """
    Calculates production costs at locations
    """

    def conversion_script(start_commodity, target_commodity):

        """
        Calculates conversion costs and conversion efficiency for given start and end commodity at location.
        Important: Only one location at each time so all data is location specific

        @param str start_commodity: name of commodity before conversion
        @param str target_commodity: name of commodity after conversion
        @return: returns conversion costs and conversion efficiency
        @rtype: (float, float)
        """

        electricity_demand = techno_economic_data_conversion[start_commodity][target_commodity]['electricity_demand']
        co2_demand = techno_economic_data_conversion[start_commodity][target_commodity]['co2_demand']
        nitrogen_demand = techno_economic_data_conversion[start_commodity][target_commodity]['nitrogen_demand']
        heat_demand = techno_economic_data_conversion[start_commodity][target_commodity]['heat_demand']

        specific_investment = techno_economic_data_conversion[start_commodity][target_commodity]['specific_investment']
        fixed_maintenance = techno_economic_data_conversion[start_commodity][target_commodity]['fixed_maintenance']
        lifetime = techno_economic_data_conversion[start_commodity][target_commodity]['lifetime']
        operating_hours = techno_economic_data_conversion[start_commodity][target_commodity]['operating_hours']

        interest_rate = locations['interest_rate']

        if heat_demand == 0:
            costs \
                = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                             operating_hours, interest_rate,
                                             electricity_costs, electricity_demand, co2_costs, co2_demand,
                                             nitrogen_costs, nitrogen_demand)

            efficiency = techno_economic_data_conversion[start_commodity][target_commodity]['efficiency_autothermal']

        else:

            heat_demand_niveau = techno_economic_data_conversion[start_commodity][target_commodity]['heat_demand_niveau']

            if (heat_demand_niveau == 'low_temperature') & config_file['low_temp_heat_available_at_start']:
                heat_costs = locations['Low_Heat_Temperature']

                costs \
                    = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                 operating_hours, interest_rate,
                                                 electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                 nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)

                efficiency = techno_economic_data_conversion[start_commodity][target_commodity][
                    'efficiency_external_heat']

            elif (heat_demand_niveau == 'mid_temperature') & config_file['mid_temp_heat_available_at_start']:
                heat_costs = locations['Mid_Heat_Temperature']

                costs \
                    = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                 operating_hours, interest_rate,
                                                 electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                 nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)
                efficiency \
                    = techno_economic_data_conversion[start_commodity][target_commodity]['efficiency_external_heat']

            elif (heat_demand_niveau == 'high_temperature') & config_file['high_temp_heat_available_at_start']:
                heat_costs = locations['High_Heat_Temperature']

                costs \
                    = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                 operating_hours, interest_rate,
                                                 electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                 nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)

                efficiency \
                    = techno_economic_data_conversion[start_commodity][target_commodity]['efficiency_external_heat']

            else:
                costs \
                    = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                 operating_hours, interest_rate,
                                                 electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                 nitrogen_costs, nitrogen_demand)

                efficiency \
                    = techno_economic_data_conversion[start_commodity][target_commodity]['efficiency_autothermal']

        return costs, efficiency

    hydrogen_costs = locations['Hydrogen_Gas']

    # some support materials are not necessarily needed -> 0 costs
    if 'Electricity' in [*techno_economic_data_conversion['cost_type'].keys()]:
        electricity_costs = locations['Electricity']
    else:
        electricity_costs = 0

    if 'Nitrogen' in [*techno_economic_data_conversion['cost_type'].keys()]:
        nitrogen_costs = locations['Nitrogen']
    else:
        nitrogen_costs = 0

    if 'CO2' in [*techno_economic_data_conversion['cost_type'].keys()]:
        co2_costs = locations['CO2']
    else:
        co2_costs = 0

    # iterate over all possible conversions from hydrogen and calculate costs and efficiency
    possible_conversions = techno_economic_data_conversion['Hydrogen_Gas']['potential_conversions']
    for commodity in possible_conversions:

        conversion_costs, conversion_efficiency = conversion_script('Hydrogen_Gas', commodity)

        locations[commodity] = (hydrogen_costs + conversion_costs) / conversion_efficiency

        # as direct conversion from hydrogen to some commodities is not possible, we have to apply second conversion
        for commodity_2 in techno_economic_data_conversion[commodity]['potential_conversions']:
            # only applicable if direct conversion from hydrogen to commodity is not possible
            if commodity_2 not in possible_conversions:

                # hydrogen gas will not be processed as it is start
                if commodity_2 == 'Hydrogen_Gas':
                    continue

                conversion_costs_2, conversion_efficiency_2 = conversion_script(commodity, commodity_2)

                new_costs = (locations[commodity] + conversion_costs_2) / conversion_efficiency_2

                # other routes might be possible as well so use the cheapest route
                if commodity_2 in locations.columns:
                    new_costs = pd.concat([new_costs, locations[commodity_2]], axis=1)
                    locations[commodity_2] = new_costs.min(axis=1)
                else:
                    locations[commodity_2] = (locations[commodity] + conversion_costs_2) / conversion_efficiency_2

    return locations


def attach_conversion_costs_and_efficiency_to_infrastructure(locations, config_file, techno_economic_data_conversion,
                                                             with_tqdm=True):

    """
    Iterates over all locations and calculates conversion costs and efficiency at specific location

    @param pandas.DataFrame locations: dataframe with the different locations and levelized costs
    @param dict config_file: dictionary with configuration
    @param dict techno_economic_data_conversion: dictionary with techno economic parameters
    @param bool with_tqdm: boolean to show progress bar (default = True)

    @return: updated dataframe with added columns showing conversion costs and efficiency at each location
    """

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
