import os
import tqdm

import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from data_processing._0_helpers_raw_data_processing import calculate_conversion_costs
from algorithm.methods_geographic import calc_distance_list_to_single


def attach_conversion_costs_and_efficiency_to_locations(options, config_file, techno_economic_data_conversion):

    def round_to_quarter(number):
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

    def apply_conversion(options_to_process, location):

        interest_rate = techno_economic_data_conversion['uniform_interest_rate']

        # some support materials are not necessarily needed -> 0 costs
        if 'Electricity' in [*config_file['cost_type'].keys()]:
            electricity_costs = options_to_process['Electricity']
        else:
            electricity_costs = 0

        if 'Nitrogen' in [*config_file['cost_type'].keys()]:
            nitrogen_costs = options_to_process['Nitrogen']
        else:
            nitrogen_costs = 0

        if 'CO2' in [*config_file['cost_type'].keys()]:
            co2_costs = options_to_process['CO2']
        else:
            co2_costs = 0

        for c1 in config_file['available_commodity']:
            for c2 in techno_economic_data_conversion[c1]['potential_conversions']:
                electricity_demand = techno_economic_data_conversion[c1][c2]['electricity_demand']
                co2_demand = techno_economic_data_conversion[c1][c2]['co2_demand']
                nitrogen_demand = techno_economic_data_conversion[c1][c2]['nitrogen_demand']
                heat_demand = techno_economic_data_conversion[c1][c2]['heat_demand']

                specific_investment = techno_economic_data_conversion[c1][c2]['specific_investment']
                fixed_maintenance = techno_economic_data_conversion[c1][c2]['fixed_maintenance']
                lifetime = techno_economic_data_conversion[c1][c2]['lifetime']
                operating_hours = techno_economic_data_conversion[c1][c2]['operating_hours']

                if heat_demand == 0:
                    conversion_costs \
                        = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                     operating_hours, interest_rate,
                                                     electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                     nitrogen_costs, nitrogen_demand)

                    conversion_efficiency = techno_economic_data_conversion[c1][c2]['efficiency_autothermal']

                else:

                    heat_demand_niveau = techno_economic_data_conversion[c1][c2]['heat_demand_niveau']

                    if (heat_demand_niveau == 'low_temperature') & config_file['low_temp_heat_available_at_' + location]:
                        heat_costs = options_to_process['Low_Temperature_Heat']

                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)

                        conversion_efficiency = techno_economic_data_conversion[c1][c2]['efficiency_external_heat']

                    elif (heat_demand_niveau == 'mid_temperature') & config_file['mid_temp_heat_available_at_' + location]:
                        heat_costs = options_to_process['Mid_Temperature_Heat']

                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)
                        conversion_efficiency = techno_economic_data_conversion[c1][c2]['efficiency_external_heat']

                    elif (heat_demand_niveau == 'high_temperature') & config_file['high_temp_heat_available_at_' + location]:
                        heat_costs = options_to_process['High_Temperature_Heat']

                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand, heat_demand, heat_costs)

                        conversion_efficiency = techno_economic_data_conversion[c1][c2]['efficiency_external_heat']

                    else:
                        conversion_costs \
                            = calculate_conversion_costs(specific_investment, lifetime, fixed_maintenance,
                                                         operating_hours, interest_rate,
                                                         electricity_costs, electricity_demand, co2_costs, co2_demand,
                                                         nitrogen_costs, nitrogen_demand)

                        conversion_efficiency = techno_economic_data_conversion[c1][c2]['efficiency_autothermal']

                options_to_process[c1 + '_' + c2 + '_conversion_costs'] = conversion_costs
                options_to_process[c1 + '_' + c2 + '_conversion_efficiency'] = conversion_efficiency

        return options_to_process

    path_raw_data = config_file['paths']['project_folder'] + config_file['paths']['raw_data']

    levelized_costs_location = pd.read_csv(path_raw_data + 'levelized_costs_locations.csv', index_col=0)
    new_index = [i.split('_')[1].split('c')[1] for i in levelized_costs_location.index]
    levelized_costs_location.index = new_index
    levelized_costs_location.sort_index(inplace=True)
    levelized_costs_location = levelized_costs_location[~levelized_costs_location.index.duplicated(keep='first')]

    levelized_costs_location['longitude'] = [int(i.split('x')[0]) / 100 for i in levelized_costs_location.index]
    levelized_costs_location['latitude'] = [int(i.split('x')[1]) / 100 for i in levelized_costs_location.index]

    levelized_costs_country = pd.read_csv(path_raw_data + 'levelized_costs_countries.csv', index_col=0)

    # add country information to options
    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)
    gdf = gpd.GeoDataFrame(options, geometry=gpd.points_from_xy(options.longitude, options.latitude))
    result = gpd.sjoin(gdf, world, how='left')

    options['country'] = result['NAME_EN']
    no_country_options = options[options['country'].isna()]
    country_options = options[~options['country'].isna()]

    # get cost of location
    new_nan_values = []
    for c in [*config_file['cost_type'].keys()]:

        if c == 'Hydrogen_Gas':
            # hydrogen is not needed since commodity is transported to i
            continue

        cost_type = config_file['cost_type'][c]

        if cost_type == 'uniform':
            country_options[c] = techno_economic_data_conversion['uniform_costs'][c]

        elif cost_type == 'country':
            for country in country_options['country'].unique():
                if country not in levelized_costs_country.index:
                    new_nan_values.append(country_options.index)

                country_options[c] = levelized_costs_country.loc[country, c]

    for i in tqdm.tqdm(country_options.index):

        if 'H' in i:
            adjusted_latitude = round_to_quarter(country_options.loc[i, 'latitude_on_coastline'])
            adjusted_longitude = round_to_quarter(country_options.loc[i, 'longitude_on_coastline'])
        else:
            adjusted_latitude = round_to_quarter(country_options.loc[i, 'latitude'])
            adjusted_longitude = round_to_quarter(country_options.loc[i, 'longitude'])

        # if is on land, continue to add information on commodities
        adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

        # get cost of location
        for c in [*config_file['cost_type'].keys()]:

            if c == 'Hydrogen_Gas':
                # hydrogen is not needed since commodity is transported to i
                continue

            if config_file['cost_type'][c] != 'location':
                continue

            if adjusted_coords in levelized_costs_location.index:
                country_options.loc[i, c] = levelized_costs_location.loc[adjusted_coords, c]
            else:

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
                    country_options.loc[i, c] = levelized_costs_location.loc[idxmin, c]

                else:
                    new_nan_values.append(i)

    new_nan_solutions = country_options.loc[new_nan_values, :]
    no_conversion_solutions = pd.concat([no_country_options, new_nan_solutions])
    no_conversion_solutions['conversion_possible'] = False

    conversion_solutions = [i for i in country_options.index if i not in new_nan_values]
    conversion_solutions = country_options.loc[conversion_solutions, :]

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
            no_conversion_solutions[c1 + '_' + c2 + '_conversion_costs'] = np.nan
            no_conversion_solutions[c1 + '_' + c2 + '_conversion_efficiency'] = np.nan

            columns_to_keep.append(c1 + '_' + c2 + '_conversion_costs')
            columns_to_keep.append(c1 + '_' + c2 + '_conversion_efficiency')

    options = pd.concat([port_options, pipeline_options, start_options, destination_options, no_conversion_solutions])

    options = options[columns_to_keep]

    return options
