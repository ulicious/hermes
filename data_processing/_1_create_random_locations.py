import math
import random
import os
import yaml

from global_land_mask import globe

import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from _0_helpers_raw_data_processing import calculate_conversion_costs
from algorithm.methods_geographic import calc_distance_list_to_single

import warnings
warnings.filterwarnings("ignore")


def randlatlon1(min_latitude, max_latitude, min_longitude, max_longitude):
    def is_within_boundaries(lat, lon):
        return min_latitude <= lat <= max_latitude and min_longitude <= lon <= max_longitude

    while True:
        pi = math.pi
        cf = 180.0 / pi  # radians to degrees Correction Factor

        # get a random Gaussian 3D vector:
        gx = random.gauss(0.0, 1.0)
        gy = random.gauss(0.0, 1.0)
        gz = random.gauss(0.0, 1.0)

        # normalize to an equidistributed (x,y,z) point on the unit sphere:
        norm2 = gx*gx + gy*gy + gz*gz
        norm1 = 1.0 / math.sqrt(norm2)
        x = gx * norm1
        y = gy * norm1
        z = gz * norm1

        radLat = math.asin(z)      # latitude in radians
        radLon = math.atan2(y, x)   # longitude in radians

        Lat = round(cf*radLat, 5)
        Lon = round(cf*radLon, 5)

        if is_within_boundaries(Lat, Lon):
            if globe.is_land(Lat, Lon):
                return Lon, Lat


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


def apply_conversion():

    def conversion_script(start_commodity, target_commodity):
        electricity_demand = techno_economic_data_conversion[start_commodity][target_commodity]['electricity_demand']
        co2_demand = techno_economic_data_conversion[start_commodity][target_commodity]['co2_demand']
        nitrogen_demand = techno_economic_data_conversion[start_commodity][target_commodity]['nitrogen_demand']
        heat_demand = techno_economic_data_conversion[start_commodity][target_commodity]['heat_demand']

        specific_investment = techno_economic_data_conversion[start_commodity][target_commodity]['specific_investment']
        fixed_maintenance = techno_economic_data_conversion[start_commodity][target_commodity]['fixed_maintenance']
        lifetime = techno_economic_data_conversion[start_commodity][target_commodity]['lifetime']
        operating_hours = techno_economic_data_conversion[start_commodity][target_commodity]['operating_hours']

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

    interest_rate = techno_economic_data_conversion['uniform_interest_rate']

    hydrogen_costs = locations['Hydrogen_Gas']

    # some support materials are not necessarily needed -> 0 costs
    if 'Electricity' in [*config_file['cost_type'].keys()]:
        electricity_costs = locations['Electricity']
    else:
        electricity_costs = 0

    if 'Nitrogen' in [*config_file['cost_type'].keys()]:
        nitrogen_costs = locations['Nitrogen']
    else:
        nitrogen_costs = 0

    if 'CO2' in [*config_file['cost_type'].keys()]:
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


path_config = os.path.dirname(os.getcwd()) + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_techno_economic_data = config_file['paths']['project_folder'] + config_file['paths']['raw_data']

if not config_file['use_minimal_example']:
    # use boundaries from config file
    minimal_latitude = config_file['minimal_latitude']
    maximal_latitude = config_file['maximal_latitude']
    minimal_longitude = config_file['minimal_longitude']
    maximal_longitude = config_file['maximal_longitude']
else:
    # if minimal example, set boundaries to Europe
    minimal_latitude, maximal_latitude = 35, 71
    minimal_longitude, maximal_longitude = -25, 45

levelized_costs_location = pd.read_csv(path_techno_economic_data + 'levelized_costs_locations.csv', index_col=0)
levelized_costs_country = pd.read_csv(path_techno_economic_data + 'levelized_costs_countries.csv', index_col=0)

yaml_file = open(path_techno_economic_data + 'techno_economic_data_conversion.yaml')
techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
world = gpd.read_file(country_shapefile)

origin_continents = config_file['origin_continents']
if config_file['use_minimal_example']:  # overwrite origin continent if minimal example
    origin_continents = ['Europe']

i = 0
processed_coords = []
locations = pd.DataFrame()
while i < config_file['number_locations']:

    restart = False

    coords = randlatlon1(minimal_latitude, maximal_latitude, minimal_longitude, maximal_longitude)

    if coords is not None:

        start_lat = coords[1]
        start_lon = coords[0]

        new_location = pd.DataFrame([start_lon, start_lat]).transpose()
        new_location.columns = ['longitude', 'latitude']

        # add country information to options
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([start_lon], [start_lat])).set_crs('EPSG:4326')
        result = gpd.sjoin(gdf, world, how='left')
        country_start = result.at[result.index[0], 'NAME_EN']
        continent_start = result.at[result.index[0], 'CONTINENT']

        if origin_continents:
            if continent_start not in origin_continents:
                continue

        adjusted_latitude = round_to_quarter(start_lat)
        adjusted_longitude = round_to_quarter(start_lon)
        adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

        # get cost of location
        for c in [*config_file['cost_type'].keys()]:

            cost_type = config_file['cost_type'][c]

            if cost_type == 'uniform':
                locations.loc[i, c] = techno_economic_data_conversion['uniform_costs'][c]
            elif cost_type == 'country':
                if country_start not in levelized_costs_country.index:
                    try:
                        locations.loc[i, c] = levelized_costs_country.loc[country_start, c]
                    except IndexError:
                        print(country_start + ' is not in levelized costs country file')
            else:

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
                    locations.loc[i, c] = levelized_costs_location.loc[idxmin, c]

                else:
                    restart = True
                    break

        if restart:
            continue

        locations.loc[i, 'country_start'] = country_start
        locations.loc[i, 'continent_start'] = continent_start

        locations.loc[i, 'longitude'] = start_lon
        locations.loc[i, 'latitude'] = start_lat

        i += 1

apply_conversion()

columns_to_keep = ['longitude', 'latitude', 'country_start', 'continent_start'] + config_file['available_commodity']
locations = locations[columns_to_keep]

locations.to_excel(config_file['paths']['project_folder'] + 'start_destination_combinations_500.xlsx')
