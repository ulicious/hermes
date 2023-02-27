import pandas as pd
from shapely.geometry import Point
# import country_converter as coco
# import countries
import pycountry_convert as pc
from geopy.geocoders import Nominatim


# Start location, destination, target product


# Production costs at start locations


# Conversion costs of commodities


# Transportation costs of commodities


def get_start_destination_combinations(location_data):

    def country_to_continent(country_name):
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name

    start_destination_combinations = []
    target_commodity_list = []
    production_costs = []
    country_start_list = []
    continent_start_list = []
    country_destination_list = []
    continent_destination_list = []

    geolocator = Nominatim(user_agent="geoapiExercises")

    for ind in location_data.index:
        target_commodity_list.append(location_data.loc[ind, 'target_commodity'])

        start_lat = float(location_data.loc[ind, 'start_lat'])
        start_lon = float(location_data.loc[ind, 'start_lon'])
        start = Point(start_lon, start_lat)

        country_start = geolocator.reverse(str(start_lat) + "," + str(start_lon),
                                           language='en').raw['address'].get('country', '')
        country_start_list.append(country_start)

        continent_start = country_to_continent(country_start)
        continent_start_list.append(continent_start)

        destination_lat = float(location_data.loc[ind, 'destination_lat'])
        destination_lon = float(location_data.loc[ind, 'destination_lon'])
        destination = Point(destination_lon, destination_lat)

        country_destination = geolocator.reverse(str(destination_lat) + "," + str(destination_lon),
                                           language='en').raw['address'].get('country', '')
        country_destination_list.append(country_destination)

        continent_destination = country_to_continent(country_destination)
        continent_destination_list.append(continent_destination)

        start_destination_combinations.append((start, destination))

        commodities = [i for i in location_data.columns if i not in ['start_lon', 'start_lat', 'destination_lat',
                                                                     'destination_lon', 'target_commodity']
                       if i != 'N/A']

        production_costs_dict = {}
        for c in commodities:
            production_costs_dict[c] = float(location_data.loc[ind, c])

        production_costs.append(production_costs_dict)

    return start_destination_combinations, target_commodity_list, production_costs, country_start_list,\
        continent_start_list, country_destination_list, continent_destination_list
