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

    geolocator = Nominatim(user_agent="geoapiExercises")

    location_data_gf = {}
    for ind in location_data.index:

        try:

            target_commodity = location_data.loc[ind, 'target_commodity']

            start_lat = float(location_data.loc[ind, 'start_lat'])
            start_lon = float(location_data.loc[ind, 'start_lon'])
            start = Point(start_lon, start_lat)

            country_start = geolocator.reverse(str(start_lat) + "," + str(start_lon),
                                               language='en').raw['address'].get('country', '')

            continent_start = country_to_continent(country_start)

            destination_lat = float(location_data.loc[ind, 'destination_lat'])
            destination_lon = float(location_data.loc[ind, 'destination_lon'])
            destination = Point(destination_lon, destination_lat)

            country_destination = geolocator.reverse(str(destination_lat) + "," + str(destination_lon),
                                                     language='en').raw['address'].get('country', '')

            continent_destination = country_to_continent(country_destination)

            commodities = [i for i in location_data.columns if i not in ['start_lon', 'start_lat', 'destination_lat',
                                                                         'destination_lon', 'target_commodity']
                           if i != 'N/A']

            production_costs = {}
            for c in commodities:
                production_costs[c] = float(location_data.loc[ind, c])

            location_data_gf[ind] = {'start_destination_combinations': (start, destination),
                                     'target_commodity': target_commodity,
                                     'production_costs': production_costs,
                                     'country_start': country_start,
                                     'continent_start': continent_start,
                                     'country_destination': country_destination,
                                     'continent_destination': continent_destination}

        except NameError:
            continue

    return location_data_gf
