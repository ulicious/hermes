import math
import random
import time

from global_land_mask import globe

import pandas as pd
from shapely.geometry import Point
# import country_converter as coco
# import countries
import pycountry_convert as pc
from geopy.geocoders import Nominatim


def randlatlon1():
    def is_in_europe(lat, lon):
        # Approximate bounds for Europe
        min_latitude, max_latitude = 35, 71
        min_longitude, max_longitude = -25, 45

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

        if is_in_europe(Lat, Lon):
            if globe.is_land(Lat, Lon):
                return Lon, Lat


def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

geolocator = Nominatim(user_agent="transportation_tool")

destination_lat = 53.551086
destination_lon = 9.993682

country_destination = geolocator.reverse(str(destination_lat) + "," + str(destination_lon),
                                         language='en').raw['address'].get('country', '')
continent_destination = country_to_continent(country_destination)

coords_df = pd.DataFrame(columns=['start_lat', 'start_lon', 'destination_lat', 'destination_lon', 'target_commodity',
                                  'Hydrogen_Gas'])

i = 0
while i < 500:

    coords = randlatlon1()

    if coords is not None:

        start_lat = coords[1]
        start_lon = coords[0]

        try:

            country_start = geolocator.reverse(str(start_lat) + "," + str(start_lon),
                                               language='en').raw['address'].get('country', '')
            continent_start = country_to_continent(country_start)

            if continent_start != 'Europe':
                continue

        except:
            continue

        coords_df.loc[i, 'country_start'] = country_start
        coords_df.loc[i, 'continent_start'] = continent_start

        coords_df.loc[i, 'country_destination'] = country_destination
        coords_df.loc[i, 'continent_destination'] = continent_destination

        coords_df.loc[i, 'start_lon'] = start_lon
        coords_df.loc[i, 'start_lat'] = start_lat

        coords_df.loc[i, 'destination_lat'] = destination_lat
        coords_df.loc[i, 'destination_lon'] = destination_lon
        coords_df.loc[i, 'target_commodity'] = 'Hydrogen_Gas,Hydrogen_Liquid,Ammonia,Methanol,Methane_Gas,Methane_Liquid,FTF'
        coords_df.loc[i, 'Hydrogen_Gas'] = 0

        i += 1

    time.sleep(1)

coords_df.to_excel('C:/Users/mt5285/Desktop/Transportmodel/Daten/start_data/start_destination_combinations_500.xlsx')
