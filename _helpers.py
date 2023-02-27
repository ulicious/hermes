import pandas as pd
import geopandas as gpd
import shapely.wkt
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim
import pycountry_convert as pc
import country_converter as coco
import json
import ast
from geopy.distance import geodesic
from math import cos, sin, asin, sqrt, radians
import math
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




def str_to_coord_point(coordinates_str):
    # Koordinaten in geometrische Daten umwandeln
    coordinates_str = coordinates_str.replace(",", "")
    coordinates_str = coordinates_str.replace("(", "")
    coordinates_str = coordinates_str.replace(")", "")
    coordinates_split = coordinates_str.split()
    latitude = float(coordinates_split[0])
    longitude = float(coordinates_split[1])
    return Point(longitude, latitude)

def get_input_data(origin_coordinates_str):
    # Inputdaten aus Excel einlesen
    input_data = pd.read_excel("input_data/Python_Input_Data.xlsx")
    origin_coordinates = str_to_coord_point(str(origin_coordinates_str))
    # Koordinaten von Zielstandort in geometrische Daten umwandeln
    destination_coordinates_str = input_data.at[0, "Zielort Koordinaten"]
    destination_coordinates = str_to_coord_point(destination_coordinates_str)

    # Land aus Koordinaten von Produktionsstandort ermitteln
    geolocator = Nominatim(user_agent="transportation_tool")
    origin_coordinates_str = origin_coordinates_str.replace("(", "")
    origin_coordinates_str = origin_coordinates_str.replace(")", "")
    i = 0

    successful = False
    while not successful:
        if i < 5:
            try:
                location_address_origin = geolocator.reverse(origin_coordinates_str, exactly_one=True, language='en')
                successful = True
            except Exception:
                i = i + 1
                pass

    city_start = location_address_origin.raw.get('address').get('city')
    state_start = location_address_origin.raw.get('address').get('state')
    country_name_start = location_address_origin.raw.get('address').get('country')
    if city_start is not None:
        location_start = city_start + ', ' + country_name_start
    elif state_start is not None:
        location_start = state_start + ', ' + country_name_start
    else:
        location_start = country_name_start

    print('country name start: ', location_start)
    print()

    # Land aus Koordinaten von Zielstandort ermitteln
    '''
    successful = False
    while not successful:
        try:
            location_address_destination = geolocator.reverse(destination_coordinates_str, exactly_one=True, language='en')
            successful = True
        except Exception:
            print('not successful destination')
            pass
    city_dest = location_address_destination.raw.get('address').get('city')
    state_dest = location_address_destination.raw.get('address').get('state')
    country_name_dest = location_address_destination.raw.get('address').get('country')
    if city_dest is not None:
        location_dest = city_dest + ', ' + country_name_dest
    elif state_dest is not None:
        location_dest = state_dest + ', ' + country_name_dest
    else:
        location_dest = country_name_dest
    print(location_dest)
    '''

    # standartmäßig Hamburg ausgewählt
    location_dest = 'Hamburg, Germany'
    country_name_dest = 'Germany'

    # Kontinent ermitteln
    def country_to_continent(country_name_start):
        country_name = coco.convert(country_name_start, to='ISO3')
        try:
            country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        except Exception:
            try:
                country_name = coco.convert(country_name_start, to='ISO2')
                country_alpha2 = pc.country_name_to_country_alpha2(country_name)
            except Exception:
                country_name = coco.convert(country_name_start, to='name_short')
                country_alpha2 = pc.country_name_to_country_alpha2(country_name)

        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name

    # Zuweisung Kontinent zu Land
    if country_name_start == 'Sahrawi Arab Democratic Republic':
        continent = 'Africa'
    else:
        continent = country_to_continent(country_name_start)

    # Inputdaten Medium
    media_list = pd.DataFrame(input_data['Kürzel'])
    media_list['Aggregatzustand'] = input_data['Aggregatzustand']
    media_list = media_list.drop(media_list[media_list.Kürzel == 'not defined'].index)
    media_list = media_list.to_string(index=False)
    print(media_list)
    medium = input_data.at[0, "Energieträger"]
    # conversion_capacity = input_data.at[0, "Anlagenleistung [MW]"]
    for i, abb in enumerate(input_data.Kürzel):
        if abb == medium:
            name = input_data.Name[i]
            state_of_aggregation = input_data.Aggregatzustand[i]
            break

    # Ausgabe Inputdaten
    print("\n")
    print("Programm Inputdaten:", "\n")
    print("Produktionsstandort:                    ", f"{origin_coordinates_str:>30}")
    # print(location_address_origin.address, '\n')
    print("Zielstandort:                           ", f"{destination_coordinates_str:>30}")
    # print(location_address_destination.address,  '\n')
    print("gewünschter Energieträger:              ", f"{name:>30}")
    print("Aggregatzustand:                        ", f"{state_of_aggregation:>30}")
    # print("zu transportierende Energiemenge [MWh]: ", f"{conversion_capacity:>30}")
    print("\n")

    data_stored = {'origin coordinates': str(origin_coordinates), 'destination coordinates': str(destination_coordinates),
                   'country name': country_name_start, 'continent': continent, 'state of aggregation': state_of_aggregation,
                   'media': media_list, 'medium': medium, 'location_dest': location_dest, 'location_start': location_start, 'country_start': country_name_start, 'country_dest': country_name_dest}

    with open('data_stored.txt', 'w') as ds:
        json.dump(data_stored, ds, indent=4)

    return data_stored


def get_continent(coordinates):
    # Land aus Koordinaten von Produktionsstandort ermitteln
    geolocator = Nominatim(user_agent="transportation_tool")
    successful = False
    while not successful:
        try:
            location_address_origin = geolocator.reverse(coordinates, exactly_one=True, language='en')
            successful = True
        except Exception:
            pass

    address = location_address_origin.raw['address']
    country_name = address.get('country')

    print('country_name: ',country_name)
    if country_name == 'Sahrawi Arab Democratic Republic':
        country_name = 'Western Sahara'
        return 'Africa'

    # Kontinent ermitteln
    def country_to_continent(country_name_start):
        country_name = coco.convert(country_name_start, to='ISO3')
        try:
            country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        except Exception:
            try:
                country_name = coco.convert(country_name_start, to='ISO2')
                country_alpha2 = pc.country_name_to_country_alpha2(country_name)
            except Exception:
                country_name = coco.convert(country_name_start, to='name_short')
                country_alpha2 = pc.country_name_to_country_alpha2(country_name)

        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name

    # Zuweisung Kontinent zu Land
    continent = country_to_continent(country_name)
    return continent

def coordinates_to_point(coordinates):
    coordinates_str = str(coordinates)
    coordinates_str = coordinates_str.replace(",", "")
    coordinates_str = coordinates_str.replace("(", "")
    coordinates_str = coordinates_str.replace(")", "")
    coordinates_split = coordinates_str.split()
    latitude_origin = float(coordinates_split[0])
    longitude_origin = float(coordinates_split[1])
    point = Point(longitude_origin, latitude_origin)
    return point

def point_to_coordinates(point):
    pnt = gpd.GeoSeries([shapely.wkt.loads(str(point))])
    pnt_longitude = float(pnt.apply(lambda p: p.x).to_string(index=False))
    pnt_latitude = float(pnt.apply(lambda p: p.y).to_string(index=False))
    coorinates = (pnt_latitude, pnt_longitude)
    return coorinates

def switch_lat_lon(coordinates):
    coordinates_str = str(coordinates)
    coordinates_str = coordinates_str.replace(",", "")
    coordinates_str = coordinates_str.replace("(", "")
    coordinates_str = coordinates_str.replace(")", "")
    coordinates_split = coordinates_str.split()
    latitude = float(coordinates_split[0])
    longitude = float(coordinates_split[1])
    coordinates_switched = longitude, latitude
    return coordinates_switched

def switch_lat_lon_list(coordinates):

    # from lat, lon to lon, lat

    coordinates_str = str(coordinates)
    coordinates_str = coordinates_str.replace(",", "")
    coordinates_str = coordinates_str.replace("(", "")
    coordinates_str = coordinates_str.replace(")", "")
    coordinates_split = coordinates_str.split()
    latitude = float(coordinates_split[0])
    longitude = float(coordinates_split[1])
    coordinates_switched = [longitude, latitude]
    return coordinates_switched


# Pipelinedaten aus Excel einlesen
gas_infrastructure = pd.read_excel("input_data/network_pipelines_gas.xlsx")
oil_infrastructure = pd.read_excel("input_data/network_pipelines_oil.xlsx")


def filter_infrastructure_data(start_coordinates, state_of_aggregation):
    continent = get_continent(start_coordinates)
    # Zuweisung Pipelinedaten Gas oder Öl
    if state_of_aggregation == 'gasförmig':
        infrastructure_data = gas_infrastructure
    elif state_of_aggregation == 'flüssig':
        infrastructure_data = oil_infrastructure

    # Filtern nach Region um Rechenzeit zu verkürzen
    if continent == 'North America':
        infrastructure_data_filtered = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'North America']
    elif continent == 'Europe':
        df1 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Europe']
        df2 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Eurasia']
        infrastructure_data_filtered = pd.concat([df1, df2], ignore_index=False)
    elif continent == 'South America':
        infrastructure_data_filtered = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Latin America and the Caribbean']
    elif continent == 'Asia':
        df1 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'East Asia']
        df2 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'SE Asia']
        df3 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'South Asia']
        df4 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Middle East and North Africa']
        df5 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Eurasia']
        infrastructure_data_filtered = pd.concat([df1, df2, df3, df4, df5], ignore_index=False)
    elif continent == 'Africa':
        df1 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Sub-Saharan Africa']
        df2 = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Middle East and North Africa']
        infrastructure_data_filtered = pd.concat([df1, df2], ignore_index=False)
    elif continent == 'Australia' or 'New Zealand' or 'Oceania':
        infrastructure_data_filtered = infrastructure_data.loc[infrastructure_data['StartRegion'] == 'Australia and New Zealand']
    return infrastructure_data_filtered


def pipeline_connection_point(pipeline_geodata, next_pipeline_geodata):
    distances = []
    lines = []
    if pipeline_geodata.intersects(next_pipeline_geodata):
        intersection_pts = pipeline_geodata.intersection(next_pipeline_geodata)
        try:
            if intersection_pts.geom_type == 'LineString':
                pipe_pt = Point(intersection_pts.coords[0])
            elif intersection_pts.geoms[0].geom_type == 'LineString':
                pipe_pt = Point(intersection_pts.geoms[0].coords[0])
            else:
                pipe_pt = intersection_pts.geoms[0]
        except AttributeError:  # nur ein Punkt
            pipe_pt = intersection_pts
    else:
        if pipeline_geodata.geom_type == 'MultiLineString':
            for line1 in pipeline_geodata.geoms:  # einzelne Lines in Multilinestring
                if next_pipeline_geodata.geom_type == 'MultiLineString':
                    for line2 in next_pipeline_geodata.geoms:  # einzelne Lines in Multilinestring
                        for e in range(len(line2.coords)):
                            line2pt = Point(line2.coords[e])
                            line2ptc = point_to_coordinates(line2pt)
                            line1pt = line1.interpolate(line1.project(line2pt))
                            line1ptc = point_to_coordinates(line1pt)
                            dist = geodesic(line1ptc, line2ptc).km
                            distances.append(dist)
                            lines.append(line2)
                    min_dist_pipes = min(distances)
                    index = distances.index(min_dist_pipes)
                    pipeline2 = lines[index]
                    for c in range(len(pipeline2.coords)):
                        pt = Point(pipeline2.coords[c])
                        pipe_pt = line1.interpolate(line1.project(pt))
                else:
                    for c in range(len(next_pipeline_geodata.coords)):  # einzelne Punkte in Linestrings
                        pt = Point(next_pipeline_geodata.coords[c])
                        pipe_pt = line1.interpolate(line1.project(pt))  # Punkt auf Pipeline2 mit geringster Entfernung zu Pipeline1

        else:
            if next_pipeline_geodata.geom_type == 'MultiLineString':
                for line in next_pipeline_geodata.geoms:  # einzelne Lines in Multilinestring
                    for c in range(len(line.coords)):  # einzelne Punkte in Linestrings
                        pt = Point(line.coords[c])
                        pipe_pt = pipeline_geodata.interpolate(pipeline_geodata.project(pt))  # Punkt auf Pipeline1 mit geringster Entfernung zu Pipeline2

            else:
                for c in range(len(next_pipeline_geodata.coords)):
                    pt = Point(next_pipeline_geodata.coords[c])
                    pipe_pt = pipeline_geodata.interpolate(pipeline_geodata.project(pt))  # Punkt auf Pipeline1 mit geringster Entfernung zu Pipeline2

    return pipe_pt


def str_to_list(list_str):
    list_str = list_str.replace('[', '')
    list_str = list_str.replace(']', '')
    list_str = list_str.replace('"', "")
    list_str = list_str.split(', ')
    list_new = []
    for pipe in range(0, len(list_str)):
        pipeline = ast.literal_eval(list_str[pipe])
        list_new.append(pipeline)
    return list_new
