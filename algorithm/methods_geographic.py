import math

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from geopandas.tools import sjoin
from math import cos, sin, asin, sqrt, radians
from shapely.geometry import Point


def calc_distance_single_to_single(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    This methods calculates direct distance between two locations
    
    @param float latitude_1: latitude first location
    @param float longitude_1: longitude first location
    @param float latitude_2: latitude second location
    @param float longitude_2: longitude second location
    @return: single direct distance values in meter
    """

    # convert decimal degrees to radians
    longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    # haversine formula
    dlon = longitude_2 - longitude_1
    dlat = latitude_2 - latitude_1
    a = sin(dlat / 2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 6371 * c * 1000

    return np.ceil(m)


def calc_distance_list_to_single(latitude_list_1, longitude_list_1, latitude_2, longitude_2):
    """
    This method calculates the direct distance between a single location and an array of locations

    @param pandas.DataFrame latitude_list_1: latitudes of start locations
    @param pandas.DataFrame longitude_list_1: longitude of start locations
    @param float latitude_2: latitude of destination
    @param float longitude_2: longitude of destination
    @return: array of direct distances in meter
    """

    # convert decimal degrees to radians
    latitude_list_1 = latitude_list_1.to_numpy(dtype=np.float32)
    longitude_list_1 = longitude_list_1.to_numpy(dtype=np.float32)

    longitude_list_1 = np.radians(longitude_list_1)
    latitude_list_1 = np.radians(latitude_list_1)
    longitude_2, latitude_2 = map(radians, [longitude_2, latitude_2])

    # haversine formula
    dlon = longitude_2 - longitude_list_1
    dlat = latitude_2 - latitude_list_1
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude_list_1) * np.cos(latitude_2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000

    return np.ceil(m)


def calc_distance_list_to_list_no_matrix(latitude_list_1, longitude_list_1, latitude_list_2, longitude_list_2):
    """
    This method calculates the direct distance between two arrays of locations

    @param pandas.DataFrame latitude_list_1: latitudes of starting locations
    @param pandas.DataFrame longitude_list_1: longitudes of starting locations
    @param pandas.DataFrame latitude_list_2: latitudes of destination locations
    @param pandas.DataFrame longitude_list_2: longitudes of destination locations
    @return: dataframe of direct distances in meter. Important: list-like not array
    """

    latitude_list_1 = latitude_list_1.to_numpy(dtype=np.float32)
    longitude_list_1 = longitude_list_1.to_numpy(dtype=np.float32)
    latitude_list_2 = latitude_list_2.to_numpy(dtype=np.float32)
    longitude_list_2 = longitude_list_2.to_numpy(dtype=np.float32)

    # convert decimal degrees to radians
    longitude_list_1 = np.radians(longitude_list_1)
    latitude_list_1 = np.radians(latitude_list_1)
    longitude_list_2 = np.radians(longitude_list_2)
    latitude_list_2 = np.radians(latitude_list_2)

    # haversine formula
    dlon = longitude_list_2 - longitude_list_1
    dlat = latitude_list_2 - latitude_list_1
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude_list_1) * np.cos(latitude_list_2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000

    return np.ceil(m)


def calc_distance_list_to_list(latitude_list_1, longitude_list_1, latitude_list_2, longitude_list_2):
    """
    This method calculates the direct distances between two lists of coordinates.

    @param pandas.DataFrame latitude_list_1: latitudes of starting locations
    @param pandas.DataFrame longitude_list_1: longitudes of starting locations
    @param pandas.DataFrame latitude_list_2: latitudes of destination locations
    @param pandas.DataFrame longitude_list_2: longitudes of destination locations
    @return: Matrix with direct distances in meter
    """

    latitude_list_1 = latitude_list_1.to_numpy(dtype=np.float32)
    longitude_list_1 = longitude_list_1.to_numpy(dtype=np.float32)
    latitude_list_2 = latitude_list_2.to_numpy(dtype=np.float32)
    longitude_list_2 = longitude_list_2.to_numpy(dtype=np.float32)

    # Convert decimal degrees to radians using Numpy arrays directly
    longitude_list_1 = np.radians(longitude_list_1)
    latitude_list_1 = np.radians(latitude_list_1)
    longitude_list_2 = np.radians(longitude_list_2)
    latitude_list_2 = np.radians(latitude_list_2)

    # Haversine formula
    dlon = np.subtract.outer(longitude_list_2, longitude_list_1)
    dlat = np.subtract.outer(latitude_list_2, latitude_list_1)

    # Use Numpy arrays directly, avoid unnecessary DataFrame conversion
    matrix_lat1_lat2 = np.outer(np.cos(latitude_list_1), np.cos(latitude_list_2))
    a = np.sin(dlat / 2) ** 2 + matrix_lat1_lat2.T * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000

    return np.ceil(m)


def get_continent_from_location(location, world=None):

    """
    This method derives continent from location coordinates. Method is used within the context of DataFrame.apply

    @param tuple location: coordinates of location as tuple (longitude, latitude)
    @param geopandas.GeoDataFrame world: shapefile of all countries
    @return: continent name of current location
    """

    location_longitude = location[0]
    location_latitude = location[1]

    if world is None:
        country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
        world = gpd.read_file(country_shapefile)

    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([location_longitude], [location_latitude])).set_crs('EPSG:4326')
    result = gpd.sjoin(gdf, world, how='left')
    continent = str(result.at[result.index[0], 'CONTINENT'])

    return continent


def check_if_reachable_on_land(target_location, list_longitude, list_latitude, coastline, get_only_availability=False,
                               get_only_poly=False):

    """
    This method checks if a location can reach different locations by checking if they are in the same polygon.
    The polygon are based on the coastlines so if not on same polygon, water is in between --> no road

    @param shapely.geometry.Point target_location: shapely.geometry.Point of start location
    @param pandas.DataFrame list_longitude: longitude of target locations
    @param pandas.DataFrame list_latitude: latitude of target locations
    @param geopandas.GeoDataFrame coastline: polygons based on coastline
    @param bool get_only_availability: array with True or False values of reachable on land
    @param bool get_only_poly: used to get only the polygon of the starting location
    @return: returns tuple with the boolean if reachable by road (within the same polygon) and the index of the polygon
    """

    coastline = coastline.copy()
    coastline = coastline.set_crs("EPSG:4326", allow_override=True)

    # Get polygon(s) of target or destination location
    points = []
    for i in list_latitude.index:
        points.append(Point([list_longitude.loc[i], list_latitude.loc[i]]))
    gdf_target = gpd.GeoDataFrame(geometry=points)
    gdf_target = gdf_target.set_crs("EPSG:4326", allow_override=True)

    gdf_start = gpd.GeoDataFrame(geometry=[target_location])
    gdf_start = gdf_start.set_crs("EPSG:4326", allow_override=True)
    if isinstance(target_location, Point):

        points = []
        for i in list_latitude.index:
            points.append(Point([list_longitude.loc[i], list_latitude.loc[i]]))

        gdf_target = gpd.GeoDataFrame(geometry=points)
        gdf_target = gdf_target.set_crs("EPSG:4326", allow_override=True)

        polygons = sjoin(gdf_start, coastline, predicate='within', how='right').dropna(subset=['index_left'])
        polygons_index = polygons.index.tolist()
    else:
        polygons_index = []
        for i in coastline.index:
            if coastline.loc[i, 'geometry'].intersects(target_location):
                polygons_index.append(i)

        polygons = coastline.loc[polygons_index, :]


    # check which infrastructure (gdf_target) is on same landmass polygon
    gdf_target['reachable'] = False
    index_poly_target = sjoin(gdf_target, coastline, predicate='within', how='right')
    index_poly_target.dropna(axis='index', subset='index_left', inplace=True)
    index_poly_target['index_left'] = index_poly_target['index_left'].astype(int)
    index_poly_target = index_poly_target.reset_index().set_index('index_left')

    for poly in index_poly_target['index'].unique():

        if poly in polygons_index:
            affected_locations = index_poly_target[index_poly_target['index'] == poly].index
            gdf_target.loc[affected_locations, 'reachable'] = True

    # not_reachable = gdf_target[~gdf_target['reachable']]
    # reachable = gdf_target[gdf_target['reachable']]

    # fig, ax = plt.subplots()
    # coastline.plot(ax=ax)
    # gdf_start.plot(ax=ax, color='yellow')
    # not_reachable.plot(ax=ax, color='red')
    # reachable.plot(ax=ax, color='orange')
    #
    # plt.show()


    # result = []
    # for i in not_reachable.index:
    #     target_point = not_reachable.at[i, 'geometry']
    #
    #     if target_location.distance(target_point) < 0.00001:
    #         result.append(True)
    #     else:
    #         result.append(False)
    #
    # not_reachable['reachable'] = result
    # gdf_target.loc[not_reachable.index, 'reachable'] = not_reachable['reachable']

    # print(len(not_reachable.index))

    return gdf_target['reachable'].tolist()
