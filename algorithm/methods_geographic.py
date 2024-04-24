import pycountry_convert as pc
import numpy as np
import reverse_geocode
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from geopandas.tools import sjoin
from math import cos, sin, asin, sqrt, radians


def calc_distance_single_to_single(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    This methods calculates direct distance between two locations
    
    @param latitude_1: latitude first location
    @param longitude_1: longitude first location
    @param latitude_2: latitude second location
    @param longitude_2: longitude second location
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
    return m


def calc_distance_list_to_single(latitude_list_1, longitude_list_1, latitude_2, longitude_2):
    """
    This method calculates the direct distance between a single location and an array of locations

    @param latitude_list_1: latitudes of start locations
    @param longitude_list_1: longitude of start locations
    @param latitude_2: latitude of destination
    @param longitude_2: longitude of destination
    @return: array of direct distances in meter
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
    This method calculates the direct distance between two arrays of locations

    @param latitude_list_1: latitudes of starting locations
    @param longitude_list_1: longitudes of starting locations
    @param latitude_list_2: latitudes of destination locations
    @param longitude_list_2: longitudes of destination locations
    @return: dataframe of direct distances in meter. Important: list-like not array
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
    This method calculates the direct distances between two lists of coordinates.

    @param latitude_list_1: latitudes of starting locations
    @param longitude_list_1: longitudes of starting locations
    @param latitude_list_2: latitudes of destination locations
    @param longitude_list_2: longitudes of destination locations
    @return: Matrix with direct distances in meter
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


def get_continent_from_location(location, world=None):

    """
    This method derives continent from location coordinates. Method is used within the context of DataFrame.apply

    @param location: coordinates of location as tuple (longitude, latitude)
    @param world: shapefile of all countries
    @return: continent
    """

    location_longitude = location[0]
    location_latitude = location[1]

    if world is None:
        country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
        world = gpd.read_file(country_shapefile)

    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([location_longitude], [location_latitude])).set_crs('EPSG:4326')
    result = gpd.sjoin(gdf, world, how='left')
    continent = result.at[result.index[0], 'CONTINENT']

    return continent


def check_if_reachable_on_land(start_location, list_longitude, list_latitude, coastline, get_only_availability=False,
                               get_only_poly=False):

    """
    This method checks if a location can reach different locations by checking if they are in the same polygon.
    The polygon are based on the coastlines so if not on same polygon, water is in between --> no road

    @param start_location: shapely.geometry.Point of start location
    @param list_longitude: longitude of target locations
    @param list_latitude: latitude of target locations
    @param coastline: polygons based on coastline
    @param get_only_availability: array with True or False values of reachable on land
    @param get_only_poly: used to get only the polygon of the starting location
    @return: returns tuple with the boolean if reachable by road (within the same polygon) and the index of the polygon
    """

    df_start = gpd.GeoSeries.from_wkt(['Point(' + str(start_location.x) + ' ' + str(start_location.y) + ')'])
    gdf_start = gpd.GeoDataFrame(df_start, geometry=0)

    points = []
    for i in list_latitude.index:
        points.append('Point(' + str(list_longitude.loc[i]) + ' ' + str(list_latitude.loc[i]) + ')')

    df_target = gpd.GeoSeries.from_wkt(points)
    gdf_target = gpd.GeoDataFrame(df_target, geometry=0)

    polygons = sjoin(gdf_start, coastline, predicate='within', how='right').dropna(subset=['index_left'])

    if len(polygons.index) > 0:

        index_poly_start = polygons.index[0]
        index_poly_target = sjoin(gdf_target, coastline, predicate='within', how='right')

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

            result = []
            for i in gdf_target.index:
                result.append(get_availability(i))

            if not get_only_availability:
                return result, index_poly_start
            else:
                return result
        else:
            return index_poly_start

    else:
        return None, None
