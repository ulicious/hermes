import geojson

import pandas as pd

from shapely.ops import nearest_points
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')


def process_ports(path_data, coastlines, use_minimal_example=False):

    """
    processes raw ports data to dataframe and connects port to the closest coastline

    @param str path_data: path to ports data
    @param geopandas.GeoDataFrame coastlines: coastlines to allow connection of ports to coastline
    @param bool use_minimal_example: removes all ports outside of Europe in case of minimal example
    @return: dataframe containing all information on ports
    """

    with open(path_data + 'seaports.geojson') as f:
        gj = geojson.load(f)
    features = gj['features']

    ports = pd.DataFrame(columns=['latitude', 'longitude', 'name', 'country', 'continent'])
    ports.drop_duplicates(subset=['latitude', 'longitude'], keep='first')

    i = 0
    for port in features:
        index = 'H' + str(i)
        ports.loc[index, 'longitude'] = port['geometry']['coordinates'][0]
        ports.loc[index, 'latitude'] = port['geometry']['coordinates'][1]
        ports.loc[index, 'name'] = port['properties']['name']
        ports.loc[index, 'country'] = port['properties']['country']
        ports.loc[index, 'continent'] = port['properties']['continent']

        # add closest point to coastline --> necessary as some ports are not connected to land
        location = Point(ports.loc[index, 'longitude'], ports.loc[index, 'latitude'])
        new_port_location = nearest_points(coastlines, location)[0]
        index_closest = coastlines.distance(location).sort_values().index[0]
        closest_point = new_port_location.loc[index_closest].values[0]

        ports.loc[index, 'longitude_on_coastline'] = closest_point.x
        ports.loc[index, 'latitude_on_coastline'] = closest_point.y

        i += 1

    if use_minimal_example:
        ports = ports[ports['continent'] == 'Europe']

    return ports
