import geojson

import pandas as pd

from vincenty import vincenty
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon

import warnings
warnings.filterwarnings('ignore')


def process_ports(path_data, coastlines, landmasses, boundaries, destination, use_minimal_example=False):

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

        location = Point(port['geometry']['coordinates'][0], port['geometry']['coordinates'][1])

        boundaries_poly = Polygon([Point([boundaries[2], boundaries[1]]),
                                   Point([boundaries[3], boundaries[1]]),
                                   Point([boundaries[3], boundaries[0]]),
                                   Point([boundaries[2], boundaries[0]])])

        if not boundaries_poly.contains(location):
            if isinstance(destination, Point):
                distance = vincenty((destination.x, destination.y), (location.x, location.y)) * 1000
                if not distance < 50000:
                    continue
            else:
                if not destination.contains(location):
                    continue

        index = 'H' + str(i)

        ports.loc[index, 'longitude'] = port['geometry']['coordinates'][0]
        ports.loc[index, 'latitude'] = port['geometry']['coordinates'][1]
        ports.loc[index, 'name'] = port['properties']['name']
        ports.loc[index, 'country'] = port['properties']['country']
        ports.loc[index, 'continent'] = port['properties']['continent']

        # add closest point to coastline --> necessary as some ports are not connected to land
        # new_port_location = nearest_points(coastlines, location)[0]
        new_port_location = nearest_points(landmasses, location)[0]
        index_closest = coastlines.distance(location).sort_values().index[0]
        closest_point = new_port_location.loc[index_closest].values[0]

        ports.loc[index, 'longitude_on_coastline'] = closest_point.x
        ports.loc[index, 'latitude_on_coastline'] = closest_point.y

        if False:  # to check if adjusted coordinates are really on landmasses
            within = closest_point.within(landmasses)
            result = within['geometry'].any()
            print(result)

        i += 1

    if use_minimal_example:
        ports = ports[ports['continent'] == 'Europe']

    return ports
