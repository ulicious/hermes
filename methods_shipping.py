import pandas as pd
import searoute as sr
from shapely.geometry import LineString, Point
import geopandas as gpd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def find_searoute(start_coordinates, destination_coordinates, existing_distance=0):

    route = sr.searoute(start_coordinates, destination_coordinates)
    if False:
        route_line = LineString(route['geometry']['coordinates'])
        route_gdf = gpd.GeoDataFrame(index=[0], geometry=[route_line])
        route_gdf.plot()
        plt.show()
    distance = (round(float(format(route.properties['length'])), 2) + existing_distance) * 1000  # m

    return Point(destination_coordinates), distance

