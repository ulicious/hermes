import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#import pygal
import pandas as pd
from shapely.wkt import loads
from textwrap import wrap
from shapely.geometry import Point

fig, ax = plt.subplots()
path_raw_data = '/home/localadmin/Dokumente/Daten_Transportmodell/gas_pipeline_data/'
file_name = 'sorted_gas_pipelines_87.csv'

data_df = pd.read_csv(path_raw_data + file_name, index_col=0, sep=';')

data_gdf = gpd.GeoDataFrame(geometry=data_df['geometry'].apply(loads))

points_gdf = gpd.GeoDataFrame(geometry=[Point([57.299959, 63.860257]), Point([63.860257, 57.299959])])

data_gdf.plot(ax=ax)
points_gdf.plot(ax=ax, color='r')

plt.show()