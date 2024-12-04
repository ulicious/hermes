import math
import os
import random

import shapely.geometry
import yaml

import pandas as pd
import geopandas as gpd
import numpy as np
from geovoronoi import voronoi_regions_from_coords
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from scipy.spatial.distance import pdist

from data_processing.helpers_create_voronoi_cells import attach_voronoi_cells_to_locations


# Create voronoi cells based on locations. This is done for each country individually to limit voronoi cells
# to country borders
path_config = os.getcwd() + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

# load locations
locations = pd.read_excel(config_file['project_folder_path'] + 'start_destination_combinations.xlsx', index_col=0)

locations = attach_voronoi_cells_to_locations(locations, config_file)

#
# locations.to_excel(config_file['project_folder_path'] + 'start_destination_combinations.xlsx')
# locations = pd.read_excel(config_file['project_folder_path'] + 'start_destination_combinations.xlsx', index_col=0)

# plot results
# fig, ax = plt.subplots()
#
# import random
#
# get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# colors = get_colors(len(locations['geometry']))
#
# regions_gdf = gpd.GeoDataFrame(geometry=locations['geometry'])
# regions_gdf.plot(ax=ax, facecolor=colors, edgecolor='black')
#
# point_gdf = gpd.GeoDataFrame(geometry=location_points)
# point_gdf.set_crs(epsg=3395, inplace=True)
# point_gdf.plot(ax=ax, facecolor='white', edgecolor='black')
#
# plt.show()

# read potential
path_raw_data = config_file['project_folder_path'] + 'raw_data/'
potential_data = pd.read_csv(path_raw_data + config_file['potential_data_name'], index_col=0)

# polygons = [Polygon([Point([potential_data.at[i, 'longitude'] + 0.125, potential_data.at[i, 'latitude'] + 0.125]),
#                      Point([potential_data.at[i, 'longitude'] - 0.125, potential_data.at[i, 'latitude'] + 0.125]),
#                      Point([potential_data.at[i, 'longitude'] - 0.125, potential_data.at[i, 'latitude'] - 0.125]),
#                      Point([potential_data.at[i, 'longitude'] + 0.125, potential_data.at[i, 'latitude'] - 0.125])])
#             for i in potential_data.index]
#
# potential_data['geometry'] = polygons
# potential_data.to_csv(path_raw_data + config_file['potential_data_name'])

polygons = gpd.GeoDataFrame(pd.Series(potential_data['geometry'].tolist()).apply(shapely.wkt.loads), columns=['geometry'])
polygons.set_geometry('geometry')
polygons = polygons['geometry'].tolist()

# locations = gpd.GeoDataFrame(pd.Series(locations['geometry'].tolist()).apply(shapely.wkt.loads), columns=['geometry'])
# locations.set_geometry('geometry')
# locations = locations['geometry'].tolist()

# Create a spatial index
from rtree import index
import random

spatial_index = index.Index()
for idx, poly in enumerate(polygons):
    spatial_index.insert(idx, poly.bounds)

potential = {}
price = {}
for m, i in enumerate(locations.index):

    poly = locations.at[i, 'geometry']

    if isinstance(poly, float):  # todo: why are there some floats in the data? All locations should have geometry
        continue

    # Query the spatial index with the bounding box of the location  # todo: there are some missing
    possible_matches = list(spatial_index.intersection(poly.bounds))

    # Filter to find polygons strictly within the location
    result = [polygons[i] for i in possible_matches if polygons[i].intersects(poly)]

    print(len(result))

    # fig, ax = plt.subplots()
    #
    # regions_gdf = gpd.GeoDataFrame(geometry=[poly])
    # regions_gdf.plot(ax=ax, facecolor='none', edgecolor='black')
    #
    # point_gdf = gpd.GeoDataFrame(geometry=result)
    # point_gdf.set_crs(epsg=3395, inplace=True)
    # point_gdf.plot(ax=ax, facecolor='blue', edgecolor='black', alpha=0.5)
    #
    # plt.show()

    # todo: make list for supply costs per location
    costs = [random.random() * 100 for n in range(len(locations.index))]

    # todo: read quantity per era cell

    total_potential = 0
    total_costs = 0
    for n, r in enumerate(result):
        # Calculate the intersection
        intersection = poly.intersection(r)

        # Calculate the overlap percentage
        overlap_percentage = (intersection.area / r.area)

        total_potential += potential_data.at[n, 'Hydrogen_Gas'] * overlap_percentage
        total_costs += total_potential * costs[m]

    if total_potential == 0:
        potential[i] = 0
        price[i] = 0

    else:
        average_costs = total_costs / total_potential

        potential[i] = total_potential
        price[i] = average_costs

print(potential)
print(price)

price = dict(sorted(price.items(), key=lambda item: item[1]))
sorted_potential = []
for k in price.keys():
    sorted_potential.append(potential[k])

get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
colors = get_colors(len(locations['geometry']))

fig, ax = plt.subplots(figsize=(10, 6))

xticks = []
for n, c in enumerate(sorted_potential):
    xticks.append(sum(sorted_potential[:n]) + sorted_potential[n] / 2)

w_new = [i / max(potential) for i in potential]
a = plt.bar(xticks, height=list(price.values()), width=sorted_potential, color=colors, alpha=0.8)
# _ = plt.xticks(xticks, x)

# plt.legend(a.patches, x)

plt.show()

total_demand = 1000000
current_supply = 0
used_voronoi_cells = []
for k in price.keys():
    current_supply += potential[k]

    used_voronoi_cells.append(locations.loc[k, 'geometry'])

    if current_supply >= total_demand:
        break

fig, ax = plt.subplots()

point_gdf = gpd.GeoDataFrame(geometry=locations['geometry'])
point_gdf.plot(ax=ax, facecolor='none', edgecolor='black')

regions_gdf = gpd.GeoDataFrame(geometry=used_voronoi_cells)
regions_gdf.plot(ax=ax, facecolor='blue', edgecolor='black', alpha=0.5)

plt.show()


