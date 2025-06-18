import math
import os
import logging
import yaml
import shapely

import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import numpy as np

from shapely.geometry import Point, Polygon
from rtree import index

from data_processing.helpers_create_voronoi_cells import attach_voronoi_cells_to_locations
from data_processing.helpers_geometry import randlatlon1, round_to_quarter, create_polygons
from data_processing.helpers_attach_costs import attach_conversion_costs_and_efficiency_to_start_locations, \
    check_if_location_is_valid, attach_feedstock_costs_and_interest_rate

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)


path_config = os.getcwd() + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_raw_data = config_file['project_folder_path'] + 'raw_data/'

update_only_techno_economic_data = config_file['update_only_conversion_costs_and_efficiency']

if not config_file['use_minimal_example']:
    # use boundaries from config file
    minimal_latitude = config_file['minimal_latitude']
    maximal_latitude = config_file['maximal_latitude']
    minimal_longitude = config_file['minimal_longitude']
    maximal_longitude = config_file['maximal_longitude']
else:
    # if minimal example, set boundaries to Europe
    minimal_latitude, maximal_latitude = 35, 71
    minimal_longitude, maximal_longitude = -25, 45

world_surface = Polygon([Point([-180, -90]), Point([-180, 90]), Point([180, 90]), Point([180, -90])])

valid_polygon = Polygon([Point([minimal_longitude, minimal_latitude]),
                         Point([minimal_longitude, maximal_latitude]),
                         Point([maximal_longitude, maximal_latitude]),
                         Point([maximal_longitude, minimal_latitude])])

levelized_costs_location = pd.read_csv(path_raw_data + config_file['location_data_name'], index_col=0)
levelized_costs_country = pd.read_csv(path_raw_data + config_file['country_data_name'], index_col=0)

yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

if not update_only_techno_economic_data:

    logging.info('Create new locations')

    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)

    state_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    states = gpd.read_file(state_shapefile)

    countries = world['NAME_EN'].tolist()

    origin_continents = config_file['origin_continents']
    if config_file['use_minimal_example']:  # overwrite origin continent if minimal example
        origin_continents = ['Europe']
        countries = world[world['CONTINENT'] == 'Europe']['NAME_EN'].tolist()

    country_shape = shapely.ops.unary_union(world['geometry'].tolist())
    country_shape = valid_polygon.intersection(country_shape)

    locations = pd.DataFrame()

    i = 0
    if config_file['location_creation_type'] == 'random':
        while i < config_file['number_locations']:
            # run algorithm as long as number locations too low or not each country has a location

            restart = False

            coords = randlatlon1(minimal_latitude, maximal_latitude, minimal_longitude, maximal_longitude)

            if coords is not None:

                start_lat = coords[1]
                start_lon = coords[0]

                new_location = pd.DataFrame([start_lon, start_lat]).transpose()
                new_location.columns = ['longitude', 'latitude']

                # add country information to options
                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([start_lon], [start_lat])).set_crs('EPSG:4326')
                result = gpd.sjoin(gdf, world, how='left')
                country_start = result.at[result.index[0], 'NAME_EN']
                continent_start = result.at[result.index[0], 'CONTINENT']

                if isinstance(country_start, float):
                    # country is nan
                    continue

                if origin_continents:
                    if continent_start not in origin_continents:
                        continue

                location_point = Point([start_lon, start_lat])
                if not location_point.within(valid_polygon):
                    continue

                adjusted_latitude = round_to_quarter(start_lat)
                adjusted_longitude = round_to_quarter(start_lon)
                adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

                restart = check_if_location_is_valid(techno_economic_data_conversion, country_start, adjusted_latitude,
                                                     adjusted_longitude, levelized_costs_location,
                                                     levelized_costs_country)

                if not restart:
                    continue

                # remove country if location will be created within
                if country_start in countries:
                    countries.remove(country_start)

                locations.loc[i, 'country_start'] = country_start
                locations.loc[i, 'continent_start'] = continent_start

                locations.loc[i, 'longitude'] = start_lon
                locations.loc[i, 'latitude'] = start_lat

                i += 1

    else:
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # because major parts of the globe are water, we need to increase number of points to get approx. correct number
        # since locations need to be distributed first and then only onshore locations are considered
        world_surface = gpd.GeoDataFrame(geometry=[world_surface])
        world_surface.crs = 'epsg:4326'
        world_surface.to_crs({'proj': 'cea'}, inplace=True)
        country_shape = gpd.GeoDataFrame(geometry=[country_shape])
        country_shape.crs = 'epsg:4326'
        country_shape.to_crs({'proj': 'cea'}, inplace=True)

        number_points = math.ceil(config_file['number_locations'] / (country_shape.area / world_surface.area) * 1.1)

        for i in range(number_points):
            z = 1 - (i / (number_points - 1)) * 2  # Map z to range [-1, 1]
            start_lat = np.degrees(np.arcsin(z))  # Latitude from -90 to 90 degrees
            start_lon = np.degrees(2 * np.pi * i / phi) % 360  # Longitude from 0 to 360
            if start_lon > 180:  # Adjust to range [-180, 180]
                start_lon -= 360

            # add country information to options
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([start_lon], [start_lat])).set_crs('EPSG:4326')
            result = gpd.sjoin(gdf, world, how='left')
            country_start = result.at[result.index[0], 'NAME_EN']
            continent_start = result.at[result.index[0], 'CONTINENT']

            if isinstance(country_start, float):
                # country is nan
                continue

            if origin_continents:
                if continent_start not in origin_continents:
                    continue

            location_point = Point([start_lon, start_lat])
            if not location_point.within(valid_polygon):
                continue

            adjusted_latitude = round_to_quarter(start_lat)
            adjusted_longitude = round_to_quarter(start_lon)
            adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

            valid = check_if_location_is_valid(techno_economic_data_conversion, country_start, adjusted_latitude,
                                               adjusted_longitude, levelized_costs_location,
                                               levelized_costs_country)

            if not valid:
                continue

            # remove country if location will be created within
            if country_start in countries:
                countries.remove(country_start)

            locations.loc[i, 'country_start'] = country_start
            locations.loc[i, 'continent_start'] = continent_start

            locations.loc[i, 'longitude'] = start_lon
            locations.loc[i, 'latitude'] = start_lat

            i += 1

    # Each country should have at least one start location if set
    if config_file['each_country_at_least_one_location']:  # todo: does not work with uniform distribution of points
        for country in countries:
            area_shape = world[world['NAME_EN'] == country].index[0]
            area_shape = world.at[area_shape, 'geometry']

            area_shape_center = area_shape.centroid

            start_lat = area_shape_center.y
            start_lon = area_shape_center.x

            new_location = pd.DataFrame([start_lon, start_lat]).transpose()
            new_location.columns = ['longitude', 'latitude']

            # add country information to options
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([start_lon], [start_lat])).set_crs('EPSG:4326')
            result = gpd.sjoin(gdf, world, how='left')
            country_start = result.at[result.index[0], 'NAME_EN']
            continent_start = result.at[result.index[0], 'CONTINENT']

            if isinstance(country_start, float):
                # country is nan
                continue

            if origin_continents:
                if continent_start not in origin_continents:
                    continue

            adjusted_latitude = round_to_quarter(start_lat)
            adjusted_longitude = round_to_quarter(start_lon)
            adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

            restart = check_if_location_is_valid(techno_economic_data_conversion, country_start, adjusted_latitude,
                                                 adjusted_longitude, levelized_costs_location,
                                                 levelized_costs_country)

            if restart:
                continue

            locations.loc[i, 'country_start'] = country_start
            locations.loc[i, 'continent_start'] = continent_start

            locations.loc[i, 'longitude'] = start_lon
            locations.loc[i, 'latitude'] = start_lat

            i += 1

    # attach voronoi cells
    if config_file['use_voronoi_cells']:
        locations = attach_voronoi_cells_to_locations(locations, config_file)

        # process polygons for better readability
        locations['geometry'] \
            = locations['geometry'].apply(lambda x: x.wkt)

        locations.reset_index(drop=True, inplace=True)

        # remove
        locations.to_csv(config_file['project_folder_path'] + 'start_destination_combinations.csv')

        # make again valid polygons
        locations['geometry'] = locations['geometry'].apply(shapely.wkt.loads)

else:
    logging.info('Update existing locations')
    # locations = pd.read_excel(config_file['project_folder_path'] + 'start_destination_combinations.xlsx', index_col=0)
    locations = pd.read_csv(config_file['project_folder_path'] + 'start_destination_combinations.csv', index_col=0)

    # remove all previous costs
    columns_to_keep = ['longitude', 'latitude', 'country_start', 'continent_start', 'geometry']
    locations = locations[columns_to_keep]

    # convert polygon strings to shapely objects
    if config_file['use_voronoi_cells']:
        locations['geometry'] \
            = locations['geometry'].apply(shapely.wkt.loads)

# convert polygon strings to shapely objects
if config_file['use_voronoi_cells']:
    levelized_costs_location['polygon'] \
        = levelized_costs_location['polygon'].apply(shapely.wkt.loads)

    # Create a spatial index
    spatial_index = index.Index()
    for idx, poly in enumerate(levelized_costs_location['polygon'].tolist()):
        spatial_index.insert(idx, poly.bounds)
else:
    spatial_index = None

for i in locations.index:
    country_start = locations.at[i, 'country_start']

    adjusted_latitude = round_to_quarter(locations.at[i, 'latitude'])
    adjusted_longitude = round_to_quarter(locations.at[i, 'longitude'])
    adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

    locations = attach_feedstock_costs_and_interest_rate(i, locations, techno_economic_data_conversion, country_start,
                                                         adjusted_latitude, adjusted_longitude,
                                                         levelized_costs_location, levelized_costs_country, config_file,
                                                         spatial_index)

logging.info('Calculate conversion costs and efficiency')
locations = attach_conversion_costs_and_efficiency_to_start_locations(locations, techno_economic_data_conversion, config_file)

columns_to_keep = ['longitude', 'latitude', 'country_start', 'continent_start', 'geometry', 'Hydrogen_Gas_Quantity'] + config_file['available_commodity']
locations = locations[columns_to_keep]

locations['geometry'] = locations['geometry'].apply(lambda x: x.wkt)
locations.to_csv(config_file['project_folder_path'] + 'start_destination_combinations.csv')
