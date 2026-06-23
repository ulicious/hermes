import math
import os
import logging
import shapely
import multiprocessing

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from shapely.geometry import Point, Polygon
from shapely.wkt import loads
from rtree import index

from algorithm.methods_geographic import calc_distance_list_to_single, check_if_reachable_on_land
from data_processing.helpers_create_voronoi_cells import attach_voronoi_cells_to_locations
from data_processing.helpers_geometry import (
    randlatlon1, round_to_quarter, create_polygons,
    get_start_location_information,
)
from data_processing.natural_earth_data import load_states, load_world
from data_processing.helpers_attach_costs import attach_conversion_costs_and_efficiency_to_start_locations, \
    check_if_location_is_valid, attach_feedstock_costs_and_interest_rate
from data_processing.configuration import (
    COUNTRY_DATA_FILE,
    LOCATION_DATA_FILE,
    load_algorithm_configuration,
    load_technology_data,
)

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)


def _as_float_or_inf(value):
    if isinstance(value, str):
        return math.inf
    return value


def calculate_start_mip_distances_like_heuristic(config_file, options, starting_location, coastlines):
    max_length_road = _as_float_or_inf(config_file['max_length_road'])
    max_length_new_segment = _as_float_or_inf(config_file['max_length_new_segment'])
    no_road_multiplier = _as_float_or_inf(config_file['no_road_multiplier'])
    max_length = max(max_length_road, max_length_new_segment) / no_road_multiplier

    options = options.copy()
    options['distance_to_start'] = calc_distance_list_to_single(
        options['latitude'], options['longitude'], starting_location.y, starting_location.x)
    options.sort_values(['distance_to_start'], inplace=True)

    first_ten_harbours = []
    for option in options.index:
        if 'H' in option:
            first_ten_harbours.append(option)
            if len(first_ten_harbours) == 10:
                break

    candidate_nodes = options[options['distance_to_start'] <= max_length].index[0:1000].tolist()
    candidate_nodes = list(set(candidate_nodes + first_ten_harbours))

    if not candidate_nodes:
        return pd.DataFrame(columns=['pointA', 'pointB', 'same_landmass', 'distance'])

    if {'longitude_on_coastline', 'latitude_on_coastline'}.issubset(options.columns):
        longitude = options.loc[candidate_nodes, 'longitude_on_coastline'].fillna(options.loc[candidate_nodes, 'longitude'])
        latitude = options.loc[candidate_nodes, 'latitude_on_coastline'].fillna(options.loc[candidate_nodes, 'latitude'])
    else:
        longitude = options.loc[candidate_nodes, 'longitude']
        latitude = options.loc[candidate_nodes, 'latitude']

    reachable = check_if_reachable_on_land(starting_location, longitude, latitude, coastlines, get_only_availability=True)
    reachable_nodes = pd.Index(candidate_nodes)[reachable].tolist()

    road_distances = pd.DataFrame({
        'pointA': 'start',
        'pointB': reachable_nodes,
        'same_landmass': True,
        'distance': options.loc[reachable_nodes, 'distance_to_start'].tolist(),
    })

    in_tolerance = road_distances[road_distances['distance'] <= config_file['tolerance_distance']].index
    road_distances.loc[in_tolerance, 'distance'] = 0

    return road_distances


def print_start_location_configuration(config_file):
    conversion_update_only = config_file['start_locations_update_only_conversion_costs_and_efficiency']
    print('Selected start-location settings:')
    print('Configuration file: ' + str(config_file.get('_configuration_path')))
    print('Project folder: ' + str(config_file['project_folder_path']))
    print('Start-location update mode - conversion costs/efficiencies only: '
          + str(config_file['start_locations_update_only_conversion_costs_and_efficiency']))
    print('MIP data enabled: ' + str(config_file['create_mip_data']))
    print('Voronoi cells enabled: ' + str(config_file['use_voronoi_cells']))
    print('Hydrogen cost weighting by quantity: ' + str(config_file['weight_hydrogen_costs_by_quantity']))
    print(
        'Heat at start low/mid/high: '
        + str(config_file['low_temp_heat_available_at_start'])
        + '/'
        + str(config_file['mid_temp_heat_available_at_start'])
        + '/'
        + str(config_file['high_temp_heat_available_at_start'])
    )
    if conversion_update_only:
        return

    print('Minimal example: ' + str(config_file['use_minimal_example']))
    print('Low memory mode: ' + str(config_file['use_low_memory']))
    print('Configured cores: ' + str(config_file['number_cores']))
    print('Start-location count setting: ' + str(config_file['number_locations']))
    print('Location creation type: ' + str(config_file['location_creation_type']))
    print('Start area type: ' + str(config_file['start_location_area_type']))
    print(
        'Start bounds: lat '
        + str(config_file['start_location_minimal_latitude'])
        + ' to '
        + str(config_file['start_location_maximal_latitude'])
        + ', lon '
        + str(config_file['start_location_minimal_longitude'])
        + ' to '
        + str(config_file['start_location_maximal_longitude'])
    )
    print('Start countries: ' + str(config_file['start_location_countries']))
    print('Origin continents: ' + str(config_file['origin_continents']))
    print('Each-country coverage enabled: ' + str(config_file['each_country_at_least_one_location']))
    print('Island locations enabled: ' + str(config_file['create_locations_for_islands']))
    print('Island area threshold: ' + str(config_file['island_area_threshold']))


config_file = load_algorithm_configuration()
print_start_location_configuration(config_file)

path_raw_data = config_file['project_folder_path'] + 'raw_data/'

num_cores = config_file['number_cores']
if num_cores == 'max':
    num_cores = multiprocessing.cpu_count() - 1
else:
    num_cores = min(num_cores, multiprocessing.cpu_count() - 1)

start_locations_update_only_conversion_costs_and_efficiency = \
    config_file['start_locations_update_only_conversion_costs_and_efficiency']
logging.info(
    'Start-location conversion update only: %s | heat start low/mid/high: %s/%s/%s',
    start_locations_update_only_conversion_costs_and_efficiency,
    config_file['low_temp_heat_available_at_start'],
    config_file['mid_temp_heat_available_at_start'],
    config_file['high_temp_heat_available_at_start'],
)

world_surface = Polygon([Point([-180, -90]), Point([-180, 90]), Point([180, 90]), Point([180, -90])])

levelized_costs_location = pd.read_csv(path_raw_data + LOCATION_DATA_FILE, index_col=0, sep=';')
levelized_costs_country = pd.read_csv(path_raw_data + COUNTRY_DATA_FILE, index_col=0)

techno_economic_data_conversion, _ = load_technology_data(config_file)

if not start_locations_update_only_conversion_costs_and_efficiency:

    logging.info('Create new locations')

    world = load_world(path_raw_data)
    states = load_states(path_raw_data)

    start_location_information = get_start_location_information(config_file, world=world, states=states)
    valid_polygon = start_location_information['location']
    minimal_longitude, minimal_latitude, maximal_longitude, maximal_latitude = valid_polygon.bounds
    countries = start_location_information['countries'].copy()

    origin_continents = config_file['origin_continents']
    if config_file['use_minimal_example']:  # overwrite origin continent if minimal example
        origin_continents = ['Europe']

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
        country_shape = gpd.GeoDataFrame(geometry=[country_shape])
        country_shape.crs = 'epsg:4326'
        country_shape.to_crs({'proj': 'cea'}, inplace=True)

        valid_polygon_shape = gpd.GeoDataFrame(geometry=[valid_polygon])
        valid_polygon_shape.crs = 'epsg:4326'
        valid_polygon_shape.to_crs({'proj': 'cea'}, inplace=True)

        number_points = math.ceil(config_file['number_locations'] / (country_shape.area / valid_polygon_shape.area) * 1.1)

        lat_min, lat_max = minimal_latitude, maximal_latitude
        lon_min, lon_max = minimal_longitude, maximal_longitude

        z_min = np.sin(np.radians(lat_min))
        z_max = np.sin(np.radians(lat_max))

        phi = (1 + np.sqrt(5)) / 2
        golden_angle = 2 * np.pi / phi

        points = np.empty((number_points, 2), dtype=float)

        for i in range(number_points):
            # --- Fibonacci / low-discrepancy parameter in [0,1) ---
            # (i + 0.5)/N avoids putting points exactly on the boundary
            t = (i + 0.5) / number_points

            # --- latitude: equal-area in the band ---
            z = z_min + t * (z_max - z_min)
            lat = np.degrees(np.arcsin(z))

            # --- longitude: golden-angle progression mapped into your window ---
            u = ((i * golden_angle) / (2 * np.pi)) % 1.0  # 0..1
            lon = lon_min + u * (lon_max - lon_min)

            points[i, 0] = lat
            points[i, 1] = lon

        # add country information to options
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:, 1], points[:, 0])).set_crs('EPSG:4326')
        result = gpd.sjoin(gdf, world, how='left')
        result.dropna(subset=["NAME_EN"], inplace=True)

        def process_points(i):
            start_lon = result.loc[i, 'geometry'].x
            start_lat = result.loc[i, 'geometry'].y

            # add country information to options
            country_start = result.at[i, 'NAME_EN']
            continent_start = result.at[i, 'CONTINENT']

            if isinstance(country_start, float):
                # country is nan
                return None

            if origin_continents:
                if continent_start not in origin_continents:
                    return None

            location_point = Point([start_lon, start_lat])
            if not location_point.within(valid_polygon):
                return None

            adjusted_latitude = round_to_quarter(start_lat)
            adjusted_longitude = round_to_quarter(start_lon)
            # adjusted_coords = str(int(adjusted_longitude * 100)) + 'x' + str(int(adjusted_latitude * 100))

            valid = check_if_location_is_valid(techno_economic_data_conversion, country_start, adjusted_latitude,
                                               adjusted_longitude, levelized_costs_location,
                                               levelized_costs_country)

            if not valid:
                return None

            # remove country if location will be created within
            if country_start in countries:
                countries.remove(country_start)

            # locations.loc[i, 'country_start'] = country_start
            # locations.loc[i, 'continent_start'] = continent_start
            #
            # locations.loc[i, 'longitude'] = start_lon
            # locations.loc[i, 'latitude'] = start_lat

            return country_start, continent_start, start_lon, start_lat

        inputs = tqdm(result.index)
        results = Parallel(n_jobs=num_cores)(delayed(process_points)(i) for i in inputs)

        i = 0
        for r in results:
            if r is not None:
                locations.loc[i, 'country_start'] = r[0]
                locations.loc[i, 'continent_start'] = r[1]

                locations.loc[i, 'longitude'] = r[2]
                locations.loc[i, 'latitude'] = r[3]

                i += 1

    # Each country should have at least one start location if set
    if config_file['each_country_at_least_one_location']:  # todo: does not work with uniform distribution of points
        for country in countries:
            area_shape = world[world['NAME_EN'] == country].index[0]
            area_shape = world.at[area_shape, 'geometry']
            area_shape = area_shape.intersection(valid_polygon)
            if area_shape.is_empty:
                continue

            area_shape_center = area_shape.representative_point()

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

    if 'polygon' in levelized_costs_location.columns:
        levelized_costs_location['polygon'] \
            = levelized_costs_location['polygon'].apply(shapely.wkt.loads)

    # Create a spatial index
    spatial_index = index.Index()
    for i in levelized_costs_location.index:

        if 'polygon' not in levelized_costs_location.columns:

            longitude = levelized_costs_location.loc[i, 'longitude']
            latitude = levelized_costs_location.loc[i, 'latitude']

            poly = Polygon([Point(longitude+0.25, latitude-0.25),
                            Point(longitude+0.25, latitude+0.25),
                            Point(longitude-0.25, latitude+0.25),
                            Point(longitude-0.25, latitude-0.25)])

        else:
            poly = levelized_costs_location.loc[i, 'polygon']

        spatial_index.insert(i, poly.bounds)
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

create_mip_data = config_file['create_mip_data']
if create_mip_data:
    path = config_file['project_folder_path'] + '/processed_data/'

    options = pd.read_csv(path + 'mip_data/' + 'options.csv', index_col=0)
    coastlines = pd.read_csv(path + 'landmasses.csv', index_col=0)
    coastlines = gpd.GeoDataFrame(coastlines, geometry=coastlines['geometry'].apply(loads))
    coastlines = coastlines.sort_index()
    coastlines.set_geometry('geometry', inplace=True)

    # distances
    for ind in tqdm(locations.index, desc='Create start MIP distances'):

        point = Point([locations.loc[ind, 'longitude'], locations.loc[ind, 'latitude']])

        road_distances = calculate_start_mip_distances_like_heuristic(config_file, options, point, coastlines)
        new_pipeline_distances = road_distances.copy()
        max_length_new_segment = _as_float_or_inf(config_file['max_length_new_segment'])
        no_road_multiplier = _as_float_or_inf(config_file['no_road_multiplier'])
        new_pipeline_distances = new_pipeline_distances[new_pipeline_distances['distance'] <= max_length_new_segment]

        road_distances['distance'] *= no_road_multiplier
        new_pipeline_distances['distance'] *= no_road_multiplier

        road_distances.to_csv(path + 'mip_data/' + str(ind) +  '_start_road_distances.csv')
        new_pipeline_distances.to_csv(path + 'mip_data/' + str(ind) + '_start_new_pipeline_distances.csv')
