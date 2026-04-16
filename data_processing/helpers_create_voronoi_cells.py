import math
import os
import random
# from xxlimited import new

import shapely.geometry
import yaml

import pandas as pd
import geopandas as gpd
import numpy as np
from geovoronoi import voronoi_regions_from_coords
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely import make_valid
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial.distance import pdist
from shapely.strtree import STRtree
from collections import defaultdict
from pyproj import Geod

from data_processing.helpers_misc import plot_points_and_areas, create_random_colors


def can_be_single_polygon(geom):
    """
    Returns True if geom is or can be represented as a single Polygon
    (touching or overlapping parts are OK).
    Returns False only if parts are completely disconnected.
    """
    if geom.geom_type == "Polygon":
        return True

    if geom.geom_type != "MultiPolygon":
        return False

    merged = unary_union(geom)
    return merged.geom_type == "Polygon"


def attach_polygon_to_other_voronoi_cells(polygon, locations):

    # check which voronoi cells are connected to unattached polygon
    connected_polygons = []

    residual_polygons = defaultdict(lambda: [])
    affected_locations = {}
    for i in locations.index:
        location_point = Point([locations.loc[i, 'longitude'], locations.loc[i, 'latitude']])
        location_polygon = locations.at[i, 'geometry']

        if isinstance(location_polygon, float):
            continue

        if location_polygon.intersects(polygon):
            # connected

            # check which part is connected (if multipolygon)
            if isinstance(location_polygon, MultiPolygon):
                for geom in location_polygon.geoms:
                    if geom.intersects(polygon):  # only the touching part
                        connected_polygons.append(geom)
                        affected_locations[i] = location_point
                    else:
                        # geovoronoi only works with polygons. If multipolygon, use only part of multipolygon which
                        # is connected to polygon and attach rest later
                        residual_polygons[i].append(geom)
            else:
                connected_polygons.append(location_polygon)  # all of it
                affected_locations[i] = location_point

    new_polygon = polygon
    for connected_polygon in connected_polygons:
        connected_polygon = connected_polygon
        new_polygon = unary_union([new_polygon, connected_polygon])

    if new_polygon.geom_type == "GeometryCollection":  # catch geometry collections
        polys = [x for x in new_polygon.geoms if x.geom_type in ("Polygon", "MultiPolygon")]
        new_polygon = unary_union(polys) if polys else new_polygon

    if isinstance(new_polygon, MultiPolygon):
        # catch multipolygons --> get the parts of the multipolygon which are connected to the initial polygon
        keep = []
        residual = []
        for geom in new_polygon.geoms:
            if geom.intersects(polygon):
                keep.append(geom)
            else:
                residual.append(geom)

        if len(affected_locations.keys()) == 1:  # if only one location apply, put them where other residuals are
            residual_polygons[[*affected_locations.keys()][0]] += residual
        elif len(affected_locations.keys()) > 1:  # else, check to which location minimal distance is
            for r in residual:
                distance = math.inf
                chosen_location = None
                for k in affected_locations.keys():
                    location_polygon = locations.at[k, 'geometry']

                    if location_polygon.distance(r) < distance:
                        distance = location_polygon.distance(r)
                        chosen_location = k

                residual_polygons[chosen_location].append(r)

        new_polygon = unary_union(keep)
        print(type(new_polygon))

    if not affected_locations:
        return locations, [polygon]

    type_poly = type(new_polygon)

    # new_polygon_gdf = gpd.GeoDataFrame(geometry=[new_polygon])
    # new_polygon_gdf['geometry'] = shapely.set_precision(new_polygon_gdf.geometry, 1e-3)
    #
    # new_polygon = new_polygon_gdf['geometry'][0]

    if type_poly != type(new_polygon):
        print('')

    if isinstance(new_polygon, MultiPolygon):

        new_polygon = [shape for shape in new_polygon.geoms if shape.area > 0.001]  # some geoms are just too small to be considered
        new_polygon = shapely.ops.unary_union(new_polygon)

        # if isinstance(new_polygon, MultiPolygon) & (len([*affected_locations.values()]) == 2):
        #     fig, ax = plt.subplots()
        #     new_polygon_gdf.plot(ax=ax, cmap='tab20', legend=False, edgecolor='black')
        #
        #     points_gdf = gpd.GeoDataFrame(geometry=list(affected_locations.values()))
        #     points_gdf.plot(ax=ax, color='yellow')
        #     plt.show()
        #
        #     for geom in new_polygon.geoms:
        #         print(geom.area)
        #         test = gpd.GeoDataFrame(geometry=[geom])
        #         test.plot()
        #         # plt.show()

    failed_polygons = []
    if len([*affected_locations.values()]) == 2:
        # simply split area into two parts
        ind1 = [*affected_locations.keys()][0]
        ind2 = [*affected_locations.keys()][1]

        lp1 = [*affected_locations.values()][0]
        lp2 = [*affected_locations.values()][1]

        split_area = divide_area_by_two(new_polygon, lp1, lp2)
        split_area = [g for g in split_area.geoms]

        for i in split_area:
            if lp1.distance(i) == 0:
                intersection_area = polygon.intersection(i)
                location_polygon = locations.loc[ind1, 'geometry']

                location_polygon = location_polygon.union(intersection_area)
                locations.loc[ind1, 'geometry'] = location_polygon

            elif lp2.distance(i) == 0:
                intersection_area = polygon.intersection(i)
                location_polygon = locations.loc[ind2, 'geometry']

                location_polygon = location_polygon.union(intersection_area)
                locations.loc[ind2, 'geometry'] = location_polygon
            else:
                for k in [*residual_polygons.keys()]:
                    if i not in residual_polygons[k]:
                        # detached polygon is not part of already existing voronoi cells
                        # --> attach to the closest location later
                        failed_polygons.append(i)

        # attach polygons which have been separated due to multipolygons
        for i in [*residual_polygons.keys()]:
            location_polygon = locations.loc[i, 'geometry']

            for rp in residual_polygons[i]:
                location_polygon = location_polygon.union(rp)

            locations.loc[i, 'geometry'] = location_polygon

    elif len([*affected_locations.values()]) > 2:
        points = [Point(p.x, p.y) for p in affected_locations.values()]

        # all latitudes cannot be the same. Same with longitude
        longitudes = set([p.x for p in points])
        if len(longitudes) == 1:  # all the same --> adjust slightly
            adjusted_point = points[0]
            adjusted_point = Point([adjusted_point.x + 0.00001, adjusted_point.y])
            points = [adjusted_point] + [p for p in points[1:]]

        latitudes = set([p.y for p in points])
        if len(latitudes) == 1:  # all the same --> adjust slightly
            adjusted_point = points[0]
            adjusted_point = Point([adjusted_point.x, adjusted_point.y + 0.00001])
            points = [adjusted_point] + [p for p in points[1:]]

        region_polys, region_pts, missing = voronoi_regions_from_coords(points, new_polygon, return_unassigned_points=True)

        if missing:
            failed_polygons.append(polygon)

        # attach new area from polygon to voronoi cells
        created_polygons = []
        for k in [*region_polys.keys()]:
            intersection_area = polygon.intersection(region_polys[k])
            index = list(affected_locations.keys())[region_pts[k][0]]
            location_polygon = locations.loc[index, 'geometry']
            location_polygon = location_polygon.union(intersection_area)
            locations.loc[index, 'geometry'] = location_polygon

            created_polygons.append(location_polygon)

        # attach polygons which have been separated due to multipolygons
        for i in [*residual_polygons.keys()]:
            location_polygon = locations.loc[i, 'geometry']

            for rp in residual_polygons[i]:
                location_polygon = location_polygon.union(rp)

            locations.loc[i, 'geometry'] = location_polygon

    elif len(affected_locations.keys()) == 1:
        # only single voronoi cell is connected to polygon --> combine
        index = list(affected_locations.keys())[0]
        locations.loc[index, 'geometry'] = new_polygon

        # attach polygons which have been separated due to multipolygons
        for i in [*residual_polygons.keys()]:
            location_polygon = locations.loc[i, 'geometry']

            for rp in residual_polygons[i]:
                location_polygon = location_polygon.union(rp)

            locations.loc[i, 'geometry'] = location_polygon

    return locations, failed_polygons


def attach_unprocessable_polygons_to_connected_voronoi_cell(polygon, locations):

    # check which voronoi cells are connected to unattached polygon
    min_distance = math.inf
    chosen_polygon = None
    chosen_index = None
    for i in locations.index:
        location_point = Point([locations.loc[i, 'longitude'], locations.loc[i, 'latitude']])
        location_polygon = locations.at[i, 'geometry']

        if isinstance(location_polygon, float):
            continue

        if location_polygon.intersects(polygon):
            # connected
            if location_point.distance(polygon) < min_distance:
                min_distance = location_point.distance(polygon)
                chosen_polygon = location_polygon
                chosen_index = i

    if chosen_index is not None:

        merged_polygon = polygon.union(chosen_polygon)
        locations.loc[chosen_index, 'geometry'] = merged_polygon

        failed = False
    else:
        failed = True

    return locations, failed


def merge_unattached_polygons(unattached_polygons):
    while True:
        old_areas = unattached_polygons.copy()

        merged = False
        new_polies = []
        processed_polies = []
        for key_1 in [*unattached_polygons.keys()]:
            poly_1 = unattached_polygons[key_1]

            if key_1 in processed_polies:
                continue

            for key_2 in [*unattached_polygons.keys()]:

                if key_2 in processed_polies:
                    continue

                if key_1 != key_2:

                    poly_2 = unattached_polygons[key_2]

                    if poly_1.intersects(poly_2):
                        poly_1 = poly_1.union(poly_2)

                        new_polies.append(poly_1)

                        # unattached_polygons[key_1] = poly_1
                        # unattached_polygons.pop(key_2)

                        processed_polies.append(key_1)
                        processed_polies.append(key_2)

                        break

        for key in [*unattached_polygons.keys()]:
            if key not in processed_polies:
                new_polies.append(unattached_polygons[key])

        new_keys = range(len(new_polies))

        unattached_polygons = dict(zip(new_keys, new_polies))

        if len(unattached_polygons) == len(old_areas):
            break

        print(len(unattached_polygons))

    return unattached_polygons


def divide_area_by_two(geom, lp1, lp2):
    # given package seems to have problems with two points
    # --> just split are in the middle between the two points

    # Get the exterior coordinates of the polygon
    distance_points = list(geom.exterior.coords)
    distances = pdist(distance_points)
    max_distance = distances.max()

    connection_line = LineString([lp1, lp2])

    # Calculate the midpoint of the line
    midpoint = connection_line.interpolate(0.5, normalized=True)  # Midpoint of the line

    # Extract line endpoints
    x1, y1 = connection_line.coords[0]
    x2, y2 = connection_line.coords[1]

    # Compute the perpendicular vector
    dx = x2 - x1
    dy = y2 - y1
    perpendicular_vector = (-dy, dx)

    # Normalize the vector and scale to desired length
    magnitude = (perpendicular_vector[0] ** 2 + perpendicular_vector[1] ** 2) ** 0.5
    unit_vector = (perpendicular_vector[0] / magnitude, perpendicular_vector[1] / magnitude)
    scaled_vector = (unit_vector[0] * max_distance / 2, unit_vector[1] * max_distance / 2)

    # Compute the endpoints of the perpendicular line
    x_mid, y_mid = midpoint.x, midpoint.y
    p1 = (x_mid + scaled_vector[0], y_mid + scaled_vector[1])
    p2 = (x_mid - scaled_vector[0], y_mid - scaled_vector[1])

    # Create the perpendicular line
    perpendicular_line = LineString([p1, p2])

    split_area = shapely.ops.split(geom, perpendicular_line)

    return split_area


def attach_voronoi_cells_to_locations(locations, config_file):

    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)

    # Create voronoi cells
    unprocessed_polygons = {}

    # create geometry column
    locations['geometry'] = math.nan

    location_points = [Point([locations.loc[i, 'longitude'], locations.loc[i, 'latitude']]) for i in locations.index]

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

    covered_area_polygon = Polygon([Point([minimal_longitude, minimal_latitude]),
                                    Point([minimal_longitude, maximal_latitude]),
                                    Point([maximal_longitude, maximal_latitude]),
                                    Point([maximal_longitude, minimal_latitude])])

    m = 0
    for c in world.index:

        country = world.at[c, 'NAME_EN']
        continent = world.at[c, 'CONTINENT']

        if config_file['use_minimal_example'] & (continent != 'Europe'):
            continue

        area = world[world['NAME_EN'] == country]
        area_shape = area.iloc[0].geometry

        # only consider the area covered by set longitudes and latitudes
        area_shape = area_shape.intersection(covered_area_polygon)

        affected_index = locations[locations['country_start'] == country].index.tolist()
        coords_in_country = [Point([locations.loc[i, 'longitude'], locations.loc[i, 'latitude']])
                             for i in affected_index]

        if isinstance(area_shape, shapely.geometry.MultiPolygon):

            intersecting_subpolygons = [poly for poly in area_shape.geoms if any(poly.intersects(point) for point in coords_in_country)]

            not_intersecting_subpolygons = [poly for poly in area_shape.geoms
                                            if poly not in intersecting_subpolygons]

            for geom in not_intersecting_subpolygons:
                unprocessed_polygons[m] = geom
                m += 1

        else:
            intersecting_subpolygons = [area_shape]

        for geom in intersecting_subpolygons:

            coords_in_country_copy = coords_in_country.copy()
            affected_index_copy = affected_index.copy()
            number_points = 0
            index_in_geom = []
            for n, p in enumerate(coords_in_country_copy):
                if p.intersects(geom):
                    number_points += 1
                    index_in_geom.append(affected_index_copy[n])

                    coords_in_country.remove(p)
                    affected_index.remove(affected_index_copy[n])

            if number_points == 0:
                # no point in geom --> will be attached to other voronoi cells
                unprocessed_polygons[m] = geom
                m += 1
                continue

            elif number_points == 1:
                # only one location in polygon --> polygon itself is voronoi cell
                i = index_in_geom[0]

                if isinstance(locations.loc[i, 'geometry'], Polygon) | isinstance(locations.loc[i, 'geometry'], MultiPolygon):
                    locations.loc[i, 'geometry'] = locations.loc[i, 'geometry'].union(geom)
                else:
                    locations.loc[i, 'geometry'] = geom
                continue

            elif number_points == 2:

                # simply split area into two parts
                ind1 = index_in_geom[0]
                ind2 = index_in_geom[1]

                lp1 = Point([locations.loc[ind1, 'longitude'], locations.loc[ind1, 'latitude']])
                lp2 = Point([locations.loc[ind2, 'longitude'], locations.loc[ind2, 'latitude']])

                split_area = divide_area_by_two(geom, lp1, lp2)

                for i in split_area.geoms:

                    if lp1.distance(i) == 0:
                        locations.loc[ind1, 'geometry'] = i
                    elif lp2.distance(i) == 0:
                        locations.loc[ind2, 'geometry'] = i
                    else:
                        unprocessed_polygons[m] = i
                        m += 1

                continue

            elif number_points > 2:
                # several points for geom --> create voronoi cells

                points = [Point([locations.loc[i, 'longitude'], locations.loc[i, 'latitude']]) for i in index_in_geom]

                # all latitudes cannot be the same. Same with longitude
                longitudes = set([p.x for p in points])
                if len(longitudes) == 1:  # all the same --> adjust slightly
                    adjusted_point = points[0]
                    adjusted_point = Point([adjusted_point.x + 0.00001, adjusted_point.y])
                    points = [adjusted_point] + [p for p in points[1:]]

                latitudes = set([p.y for p in points])
                if len(latitudes) == 1:  # all the same --> adjust slightly
                    adjusted_point = points[0]
                    adjusted_point = Point([adjusted_point.x, adjusted_point.y + 0.00001])
                    points = [adjusted_point] + [p for p in points[1:]]

                point_gdf = gpd.GeoDataFrame(geometry=points)
                point_gdf.set_crs(epsg=3395, inplace=True)

                region_polys, region_pts, missing = voronoi_regions_from_coords(point_gdf['geometry'], geom,
                                                                                return_unassigned_points=True)

                if missing:
                    print(missing)

                indexes = []
                for k in [*region_polys.keys()]:
                    index = index_in_geom[region_pts[k][0]]
                    locations.loc[index, 'geometry'] = region_polys[k]

                    indexes.append(index)

    # we need to process all polygons without a location point inside (some countries, islands, exclaves)
    # --> merge with closest voronoi cells

    # merge polygons which are adjacent since they should not be surrounded by other polygons without voronoi cells
    # unprocessed_polygons = merge_unattached_polygons(unprocessed_polygons)
    unprocessed_polygons = shapely.ops.unary_union([*unprocessed_polygons.values()])
    unprocessed_polygons = list(unprocessed_polygons.geoms)

    failed_polygons = []
    for area in unprocessed_polygons:

        if isinstance(area, shapely.geometry.Polygon):
            locations, not_processable_polygons = attach_polygon_to_other_voronoi_cells(area, locations)
            failed_polygons += not_processable_polygons
        else:
            for polygon in area.geoms:
                locations, not_processable_polygons = attach_polygon_to_other_voronoi_cells(polygon, locations)
                failed_polygons += not_processable_polygons

    # finally, some polygons can still not be processed (problem with geovoronoi)
    # --> attach them to closest voronoi cell if connected
    still_failed = []
    for poly in failed_polygons:
        locations, failed = attach_unprocessable_polygons_to_connected_voronoi_cell(poly, locations)
        if failed:
            still_failed.append(poly)

    # there are several islands and stuff which is not connected since there is no location on the same landmass.
    # if desired, add a location for each residual polygon of certain size
    if config_file['create_locations_for_islands']:
        geod = Geod(ellps="WGS84")
        max_location_ind = locations.index[-1] + 1

        country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
        world = gpd.read_file(country_shapefile)

        for poly in still_failed:
            lon, lat = poly.exterior.coords.xy
            area, _ = geod.polygon_area_perimeter(lon, lat)
            area = abs(area / 1e6)

            if area > config_file['island_area_threshold']:
                center = poly.representative_point()

                longitude = center.x
                latitude = center.y

                gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([longitude], [latitude])).set_crs('EPSG:4326')
                result = gpd.sjoin(gdf, world, how='left')
                country_start = result.at[result.index[0], 'NAME_EN']
                continent_start = result.at[result.index[0], 'CONTINENT']

                if continent_start == 'Antarctica':
                    continue

                locations.loc[max_location_ind, 'country_start'] = country_start
                locations.loc[max_location_ind, 'continent_start'] = continent_start

                locations.loc[max_location_ind, 'longitude'] = longitude
                locations.loc[max_location_ind, 'latitude'] = latitude

                locations.loc[max_location_ind, 'geometry'] = poly
                location_points.append(Point([longitude, latitude]))

                max_location_ind += 1

    colors = create_random_colors(len(locations.index))

    world_gdf = gpd.GeoDataFrame(geometry=locations['geometry'])
    locations_gdf = gpd.GeoDataFrame(geometry=location_points)
    fig, ax = plt.subplots(figsize=(15, 10))
    world_gdf.plot(ax=ax, color=colors)
    locations_gdf.plot(ax=ax, color=colors, edgecolor='white', markersize=0.1)

    plt.savefig(config_file['project_folder_path'] + '/starting_locations.svg')

    nan_values = locations[locations['geometry'].isnull()].index
    locations.drop(nan_values, inplace=True)

    return locations
