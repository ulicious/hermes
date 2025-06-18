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

from data_processing.helpers_misc import plot_points_and_areas, create_random_colors


def attach_polygon_to_other_voronoi_cells(polygon, locations):

    # check which voronoi cells are connected to unattached polygon
    connected_polygons = []

    residual_polygons = {}
    affected_locations = {}
    for i in locations.index:
        location_point = Point([locations.loc[i, 'longitude'], locations.loc[i, 'latitude']])
        location_polygon = locations.at[i, 'geometry']
        first = True

        if isinstance(location_polygon, float):
            continue

        if location_polygon.distance(polygon) == 0:
            # connected

            if isinstance(location_polygon, MultiPolygon):
                for geom in location_polygon.geoms:
                    if geom.distance(polygon) == 0:
                        connected_polygons.append(geom)
                        affected_locations[i] = location_point
                    else:
                        # geovoronoi only works with polygons. If multipolygon, use only part of multipolygon which
                        # is connected to polygon and attach rest later
                        if first:
                            residual_polygons[i] = [geom]
                            first = False
                        else:
                            residual_polygons[i].append(geom)
            else:
                connected_polygons.append(location_polygon)
                affected_locations[i] = location_point

    new_polygon = polygon
    for connected_polygon in connected_polygons:
        new_polygon = new_polygon.union(connected_polygon)

    new_polygon_gdf = gpd.GeoDataFrame(geometry=[new_polygon])
    new_polygon_gdf['geometry'] = shapely.set_precision(new_polygon_gdf.geometry, 1e-3)

    new_polygon = new_polygon_gdf['geometry'][0]

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
        region_polys, region_pts, missing = voronoi_regions_from_coords([*affected_locations.values()],
                                                                        new_polygon,
                                                                        return_unassigned_points=True)

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

    return locations


def merge_unattached_polygons(unattached_polygons):
    while True:
        old_areas = unattached_polygons.copy()

        merged = False
        for key_1 in [*unattached_polygons.keys()]:
            poly_1 = unattached_polygons[key_1]

            for key_2 in [*unattached_polygons.keys()]:

                if key_1 != key_2:

                    poly_2 = unattached_polygons[key_2]

                    if poly_1.intersects(poly_2):
                        poly_1 = poly_1.union(poly_2)

                        unattached_polygons[key_1] = poly_1
                        unattached_polygons.pop(key_2)

                        merged = True

                        break
            if merged:
                break

        if unattached_polygons == old_areas:
            break

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

            intersecting_subpolygons = [poly for poly in area_shape.geoms
                                        if any(poly.intersects(point) for point in coords_in_country)]

            not_intersecting_subpolygons = [poly for poly in area_shape.geoms
                                            if not any(poly.intersects(point) for point in coords_in_country)]

            for geom in not_intersecting_subpolygons:
                unprocessed_polygons[m] = geom
                m += 1

        else:
            intersecting_subpolygons = [area_shape]

        for geom in intersecting_subpolygons:

            points_in_geom = {}
            coords_in_country_copy = coords_in_country.copy()
            affected_index_copy = affected_index.copy()
            for n, p in enumerate(coords_in_country_copy):
                if p.intersects(geom):
                    points_in_geom[affected_index_copy[n]] = p

                    # remove used coords and index to not iterate repeatedly over the same coordinates
                    coords_in_country.remove(p)
                    affected_index.remove(affected_index_copy[n])

            if len(points_in_geom.keys()) == 0:
                # no point in geom --> will be attached to other voronoi cells
                unprocessed_polygons[m] = geom
                m += 1
                continue

            elif len(points_in_geom.keys()) == 1:
                # only one location in polygon --> polygon itself is voronoi cell
                i = [*points_in_geom.keys()][0]

                if isinstance(locations.loc[i, 'geometry'], Polygon) | isinstance(locations.loc[i, 'geometry'], MultiPolygon):
                    locations.loc[i, 'geometry'] = locations.loc[i, 'geometry'].union(geom)
                else:
                    locations.loc[i, 'geometry'] = geom
                continue

            elif len(points_in_geom.keys()) == 2:

                # simply split area into two parts
                ind1 = [*points_in_geom.keys()][0]
                ind2 = [*points_in_geom.keys()][1]

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

            elif len(points_in_geom.keys()) > 2:
                # several points for geom --> create voronoi cells

                point_gdf = gpd.GeoDataFrame(geometry=list(points_in_geom.values()))
                point_gdf.set_crs(epsg=3395, inplace=True)

                region_polys, region_pts, missing = voronoi_regions_from_coords(point_gdf['geometry'], geom,
                                                                                return_unassigned_points=True)

                if missing:
                    print(missing)

                indexes = []
                for k in [*region_polys.keys()]:
                    index = [*points_in_geom.keys()][region_pts[k][0]]
                    locations.loc[index, 'geometry'] = region_polys[k]

                    indexes.append(index)

    # we need to process all polygons without a location point inside (some countries, islands, exclaves)
    # --> merge with closest voronoi cells

    # merge polygons which are adjacent since they should not be surrounded by other polygons without voronoi cells
    unprocessed_polygons = merge_unattached_polygons(unprocessed_polygons)

    failed_polygons = []
    for poly in [*unprocessed_polygons.keys()]:

        area = unprocessed_polygons[poly]

        if isinstance(area, shapely.geometry.Polygon):
            polygon = area
            locations, not_processable_polygons = attach_polygon_to_other_voronoi_cells(polygon, locations)
            failed_polygons += not_processable_polygons
        else:
            for polygon in area.geoms:
                locations, not_processable_polygons = attach_polygon_to_other_voronoi_cells(polygon, locations)
                failed_polygons += not_processable_polygons

    # finally, some polygons can still not be processed (problem with geovoronoi)
    # --> attach them to closest voronoi cell if connected

    print(failed_polygons)
    for poly in failed_polygons:
        locations = attach_unprocessable_polygons_to_connected_voronoi_cell(poly, locations)

    colors = create_random_colors(len(locations.index))

    world_gdf = gpd.GeoDataFrame(geometry=locations['geometry'])
    locations_gdf = gpd.GeoDataFrame(geometry=location_points)
    fig, ax = plt.subplots()
    world_gdf.plot(ax=ax, color=colors)
    locations_gdf.plot(ax=ax, color=colors, edgecolor='white')
    plt.show()

    nan_values = locations[locations['geometry'].isnull()].index
    locations.drop(nan_values, inplace=True)

    return locations
