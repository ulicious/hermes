from algorithm.methods_geographic import calc_distance_list_to_list_no_matrix

import pandas as pd
from shapely.ops import unary_union

import geopandas as gpd
import cartopy.io.shapereader as shpreader

from itertools import combinations

def calculate_road_distances(tolerance, infrastructure, single_point=None, single_point_name=''):

    # todo: road distances between ports if they are not on same landmass --> should not exist

    # Get path to high-resolution land shapefile (global)
    shp_path = shpreader.natural_earth(resolution='10m', category='physical', name='land')

    # Load with GeoPandas
    land = gpd.read_file(shp_path)

    # Explode into distinct landmasses (connected polygons)
    land_union = unary_union(land.geometry)
    landmasses = gpd.GeoSeries(land_union).explode(index_parts=False)
    landmass_gdf = gpd.GeoDataFrame(geometry=landmasses, crs=land.crs)
    landmass_gdf['landmass_id'] = range(len(landmass_gdf.index))

    coords_gdf = gpd.GeoDataFrame(
        infrastructure,
        geometry=gpd.points_from_xy(infrastructure['longitude'], infrastructure['latitude']),
        crs=land.crs
    )

    if single_point is not None:
        single = gpd.GeoDataFrame({'longitude': single_point.x, 'latitude': single_point.y},
                                  index=[single_point_name],
                                  geometry=[single_point], columns=['longitude', 'latitude'],
                                  crs=land.crs
        )

        df_list = [coords_gdf, single]
        coords_gdf = gpd.GeoDataFrame(pd.concat(df_list, axis=0), crs=df_list[0].crs)

    # Spatial join: Assign each point to a landmass
    coords_with_landmass = gpd.sjoin(coords_gdf, landmass_gdf, how='left', predicate='within')
    # Now has 'landmass_id' for each point

    # Get all unique pairs
    names = coords_gdf.index.tolist()
    if single_point is None:
        pairs = list(combinations(names, 2))
    else:
        pairs = [(single_point_name, i) for i in names if i != single_point_name]

    # Prepare result
    results = []
    landmass_lookup = coords_with_landmass['landmass_id'].to_dict()

    for a, b in pairs:
        same = (landmass_lookup[a] == landmass_lookup[b]) and (landmass_lookup[a] is not None)
        results.append({'pointA': a, 'pointB': b, 'same_landmass': same})

    results_df = pd.DataFrame(results)
    results_df_true = results_df[results_df['same_landmass']]

    if single_point is None:
        list_longitude_1 = infrastructure.loc[results_df_true['pointA'], 'longitude']
        list_latitude_1 = infrastructure.loc[results_df_true['pointA'], 'latitude']
        list_longitude_2 = infrastructure.loc[results_df_true['pointB'], 'longitude']
        list_latitude_2 = infrastructure.loc[results_df_true['pointB'], 'latitude']
    else:
        list_longitude_1 = single['longitude']
        list_latitude_1 = single['latitude']
        list_longitude_2 = infrastructure.loc[results_df_true['pointB'], 'longitude']
        list_latitude_2 = infrastructure.loc[results_df_true['pointB'], 'latitude']

    distances = calc_distance_list_to_list_no_matrix(list_latitude_1, list_longitude_1, list_latitude_2, list_longitude_2)

    results_df_true['distance'] = distances

    in_tolerance_distances = results_df_true[results_df_true['distance'] <= tolerance].index

    results_df_true.loc[in_tolerance_distances, 'distance'] = 0

    return results_df_true


def calculate_efficiencies(distances, durations, boil_off, uses_commodity_as_shipping_fuel, self_consumption):

    total_boil_off = durations / 24 * boil_off
    total_self_consumption = distances / 1000 * self_consumption

    if uses_commodity_as_shipping_fuel:
        # if boil off is higher than self consumption, boil off will set efficiency, else self consumption
        efficiency = total_boil_off.combine(total_self_consumption, func=lambda s1, s2: s1.where(s1 > s2, s2))

        return efficiency
    else:
        return total_boil_off
