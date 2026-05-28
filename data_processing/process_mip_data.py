import os
import logging
import math

from algorithm.methods_geographic import calc_distance_list_to_list_no_matrix

import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

import geopandas as gpd
import cartopy.io.shapereader as shpreader

from itertools import combinations
from tqdm import tqdm

from mixed_integer_program.mip_data_helpers import create_transport_edges


logger = logging.getLogger(__name__)


def create_bidirectional_distances(distances):
    """Expand unordered physical connections into both transport directions."""
    reverse_distances = distances.copy()
    reverse_distances[['pointA', 'pointB']] = distances[['pointB', 'pointA']].to_numpy()
    return pd.concat([distances, reverse_distances], ignore_index=True)


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


def calculate_uniform_conversion_data(nodes, commodities, techno_economic_data_conversion):
    """Create conversion inputs for an in-memory infrastructure without local cost data."""
    uniform_costs = techno_economic_data_conversion['uniform_costs']
    interest_rate = uniform_costs['interest_rate']
    conversion_data = pd.DataFrame(index=nodes)
    conversion_data['conversion_possible'] = True

    for commodity_start in commodities:
        for commodity_end in techno_economic_data_conversion[commodity_start]['potential_conversions']:
            technology = techno_economic_data_conversion[commodity_start][commodity_end]
            annuity_factor = (interest_rate * (1 + interest_rate) ** technology['lifetime']) / \
                ((1 + interest_rate) ** technology['lifetime'] - 1)
            conversion_costs = technology['specific_investment'] * \
                (annuity_factor + technology['fixed_maintenance']) / technology['operating_hours'] + \
                uniform_costs['Electricity'] * technology['electricity_demand'] + \
                uniform_costs['CO2'] * technology['co2_demand'] + \
                uniform_costs['Nitrogen'] * technology['nitrogen_demand']

            conversion_data[commodity_start + '-' + commodity_end + '-conversion_costs'] = conversion_costs
            conversion_data[commodity_start + '-' + commodity_end + '-conversion_efficiency'] = \
                technology['efficiency_autothermal']

    return conversion_data


def _stack_distance_matrices(distance_matrices, show_progress=False, description='Stack pipeline matrices'):
    """Convert square distance matrices into directed physical segments."""
    distances = []
    iterator = tqdm(distance_matrices, desc=description, disable=not show_progress)
    for distance_matrix in iterator:
        directed = distance_matrix.stack().reset_index()
        directed.columns = ['pointA', 'pointB', 'distance']
        directed = directed[directed['pointA'] != directed['pointB']]
        distances.append(directed)

    if not distances:
        return pd.DataFrame(columns=['pointA', 'pointB', 'distance'])
    return pd.concat(distances, ignore_index=True)


def build_static_mip_graph(infrastructure_data, config_file, techno_economic_data_conversion,
                           techno_economic_data_transport, conversion_costs_and_efficiencies=None,
                           show_progress=False):
    """
    Build all MIP nodes and edges that do not depend on a chosen origin or destination.

    Start-to-infrastructure links and final sink links are deliberately omitted;
    `prepare_data` appends those for the current optimization run.
    """
    all_nodes = infrastructure_data['options'].index.tolist()
    all_commodities = config_file['available_commodity']
    logger.info('Build static MIP graph for %s infrastructure nodes and %s commodities',
                len(all_nodes), len(all_commodities))

    if conversion_costs_and_efficiencies is None:
        logger.info('Calculate uniform conversion inputs for in-memory infrastructure')
        conversion_costs_and_efficiencies = calculate_uniform_conversion_data(
            all_nodes, all_commodities, techno_economic_data_conversion)

    edges = {}
    conversion_edges = {}
    node_iterator = tqdm(all_nodes, desc='Create conversion edges', disable=not show_progress)
    for node in node_iterator:
        if not conversion_costs_and_efficiencies.loc[node, 'conversion_possible']:
            continue
        for commodity_start in all_commodities:
            for commodity_end in techno_economic_data_conversion[commodity_start]['potential_conversions']:
                if commodity_end not in all_commodities:
                    continue
                conversion_costs = conversion_costs_and_efficiencies.loc[
                    node, commodity_start + '-' + commodity_end + '-conversion_costs']
                if conversion_costs == math.inf:
                    continue
                conversion_loss = 1 - conversion_costs_and_efficiencies.loc[
                    node, commodity_start + '-' + commodity_end + '-conversion_efficiency']
                key = node + '+' + commodity_start + '-' + node + '+' + commodity_end
                conversion_edges[key] = (node + '+' + commodity_start, node + '+' + commodity_end,
                                         conversion_costs, conversion_loss, commodity_end)
                edges[key] = ('conversion',) + conversion_edges[key]

    logger.info('Prepare directed infrastructure distances for static transport edges')
    port_distances = infrastructure_data['port_distances'].stack().reset_index()
    port_distances.columns = ['pointA', 'pointB', 'distance']
    distance_options = {
        'Road': create_bidirectional_distances(infrastructure_data['road_distances']),
        'New_Pipeline_Gas': create_bidirectional_distances(infrastructure_data['new_pipeline_distances']),
        'New_Pipeline_Liquid': create_bidirectional_distances(infrastructure_data['new_pipeline_distances']),
        'Shipping': port_distances,
        'Pipeline_Gas': _stack_distance_matrices(
            infrastructure_data['gas_pipeline_matrices'], show_progress, 'Stack gas pipeline matrices'),
        'Pipeline_Liquid': _stack_distance_matrices(
            infrastructure_data['oil_pipeline_matrices'], show_progress, 'Stack liquid pipeline matrices'),
    }
    distance_options = {mean: distance_options[mean] for mean in config_file['available_transport_means']
                        if mean in distance_options}
    transport_edges, max_costs = create_transport_edges(
        distance_options, all_commodities, techno_economic_data_transport, show_progress=show_progress)
    edges.update(transport_edges)
    logger.info('Static MIP graph contains %s conversion edges and %s transport edges',
                len(conversion_edges), len(transport_edges))

    columns_conversion = ['start', 'end', 'costs', 'efficiency', 'end_commodity']
    columns_transport = ['start', 'end', 'costs', 'efficiency', 'commodity', 'mean']
    return {
        'nodes': [node + '+' + commodity for node in all_nodes for commodity in all_commodities],
        'physical_nodes': all_nodes,
        'edges': edges,
        'conversion_edges': pd.DataFrame.from_dict(conversion_edges, orient='index',
                                                   columns=columns_conversion),
        'transport_edges': pd.DataFrame.from_dict(
            {key: value[1:] for key, value in transport_edges.items()},
            orient='index', columns=columns_transport),
        'max_costs': max_costs,
    }


def save_static_mip_graph(static_graph, path_mip_data):
    """
    Persist all origin-independent nodes and edges in one binary graph file.

    CSV was useful while inspecting the initial graph expansion, but loading
    it requires parsing large text tables and rebuilding every edge tuple.
    The pickle contains the exact in-memory representation needed by the MIP.
    """
    graph_file = path_mip_data + 'static_graph.pkl'
    logger.info('Save static MIP graph with %s nodes and %s edges to %s',
                len(static_graph['nodes']), len(static_graph['edges']), graph_file)
    pd.to_pickle(static_graph, graph_file)


def prepare_global_mip_data(options, ports, config_file, techno_economic_data_conversion,
                            techno_economic_data_transport, conversion_costs_and_efficiencies,
                            path_processed_data):
    """Calculate and persist all MIP input that is common to every optimization location."""
    path_mip_data = path_processed_data + 'mip_data/'
    logger.info('Prepare global MIP data independent from origin and destination')
    mip_options = options.copy()
    mip_options.loc[ports.index, 'longitude'] = mip_options.loc[ports.index, 'longitude_on_coastline']
    mip_options.loc[ports.index, 'latitude'] = mip_options.loc[ports.index, 'latitude_on_coastline']
    columns_to_drop = ['name', 'country', 'continent', 'longitude_on_coastline', 'latitude_on_coastline']
    mip_options.drop(columns=[column for column in columns_to_drop if column in mip_options.columns], inplace=True)
    mip_options.to_csv(path_mip_data + 'options.csv', encoding='utf-8', index=True)
    logger.info('Saved %s global infrastructure options', len(mip_options))

    logger.info('Calculate global road and new-pipeline distances')
    road_distances = calculate_road_distances(config_file['tolerance_distance'], mip_options)
    new_pipeline_distances = road_distances[
        road_distances['distance'] <= config_file['max_length_new_segment']].copy()
    road_distances = road_distances.copy()
    road_distances['distance'] *= config_file['no_road_multiplier']
    new_pipeline_distances['distance'] *= config_file['no_road_multiplier']
    road_distances.to_csv(path_mip_data + 'road_distances.csv', encoding='utf-8', index=True)
    new_pipeline_distances.to_csv(path_mip_data + 'new_pipeline_distances.csv', encoding='utf-8', index=True)
    logger.info('Saved %s road and %s new-pipeline physical connections',
                len(road_distances), len(new_pipeline_distances))

    logger.info('Calculate shipping efficiencies per commodity')
    port_distances = pd.read_csv(path_mip_data + 'port_distances.csv', index_col=0)
    commodity_iterator = tqdm(config_file['available_commodity'], desc='Shipping efficiencies')
    for commodity in commodity_iterator:
        technology = techno_economic_data_transport[commodity]
        if 'Shipping' not in technology['potential_transportation']:
            continue
        port_durations = port_distances / technology['Shipping_Speed'] / 1000
        efficiency = calculate_efficiencies(
            port_distances, port_durations, technology['Boil_Off'],
            technology['Uses_Commodity_as_Shipping_Fuel'], technology['Self_Consumption'])
        efficiency.to_csv(path_mip_data + commodity + '_efficiencies.csv', encoding='utf-8', index=True)

    logger.info('Save global conversion costs and read existing pipeline distance matrices')
    conversion_costs_and_efficiencies.to_csv(
        path_mip_data + 'conversion_costs_and_efficiency.csv', encoding='utf-8', index=True)
    gas_files = [file for file in os.listdir(path_mip_data) if file.startswith('PG')]
    oil_files = [file for file in os.listdir(path_mip_data) if file.startswith('PL')]
    gas_matrices = [pd.read_csv(path_mip_data + file, index_col=0)
                    for file in tqdm(gas_files, desc='Load gas pipeline matrices')]
    oil_matrices = [pd.read_csv(path_mip_data + file, index_col=0)
                    for file in tqdm(oil_files, desc='Load liquid pipeline matrices')]
    infrastructure_data = {
        'options': mip_options,
        'road_distances': road_distances,
        'new_pipeline_distances': new_pipeline_distances,
        'port_distances': port_distances,
        'gas_pipeline_matrices': gas_matrices,
        'oil_pipeline_matrices': oil_matrices,
    }
    static_graph = build_static_mip_graph(
        infrastructure_data, config_file, techno_economic_data_conversion,
        techno_economic_data_transport, conversion_costs_and_efficiencies, show_progress=True)
    save_static_mip_graph(static_graph, path_mip_data)
    logger.info('Finished global MIP preparation')
    return mip_options


def prepare_minimal_mip_case(config_file, techno_economic_data_conversion,
                             techno_economic_data_transport, path_processed_data):
    """
    Persist the hardcoded MIP test case before optimization.

    The case file contains the already expanded static graph plus only the
    start/end-specific inputs that `prepare_data` must add at run time.
    """
    logger.info('Prepare hardcoded minimal MIP case')
    minimal_nodes = ['s_0', 's_1', 's_2', 's_3',
                     'PG_0', 'PG_1', 'PG_2',
                     'PL_0', 'PL_1']
    minimal_infrastructure = {
        'options': pd.DataFrame(index=minimal_nodes),
        'road_distances': pd.DataFrame([
            ('PG_1', 's_0', 1000),
            ('PG_2', 's_0', 30_000_000),
            ('PG_2', 's_2', 25_000_000),
            ('s_3', 'PL_0', 30_000_000),
        ], columns=['pointA', 'pointB', 'distance']),
        'start_road_distances': pd.DataFrame([
            ('start', 's_0', 100_000_000_000),
            ('start', 's_2', 50_000_000),
        ], columns=['pointA', 'pointB', 'distance']),
        'new_pipeline_distances': pd.DataFrame([
            ('s_1', 'PL_0', 1000),
            ('s_1', 'PL_1', 1_000_000_000),
            ('s_3', 'PL_0', 1_000_000_000),
        ], columns=['pointA', 'pointB', 'distance']),
        'start_new_pipeline_distances': pd.DataFrame([
            ('start', 'PG_0', 1000),
            ('start', 'PG_1', 100_000_000_000),
            ('start', 'PL_0', 100_000_000_000),
            ('start', 'PG_2', 50_000_000),
        ], columns=['pointA', 'pointB', 'distance']),
        'port_distances': pd.DataFrame([
            [0, 1000, 2_000_000, 3_000_000],
            [1000, 0, 2_000_000, 2_500_000],
            [2_000_000, 2_000_000, 0, 1_500_000],
            [3_000_000, 2_500_000, 1_500_000, 0],
        ], index=['s_0', 's_1', 's_2', 's_3'],
           columns=['s_0', 's_1', 's_2', 's_3']),
        'gas_pipeline_matrices': [pd.DataFrame([
            [0, 1000, 40_000_000],
            [1000, 0, 20_000_000],
            [40_000_000, 20_000_000, 0],
        ], index=['PG_0', 'PG_1', 'PG_2'],
           columns=['PG_0', 'PG_1', 'PG_2'])],
        'oil_pipeline_matrices': [pd.DataFrame([
            [0, 1000],
            [1000, 0],
        ], index=['PL_0', 'PL_1'], columns=['PL_0', 'PL_1'])],
    }
    static_graph = build_static_mip_graph(
        minimal_infrastructure, config_file, techno_economic_data_conversion,
        techno_economic_data_transport)
    minimal_path = path_processed_data + 'mip_data/minimal/'
    os.makedirs(minimal_path, exist_ok=True)
    save_static_mip_graph(static_graph, minimal_path)

    minimal_case = {
        'static_graph': static_graph,
        'start_location_data': pd.Series({
            'Hydrogen_Gas': 0, 'Ammonia': 10, 'Methane_Gas': 5,
            'Methane_Liquid': 10, 'Methanol': 12, 'FTF': 15,
        }),
        'start_road_distances': minimal_infrastructure['start_road_distances'],
        'start_new_pipeline_distances': minimal_infrastructure['start_new_pipeline_distances'],
        'end_location': ['PL_0'],
        'warm_start_route': [
            'start+Hydrogen_Gas-PG_0+Hydrogen_Gas-New_Pipeline_Gas',
            'PG_0+Hydrogen_Gas-PG_1+Hydrogen_Gas-Pipeline_Gas',
            'PG_1+Hydrogen_Gas-s_0+Hydrogen_Gas-Road',
            's_0+Hydrogen_Gas-s_0+FTF',
            's_0+FTF-s_1+FTF-Shipping',
            's_1+FTF-PL_0+FTF-New_Pipeline_Liquid',
            'PL_0+FTF-end',
        ],
    }
    case_file = path_processed_data + 'mip_data/minimal_case.pkl'
    pd.to_pickle(minimal_case, case_file)
    logger.info('Saved minimal MIP case with %s static nodes and %s static edges to %s',
                len(minimal_case['static_graph']['nodes']),
                len(minimal_case['static_graph']['edges']), case_file)
