import os

from algorithm.methods_geographic import calc_distance_list_to_list_no_matrix

import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

import geopandas as gpd
import cartopy.io.shapereader as shpreader

from itertools import combinations


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


def _stack_distance_matrices(distance_matrices):
    """Convert square distance matrices into directed physical segments."""
    distances = []
    for distance_matrix in distance_matrices:
        directed = distance_matrix.stack().reset_index()
        directed.columns = ['pointA', 'pointB', 'distance']
        directed = directed[directed['pointA'] != directed['pointB']]
        distances.append(directed)

    if not distances:
        return pd.DataFrame(columns=['pointA', 'pointB', 'distance'])
    return pd.concat(distances, ignore_index=True)


def create_transport_edges(distance_options, commodities, techno_economic_data_transport):
    """Attach permitted commodities and techno-economic values to directed segments."""
    edges = {}
    max_costs = 0

    for transport_mean, distances in distance_options.items():
        if distances.empty:
            continue
        for row in distances.itertuples():
            if row.pointA == row.pointB:
                continue
            for commodity in commodities:
                if transport_mean not in techno_economic_data_transport[commodity]['potential_transportation']:
                    continue

                transport_costs = row.distance / 1000 * \
                    techno_economic_data_transport[commodity][transport_mean] / 1000
                transport_losses = 0
                max_costs = max(max_costs, transport_costs)

                if transport_mean == 'Shipping':
                    technology = techno_economic_data_transport[commodity]
                    boil_off = 0
                    if technology['Boil_Off'] > 0:
                        duration = row.distance / 1000 / technology['Shipping_Speed']
                        boil_off = duration / 24 * technology['Boil_Off']

                    self_consumption = 0
                    if technology['Uses_Commodity_as_Shipping_Fuel']:
                        self_consumption = row.distance / 1000 * technology['Self_Consumption']
                        transport_costs = 0

                    transport_losses = max(boil_off, self_consumption)

                key = row.pointA + '+' + commodity + '-' + row.pointB + '+' + commodity + '-' + transport_mean
                edges[key] = ('transport', row.pointA + '+' + commodity, row.pointB + '+' + commodity,
                              transport_costs, transport_losses, commodity, transport_mean)

    return edges, max_costs


def build_static_mip_graph(infrastructure_data, config_file, techno_economic_data_conversion,
                           techno_economic_data_transport, conversion_costs_and_efficiencies=None):
    """
    Build all MIP nodes and edges that do not depend on a chosen origin or destination.

    Start-to-infrastructure links and final sink links are deliberately omitted;
    `prepare_data` appends those for the current optimization run.
    """
    all_nodes = infrastructure_data['options'].index.tolist()
    all_commodities = config_file['available_commodity']

    if conversion_costs_and_efficiencies is None:
        conversion_costs_and_efficiencies = calculate_uniform_conversion_data(
            all_nodes, all_commodities, techno_economic_data_conversion)

    edges = {}
    conversion_edges = {}
    for node in all_nodes:
        if not conversion_costs_and_efficiencies.loc[node, 'conversion_possible']:
            continue
        for commodity_start in all_commodities:
            for commodity_end in techno_economic_data_conversion[commodity_start]['potential_conversions']:
                if commodity_end not in all_commodities:
                    continue
                conversion_costs = conversion_costs_and_efficiencies.loc[
                    node, commodity_start + '-' + commodity_end + '-conversion_costs']
                conversion_loss = 1 - conversion_costs_and_efficiencies.loc[
                    node, commodity_start + '-' + commodity_end + '-conversion_efficiency']
                key = node + '+' + commodity_start + '-' + node + '+' + commodity_end
                conversion_edges[key] = (node + '+' + commodity_start, node + '+' + commodity_end,
                                         conversion_costs, conversion_loss, commodity_end)
                edges[key] = ('conversion',) + conversion_edges[key]

    port_distances = infrastructure_data['port_distances'].stack().reset_index()
    port_distances.columns = ['pointA', 'pointB', 'distance']
    distance_options = {
        'Road': create_bidirectional_distances(infrastructure_data['road_distances']),
        'New_Pipeline_Gas': create_bidirectional_distances(infrastructure_data['new_pipeline_distances']),
        'New_Pipeline_Liquid': create_bidirectional_distances(infrastructure_data['new_pipeline_distances']),
        'Shipping': port_distances,
        'Pipeline_Gas': _stack_distance_matrices(infrastructure_data['gas_pipeline_matrices']),
        'Pipeline_Liquid': _stack_distance_matrices(infrastructure_data['oil_pipeline_matrices']),
    }
    distance_options = {mean: distance_options[mean] for mean in config_file['available_transport_means']
                        if mean in distance_options}
    transport_edges, max_costs = create_transport_edges(
        distance_options, all_commodities, techno_economic_data_transport)
    edges.update(transport_edges)

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
    """Persist global graph components for reuse by each origin/destination run."""
    pd.DataFrame(index=static_graph['nodes']).to_csv(path_mip_data + 'static_nodes.csv')
    static_graph['conversion_edges'].to_csv(path_mip_data + 'static_conversion_edges.csv')
    static_graph['transport_edges'].to_csv(path_mip_data + 'static_transport_edges.csv')


def load_static_mip_graph(path_mip_data):
    """Load previously processed global nodes and reconstruct edge tuples."""
    nodes = pd.read_csv(path_mip_data + 'static_nodes.csv', index_col=0).index.tolist()
    conversion_data = pd.read_csv(path_mip_data + 'static_conversion_edges.csv', index_col=0)
    transport_data = pd.read_csv(path_mip_data + 'static_transport_edges.csv', index_col=0)
    conversion_edges = {
        key: (row.start, row.end, row.costs, row.efficiency, row.end_commodity)
        for key, row in conversion_data.iterrows()
    }
    transport_edges = {
        key: (row.start, row.end, row.costs, row.efficiency, row.commodity, row['mean'])
        for key, row in transport_data.iterrows()
    }
    edges = {key: ('conversion',) + value for key, value in conversion_edges.items()}
    edges.update({key: ('transport',) + value for key, value in transport_edges.items()})
    max_costs = transport_data['costs'].max() if not transport_data.empty else 0
    return {
        'nodes': nodes,
        'edges': edges,
        'conversion_edges': conversion_data,
        'transport_edges': transport_data,
        'max_costs': max_costs,
    }


def prepare_global_mip_data(options, ports, config_file, techno_economic_data_conversion,
                            techno_economic_data_transport, conversion_costs_and_efficiencies,
                            path_processed_data):
    """Calculate and persist all MIP input that is common to every optimization location."""
    path_mip_data = path_processed_data + 'mip_data/'
    mip_options = options.copy()
    mip_options.loc[ports.index, 'longitude'] = mip_options.loc[ports.index, 'longitude_on_coastline']
    mip_options.loc[ports.index, 'latitude'] = mip_options.loc[ports.index, 'latitude_on_coastline']
    columns_to_drop = ['name', 'country', 'continent', 'longitude_on_coastline', 'latitude_on_coastline']
    mip_options.drop(columns=[column for column in columns_to_drop if column in mip_options.columns], inplace=True)
    mip_options.to_csv(path_mip_data + 'options.csv', encoding='utf-8', index=True)

    road_distances = calculate_road_distances(config_file['tolerance_distance'], mip_options)
    new_pipeline_distances = road_distances[
        road_distances['distance'] <= config_file['max_length_new_segment']].copy()
    road_distances = road_distances.copy()
    road_distances['distance'] *= config_file['no_road_multiplier']
    new_pipeline_distances['distance'] *= config_file['no_road_multiplier']
    road_distances.to_csv(path_mip_data + 'road_distances.csv', encoding='utf-8', index=True)
    new_pipeline_distances.to_csv(path_mip_data + 'new_pipeline_distances.csv', encoding='utf-8', index=True)

    port_distances = pd.read_csv(path_mip_data + 'port_distances.csv', index_col=0)
    for commodity in config_file['available_commodity']:
        technology = techno_economic_data_transport[commodity]
        if 'Shipping' not in technology['potential_transportation']:
            continue
        port_durations = port_distances / technology['Shipping_Speed'] / 1000
        efficiency = calculate_efficiencies(
            port_distances, port_durations, technology['Boil_Off'],
            technology['Uses_Commodity_as_Shipping_Fuel'], technology['Self_Consumption'])
        efficiency.to_csv(path_mip_data + commodity + '_efficiencies.csv', encoding='utf-8', index=True)

    conversion_costs_and_efficiencies.to_csv(
        path_mip_data + 'conversion_costs_and_efficiency.csv', encoding='utf-8', index=True)
    gas_matrices = [pd.read_csv(path_mip_data + file, index_col=0)
                    for file in os.listdir(path_mip_data) if file.startswith('PG')]
    oil_matrices = [pd.read_csv(path_mip_data + file, index_col=0)
                    for file in os.listdir(path_mip_data) if file.startswith('PL')]
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
        techno_economic_data_transport, conversion_costs_and_efficiencies)
    save_static_mip_graph(static_graph, path_mip_data)
    return mip_options


def prepare_destination_mip_data(options, destination, path_processed_data=None):
    """Determine the infrastructure nodes accepted as sinks for one destination."""
    destination_infrastructure = []
    if hasattr(destination, 'contains'):
        for option in options.index:
            option_point = Point([options.loc[option, 'longitude'], options.loc[option, 'latitude']])
            if destination.contains(option_point):
                destination_infrastructure.append(option)

    result = pd.DataFrame(destination_infrastructure, columns=['destination_infrastructure'])
    if path_processed_data is not None:
        result.to_csv(path_processed_data + 'mip_data/destination_infrastructure.csv',
                      encoding='utf-8', index=True)
    return result
