import math
import gc
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

from shapely.geometry import Point

from algorithm.methods_geographic import calc_distance_list_to_single, calc_distance_list_to_list
from algorithm.methods_cost_approximations import calculate_cheapest_option_to_closest_infrastructure, \
    calculate_cheapest_option_to_final_destination
from algorithm.methods_algorithm import remove_duplicate_branches
from algorithm.tracking import branch_count, get_tracker

import warnings
warnings.filterwarnings("ignore")


def _load_shipping_distances(path_processed_data):
    path_file = path_processed_data + 'inner_infrastructure_distances/port_distances.csv'
    if not os.path.exists(path_file):
        return pd.DataFrame()
    shipping_distances = pd.read_csv(path_file, index_col=0, header=0, dtype=str, sep=None, engine='python',
                                     keep_default_na=False)
    if shipping_distances.empty:
        return shipping_distances
    return np.ceil(shipping_distances.apply(pd.to_numeric, errors='raise'))


def _build_options_from_mask(values, row_index, column_index, mask):
    row_positions, column_positions = np.nonzero(mask)
    if len(row_positions) == 0:
        return pd.DataFrame(columns=['previous_branch', 'current_node', 'current_distance'])

    row_index = np.asarray(row_index, dtype=object)
    column_index = np.asarray(column_index, dtype=object)
    return pd.DataFrame({
        'previous_branch': column_index[column_positions],
        'current_node': row_index[row_positions],
        'current_distance': values[row_positions, column_positions],
    })


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _filter_shipping_infrastructure_by_destination_continents(shipping_infrastructure, destination_continents):
    destination_continents = _as_list(destination_continents)
    target_continents = set(destination_continents)

    if target_continents.intersection({'Europe', 'Asia', 'Africa'}):
        target_continents.update(['Europe', 'Asia', 'Africa'])

    return shipping_infrastructure[shipping_infrastructure['continent'].isin(target_continents)]


def _is_valid_continent(continent):
    return continent is not None and pd.notna(continent) and str(continent).lower() != 'nan'


def _get_reachable_continents(continent, data):
    if not _is_valid_continent(continent):
        return None

    continent = str(continent)
    continent_connections = data.get('continent_connections', {})
    reachable_continents = continent_connections.get('reachable_continents', {})
    return set(reachable_continents.get(continent, [continent]))


def _get_branch_reachable_continent_union(branches, data):
    if 'current_continent' not in branches.columns:
        return None

    reachable_union = set()
    for continent in branches['current_continent'].dropna().unique():
        reachable = _get_reachable_continents(continent, data)
        if reachable is not None:
            reachable_union.update(reachable)

    if not reachable_union:
        return None
    return reachable_union


def _filter_infrastructure_by_reachable_continents(infrastructure_index, complete_infrastructure, branches, data):
    if 'continent' not in complete_infrastructure.columns:
        return infrastructure_index

    reachable_union = _get_branch_reachable_continent_union(branches, data)
    if reachable_union is None:
        return infrastructure_index

    infrastructure_continents = complete_infrastructure.loc[infrastructure_index, 'continent']
    keep_mask = (
        infrastructure_continents.apply(lambda value: not _is_valid_continent(value))
        | infrastructure_continents.astype(str).isin(reachable_union)
    )
    return list(pd.Index(infrastructure_index)[keep_mask.to_numpy()])


def _filter_infrastructure_by_reachable_set(infrastructure_index, complete_infrastructure, reachable_continents):
    if reachable_continents is None or 'continent' not in complete_infrastructure.columns:
        return list(infrastructure_index)

    infrastructure_continents = complete_infrastructure.loc[infrastructure_index, 'continent']
    keep_mask = (
        infrastructure_continents.apply(lambda value: not _is_valid_continent(value))
        | infrastructure_continents.astype(str).isin(reachable_continents)
    )
    return list(pd.Index(infrastructure_index)[keep_mask.to_numpy()])


def _build_reachable_distance_blocks(complete_infrastructure, infrastructure_index, branches_no_duplicates,
                                     branches, data):
    if (
        'continent' not in complete_infrastructure.columns
        or 'current_continent' not in branches.columns
        or branches_no_duplicates.empty
    ):
        distances = calc_distance_list_to_list(
            complete_infrastructure.loc[infrastructure_index, 'latitude'],
            complete_infrastructure.loc[infrastructure_index, 'longitude'],
            branches_no_duplicates['latitude'],
            branches_no_duplicates['longitude'])
        return [{
            'row_index': pd.Index(infrastructure_index),
            'column_nodes': pd.Index(branches_no_duplicates['current_node']),
            'values': np.asarray(distances).transpose(),
        }]

    branch_continents = branches_no_duplicates['current_continent']
    grouped_nodes = {}
    for row_index, continent in branch_continents.items():
        reachable = _get_reachable_continents(continent, data)
        if reachable is None:
            reachable_key = None
        else:
            reachable_key = tuple(sorted(reachable))
        grouped_nodes.setdefault(reachable_key, []).append(row_index)

    distance_blocks = []
    for reachable_key, branch_indices in grouped_nodes.items():
        branch_block = branches_no_duplicates.loc[branch_indices]
        reachable = set(reachable_key) if reachable_key is not None else None
        infrastructure_block_index = _filter_infrastructure_by_reachable_set(
            infrastructure_index, complete_infrastructure, reachable)
        if len(infrastructure_block_index) == 0 or branch_block.empty:
            continue

        distances = calc_distance_list_to_list(
            complete_infrastructure.loc[infrastructure_block_index, 'latitude'],
            complete_infrastructure.loc[infrastructure_block_index, 'longitude'],
            branch_block['latitude'],
            branch_block['longitude'])
        distance_blocks.append({
            'row_index': pd.Index(infrastructure_block_index),
            'column_nodes': pd.Index(branch_block['current_node']),
            'values': np.asarray(distances).transpose(),
        })

    return distance_blocks


def _apply_reachable_continent_mask(mask, row_index, column_index, branches, complete_infrastructure, data):
    if (
        mask.size == 0
        or 'current_continent' not in branches.columns
        or 'continent' not in complete_infrastructure.columns
    ):
        return 0

    row_index = pd.Index(row_index)
    column_index = pd.Index(column_index)
    row_continents = complete_infrastructure.reindex(row_index)['continent']
    if row_continents.empty:
        return 0

    before = int(mask.sum())
    branch_continents = branches.reindex(column_index)['current_continent']

    for continent in branch_continents.dropna().unique():
        reachable = _get_reachable_continents(continent, data)
        if reachable is None:
            continue

        affected_columns = np.flatnonzero(branch_continents.astype(str).to_numpy() == str(continent))
        if len(affected_columns) == 0:
            continue

        allowed_rows = (
            row_continents.apply(lambda value: not _is_valid_continent(value))
            | row_continents.astype(str).isin(reachable)
        ).to_numpy()
        mask[np.ix_(~allowed_rows, affected_columns)] = False

    return before - int(mask.sum())


def _remove_visited_options_from_mask(mask, row_index, column_index, visited_infrastructure):
    row_lookup = pd.Index(row_index)
    column_lookup = {branch: position for position, branch in enumerate(column_index)}

    for infrastructure_data in visited_infrastructure.values():
        affected_columns = [column_lookup[branch]
                            for branch in infrastructure_data['branches']
                            if branch in column_lookup]
        if not affected_columns:
            continue

        row_positions = row_lookup.get_indexer(infrastructure_data['nodes'])
        row_positions = row_positions[row_positions >= 0]
        if len(row_positions) == 0:
            continue

        mask[np.ix_(row_positions, affected_columns)] = False


def get_complete_infrastructure(data, config_file):

    """
    Method to collect all ports, nodes and destination in one single dataframe

    @param dict data: dictionary with common data
    @param dict config_file: contains all configurations

    @return: pandas.DataFrame with all nodes, ports and destination
    """

    complete_infrastructure = pd.DataFrame()
    infrastructure_to_concat = []
    final_destination = data['destination']['location']
    for m in ['Road', 'Shipping', 'Pipeline_Gas', 'Pipeline_Liquid']:

        if m == 'Road':

            if (config_file['destination_type'] == 'location'
                    or 'Destination' in _as_list(data['destination'].get('infrastructure'))):
                # Check final destination and add to option outside tolerance if applicable
                if isinstance(final_destination, pd.Series):
                    destination_point = Point([final_destination['longitude'], final_destination['latitude']])
                elif hasattr(final_destination, 'representative_point') and not hasattr(final_destination, 'x'):
                    destination_point = final_destination.representative_point()
                else:
                    destination_point = final_destination

                complete_infrastructure.loc['Destination', 'latitude'] = destination_point.y
                complete_infrastructure.loc['Destination', 'longitude'] = destination_point.x
                complete_infrastructure.loc['Destination', 'current_transport_mean'] = m
                complete_infrastructure.loc['Destination', 'graph'] = None
                complete_infrastructure.loc['Destination', 'continent'] = _as_list(data['destination']['continent'])[0]

            continue

        # get all complete_infrastructure of current mean of transport
        if m == 'Shipping':

            # get all complete_infrastructure of current mean of transport
            shipping_infrastructure = data[m]['ports']
            shipping_infrastructure['current_transport_mean'] = m
            shipping_infrastructure['graph'] = None

            infrastructure_to_concat.append(shipping_infrastructure)

        else:
            networks = data[m].keys()
            for n in networks:
                network_infrastructure = data[m][n]['NodeLocations'].copy()
                network_infrastructure['current_transport_mean'] = m
                infrastructure_to_concat.append(network_infrastructure)

    complete_infrastructure = pd.concat([complete_infrastructure] + infrastructure_to_concat)

    # create common infrastructure column
    complete_infrastructure['infrastructure'] = complete_infrastructure.index
    graph_df = complete_infrastructure[complete_infrastructure['graph'].apply(lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) if isinstance(x, (list, float)) else False)]
    complete_infrastructure.loc[graph_df.index, 'infrastructure'] = complete_infrastructure.loc[graph_df.index, 'infrastructure']

    return complete_infrastructure


def create_branches_from_in_tolerance_locations(data, branches, complete_infrastructure, benchmarks, configuration):

    """
    Create zero-distance follow-up branches for already assessed infrastructure branches based on
    data['in_tolerance_locations'].

    @param dict data: dictionary with common data
    @param pandas.DataFrame branches: dataframe with already generated and assessed branches
    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param dict benchmarks: current benchmarks
    @param dict configuration: dictionary with configuration

    @return: pandas.DataFrame with new zero-distance branches
    """

    if branches.empty:
        return pd.DataFrame()

    in_tolerance_locations = data['in_tolerance_locations']

    previous_branches = []
    starting_points = []
    current_nodes = []

    for branch in branches.itertuples():
        location_name = branch.current_node
        next_locations = in_tolerance_locations.get(location_name, [])

        if not next_locations:
            continue

        visited_infrastructure = set(branch.all_previous_infrastructure)

        for next_location in next_locations:
            if next_location == location_name:
                continue

            if next_location not in complete_infrastructure.index:
                continue

            target_graph = complete_infrastructure.at[next_location, 'graph']
            if isinstance(target_graph, str):
                if target_graph in visited_infrastructure:
                    continue
            elif next_location in visited_infrastructure:
                continue

            previous_branches.append(branch.Index)
            starting_points.append(location_name)
            current_nodes.append(next_location)

    if not previous_branches:
        return pd.DataFrame()

    direct_branches = pd.DataFrame({
        'previous_branch': previous_branches,
        'current_node': current_nodes,
        'current_distance': np.zeros(len(previous_branches), dtype=np.float32),
        'starting_point': starting_points,
        'current_transport_mean': 'Road',
        'current_infrastructure': None,
        'specific_transportation_costs': np.zeros(len(previous_branches), dtype=np.float32),
        'current_transportation_costs': np.zeros(len(previous_branches), dtype=np.float32),
        'current_total_costs': branches.loc[previous_branches, 'current_total_costs'].to_numpy(),
        'total_efficiency': branches.loc[previous_branches, 'total_efficiency'].to_numpy(),
        'current_commodity': branches.loc[previous_branches, 'current_commodity'].tolist(),
        'current_commodity_object': branches.loc[previous_branches, 'current_commodity_object'].tolist(),
    })

    direct_branches['taken_route'] \
        = list(zip(direct_branches['starting_point'],
                   ['Road'] * len(direct_branches.index),
                   direct_branches['current_distance'],
                   direct_branches['current_node'],
                   [1] * len(direct_branches.index)))

    direct_branches['benchmark'] = direct_branches['current_commodity'].map(benchmarks)
    direct_branches = direct_branches[direct_branches['current_total_costs'] <= direct_branches['benchmark']]

    if direct_branches.empty:
        return pd.DataFrame()

    nodes_list = direct_branches['current_node'].tolist()
    direct_branches['latitude'] = complete_infrastructure.loc[nodes_list, 'latitude'].tolist()
    direct_branches['longitude'] = complete_infrastructure.loc[nodes_list, 'longitude'].tolist()

    direct_branches.sort_values(['current_total_costs'], inplace=True)
    direct_branches = remove_duplicate_branches(direct_branches)

    final_destination = data['destination']['location']

    if configuration['destination_type'] == 'location':
        direct_branches['distance_to_final_destination'] \
            = calc_distance_list_to_single(direct_branches['latitude'], direct_branches['longitude'],
                                           final_destination.y, final_destination.x)
    else:
        infrastructure_in_destination = data['destination']['infrastructure']
        distances = calc_distance_list_to_list(direct_branches['latitude'], direct_branches['longitude'],
                                               infrastructure_in_destination['latitude'],
                                               infrastructure_in_destination['longitude'])
        direct_branches['distance_to_final_destination'] = np.asarray(distances).min(axis=0)

    in_destination_tolerance \
        = direct_branches[direct_branches['distance_to_final_destination']
                          <= configuration['to_final_destination_tolerance']].index
    direct_branches.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

    min_values, min_commodities \
        = calculate_cheapest_option_to_closest_infrastructure(data, direct_branches, configuration,
                                                              benchmarks, 'current_total_costs')

    direct_branches['minimal_total_costs'] = min_values
    direct_branches['minimal_commodity'] = min_commodities

    direct_branches['benchmark'] = direct_branches['minimal_commodity'].map(benchmarks)
    direct_branches = direct_branches[direct_branches['minimal_total_costs'] <= direct_branches['benchmark']]

    if direct_branches.empty:
        return pd.DataFrame()

    return direct_branches


def process_out_tolerance_branches(complete_infrastructure, branches, configuration, iteration, data, benchmarks,
                                   use_minimal_distance=False, limitation=None):

    """
    Method to assess potential transportation destinations via road or new pipelines based on current branches
    and all available infrastructure.

    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param pandas.DataFrame branches: DataFrame with current branches
    @param dict configuration: dictionary with configuration
    @param int iteration: current iteration
    @param dict data: dictionary with common data
    @param dict benchmarks: current benchmarks
    @param bool use_minimal_distance: (optional) boolean to set if minimal distances are used to assess locations
    @param str limitation: (optional) determines which infrastructure can actually be used

    @return: pandas.DataFrame with new branches
    """

    tolerance_distance = configuration['tolerance_distance']
    max_length_new_segment = configuration['max_length_new_segment']
    max_length_road = configuration['max_length_road']
    no_road_multiplier = configuration['no_road_multiplier']
    tracker = get_tracker(data)
    method = 'process_out_tolerance_branches'
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_out', method=method,
                      event='input', before=branch_count(branches),
                      runtime_s=0.0,
                      details={'use_minimal_distance': use_minimal_distance,
                               'limitation': limitation})
    time_candidate_generation = time.perf_counter()

    if iteration == 0:
        # if iteration is 0, we don't make any preselection since we have only a very limited amount of branches
        # and calculating distances for these few branches is possible without long computation times

        # only use options which are actually reachable from start
        complete_infrastructure = complete_infrastructure[complete_infrastructure['reachable_from_start']]

        # always consider infrastructure in destination
        in_tolerance_to_destination_infrastructure = complete_infrastructure[complete_infrastructure['distance_to_destination'] < configuration['to_final_destination_tolerance']].index.tolist()

        if limitation == 'no_pipeline_gas':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PG' not in i]
            if not use_minimal_distance:
                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'no_pipeline_liquid':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PL' not in i]
            if not use_minimal_distance:
                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'no_pipelines':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i]
            if not use_minimal_distance:
                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'only_in_tolerance':  # these will immediately terminate since destinations are in tolerance
            reduced_infrastructure_index = in_tolerance_to_destination_infrastructure
            if not use_minimal_distance:
                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])


        else:  # don't limit infrastructure at all
            reduced_infrastructure_index = complete_infrastructure.index
            if not use_minimal_distance:
                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        road_transportation_costs = {}

        new_transportation_costs = {}

        branches_to_keep_road = []
        branches_to_keep_new = []

        distance_values = np.asarray(distances).transpose()
        reduced_infrastructure_index = np.asarray(reduced_infrastructure_index, dtype=object)
        road_values = distance_values.copy()
        road_mask = np.zeros(distance_values.shape, dtype=bool)
        new_infrastructure_mask = np.zeros(distance_values.shape, dtype=bool)

        # for each branch, assess if new pipelines or road is applicable based on current commodity
        for branch_index in branches.index:
            current_commodity_object = branches.at[branch_index, 'current_commodity_object']
            branch_position = branches.index.get_loc(branch_index)

            pipeline_applicable \
                = current_commodity_object.get_transportation_options_specific_mean_of_transport('New_Pipeline_Gas') \
                | current_commodity_object.get_transportation_options_specific_mean_of_transport('New_Pipeline_Liquid')

            # check which transport mean was used in previous iteration
            was_not_road = branches.at[branch_index, 'current_transport_mean'] != 'Road'
            was_not_new = branches.at[branch_index, 'current_transport_mean'] not in ['New_Pipeline_Gas', 'New_Pipeline_Liquid']

            # we cannot use road or pipeline twice in a row
            road_applicable = current_commodity_object.get_transportation_options_specific_mean_of_transport('Road')
            road_applicable = road_applicable & was_not_road & was_not_new

            # check which infrastructure can be transported via road
            if road_applicable:
                # branches where last one was not road or new & commodity can be transported via road
                road_transportation_costs[branch_index]\
                    = current_commodity_object.get_transportation_costs_specific_mean_of_transport('Road')
                branches_to_keep_road.append(branch_index)
                road_mask[:, branch_position] = distance_values[:, branch_position] <= max_length_road / no_road_multiplier
            else:
                # branches where the above does not allow new road but as infrastructure is within tolerance, we can
                # ignore transport mean as in this case we assume that we are already there
                in_tolerance_options = distance_values[:, branch_position] <= configuration['tolerance_distance']
                road_values[in_tolerance_options, branch_position] = 0
                road_mask[:, branch_position] = in_tolerance_options

                road_transportation_costs[branch_index] = 0
                branches_to_keep_road.append(branch_index)

            # we cannot use new pipeline or road pipeline twice in a row
            pipeline_applicable = pipeline_applicable & was_not_new & was_not_road

            # check if new pipelines are allowed and if so, which branches can use them
            if pipeline_applicable:
                if current_commodity_object.get_transportation_options_specific_mean_of_transport('New_Pipeline_Gas'):
                    # gas pipeline
                    new_transportation_costs[branch_index] \
                        = current_commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Gas')
                else:
                    # oil pipeline
                    new_transportation_costs[branch_index]\
                        = current_commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Liquid')
                branches_to_keep_new.append(branch_index)
                new_infrastructure_mask[:, branch_position] \
                    = distance_values[:, branch_position] <= max_length_new_segment / no_road_multiplier

        if not use_minimal_distance:
            _apply_reachable_continent_mask(road_mask, reduced_infrastructure_index, branches.index,
                                            branches, complete_infrastructure, data)
            _apply_reachable_continent_mask(new_infrastructure_mask, reduced_infrastructure_index, branches.index,
                                            branches, complete_infrastructure, data)

        road_transportation_costs = pd.Series(road_transportation_costs.values(), index=road_transportation_costs.keys())
        new_transportation_costs = pd.Series(new_transportation_costs.values(), index=new_transportation_costs.keys())

        road_options = _build_options_from_mask(road_values, reduced_infrastructure_index, branches.index, road_mask)
        new_infrastructure_options = _build_options_from_mask(distance_values, reduced_infrastructure_index,
                                                              branches.index, new_infrastructure_mask)

    else:
        # if iteration is not 0, there might be a large number of branches. Therefore, we need to preselect
        # potential branches

        road_transportation_costs = branches['road_transportation_costs']

        new_transportation_costs = branches['new_transportation_costs']

        minimal_distances = data['minimal_distances']

        all_road_distances = []
        all_new_distances = []

        if not use_minimal_distance:
            # if we don't use minimal distances, we have to assess all locations

            # if several branches with different commodities are at same location, we still need distances for this
            # location only once. Therefore, remove duplicates
            branches_no_duplicates = branches.drop_duplicates(subset=['current_node'], keep='first')

            in_tolerance_to_destination_infrastructure = complete_infrastructure[
                complete_infrastructure['distance_to_destination'] < configuration[
                    'to_final_destination_tolerance']].index.tolist()

            if limitation == 'no_pipeline_gas':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PG' not in i]

                if configuration['destination_type'] == 'country':  # destination not necessary with polygons
                    reduced_infrastructure_index = [i for i in reduced_infrastructure_index if i != 'Destination']

                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)

            elif limitation == 'no_pipeline_liquid':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PL' not in i]

                if configuration['destination_type'] == 'country':  # destination not necessary with polygons
                    reduced_infrastructure_index = [i for i in reduced_infrastructure_index if i != 'Destination']

                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)

            elif limitation == 'no_pipelines':
                if configuration['destination_type'] == 'location':
                    reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i] + ['Destination']
                else:
                    reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i]

                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)

            elif limitation == 'only_in_tolerance':  # these will immediately terminate since destinations are in tolerance
                reduced_infrastructure_index = in_tolerance_to_destination_infrastructure
                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)

            else:
                reduced_infrastructure_index = complete_infrastructure.index.tolist()
                if configuration['destination_type'] == 'country':  # destination not necessary with polygons
                    reduced_infrastructure_index = [i for i in complete_infrastructure.index if i != 'Destination']

                reduced_infrastructure_index = _filter_infrastructure_by_reachable_continents(
                    reduced_infrastructure_index, complete_infrastructure, branches, data)

            distance_blocks = _build_reachable_distance_blocks(
                complete_infrastructure, reduced_infrastructure_index,
                branches_no_duplicates, branches, data)

        else:
            # Check distance not for all infrastructure but just closest one to assess, if the closest
            # infrastructure is already too expensive. This approach helps to remove branches

            branches_no_duplicates = branches.drop_duplicates(subset=['current_node'], keep='first')

            branches_no_duplicates['minimal_distance'] \
                = minimal_distances.loc[branches_no_duplicates['current_node'].tolist(), 'minimal_distance'].tolist()
            branches_no_duplicates['closest_node'] \
                = minimal_distances.loc[branches_no_duplicates['current_node'].tolist(), 'closest_node'].tolist()

            distances \
                = branches_no_duplicates.set_index([branches_no_duplicates['current_node'],
                                                    branches_no_duplicates['closest_node']])['minimal_distance'].unstack().transpose()
            distance_blocks = [{
                'row_index': distances.index,
                'column_nodes': distances.columns,
                'values': distances.to_numpy(copy=False),
            }]
        if tracker is not None:
            matrix_rows = sum(len(block['row_index']) for block in distance_blocks)
            matrix_columns = sum(len(block['column_nodes']) for block in distance_blocks)
            matrix_cells = sum(len(block['row_index']) * len(block['column_nodes'])
                               for block in distance_blocks)
            tracker.event(iteration=iteration, phase='routing_out', method=method,
                          event='distance_matrix_created',
                          details={'rows': matrix_rows,
                                   'columns': matrix_columns,
                                   'blocks': len(distance_blocks),
                                   'cells': matrix_cells,
                                   'use_minimal_distance': use_minimal_distance,
                                   'limitation': limitation})

        # Instead of iterating over all visited infrastructure for all branches, we iterate over each visited
        # infrastructure once and check which branch has been there
        def flatten_list(nested_list):  # get single list of all previously visited nodes and ports
            flattened_list = []
            for element in nested_list:
                if isinstance(element, list):
                    flattened_list.extend(flatten_list(element))
                else:
                    flattened_list.append(element)
            return flattened_list

        if not use_minimal_distance:
            all_previous_infrastructure = list(set(flatten_list(branches['all_previous_infrastructure'].tolist())))

            # for each already visited node or port, get all branches which have been there already
            branches_to_remove_based_on_visited_infrastructure = {}
            for branch_index in all_previous_infrastructure:
                if branch_index is not None:

                    # check which branch has the visited infrastructure in all_previous_infrastructure
                    affected_branches = branches[branches['all_previous_infrastructure'].apply(lambda x: branch_index in x)].index

                    if 'PG' in branch_index:
                        branches_to_remove_based_on_visited_infrastructure[branch_index] \
                            = {'nodes': data['Pipeline_Gas'][branch_index]['NodeLocations'].index.tolist(),
                               'branches': affected_branches}

                    elif 'PL' in branch_index:
                        branches_to_remove_based_on_visited_infrastructure[branch_index] \
                            = {'nodes': data['Pipeline_Liquid'][branch_index]['NodeLocations'].index.tolist(),
                               'branches': affected_branches}

                    else:
                        branches_to_remove_based_on_visited_infrastructure[branch_index] \
                            = {'nodes': [branch_index],
                               'branches': affected_branches}

        # iterate over all commodities. Necessary to look at each commodity to check if applicable for road or
        # new pipeline and to get costs of transport
        for distance_block in distance_blocks:
            block_column_nodes = distance_block['column_nodes']
            row_index = distance_block['row_index']
            block_values = distance_block['values']

            for c in branches['current_commodity'].unique():
                c_branches = branches[branches['current_commodity'] == c]
                if c_branches.empty:
                    continue

                commodity_object = data['commodities']['commodity_objects'][c]

                # exchange current_node columns with corresponding branch names
                node_to_branch = dict(zip(c_branches['current_node'], c_branches.index))
                block_column_lookup = {node: position for position, node in enumerate(block_column_nodes)}
                columns_to_keep = [n for n in block_column_nodes if n in node_to_branch]
                if not columns_to_keep:
                    continue

                column_positions = [block_column_lookup[n] for n in columns_to_keep]
                column_index = np.asarray([node_to_branch[n] for n in columns_to_keep], dtype=object)
                distance_values = block_values[:, column_positions]
                branch_meta = c_branches.loc[column_index]

                # some locations are within tolerance. These are processed separately as we don't need transportation
                in_tolerance_mask = distance_values <= configuration['tolerance_distance']
                if not use_minimal_distance:
                    _apply_reachable_continent_mask(in_tolerance_mask, row_index, column_index,
                                                    c_branches, complete_infrastructure, data)
                if np.any(in_tolerance_mask):
                    in_tolerance_distances = _build_options_from_mask(distance_values, row_index, column_index,
                                                                      in_tolerance_mask)
                    in_tolerance_distances['current_distance'] = 0  # in tolerance means 0 distance
                    all_road_distances.append(in_tolerance_distances)

                if commodity_object.get_transportation_options()['Road']:

                    # remove all branches where road is not applicable (remove rows)
                    road_applicable = branch_meta['Road_applicable'].to_numpy(dtype=bool)

                    # remove all options where max length exceeds distance (remove columns)
                    max_length_road_costs \
                        = (benchmarks[commodity_object.get_name()] - branch_meta['current_total_costs']).to_numpy() * 1000 \
                        / branch_meta['road_transportation_costs'].to_numpy() / no_road_multiplier
                    max_length_road_array = np.minimum(max_length_road_costs, max_length_road / no_road_multiplier)
                    road_mask = (distance_values <= max_length_road_array[None, :]) & road_applicable[None, :]
                    if not use_minimal_distance:
                        _apply_reachable_continent_mask(road_mask, row_index, column_index,
                                                        c_branches, complete_infrastructure, data)

                    # remove options based on previous used infrastructure
                    if not use_minimal_distance:
                        _remove_visited_options_from_mask(road_mask, row_index, column_index,
                                                          branches_to_remove_based_on_visited_infrastructure)

                    road_distances = _build_options_from_mask(distance_values, row_index, column_index, road_mask)

                    # todo: some values are b'' --> why?

                    if not road_distances.empty:
                        all_road_distances.append(road_distances)

                # create and process new infrastructure distances
                if (commodity_object.get_transportation_options()['New_Pipeline_Gas']
                        | commodity_object.get_transportation_options()['New_Pipeline_Liquid']):

                    # add information before any change to distances is made
                    pipeline_applicable = (branch_meta['New_Pipeline_Gas_applicable'].to_numpy(dtype=bool)
                                           | branch_meta['New_Pipeline_Liquid_applicable'].to_numpy(dtype=bool))
                    minimal_distance = minimal_distances.loc[columns_to_keep, 'minimal_distance'].to_numpy()

                    # remove branches where all minimal distances are already higher than minimal distance to next node,
                    # choose branches which are applicable for new infrastructure, and remove distances above max length.
                    new_branch_mask = (minimal_distance <= max_length_new_segment / no_road_multiplier) & pipeline_applicable
                    new_mask = (distance_values <= max_length_new_segment / no_road_multiplier) & new_branch_mask[None, :]
                    if not use_minimal_distance:
                        _apply_reachable_continent_mask(new_mask, row_index, column_index,
                                                        c_branches, complete_infrastructure, data)

                    # remove used infrastructure
                    if not use_minimal_distance:
                        _remove_visited_options_from_mask(new_mask, row_index, column_index,
                                                          branches_to_remove_based_on_visited_infrastructure)

                    new_distances = _build_options_from_mask(distance_values, row_index, column_index, new_mask)
                    if not new_distances.empty:
                        all_new_distances.append(new_distances)

        if all_road_distances:
            road_options = pd.concat(all_road_distances, ignore_index=True)
        else:
            road_options = pd.DataFrame()

        if all_new_distances:
            new_infrastructure_options = pd.concat(all_new_distances, ignore_index=True)
        else:
            new_infrastructure_options = pd.DataFrame()

        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_out', method=method,
                          event='raw_candidates_created',
                          created=branch_count(road_options) + branch_count(new_infrastructure_options),
                          runtime_s=time.perf_counter() - time_candidate_generation,
                          details={'road_candidates': branch_count(road_options),
                                   'new_pipeline_candidates': branch_count(new_infrastructure_options),
                                   'use_minimal_distance': use_minimal_distance,
                                   'limitation': limitation})

    # Create new branches for road options
    if not road_options.empty:  # todo: Man könnte noch alle optionen entfernen, die gleiche start und end node haben (--> kein transport)

        # all distance below tolerance are 0
        below_tolerance = road_options[road_options['current_distance'] <= tolerance_distance].index
        road_options.loc[below_tolerance, 'current_distance'] = 0

        # add further information
        road_options['current_distance'] = road_options['current_distance'] * no_road_multiplier

        branch_list = road_options['previous_branch'].tolist()
        options_list = road_options['current_node'].tolist()

        road_options['current_commodity'] = branches.loc[branch_list, 'current_commodity'].tolist()
        road_options['current_commodity_object'] = branches.loc[branch_list, 'current_commodity_object'].tolist()
        road_options['specific_transportation_costs'] = road_transportation_costs.loc[branch_list].tolist()
        road_options['previous_total_costs'] = branches.loc[branch_list, 'current_total_costs'].tolist()
        road_options['current_transport_mean'] = 'Road'
        road_options['latitude'] = complete_infrastructure.loc[options_list, 'latitude'].tolist()
        road_options['longitude'] = complete_infrastructure.loc[options_list, 'longitude'].tolist()
        road_options['current_continent'] = branches.loc[branch_list, 'current_continent'].tolist()

        taken_route = [(branches.at[road_options.at[i, 'previous_branch'], 'current_node'], 'Road',
                        road_options.at[i, 'current_distance'], road_options.at[i, 'current_node'], 1)
                       for i in road_options.index]
        road_options['taken_route'] = taken_route

        road_options['total_efficiency'] = branches.loc[branch_list, 'total_efficiency'].tolist()

        # calculate costs and remove all above benchmark
        road_options['current_transportation_costs'] \
            = road_options['current_distance'] * road_options['specific_transportation_costs'] / 1000
        road_options['current_total_costs'] \
            = road_options['previous_total_costs'] + road_options['current_transportation_costs']

        # compare to total costs minus fuel price
        # to compare the different commodities, the benchmark is adjusted by the fuel price
        road_options['benchmark'] = road_options['current_commodity'].map(benchmarks)
        before_road_benchmark = branch_count(road_options)
        time_road_benchmark = time.perf_counter()
        road_options = road_options[road_options['current_total_costs'] <= road_options['benchmark']]
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_out', method=method,
                          event='filter_road_costs_vs_benchmark',
                          before=before_road_benchmark, after=branch_count(road_options),
                          removed=before_road_benchmark - branch_count(road_options),
                          runtime_s=time.perf_counter() - time_road_benchmark,
                          details={'use_minimal_distance': use_minimal_distance,
                                   'limitation': limitation})

        road_options['distance_type'] = 'road'

        # remove duplicates based on node/port, commodity, future transport state and visited infrastructure
        road_options.sort_values(['current_total_costs'], inplace=True)

        if not use_minimal_distance:
            before_road_dedup = branch_count(road_options)
            time_road_dedup = time.perf_counter()
            road_options = remove_duplicate_branches(road_options, branches)
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_out', method=method,
                              event='deduplicate_road',
                              before=before_road_dedup, after=branch_count(road_options),
                              removed=before_road_dedup - branch_count(road_options),
                              runtime_s=time.perf_counter() - time_road_dedup,
                              details={'limitation': limitation})

    else:
        road_options = pd.DataFrame()

    # Create new branches for infrastructure options
    if not new_infrastructure_options.empty:

        # all distance below tolerance are 0
        below_tolerance = new_infrastructure_options[new_infrastructure_options['current_distance'] <= tolerance_distance].index
        new_infrastructure_options.loc[below_tolerance, 'current_distance'] = 0

        new_infrastructure_options['current_distance'] = new_infrastructure_options['current_distance'] * no_road_multiplier

        branch_list = new_infrastructure_options['previous_branch'].tolist()
        options_list = new_infrastructure_options['current_node'].tolist()

        # Add additional information
        new_infrastructure_options['current_commodity'] = branches.loc[branch_list, 'current_commodity'].tolist()
        new_infrastructure_options['current_commodity_object'] = branches.loc[branch_list, 'current_commodity_object'].tolist()
        new_infrastructure_options['previous_total_costs'] = branches.loc[branch_list, 'current_total_costs'].tolist()
        new_infrastructure_options['current_continent'] = branches.loc[branch_list, 'current_continent'].tolist()

        new_infrastructure_options['specific_transportation_costs'] = new_transportation_costs.loc[branch_list].tolist()

        new_infrastructure_options['latitude'] = complete_infrastructure.loc[options_list, 'latitude'].tolist()
        new_infrastructure_options['longitude'] = complete_infrastructure.loc[options_list, 'longitude'].tolist()

        new_infrastructure_options['current_transportation_costs'] \
            = new_infrastructure_options['current_distance'] * new_infrastructure_options['specific_transportation_costs'] / 1000

        new_infrastructure_options['current_total_costs'] \
            = new_infrastructure_options['previous_total_costs'] + new_infrastructure_options['current_transportation_costs']

        pipeline_gas_branches = branches[branches['New_Pipeline_Gas_applicable']].index
        pg_options \
            = new_infrastructure_options[new_infrastructure_options['previous_branch'].isin(pipeline_gas_branches)].index
        new_infrastructure_options.loc[pg_options, 'current_transport_mean'] = 'New_Pipeline_Gas'

        pipeline_liquid_branches = branches[branches['New_Pipeline_Liquid_applicable']].index
        pl_options \
            = new_infrastructure_options[new_infrastructure_options['previous_branch'].isin(pipeline_liquid_branches)].index
        new_infrastructure_options.loc[pl_options, 'current_transport_mean'] = 'New_Pipeline_Liquid'

        taken_route = [(branches.at[new_infrastructure_options.at[i, 'previous_branch'], 'current_node'],
                        new_infrastructure_options.at[i, 'current_transport_mean'],
                        new_infrastructure_options.at[i, 'current_distance'],
                        new_infrastructure_options.at[i, 'current_node'],
                        1) for i in new_infrastructure_options.index]
        new_infrastructure_options['taken_route'] = taken_route

        new_infrastructure_options['total_efficiency'] = branches.loc[branch_list, 'total_efficiency'].tolist()

        # compare to total costs minus fuel price
        # to compare the different commodities, the benchmark is adjusted by the fuel price
        new_infrastructure_options['benchmark'] = new_infrastructure_options['current_commodity'].map(benchmarks)
        before_new_benchmark = branch_count(new_infrastructure_options)
        time_new_benchmark = time.perf_counter()
        new_infrastructure_options = new_infrastructure_options[new_infrastructure_options['current_total_costs'] <= new_infrastructure_options['benchmark']]
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_out', method=method,
                          event='filter_new_pipeline_costs_vs_benchmark',
                          before=before_new_benchmark, after=branch_count(new_infrastructure_options),
                          removed=before_new_benchmark - branch_count(new_infrastructure_options),
                          runtime_s=time.perf_counter() - time_new_benchmark,
                          details={'use_minimal_distance': use_minimal_distance,
                                   'limitation': limitation})

        new_infrastructure_options['distance_type'] = 'new'

        # remove duplicates based on node/port, commodity, future transport state and visited infrastructure
        new_infrastructure_options.sort_values(['current_total_costs'], inplace=True)

        if not use_minimal_distance:
            before_new_dedup = branch_count(new_infrastructure_options)
            time_new_dedup = time.perf_counter()
            new_infrastructure_options = remove_duplicate_branches(new_infrastructure_options, branches)
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_out', method=method,
                              event='deduplicate_new_pipeline',
                              before=before_new_dedup, after=branch_count(new_infrastructure_options),
                              removed=before_new_dedup - branch_count(new_infrastructure_options),
                              runtime_s=time.perf_counter() - time_new_dedup,
                              details={'limitation': limitation})
    else:
        new_infrastructure_options = pd.DataFrame()

    # Concatenate all options
    time_combine_options = time.perf_counter()
    outside_options = pd.concat([road_options, new_infrastructure_options], ignore_index=True)
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_out', method=method,
                      event='combined_transport_options',
                      after=branch_count(outside_options),
                      runtime_s=time.perf_counter() - time_combine_options,
                      details={'road_options': branch_count(road_options),
                               'new_pipeline_options': branch_count(new_infrastructure_options),
                               'use_minimal_distance': use_minimal_distance,
                               'limitation': limitation})

    if not outside_options.empty:

        final_destination = data['destination']['location']

        if configuration['destination_type'] == 'location':
            outside_options['distance_to_final_destination'] = calc_distance_list_to_single(outside_options['latitude'],
                                                                                            outside_options['longitude'],
                                                                                            final_destination.y,
                                                                                            final_destination.x)
        else:
            # destination is polygon -> each infrastructure has different closest point to destination
            infrastructure_in_destination = data['destination']['infrastructure']
            distances = calc_distance_list_to_list(outside_options['latitude'], outside_options['longitude'],
                                                   infrastructure_in_destination['latitude'],
                                                   infrastructure_in_destination['longitude'])
            outside_options['distance_to_final_destination'] = np.asarray(distances).min(axis=0)

        #     print(distances.idxmin('columns'))
        #     print(distances['PG_Graph_7_Node_2947'])
        #     print(outside_options[['current_node', 'distance_to_final_destination']])
        #
        # from shapely.geometry import Point
        # import geopandas as gpd
        #
        # current_nodes = complete_infrastructure.loc[outside_options['current_node']]
        # current_nodes = [Point([complete_infrastructure.at[i, 'longitude'], complete_infrastructure.at[i, 'latitude']]) for i in current_nodes.index]
        # current_nodes = gpd.GeoDataFrame(geometry=current_nodes)
        #
        # closest_node = complete_infrastructure.loc[distances.idxmin('columns')]
        # closest_node = [
        #     Point([complete_infrastructure.loc[i, 'longitude'], complete_infrastructure.loc[i, 'latitude']]) for i in closest_node.index]
        # closest_node = gpd.GeoDataFrame(geometry=closest_node)
        #
        # benchmark_node = [Point([complete_infrastructure.loc['PG_Graph_7_Node_2947', 'longitude'], complete_infrastructure.loc['PG_Graph_7_Node_2947', 'latitude']])]
        # benchmark_node = gpd.GeoDataFrame(geometry=benchmark_node)
        #
        # polygon = data['destination']['location']
        # polygon = gpd.GeoDataFrame(geometry=[polygon])
        #
        # fig, ax = plt.subplots()
        # polygon.plot(ax=ax, fc='none', ec='black')
        # current_nodes.plot(ax=ax)
        # closest_node.plot(ax=ax, color='red')
        # benchmark_node.plot(ax=ax, color='yellow')
        #
        # plt.show()

        in_destination_tolerance \
            = outside_options[outside_options['distance_to_final_destination']
                              <= configuration['to_final_destination_tolerance']].index
        outside_options.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

        # get costs for all options outside tolerance
        min_values, min_commodities = calculate_cheapest_option_to_final_destination(data, outside_options, benchmarks, 'current_total_costs')

        outside_options['minimal_total_costs'] = min_values
        outside_options['minimal_commodity'] = min_commodities

        # throw out options to expensive
        outside_options['benchmark'] = outside_options['minimal_commodity'].map(benchmarks)
        before_final_cost_filter = branch_count(outside_options)
        time_final_cost_filter = time.perf_counter()
        outside_options = outside_options[outside_options['minimal_total_costs'] <= outside_options['benchmark']]
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_out', method=method,
                          event='filter_minimal_total_vs_benchmark',
                          before=before_final_cost_filter, after=branch_count(outside_options),
                          removed=before_final_cost_filter - branch_count(outside_options),
                          runtime_s=time.perf_counter() - time_final_cost_filter,
                          details={'use_minimal_distance': use_minimal_distance,
                                   'limitation': limitation})

        # print('after')
        # print(outside_options[['current_node', 'distance_to_final_destination']])

        # add further information
        outside_options['current_infrastructure'] = None

    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_out', method=method,
                      event='output', after=branch_count(outside_options),
                      runtime_s=0.0,
                      details={'use_minimal_distance': use_minimal_distance,
                               'limitation': limitation})

    return outside_options


def process_in_tolerance_branches_high_memory(data, branches, complete_infrastructure, benchmarks, configuration,
                                              with_assessment=True):

    """
    This method iterates over all branches, gets the distance to all connected nodes or ports, and then processes
    all branches together. Processing all branches together results in high memory usage but is faster

    @param dict data: dictionary with common data
    @param pandas.DataFrame branches: DataFrame with current branches
    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param dict benchmarks: current benchmarks
    @param dict configuration: dictionary with configuration
    @param bool with_assessment: boolean to start assessment of resulting dataframe

    @return: pandas.DataFrame with new branches
    """

    destination_continents = data['destination']['continent']
    tracker = get_tracker(data)
    iteration = data.get('current_iteration') if isinstance(data, dict) else None
    method = 'process_in_tolerance_branches_high_memory'
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_in', method=method,
                      event='input', before=branch_count(branches),
                      runtime_s=0.0,
                      details={'with_assessment': with_assessment})
    time_candidate_generation = time.perf_counter()

    infrastructure_chunks = []

    for mot in branches['current_transport_mean'].unique():
        options_m = branches.loc[branches['current_transport_mean'] == mot]

        if options_m.empty:
            continue

        if mot == 'Shipping':

            shipping_infrastructure = data['Shipping']['ports']

            # Only use ports which are on the same continent as the final destination
            shipping_infrastructure = _filter_shipping_infrastructure_by_destination_continents(
                shipping_infrastructure, destination_continents)

            shipping_distances = _load_shipping_distances(configuration['path_processed_data'])
            if shipping_distances.empty:
                continue

            # create one big target_infrastructure dataframe for all shipping options
            for s in options_m.index:

                current_commodity_object = options_m.at[s, 'current_commodity_object']
                if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                    continue

                previous_transport_means = options_m.at[s, 'all_previous_transport_means']
                if 'Shipping' in previous_transport_means:
                    # pass branch because cannot ship twice
                    continue

                transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
                current_total_costs = options_m.at[s, 'current_total_costs']

                used_infrastructure = options_m.at[s, 'all_previous_infrastructure']

                start_infrastructure = options_m.loc[s, 'current_node']
                if start_infrastructure in used_infrastructure:
                    # pass branch because infrastructure was already used
                    continue

                distances = shipping_distances.loc[start_infrastructure, shipping_infrastructure.index]

                distances = distances[distances.index != start_infrastructure]  # remove transport between same node
                if distances.empty:
                    continue

                durations = distances / 1000 / current_commodity_object.get_shipping_speed()

                efficiency, current_total_costs_distances \
                    = current_commodity_object.get_distance_and_duration_based_costs_and_efficiency_shipping(distances, durations, current_total_costs)

                # assess for benchmark
                current_total_costs_distances_benchmark = current_total_costs_distances[current_total_costs_distances <= benchmarks[current_commodity_object.get_name()]].dropna()

                current_total_costs_distances = current_total_costs_distances.loc[current_total_costs_distances_benchmark.index]
                efficiency = efficiency.loc[current_total_costs_distances.index]

                transportation_costs = current_total_costs_distances - current_total_costs
                transportation_costs = transportation_costs.loc[current_total_costs_distances.index] / distances.loc[current_total_costs_distances.index] * 1000

                if current_total_costs_distances.empty:
                    continue

                current_index = current_total_costs_distances.index
                infrastructure = pd.DataFrame({
                    'previous_branch': s,
                    'current_node': current_index,
                    'current_distance': distances.loc[current_index].to_numpy(),
                    'starting_point': start_infrastructure,
                    'current_transport_mean': 'Shipping',
                    'current_infrastructure': current_index,
                    'current_total_costs': current_total_costs_distances.to_numpy(),
                    'specific_transportation_costs': transportation_costs.to_numpy(),
                    'taken_route': [(start_infrastructure, mot, distances.at[i], i, efficiency.loc[i])
                                    for i in current_index],
                    'total_efficiency': [branches.loc[s, 'total_efficiency'] * e for e in efficiency],
                    'current_commodity': current_commodity_object.get_name(),
                    'current_commodity_object': current_commodity_object,
                })
                infrastructure_chunks.append(infrastructure)

        else:

            graph_distances = {}
            if configuration['use_low_storage']:  # todo: check if it works as it's planned
                # if not precalculated, calculate network distances for respective nodes

                for g in options_m['graph'].unique():
                    graph = data[mot][g]['Graph']

                    g_options_m = options_m[options_m['graph'] == g]
                    nodes = g_options_m['current_node'].unique()

                    for n in nodes:
                        distances = nx.single_source_dijkstra_path_length(graph, n)
                        distances = pd.Series(distances, dtype='int32')

                        graph_distances[n] = distances

            for s in options_m.index:

                current_commodity_object = options_m.at[s, 'current_commodity_object']
                if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                    continue

                if (current_commodity_object.get_name() == 'Hydrogen_Gas') & (not configuration['H2_ready_infrastructure']):
                    # if pipelines are not H2 ready, we cannot use pipelines if current commodity is H2
                    continue

                transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
                current_total_costs = options_m.at[s, 'current_total_costs']

                # for removing already used infrastructure
                used_infrastructure = options_m.at[s, 'all_previous_infrastructure']

                graph_id = options_m.at[s, 'graph']

                if graph_id in used_infrastructure:
                    continue

                start_infrastructure = options_m.at[s, 'current_node']

                if not configuration['use_low_storage']:
                    path_processed_data = configuration['path_processed_data']
                    distances = pd.read_hdf(path_processed_data + '/inner_infrastructure_distances/'
                                            + start_infrastructure + '.h5', mode='r', title=graph_id)
                    distances = pd.Series(np.ceil(distances.iloc[:, 0].to_numpy()).astype('int32', copy=False),
                                          index=distances.index)
                    distances = distances.loc[data[mot][graph_id]['NodeLocations'].index]
                else:
                    distances = graph_distances[start_infrastructure]

                distances = distances[distances.index != start_infrastructure]
                if distances.empty:
                    continue

                current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs

                # assess for benchmark
                current_total_costs_distances_benchmark = current_total_costs_distances[current_total_costs_distances <= benchmarks[current_commodity_object.get_name()]].dropna()

                current_total_costs_distances = current_total_costs_distances.loc[current_total_costs_distances_benchmark.index]

                if current_total_costs_distances.empty:
                    continue

                current_index = current_total_costs_distances.index
                length_index = len(current_index)
                infrastructure = pd.DataFrame({
                    'previous_branch': s,
                    'current_node': current_index,
                    'current_distance': distances.loc[current_index].to_numpy(),
                    'starting_point': start_infrastructure,
                    'current_transport_mean': mot,
                    'current_infrastructure': graph_id,
                    'current_total_costs': current_total_costs_distances.to_numpy(),
                    'specific_transportation_costs': transportation_costs,
                    'taken_route': [(start_infrastructure, mot, distances.at[i], i, 1) for i in current_index],
                    'total_efficiency': branches.loc[[s], 'total_efficiency'].tolist() * length_index,
                    'current_commodity': current_commodity_object.get_name(),
                    'current_commodity_object': current_commodity_object,
                })
                infrastructure_chunks.append(infrastructure)

    if infrastructure_chunks:

        all_infrastructures = pd.concat(infrastructure_chunks, ignore_index=True)
        del infrastructure_chunks
        gc.collect()
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='raw_candidates_created',
                          created=branch_count(all_infrastructures),
                          after=branch_count(all_infrastructures),
                          runtime_s=time.perf_counter() - time_candidate_generation,
                          details={'with_assessment': with_assessment})

        if all_infrastructures.empty:
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_in', method=method,
                              event='output', after=0,
                              runtime_s=0.0,
                              details={'with_assessment': with_assessment})
            return pd.DataFrame()

        nodes_list = all_infrastructures['current_node'].tolist()
        all_infrastructures['latitude'] = complete_infrastructure.loc[nodes_list, 'latitude'].tolist()
        all_infrastructures['longitude'] = complete_infrastructure.loc[nodes_list, 'longitude'].tolist()

        # remove duplicates
        time_deduplicate = time.perf_counter()
        all_infrastructures.sort_values(['current_total_costs'], inplace=True)
        before_dedup = branch_count(all_infrastructures)
        all_infrastructures = remove_duplicate_branches(all_infrastructures, branches)
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='deduplicate_comparison_index',
                          before=before_dedup, after=branch_count(all_infrastructures),
                          removed=before_dedup - branch_count(all_infrastructures),
                          runtime_s=time.perf_counter() - time_deduplicate,
                          details={'with_assessment': with_assessment})

        # costs assessment for benchmark comparing and anticipation of costs to the closest infrastructure
        if with_assessment:

            # add costs to options
            all_infrastructures['current_transportation_costs'] \
                = all_infrastructures['current_distance'] / 1000 * all_infrastructures['specific_transportation_costs']

            # calculate minimal potential costs to final destination
            final_destination = data['destination']['location']

            if configuration['destination_type'] == 'location':
                all_infrastructures['distance_to_final_destination'] \
                    = calc_distance_list_to_single(all_infrastructures['latitude'], all_infrastructures['longitude'],
                                                   final_destination.y, final_destination.x)
            else:
                # destination is polygon -> each infrastructure has different closest point to destination
                infrastructure_in_destination = data['destination']['infrastructure']
                distances = calc_distance_list_to_list(all_infrastructures['latitude'], all_infrastructures['longitude'],
                                                       infrastructure_in_destination['latitude'],
                                                       infrastructure_in_destination['longitude'])

                all_infrastructures['distance_to_final_destination'] = np.asarray(distances).min(axis=0)

            # asses costs to final destination based on distance to final destination
            # get options in tolerance to final destination and set distance to 0
            in_destination_tolerance \
                = all_infrastructures[all_infrastructures['distance_to_final_destination']
                                      <= configuration['to_final_destination_tolerance']].index
            all_infrastructures.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

            # get costs for all options outside tolerance

            min_values, min_commodities = calculate_cheapest_option_to_closest_infrastructure(data, all_infrastructures, configuration,
                                                                                              benchmarks, 'current_total_costs')

            all_infrastructures['minimal_total_costs'] = min_values
            all_infrastructures['minimal_commodity'] = min_commodities

            # throw out options to expensive
            all_infrastructures['benchmark'] = all_infrastructures['minimal_commodity'].map(benchmarks)
            before_benchmark = branch_count(all_infrastructures)
            time_benchmark = time.perf_counter()
            all_infrastructures = all_infrastructures[all_infrastructures['minimal_total_costs'] <= all_infrastructures['benchmark']]
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_in', method=method,
                              event='filter_minimal_total_vs_benchmark',
                              before=before_benchmark, after=branch_count(all_infrastructures),
                              removed=before_benchmark - branch_count(all_infrastructures),
                              runtime_s=time.perf_counter() - time_benchmark,
                              details={'with_assessment': with_assessment})

            # next iteration either uses road or new pipeline. Remove options where the closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            all_infrastructures['minimal_distances'] \
                = minimal_distances.loc[all_infrastructures['current_node'].tolist(), 'minimal_distance'].tolist()

            before_minimal_distance = branch_count(all_infrastructures)
            time_minimal_distance = time.perf_counter()
            all_infrastructures = all_infrastructures[all_infrastructures['minimal_distances'] <= max_length]
            all_infrastructures.drop(['minimal_distances'], axis=1, inplace=True)
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_in', method=method,
                              event='filter_minimal_distance_to_next_connector',
                              before=before_minimal_distance, after=branch_count(all_infrastructures),
                              removed=before_minimal_distance - branch_count(all_infrastructures),
                              runtime_s=time.perf_counter() - time_minimal_distance,
                              details={'with_assessment': with_assessment})

        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='output', after=branch_count(all_infrastructures),
                          runtime_s=0.0,
                          details={'with_assessment': with_assessment})
        return all_infrastructures
    else:
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='output', after=0,
                          runtime_s=0.0,
                          details={'with_assessment': with_assessment})
        return pd.DataFrame()


def process_in_tolerance_branches_low_memory(data, branches, complete_infrastructure, benchmarks, configuration,
                                             with_assessment=True):

    """
    This method iterates over all branches, gets the distance to all connected nodes or ports, and then processes
    each branch alone. Processing all branches alone saves memory but takes more time

    @param dict data: dictionary with common data
    @param pandas.DataFrame branches: DataFrame with current branches
    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param dict benchmarks: current benchmarks
    @param dict configuration: dictionary with configuration
    @param bool with_assessment: boolean to start assessment of resulting dataframe

    @return: pandas.DataFrame with new branches
    """

    destination_continents = data['destination']['continent']
    tracker = get_tracker(data)
    iteration = data.get('current_iteration') if isinstance(data, dict) else None
    method = 'process_in_tolerance_branches_low_memory'
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_in', method=method,
                      event='input', before=branch_count(branches),
                      runtime_s=0.0,
                      details={'with_assessment': with_assessment})
    infrastructure_chunks = []

    for o in branches.index:

        mot = branches.at[o, 'current_transport_mean']
        if mot == 'Shipping':

            shipping_infrastructure = data['Shipping']['ports']

            shipping_distances = _load_shipping_distances(configuration['path_processed_data'])
            if shipping_distances.empty:
                continue

            # Only use ports which are on the same continent as the final destination
            shipping_infrastructure = _filter_shipping_infrastructure_by_destination_continents(
                shipping_infrastructure, destination_continents)

            shipping_distances_columns = [c for c in shipping_distances.columns if c in shipping_infrastructure.index]
            shipping_distances = shipping_distances[shipping_distances_columns]

            # create one big target_infrastructure dataframe for all shipping options
            current_commodity_object = branches.at[o, 'current_commodity_object']
            if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                # pass branch because commodity cannot be transported via ship
                continue

            previous_transport_means = branches.at[o, 'all_previous_transport_means']
            if 'Shipping' in previous_transport_means:
                # pass branch because cannot ship twice
                continue

            transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
            current_total_costs = branches.at[o, 'current_total_costs']

            used_infrastructure = branches.at[o, 'all_previous_infrastructure']

            # pass branch if port has already been used
            start_infrastructure = branches.loc[o, 'current_node']
            if start_infrastructure in used_infrastructure:
                continue

            distances = shipping_distances.loc[start_infrastructure, :]
            distances = distances[distances.index != start_infrastructure]
            if distances.empty:
                continue
            durations = distances / 1000 / current_commodity_object.get_shipping_speed()

            efficiency, current_total_costs_distances \
                = current_commodity_object.get_distance_and_duration_based_costs_and_efficiency_shipping(distances,
                                                                                                         durations,
                                                                                                         current_total_costs)

            # assess for benchmark
            current_total_costs_distances_benchmark = current_total_costs_distances[current_total_costs_distances <= benchmarks[current_commodity_object.get_name()]].dropna()
            current_total_costs_distances = current_total_costs_distances.loc[current_total_costs_distances_benchmark.index]

            transportation_costs = current_total_costs_distances - current_total_costs
            transportation_costs = transportation_costs.loc[current_total_costs_distances.index]  # todo: row or columns?

            current_infrastructure = 'Shipping'

            taken_route = []
            for i in distances.index:
                taken_route.append((start_infrastructure, mot, distances.at[i], i))

            total_efficiency = branches.at[o, 'total_efficiency'] * efficiency

        else:

            graph_id = branches.at[o, 'graph']

            current_commodity_object = branches.at[o, 'current_commodity_object']
            if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                # pass branch if commodity cannot be transported via pipeline
                continue

            if (current_commodity_object.get_name() == 'Hydrogen_Gas') & (not configuration['H2_ready_infrastructure']):
                # pass branch if commodity is H2 but pipeline is set to not H2 ready pipelines
                continue

            transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
            current_total_costs = branches.at[o, 'current_total_costs']

            used_infrastructure = branches.at[o, 'all_previous_infrastructure']
            if graph_id in used_infrastructure:
                # pass branch if node has already been used
                continue

            start_infrastructure = branches.at[o, 'current_node']

            if not configuration['use_low_storage']:
                # uses precalculated distances
                path_processed_data = configuration['path_processed_data']
                distances = pd.read_hdf(path_processed_data + '/inner_infrastructure_distances/'
                                        + start_infrastructure + '.h5', mode='r', title=graph_id)
                distances = pd.Series(np.ceil(distances.iloc[:, 0].to_numpy()), index=distances.index)
            else:
                # calculates distances from current node
                graph = data[mot][graph_id]['Graph']
                distances = nx.single_source_dijkstra_path_length(graph, start_infrastructure)
                distances = pd.Series(distances)  # todo: check how it looks like
            distances = distances[distances.index != start_infrastructure]
            if distances.empty:
                continue

            current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs

            current_infrastructure = graph_id

            total_efficiency = branches.at[o, 'total_efficiency']

            taken_route = []
            for i in distances.index:
                taken_route.append((start_infrastructure, mot, distances.at[i], i))

        infrastructure = pd.DataFrame(distances.values, index=distances.index, columns=['current_distance'])
        infrastructure['previous_branch'] = o
        infrastructure['current_node'] = distances.index

        if infrastructure.empty:
            return pd.DataFrame()

        infrastructure['current_total_costs'] = current_total_costs_distances
        infrastructure['taken_route'] = taken_route

        infrastructure['benchmark'] = benchmarks[current_commodity_object.get_name()]
        before_benchmark_direct = branch_count(infrastructure)
        time_benchmark_direct = time.perf_counter()
        infrastructure = infrastructure[infrastructure['current_total_costs'] <= infrastructure['benchmark']]
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='filter_transport_costs_vs_benchmark',
                          before=before_benchmark_direct, after=branch_count(infrastructure),
                          removed=before_benchmark_direct - branch_count(infrastructure),
                          runtime_s=time.perf_counter() - time_benchmark_direct,
                          details={'branch': o, 'transport_mean': mot})

        nodes_list = infrastructure['current_node'].tolist()

        infrastructure['starting_point'] = start_infrastructure
        infrastructure['previous_branch'] = o
        infrastructure['current_transport_mean'] = mot
        infrastructure['current_infrastructure'] = current_infrastructure
        infrastructure['specific_transportation_costs'] = transportation_costs

        infrastructure['current_commodity'] = current_commodity_object.get_name()
        infrastructure['current_commodity_object'] = current_commodity_object
        infrastructure['latitude'] = complete_infrastructure.loc[nodes_list, 'latitude'].tolist()
        infrastructure['longitude'] = complete_infrastructure.loc[nodes_list, 'longitude'].tolist()

        infrastructure['total_efficiency'] = total_efficiency

        # remove duplicates
        time_deduplicate = time.perf_counter()
        infrastructure.sort_values(['current_total_costs'], inplace=True)
        before_dedup = branch_count(infrastructure)
        infrastructure = remove_duplicate_branches(infrastructure, branches)
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='deduplicate_comparison_index',
                          before=before_dedup, after=branch_count(infrastructure),
                          removed=before_dedup - branch_count(infrastructure),
                          runtime_s=time.perf_counter() - time_deduplicate,
                          details={'branch': o, 'transport_mean': mot})

        # costs assessment for benchmark comparing and anticipation of costs to the closest infrastructure
        if with_assessment:

            # add costs to options
            infrastructure['current_transportation_costs'] \
                = infrastructure['current_distance'] / 1000 * infrastructure['specific_transportation_costs']

            # calculate minimal potential costs to final destination
            final_destination = data['destination']['location']

            if configuration['destination_type'] == 'location':
                infrastructure['distance_to_final_destination']\
                    = calc_distance_list_to_single(infrastructure['latitude'], infrastructure['longitude'],
                                                   final_destination.y, final_destination.x)
            else:
                # destination is polygon -> each infrastructure has different closest point to destination
                infrastructure_in_destination = data['destination']['infrastructure']
                distances = calc_distance_list_to_list(infrastructure['latitude'], infrastructure['longitude'],
                                                       infrastructure_in_destination['latitude'],
                                                       infrastructure_in_destination['longitude'])
                infrastructure['distance_to_final_destination'] = np.asarray(distances).min(axis=0)

            # asses costs to final destination based on distance to final destination
            # get options in tolerance to final destination and set distance to 0
            in_destination_tolerance \
                = infrastructure[infrastructure['distance_to_final_destination']
                                 <= configuration['to_final_destination_tolerance']].index
            infrastructure.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

            # get costs for all options outside tolerance
            min_values, min_commodities = calculate_cheapest_option_to_closest_infrastructure(data, infrastructure,
                                                                                              configuration,
                                                                                              benchmarks,
                                                                                              'current_total_costs')

            infrastructure['minimal_total_costs'] = min_values
            infrastructure['minimal_commodity'] = min_commodities

            # throws out options to expensive
            infrastructure['benchmark'] = infrastructure['minimal_commodity'].map(benchmarks)
            before_benchmark = branch_count(infrastructure)
            time_benchmark = time.perf_counter()
            infrastructure = infrastructure[infrastructure['minimal_total_costs'] <= infrastructure['benchmark']]
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_in', method=method,
                              event='filter_minimal_total_vs_benchmark',
                              before=before_benchmark, after=branch_count(infrastructure),
                              removed=before_benchmark - branch_count(infrastructure),
                              runtime_s=time.perf_counter() - time_benchmark,
                              details={'branch': o, 'transport_mean': mot})

            # next iteration either uses road or new pipeline. Remove options where the closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            infrastructure['minimal_distances'] \
                = minimal_distances.loc[infrastructure['current_node'].tolist(), 'minimal_distance'].tolist()

            before_minimal_distance = branch_count(infrastructure)
            time_minimal_distance = time.perf_counter()
            infrastructure = infrastructure[infrastructure['minimal_distances'] <= max_length]
            infrastructure.drop(['minimal_distances'], axis=1, inplace=True)
            if tracker is not None:
                tracker.event(iteration=iteration, phase='routing_in', method=method,
                              event='filter_minimal_distance_to_next_connector',
                              before=before_minimal_distance, after=branch_count(infrastructure),
                              removed=before_minimal_distance - branch_count(infrastructure),
                              runtime_s=time.perf_counter() - time_minimal_distance,
                              details={'branch': o, 'transport_mean': mot})

        if not infrastructure.empty:
            infrastructure_chunks.append(infrastructure)

    if infrastructure_chunks:
        all_infrastructures = pd.concat(infrastructure_chunks, ignore_index=False)
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_in', method=method,
                          event='output', after=branch_count(all_infrastructures),
                          runtime_s=0.0,
                          details={'with_assessment': with_assessment})
        return all_infrastructures
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_in', method=method,
                      event='output', after=0,
                      runtime_s=0.0,
                      details={'with_assessment': with_assessment})
    return pd.DataFrame()
