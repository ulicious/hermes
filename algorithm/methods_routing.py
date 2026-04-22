import math
import gc

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

from algorithm.methods_geographic import calc_distance_list_to_single, calc_distance_list_to_list
from algorithm.methods_cost_approximations import calculate_cheapest_option_to_closest_infrastructure, \
    calculate_cheapest_option_to_final_destination

import warnings
warnings.filterwarnings("ignore")


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

            if config_file['destination_type'] == 'location':
                # Check final destination and add to option outside tolerance if applicable
                complete_infrastructure.loc['Destination', 'latitude'] = final_destination.y
                complete_infrastructure.loc['Destination', 'longitude'] = final_destination.x
                complete_infrastructure.loc['Destination', 'current_transport_mean'] = m
                complete_infrastructure.loc['Destination', 'graph'] = None
                complete_infrastructure.loc['Destination', 'continent'] = data['destination']['continent']

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

    direct_branches['comparison_index'] \
        = direct_branches['current_node'] + '-' + direct_branches['current_commodity']
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
    direct_branches = direct_branches.drop_duplicates(subset=['comparison_index'], keep='first')

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

    if iteration == 0:
        # if iteration is 0, we don't make any preselection since we have only a very limited amount of branches
        # and calculating distances for these few branches is possible without long computation times

        # only use options which are actually reachable from start
        complete_infrastructure = complete_infrastructure[complete_infrastructure['reachable_from_start']]

        if limitation == 'no_pipeline_gas':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PG' not in i]
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'no_pipeline_liquid':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PL' not in i]
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'no_pipelines':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i]
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        else:  # don't limit infrastructure at all
            reduced_infrastructure_index = complete_infrastructure.index
            distances = calc_distance_list_to_list(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
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
                = current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas') \
                | current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid')

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

            if limitation == 'no_pipeline_gas':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PG' not in i]

                if configuration['destination_type'] == 'country':  # destination not necessary with polygons
                    reduced_infrastructure_index = [i for i in reduced_infrastructure_index if i != 'Destination']

                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            elif limitation == 'no_pipeline_liquid':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PL' not in i]

                if configuration['destination_type'] == 'country':  # destination not necessary with polygons
                    reduced_infrastructure_index = [i for i in reduced_infrastructure_index if i != 'Destination']

                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            elif limitation == 'no_pipelines':
                if configuration['destination_type'] == 'location':
                    reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i] + ['Destination']
                else:
                    reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i]

                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            else:
                reduced_infrastructure_index = complete_infrastructure.index
                if configuration['destination_type'] == 'country':  # destination not necessary with polygons
                    reduced_infrastructure_index = [i for i in complete_infrastructure.index if i != 'Destination']

                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            distances = pd.DataFrame(distances.transpose(),
                                     index=reduced_infrastructure_index,
                                     columns=branches_no_duplicates['current_node'], dtype='int32')

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
        for c in branches['current_commodity'].unique():
            c_branches = branches[branches['current_commodity'] == c]
            if c_branches.empty:
                continue

            commodity_object = data['commodities']['commodity_objects'][c]

            # exchange current_node columns with corresponding branch names
            node_to_branch = dict(zip(c_branches['current_node'], c_branches.index))
            columns_to_keep = [n for n in distances.columns if n in node_to_branch]
            if not columns_to_keep:
                continue

            column_index = np.asarray([node_to_branch[n] for n in columns_to_keep], dtype=object)
            row_index = distances.index
            distance_values = distances.loc[:, columns_to_keep].to_numpy(copy=False)
            branch_meta = c_branches.loc[column_index]

            # some locations are within tolerance. These are processed separately as we don't need transportation
            in_tolerance_mask = distance_values <= configuration['tolerance_distance']
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

                # remove options based on previous used infrastructure
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
                pipeline_applicable = (branch_meta['Pipeline_Gas_applicable'].to_numpy(dtype=bool)
                                       | branch_meta['Pipeline_Liquid_applicable'].to_numpy(dtype=bool))
                minimal_distance = minimal_distances.loc[columns_to_keep, 'minimal_distance'].to_numpy()

                # remove branches where all minimal distances are already higher than minimal distance to next node,
                # choose branches which are applicable for new infrastructure, and remove distances above max length.
                new_branch_mask = (minimal_distance <= max_length_new_segment / no_road_multiplier) & pipeline_applicable
                new_mask = (distance_values <= max_length_new_segment / no_road_multiplier) & new_branch_mask[None, :]

                # remove used infrastructure
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
        road_options = road_options[road_options['current_total_costs'] <= road_options['benchmark']]

        road_options['distance_type'] = 'road'

        # remove duplicates based on node/port, commodity and costs
        road_options['comparison_index'] = [road_options.at[ind, 'current_node'] + '-'
                                            + road_options.at[ind, 'current_commodity']
                                            for ind in road_options.index]
        road_options.sort_values(['current_total_costs'], inplace=True)
        road_options = road_options.drop_duplicates(subset=['comparison_index'], keep='first')

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
        new_infrastructure_options['current_continent'] = branches.loc[branch_list, 'current_commodity'].tolist()

        new_infrastructure_options['specific_transportation_costs'] = new_transportation_costs.loc[branch_list].tolist()

        new_infrastructure_options['latitude'] = complete_infrastructure.loc[options_list, 'latitude'].tolist()
        new_infrastructure_options['longitude'] = complete_infrastructure.loc[options_list, 'longitude'].tolist()

        new_infrastructure_options['current_transportation_costs'] \
            = new_infrastructure_options['current_distance'] * new_infrastructure_options['specific_transportation_costs'] / 1000

        new_infrastructure_options['current_total_costs'] \
            = new_infrastructure_options['previous_total_costs'] + new_infrastructure_options['current_transportation_costs']

        pipeline_gas_branches = branches[branches['Pipeline_Gas_applicable']].index
        pg_options \
            = new_infrastructure_options[new_infrastructure_options['previous_branch'].isin(pipeline_gas_branches)].index
        new_infrastructure_options.loc[pg_options, 'current_transport_mean'] = 'New_Pipeline_Gas'

        pipeline_liquid_branches = branches[branches['Pipeline_Liquid_applicable']].index
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
        new_infrastructure_options = new_infrastructure_options[new_infrastructure_options['current_total_costs'] <= new_infrastructure_options['benchmark']]

        new_infrastructure_options['distance_type'] = 'new'

        # remove duplicates based on node/port, commodity and costs
        new_infrastructure_options['comparison_index'] = [new_infrastructure_options.at[ind, 'current_node'] + '-'
                                                          + new_infrastructure_options.at[ind, 'current_commodity']
                                                          for ind in new_infrastructure_options.index]
        new_infrastructure_options.sort_values(['current_total_costs'], inplace=True)
        new_infrastructure_options = new_infrastructure_options.drop_duplicates(subset=['comparison_index'], keep='first')
    else:
        new_infrastructure_options = pd.DataFrame()

    # Concatenate all options
    outside_options = pd.concat([road_options, new_infrastructure_options], ignore_index=True)

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
        outside_options = outside_options[outside_options['minimal_total_costs'] <= outside_options['benchmark']]

        # print('after')
        # print(outside_options[['current_node', 'distance_to_final_destination']])

        # add further information
        outside_options['current_infrastructure'] = None

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

    destination_continent = data['destination']['continent']

    infrastructure_chunks = []

    for mot in branches['current_transport_mean'].unique():
        options_m = branches.loc[branches['current_transport_mean'] == mot]

        if options_m.empty:
            continue

        if mot == 'Shipping':

            shipping_infrastructure = data['Shipping']['ports']

            # Only use ports which are on the same continent as the final destination
            if destination_continent in ['Europe', 'Asia', 'Africa']:
                shipping_infrastructure = shipping_infrastructure[shipping_infrastructure['continent'].isin(['Europe',
                                                                                                             'Asia',
                                                                                                             'Africa'])]
            else:
                shipping_infrastructure = shipping_infrastructure[
                    shipping_infrastructure['continent'].isin([destination_continent])]

            shipping_distances = pd.read_csv(configuration['path_processed_data']
                                             + 'inner_infrastructure_distances/port_distances.csv',
                                             index_col=0)

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
                    'comparison_index': [i + '-' + current_commodity_object.get_name() for i in current_index],
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
                    'comparison_index': list(map(lambda x: x + '-' + current_commodity_object.get_name(),
                                                 current_index.tolist())),
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

        if all_infrastructures.empty:
            return pd.DataFrame()

        nodes_list = all_infrastructures['current_node'].tolist()
        all_infrastructures['latitude'] = complete_infrastructure.loc[nodes_list, 'latitude'].tolist()
        all_infrastructures['longitude'] = complete_infrastructure.loc[nodes_list, 'longitude'].tolist()

        # remove duplicates
        all_infrastructures.sort_values(['current_total_costs'], inplace=True)
        all_infrastructures = all_infrastructures.drop_duplicates(subset=['comparison_index'], keep='first')

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
            all_infrastructures = all_infrastructures[all_infrastructures['minimal_total_costs'] <= all_infrastructures['benchmark']]

            # next iteration either uses road or new pipeline. Remove options where the closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            all_infrastructures['minimal_distances'] \
                = minimal_distances.loc[all_infrastructures['current_node'].tolist(), 'minimal_distance'].tolist()

            all_infrastructures = all_infrastructures[all_infrastructures['minimal_distances'] <= max_length]
            all_infrastructures.drop(['minimal_distances'], axis=1, inplace=True)

        return all_infrastructures
    else:
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

    destination_continent = data['destination']['continent']
    infrastructure_chunks = []

    for o in branches.index:

        mot = branches.at[o, 'current_transport_mean']
        if mot == 'Shipping':

            shipping_infrastructure = data['Shipping']['ports']

            shipping_distances = pd.read_csv(configuration['path_processed_data']
                                             + 'inner_infrastructure_distances/port_distances.csv',
                                             index_col=0)

            # Only use ports which are on the same continent as the final destination
            if destination_continent in ['Europe', 'Asia', 'Africa']:
                shipping_infrastructure = shipping_infrastructure[shipping_infrastructure['continent'].isin(['Europe',
                                                                                                             'Asia',
                                                                                                             'Africa'])]
            else:
                shipping_infrastructure = shipping_infrastructure[
                    shipping_infrastructure['continent'].isin([destination_continent])]

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

            comparison_index = []
            taken_route = []
            for i in distances.index:
                comparison_index.append(i + '-' + current_commodity_object.get_name())
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

            current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs

            current_infrastructure = graph_id

            total_efficiency = branches.at[o, 'total_efficiency']

            comparison_index = []
            taken_route = []
            for i in distances.index:
                comparison_index.append(i + '-' + current_commodity_object.get_name())
                taken_route.append((start_infrastructure, mot, distances.at[i], i))

        infrastructure = pd.DataFrame(distances.values, index=distances.index, columns=['current_distance'])
        infrastructure['previous_branch'] = o
        infrastructure['current_node'] = distances.index

        if infrastructure.empty:
            return pd.DataFrame()

        infrastructure['current_total_costs'] = current_total_costs_distances
        infrastructure['comparison_index'] = comparison_index
        infrastructure['taken_route'] = taken_route

        infrastructure['benchmark'] = benchmarks[current_commodity_object.get_name()]
        infrastructure = infrastructure[infrastructure['current_total_costs'] <= infrastructure['benchmark']]

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
        infrastructure.sort_values(['current_total_costs'], inplace=True)
        infrastructure = infrastructure.drop_duplicates(subset=['comparison_index'], keep='first')

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
            infrastructure = infrastructure[infrastructure['minimal_total_costs'] <= infrastructure['benchmark']]

            # next iteration either uses road or new pipeline. Remove options where the closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            infrastructure['minimal_distances'] \
                = minimal_distances.loc[infrastructure['current_node'].tolist(), 'minimal_distance'].tolist()

            infrastructure = infrastructure[infrastructure['minimal_distances'] <= max_length]
            infrastructure.drop(['minimal_distances'], axis=1, inplace=True)

        if not infrastructure.empty:
            infrastructure_chunks.append(infrastructure)

    if infrastructure_chunks:
        return pd.concat(infrastructure_chunks, ignore_index=False)
    return pd.DataFrame()
