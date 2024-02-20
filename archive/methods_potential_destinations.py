import math
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points
import networkx as nx
import numpy as np
import searoute as sr
from tables import *

import time

from _helpers import calc_distance_single_to_single, calc_distance_list_to_single,\
    calculate_cheapest_option_to_final_destination, check_if_reachable_on_land
from methods_road_transport import get_road_distance_to_single_option, get_road_distance_to_options
from methods_networks import attach_new_node_to_graph

# Ignore runtime warnings as they
# import os
# os.environ['PYTHONWARNINGS'] = 'ignore::[RuntimeWarning]'
# os.environ['PYTHONWARNINGS'] = 'ignore::[FutureWarning]'

import warnings
warnings.filterwarnings("ignore")


def _get_all_options(data, configuration, solution, benchmark, new_node_count, iteration):
    location = solution.get_current_location()
    final_destination = solution.get_destination()

    distance_to_final_destination = calc_distance_single_to_single(location.y, location.x,
                                                                   final_destination.y, final_destination.x)
    if configuration['to_final_destination_tolerance'] >= distance_to_final_destination:
        # if solution is at final destination, no routing is applied. Solution will be dropped as it was at the final
        # destination in the previous conversion step meaning that it has been conversed to the right commodity and was
        # at right location
        return pd.DataFrame(), new_node_count, data

    means_of_transport = data['Means_of_Transport']

    # Iterate through means of transport and get all possible destinations
    total_costs = solution.get_total_costs()
    commodity = solution.get_current_commodity_object()
    transportation_options = commodity.get_transportation_options()
    used_infrastructure = [element for inner_list in solution.get_used_infrastructure().values() for element in inner_list]

    options = pd.DataFrame(columns=['latitude', 'longitude',
                                    'name', 'country', 'direct_distance', 'direct_distance_costs',
                                    'mean_of_transport', 'graph', 'used_infrastructure'])

    options_to_concat = []
    for m in means_of_transport:

        if not transportation_options[m]:
            # commodity cannot be transported via m
            continue

        if 'New' in m:
            # new infrastructure is considered at other place of code
            continue

        transportation_costs_road = commodity.get_transportation_costs_specific_mean_of_transport('Road')

        if m == 'Road':
            # Check final destination and add to option outside tolerance if applicable
            options.loc['Destination', 'latitude'] = final_destination.y
            options.loc['Destination', 'longitude'] = final_destination.x
            options.loc['Destination', 'mean_of_transport'] = m

            continue

        # get all options of current mean of transport
        if m == 'Shipping':

            if solution.get_used_transport_means().count('Shipping') == configuration['Shipping']['number_of_times_possible']:
                # limit the number of times shipping can be used
                continue

            # get all options of current mean of transport
            options_shipping = data[m]['ports']
            options_shipping['mean_of_transport'] = m
            options_to_concat.append(options_shipping)

            # todo: remove ports already

            # print('Final options ' + m + ': ' + str(len(options_in_tolerance) + len(options_outside_tolerance)))

        else:

            # remove networks which have been used already
            networks = list(set(data[m].keys()) - set(used_infrastructure))

            for n in networks:

                graph_object = data[m][n]['GraphObject']
                geodata = data[m][n]['GeoData']

                # Add node to each network which is closest to final destination
                # This is done only once. Therefore, do it at the beginning (where no infrastructure was used)
                # todo: could be done only with infrastructure which is useful to reduce effort --> define what is
                #  useful

                # Check if direct path to network does not already cost more than benchmark
                geodata['direct_distance'] = calc_distance_list_to_single(geodata['latitude'],
                                                                          geodata['longitude'],
                                                                          location.y, location.x)
                distance_to_closest = geodata['direct_distance'].min()
                closest_node = geodata['direct_distance'].idxmin()

                # if distance is within tolerance, then distance is 0
                if distance_to_closest <= configuration['tolerance_distance']:
                    costs_to_closest = 0
                else:
                    distance_to_closest = distance_to_closest * configuration['no_road_multiplier']
                    # if distance is not within tolerance, check if infrastructure can be reached
                    if not transportation_options['Road']:
                        # if commodity cannot be transported to infrastructure via road, it is only possible if
                        # a new infrastructure to the existing one is built
                        if configuration[m]['build_new_infrastructure']:
                            if distance_to_closest > configuration[m]['max_length_new_segment']:
                                # if this is not possible as well, than infrastructure not considered
                                continue
                            else:
                                # if infrastructure can be build and is in distance
                                # --> calculate based on new infrastructure costs
                                costs_to_closest = total_costs + distance_to_closest / 1000 \
                                    * commodity.get_transportation_costs_specific_mean_of_transport('New_' + m)

                        else:
                            # if this is not possible as well, than infrastructure not considered
                            continue
                    else:
                        # calculate road transportation
                        costs_to_closest = total_costs + distance_to_closest / 1000 * transportation_costs_road
                        costs_to_closest_new = math.inf

                        # if new infrastructure is possible, calculate costs and replace road transport if cheaper
                        if configuration[m]['build_new_infrastructure']:
                            if configuration[m]['build_consecutive_new_infrastructure']:
                                if distance_to_closest <= configuration[m]['max_length_new_segment']:
                                    # if infrastructure can be build and is in distance
                                    # --> calculate based on new infrastructure costs
                                    costs_to_closest_new = total_costs + distance_to_closest / 1000 \
                                        * commodity.get_transportation_costs_specific_mean_of_transport('New_' + m)
                            else:
                                # if no consecutive new segments can be placed, calculate costs only based on
                                # Road if last was newly built segment
                                if solution.get_last_used_transport_mean() != 'New_' + m:
                                    if distance_to_closest <= configuration[m]['max_length_new_segment']:
                                        # if infrastructure can be build and is in distance
                                        # --> calculate based on new infrastructure costs
                                        costs_to_closest_new \
                                            = total_costs + distance_to_closest / 1000 \
                                              * commodity.get_transportation_costs_specific_mean_of_transport('New_' + m)

                        # compare new infrastructure (if applicable) to road transport
                        if costs_to_closest > costs_to_closest_new:
                            costs_to_closest = costs_to_closest_new

                if costs_to_closest <= benchmark:

                    if len(data[m][n]['GeoData'].index) == 0:
                        continue

                    geodata = data[m][n]['GeoData']
                    graph = data[m][n]['Graph']
                    graph_data = data[m][n]['GraphData']

                    if iteration == '0T':
                        if not configuration[m]['use_only_existing_nodes']:

                            # todo: if new nodes are attached, we need to update the infrastructure distances
                            # Check if the closest point from location to network is existing node or is new node.
                            new_node_index = geodata[(geodata['longitude'] == round(closest_node.x, 4))
                                                     & (geodata['latitude'] == round(closest_node.y, 4))].index

                            if len(new_node_index) == 0:
                                # not existing node -> add new node in graph

                                geodata, graph, graph_data, graph_object, new_node_count \
                                    = attach_new_node_to_graph(geodata, graph, graph_data, graph_object, n,
                                                               closest_node, new_node_count)

                                data[m][n]['GeoData'] = geodata
                                data[m][n]['Graph'] = graph
                                data[m][n]['GraphData'] = graph_data
                                data[m][n]['GraphObject'] = graph_object

                    geodata['mean_of_transport'] = m
                    options_to_concat.append(geodata)

    options = pd.concat([options] + options_to_concat)

    options['transported_commodity'] = commodity.get_name()

    return options, new_node_count, data


def _process_options(data, configuration, solution, options, local_benchmarks, solutions_to_remove):

    # todo: move code which processes in or outside tolerance to specific method

    location = solution.get_current_location()
    final_destination = solution.get_destination()
    current_continent = solution.get_current_continent()
    current_commodity = solution.get_current_commodity_object()
    transportation_options = current_commodity.get_transportation_options()

    tolerance_distance = configuration['tolerance_distance']

    means_of_transport = data['Means_of_Transport']

    # first of all, there is no returning to old locations --> remove all options which have been visited
    if True:
        locations_visited = solution.get_locations()
        locations_visited_list = list(locations_visited.keys())
        if len(locations_visited_list) > 3:
            # first three locations are always destination, first location (transport) and first location (conversion)
            # Therefore, it doesn't make sense to look at these
            for l_key in locations_visited_list[:-2]:
                # then we don't look at the last two locations as sometimes two infrastructure points are at the same
                # location and we would remove the possibility to switch between them
                past_location = locations_visited[l_key]
                if past_location != final_destination:  # we never remove the final destination
                    options = options[(options['latitude'] != past_location.y) & (options['longitude'] != past_location.x)].copy()
    # store targets in column
    # options['target_infrastructure'] = options.index.tolist()

    # attach distance to final destination
    options['distance_to_final_destination'] \
        = calc_distance_list_to_single(options['latitude'], options['longitude'],
                                       final_destination.y, final_destination.x)

    # Remove ports which are not on same continent as current location
    if 'continent' in options.columns:
        options = _remove_ports_based_on_continent(options, current_continent)

    # Calculate direct distance to all options and sort them ascending
    if len(options.index) == 1:
        options['direct_distance'] = calc_distance_single_to_single(options['latitude'],
                                                                    options['longitude'],
                                                                    location.y, location.x)
    else:
        options['direct_distance'] = calc_distance_list_to_single(options['latitude'],
                                                                  options['longitude'],
                                                                  location.y, location.x)
    options.sort_values(['direct_distance'], inplace=True)
    local_infrastructure_local = options[options['direct_distance'] <= tolerance_distance].copy()

    # If final destination is in tolerance, return nothing. Such solutions should not exist
    if 'Destination' in options.index:
        if options.loc['Destination', 'direct_distance'] <= configuration['to_final_destination_tolerance']:
            return pd.DataFrame(), pd.DataFrame(), local_infrastructure_local, \
                   local_benchmarks, solutions_to_remove

            # options = options.loc[['Destination']]
            # options.loc['Destination', 'direct_distance'] = 0

    # If last mean of transport was Road, it cannot use Road again and, therefore,
    # all destinations further away than tolerance are dropped
    if solution.get_last_used_transport_mean() == 'Road':
        options = options[options['direct_distance'] <= tolerance_distance]

    # Remove infrastructure which has been used already
    used_infrastructure = [element for inner_list in solution.get_used_infrastructure().values() for element in
                           inner_list]
    if used_infrastructure:
        # remove infrastructure based on index
        not_used_infrastructure = list(set(options.index.tolist()) - set(used_infrastructure))
        options = options.loc[not_used_infrastructure, :]

        # remove infrastructure based on graph
        options = options[~options['graph'].isin(used_infrastructure)].copy()

    if len(options.index) == 0:
        return pd.DataFrame(), pd.DataFrame(), local_infrastructure_local, \
               local_benchmarks, solutions_to_remove

    # Separate options regarding in tolerance or outside
    options_in_tolerance_local = options[options['direct_distance'] <= tolerance_distance]
    options_outside_tolerance_local = options[options['direct_distance'] > tolerance_distance]

    # if option is in tolerance, than no costs from current location to option will occur. Therefore, we can
    # set costs to option equal to costs to current location
    if len(options_in_tolerance_local.index) > 0:

        options_in_tolerance_local['local_benchmark'] = math.inf

        old_index = options_in_tolerance_local.index

        options_in_tolerance_local_adjusted_index = pd.Index([(Point([options_in_tolerance_local.at[ind, 'longitude'],
                                                                      options_in_tolerance_local.at[ind, 'latitude']]),
                                                               current_commodity.get_name())
                                                              for ind in options_in_tolerance_local.index])
        options_in_tolerance_local['adjusted_index'] = options_in_tolerance_local_adjusted_index

        common_local_benchmarks = options_in_tolerance_local_adjusted_index.intersection(local_benchmarks.index)

        if not common_local_benchmarks.empty:
            benchmarks = []
            for k in options_in_tolerance_local_adjusted_index:
                if k in common_local_benchmarks:
                    benchmarks.append(local_benchmarks.at[k, 'total_costs'])
                else:
                    benchmarks.append(math.inf)

            options_in_tolerance_local['local_benchmark'] = benchmarks

        options_in_tolerance_local['costs_for_benchmark_comparison'] = solution.get_total_costs()

        if True:
            # replace local benchmark where new options are below local benchmark
            # possible because only options in tolerance --> equal to costs of current location, which are known
            index_below_original \
                = options_in_tolerance_local[options_in_tolerance_local['local_benchmark'] > solution.get_total_costs()].index

            if not index_below_original.empty:
                # as we set the column local_benchmark to math.inf, it might be quite common that the local benchmark
                # is more expensive than solution costs. Therefore, compare again to local benchmark index
                index_below_in_local_benchmark = pd.Index(options_in_tolerance_local.loc[index_below_original, 'adjusted_index'])
                index_below_in_local_benchmark = index_below_in_local_benchmark.intersection(local_benchmarks.index)
                if not index_below_in_local_benchmark.empty:
                    local_benchmarks.loc[index_below_in_local_benchmark, 'total_costs'] = solution.get_total_costs()

        if True:
            # add local benchmark if not existing
            index_local_benchmark = pd.Index(options_in_tolerance_local['adjusted_index'])
            not_common_local_benchmarks = index_local_benchmark.difference(local_benchmarks.index)
            if not not_common_local_benchmarks.empty:
                new_location_benchmark = pd.DataFrame(index=not_common_local_benchmarks)
                new_location_benchmark['total_costs'] = solution.get_total_costs()
                new_location_benchmark['solution'] = None
                local_benchmarks = pd.concat([local_benchmarks, new_location_benchmark])

        # remove options above local benchmark
        options_in_tolerance_local.index = old_index
        options_in_tolerance_local \
            = options_in_tolerance_local[options_in_tolerance_local['local_benchmark'] <= solution.get_total_costs()].copy()

    # Only choose closest target of network if set in configuration
    # in tolerance
    list_df_to_concat = []
    if len(options_in_tolerance_local.index) > 0:
        df_reduced_options = pd.DataFrame()
        for m_local in means_of_transport:
            if m_local == 'Road':
                df_reduced_options_m_local \
                    = options_in_tolerance_local[options_in_tolerance_local['mean_of_transport'] == m_local].copy()
                df_reduced_options = pd.concat([df_reduced_options, df_reduced_options_m_local])
                continue

            if m_local == 'Shipping':
                df_reduced_options_m_local\
                    = options_in_tolerance_local[options_in_tolerance_local['mean_of_transport'] == m_local].copy()

                if configuration[m_local]['find_only_closest_in_tolerance']:
                    if len(df_reduced_options_m_local.index) > 0:
                        df_reduced_options_m_local = df_reduced_options_m_local.drop(
                            df_reduced_options_m_local.index[1:].tolist())

                list_df_to_concat.append(df_reduced_options_m_local)

            else:
                df_options_m_local \
                    = options_in_tolerance_local[options_in_tolerance_local['mean_of_transport'] == m_local].copy()

                if configuration[m_local]['find_only_closest_in_tolerance']:

                    for g in df_options_m_local['graph'].unique():
                        df_options_m_local_g = df_options_m_local[df_options_m_local['graph'] == g].copy()

                        if len(df_options_m_local_g.index) > 0:
                            df_options_m_local_g = df_options_m_local_g.drop(
                                df_options_m_local_g.index[1:].tolist())

                        list_df_to_concat.append(df_options_m_local_g)

                else:
                    list_df_to_concat.append(df_options_m_local)

        options_in_tolerance_local = pd.concat(list_df_to_concat)

    # outside tolerance
    list_df_to_concat = []
    if len(options_outside_tolerance_local.index) > 0:
        for m_local in means_of_transport:
            if m_local == 'Road':
                df_reduced_options_m_local \
                    = options_outside_tolerance_local[options_outside_tolerance_local['mean_of_transport']
                                                      == m_local].copy()
                list_df_to_concat.append(df_reduced_options_m_local)
                continue

            if m_local == 'Shipping':
                df_reduced_options_m_local \
                    = options_outside_tolerance_local[options_outside_tolerance_local['mean_of_transport']
                                                      == m_local].copy()

                if configuration[m_local]['find_only_closest_in_tolerance']:
                    if len(df_reduced_options_m_local.index) > 0:
                        df_reduced_options_m_local = df_reduced_options_m_local.drop(
                            df_reduced_options_m_local.index[1:].tolist())

                list_df_to_concat.append(df_reduced_options_m_local)

            else:
                df_options_m_local \
                    = options_outside_tolerance_local[options_outside_tolerance_local['mean_of_transport']
                                                      == m_local].copy()

                if configuration[m_local]['find_only_closest_in_tolerance']:

                    for g in df_options_m_local['graph'].unique():
                        df_options_m_local_g = df_options_m_local[df_options_m_local['graph'] == g].copy()

                        if len(df_options_m_local_g.index) > 0:
                            df_options_m_local_g = df_options_m_local_g.drop(df_options_m_local_g.index[1:].tolist())

                        list_df_to_concat.append(df_options_m_local_g)

                else:
                    list_df_to_concat.append(df_options_m_local)

        options_outside_tolerance_local = pd.concat(list_df_to_concat)

    # Assess options outside tolerance
    if len(options_outside_tolerance_local.index) > 0:

        # options outside tolerance have to be reached with road transport. If set in the configurations, also new
        # pipelines can be build.

        if len(local_infrastructure_local.index) > 0:
            start_location = local_infrastructure_local.index[0]
        else:
            start_location = 'Start'

        # todo: attaching the index takes quite a long time
        new_network_segment_options = []
        for network in ['Pipeline_Gas', 'Pipeline_Liquid']:  # todo: add railroad
            if network in options_outside_tolerance_local['mean_of_transport'].values.tolist():
                if configuration[network]['build_new_infrastructure']:
                    if not configuration[network]['build_consecutive_new_infrastructure']:
                        # If set in settings, don't build consecutive new infrastructure (connecting new infrastructure
                        # to new infrastructure)

                        if solution.get_last_used_transport_mean() == 'New_' + network:
                            # if last mean of transport is new infrastructure & consecutive new ones are not allowed
                            # --> skip
                            continue

                    max_length_new_infrastructure = configuration[network]['max_length_new_segment']
                    possible_new_network_segment_options \
                        = options_outside_tolerance_local[options_outside_tolerance_local['direct_distance']
                                                          <= max_length_new_infrastructure].copy()

                    print(possible_new_network_segment_options)

                    possible_new_network_segment_options['mean_of_transport'] = 'New_' + network

                    # attach multi index to dataframe
                    if len(possible_new_network_segment_options.index) > 0:
                        new_index = pd.MultiIndex.from_tuples([(start_location, i, 'New_' + network)
                                                               for i in possible_new_network_segment_options.index])
                        possible_new_network_segment_options.index = new_index
                    else:
                        new_index = pd.MultiIndex(levels=[[], [], []],
                                                  codes=[[], [], []])
                        possible_new_network_segment_options.index = new_index

                    new_network_segment_options.append(possible_new_network_segment_options)

        # after new segments have been installed (if possible),
        # remove road options if they are not applicable for commodity
        if transportation_options['Road']:
            options_outside_tolerance_local['mean_of_transport'] = 'Road'
        else:
            options_outside_tolerance_local = pd.DataFrame()

        # attach multi index to dataframe
        if len(options_outside_tolerance_local.index) > 0:  # new index only if dataframe is not empty
            new_index = pd.MultiIndex.from_tuples([(start_location, o, 'Road')
                                                   for o in options_outside_tolerance_local.index])
            options_outside_tolerance_local.index = new_index
        else:
            new_index = pd.MultiIndex(levels=[[], [], []],
                                      codes=[[], [], []])
            options_outside_tolerance_local.index = new_index

        new_network_segment_options.append(options_outside_tolerance_local)
        options_outside_tolerance_local = pd.concat(new_network_segment_options)

    return options_in_tolerance_local, options_outside_tolerance_local, local_infrastructure_local, \
        local_benchmarks, solutions_to_remove


def _remove_ports_based_on_continent(options, considered_continent):
    if True:
        if considered_continent in ['Europe', 'Asia', 'Africa']:
            considered_continent = ['Europe', 'Asia', 'Africa']
        else:
            considered_continent = [considered_continent]

        options_shipping = options[options.mean_of_transport == 'Shipping'].copy()
        options_shipping = options_shipping[options_shipping.continent.isin(considered_continent)].copy()

        options_not_shipping = options.loc[options.mean_of_transport != 'Shipping', :].copy()

        options = pd.concat([options_shipping, options_not_shipping])

    else:  # currently must be on the same continent

        # todo: Also for part above

        options_shipping = options.loc[options.mean_of_transport == 'Shipping', :]
        options_shipping = options_shipping.loc[options_shipping.continent == considered_continent]

        options_not_shipping = options.loc[options.mean_of_transport != 'Shipping', :]

        options = pd.concat([options_shipping, options_not_shipping])

    return options


def _remove_options_based_on_used_infrastructure(solution, options):
    # Remove infrastructure which has been used already
    used_infrastructure = list(solution.get_used_infrastructure().values())

    # todo: apply without for loop

    if used_infrastructure:
        not_used_infrastructure = []
        for o_local in options.index:
            if o_local not in used_infrastructure:
                not_used_infrastructure.append(o_local)

            if 'graph' in options.columns:
                graph_local = options.loc[o_local, 'graph']
                if graph_local not in used_infrastructure:
                    if o_local not in not_used_infrastructure:
                        not_used_infrastructure.append(o_local)

        options = options.loc[not_used_infrastructure]

    return options


def _assess_options_in_tolerance(data, configuration, solution, options_within_infrastructure, benchmark,
                                 local_benchmarks, graph_data_local):

    def _assess_targets():

        benchmark_local = benchmark
        local_benchmarks_updated = local_benchmarks

        # For each potential destination, calculate distance + costs to potential destination and afterwards
        # calculate distance and costs from potential destination to final destination
        # Remove potential destination if any costs are above benchmark

        # get those options which have the same network
        if m_local != 'Shipping':
            options_of_current_infrastructure = \
                options_within_infrastructure_m_local[options_within_infrastructure_m_local['graph'] == graph_id].copy()
        else:
            options_of_current_infrastructure = options_within_infrastructure_m_local

        start_location_df_list = []
        for start_location in options_of_current_infrastructure.index:

            if m_local != 'Shipping':
                # todo: in case of pipelines, start_location = node_start -> second iteration is not necessary
                if not start_location == n_start:
                    continue

            target_infrastructure_local = target_infrastructure.copy()

            if True:

                # No transportation from one location to same location
                if start_location in target_infrastructure_local.index:
                    target_infrastructure_local.drop([start_location], inplace=True)

                # add distance column to target and fill with information from infrastructure distances
                target_infrastructure_local['distance'] \
                    = infrastructure_distances.loc[start_location, target_infrastructure_local.index.tolist()]

            else:
                target_infrastructure_local['distance'] \
                    = pd.DataFrame(infrastructure_distances[start_location], index=[start_location])[target_infrastructure_local.index.tolist()]

            # Before assessing the targets, check if one is at final destination. Set these one to 0
            target_infrastructure_local['distance_to_final_destination'] = \
                calc_distance_list_to_single(target_infrastructure_local['destination_latitude'],
                                             target_infrastructure_local['destination_longitude'],
                                             final_destination.y, final_destination.x)

            ind_at_final_destination \
                = target_infrastructure_local[target_infrastructure_local['distance_to_final_destination']
                                              <= configuration['to_final_destination_tolerance']].index
            if len(ind_at_final_destination) > 0:
                target_infrastructure_local.loc[ind_at_final_destination, 'distance_to_final_destination'] = 0

            # Distance and costs from current location to potential destination
            target_infrastructure_local['costs_to_destination'] \
                = target_infrastructure_local['distance'] / 1000 * transportation_costs_m_local
            target_infrastructure_local['total_costs_to_destination'] \
                = total_costs + target_infrastructure_local['distance'] / 1000 \
                * transportation_costs_m_local

            # separate options based on check_benchmark column
            target_infrastructure_local \
                = target_infrastructure_local[target_infrastructure_local['total_costs_to_destination']
                                              <= benchmark].copy()

            if len(target_infrastructure_local.index) == 0:
                continue

            # add local benchmark columns for comparison
            target_infrastructure_local['local_benchmark'] = math.inf

            old_index = target_infrastructure_local.index
            target_infrastructure_local_adjusted_index = pd.Index([(Point([target_infrastructure_local.at[ind, 'destination_longitude'],
                                                                           target_infrastructure_local.at[ind, 'destination_latitude']]),
                                                                    commodity.get_name())
                                                                   for ind in target_infrastructure_local.index])
            target_infrastructure_local['adjusted_index'] = target_infrastructure_local_adjusted_index

            common_local_benchmarks = target_infrastructure_local_adjusted_index.intersection(local_benchmarks_updated.index)
            common_local_benchmarks_original_index \
                = target_infrastructure_local[target_infrastructure_local['adjusted_index'].isin(common_local_benchmarks)].index

            if not common_local_benchmarks_original_index.empty:

                benchmarks = []
                for k in target_infrastructure_local_adjusted_index:
                    if k in local_benchmarks_updated.index:
                        benchmarks.append(local_benchmarks_updated.at[k, 'total_costs'])
                    else:
                        benchmarks.append(math.inf)

                target_infrastructure_local['local_benchmark'] = benchmarks

            target_infrastructure_local.index = old_index

            target_infrastructure_local \
                = target_infrastructure_local[target_infrastructure_local['total_costs_to_destination']
                                              <= target_infrastructure_local['local_benchmark']].copy()

            if len(target_infrastructure_local.index) == 0:
                continue

            # here we can update the local benchmark as we know the exact costs to the target infrastructure
            if False:
                # replace local benchmark where new options are below local benchmark
                # possible because only options in tolerance --> equal to costs of current location, which are known
                index_below_original \
                    = target_infrastructure_local[
                    target_infrastructure_local['local_benchmark'] > target_infrastructure_local['total_costs_to_destination']].index

                if not index_below_original.empty:
                    # as we set the column local_benchmark to math.inf, it might be quite common that the local benchmark
                    # is more expensive than solution costs. Therefore, compare again to local benchmark index
                    index_below_in_local_benchmark = pd.Index(
                        target_infrastructure_local.loc[index_below_original, 'adjusted_index'])
                    index_below_in_local_benchmark = index_below_in_local_benchmark.intersection(local_benchmarks_updated.index)
                    if not index_below_in_local_benchmark.empty:
                        benchmarks = []
                        for k in target_infrastructure_local.index:
                            if target_infrastructure_local.at[k, 'adjusted_index'] in index_below_in_local_benchmark:
                                benchmarks.append(target_infrastructure_local.at[k, 'total_costs_to_destination'])
                        local_benchmarks_updated.loc[index_below_in_local_benchmark, 'total_costs'] = benchmarks
                        local_benchmarks_updated.loc[index_below_in_local_benchmark, 'solution'] = None

                # add local benchmark if not existing
                index_local_benchmark = pd.Index(target_infrastructure_local['adjusted_index'])
                not_common_local_benchmarks = index_local_benchmark.difference(local_benchmarks.index)
                if not not_common_local_benchmarks.empty:
                    benchmarks = []
                    for k in target_infrastructure_local.index:
                        if target_infrastructure_local.at[k, 'adjusted_index'] in not_common_local_benchmarks:
                            benchmarks.append(target_infrastructure_local.at[k, 'total_costs_to_destination'])

                    new_location_benchmark = pd.DataFrame(index=not_common_local_benchmarks)
                    new_location_benchmark['total_costs'] = benchmarks
                    new_location_benchmark['solution'] = None
                    local_benchmarks_updated = pd.concat([local_benchmarks, new_location_benchmark])

            # Distance and costs from potential destination to final destination
            target_infrastructure_local \
                = calculate_cheapest_option_to_final_destination(data, target_infrastructure_local, solution=solution)

            target_infrastructure_local['total_costs_to_final_destination'] \
                = total_costs + target_infrastructure_local['distance'] / 1000 \
                * transportation_costs_m_local \
                + target_infrastructure_local['costs_to_final_destination']

            target_infrastructure_local \
                = target_infrastructure_local[target_infrastructure_local['total_costs_to_final_destination'] <= benchmark].copy()

            if len(target_infrastructure_local.index) == 0:
                continue

            # Check if one of the options is within tolerance to the final destination. If so, then all options
            # which are more expensive will be removed. The solution which is at the final destination will set the
            # new benchmark and all other options will be removed on this basis
            if solution.get_current_commodity_name() in solution.get_final_commodity():
                targets_in_tolerance_to_destination \
                    = target_infrastructure_local[target_infrastructure_local['distance_to_final_destination']
                                                  <= configuration['to_final_destination_tolerance']].index

                if len(targets_in_tolerance_to_destination) > 0:
                    # max_costs_to_final_destination = future benchmark
                    # as the potential destination will have no costs to final destination
                    max_costs_to_final_destination \
                        = target_infrastructure_local.loc[targets_in_tolerance_to_destination,
                                                          'total_costs_to_final_destination'].values[0]

                    # get all potential destinations which minimal costs to next destination or
                    # final destination is above max value (= new benchmark)
                    targets_to_remove = \
                        target_infrastructure_local[target_infrastructure_local['total_costs_to_destination']
                                                    > max_costs_to_final_destination].index.tolist()
                    targets_to_remove += \
                        target_infrastructure_local[target_infrastructure_local['total_costs_to_final_destination']
                                                    > max_costs_to_final_destination].index.tolist()
                    target_infrastructure_local.drop(targets_to_remove, inplace=True)

                    # update benchmark
                    benchmark_local = max_costs_to_final_destination

            if len(target_infrastructure_local.index) == 0:
                continue

            # If set in configuration, only the closest node to the destination of a infrastructure
            if configuration[m_local]['find_only_closest_to_destination']:
                target_infrastructure_local.sort_values(['distance_to_final_destination'])
                target_infrastructure_local \
                    = target_infrastructure_local.drop(target_infrastructure_local.index[1:])

            if len(target_infrastructure_local.index) == 0:
                continue

            # Left destinations are processed to nice dataframe
            # Here: change index to unique index: start_end
            start_location_df = pd.DataFrame(index=[start_location + '_' + p
                                                    for p in target_infrastructure_local.index])
            start_location_df['destination_latitude'] = target_infrastructure_local['destination_latitude'].values
            start_location_df['destination_longitude'] = target_infrastructure_local['destination_longitude'].values
            start_location_df['costs_to_destination'] = target_infrastructure_local['costs_to_destination'].values
            start_location_df['total_costs_to_destination'] = target_infrastructure_local['costs_to_destination'].values + solution.get_total_costs()
            start_location_df['distance'] = target_infrastructure_local['distance'].values
            start_location_df['start_latitude'] = options_within_infrastructure_m_local.loc[start_location, 'latitude']
            start_location_df['start_longitude']\
                = options_within_infrastructure_m_local.loc[start_location, 'longitude']
            start_location_df['start_infrastructure'] = start_location
            start_location_df['target_infrastructure'] = target_infrastructure_local.index.tolist()
            start_location_df['used_infrastructure'] = start_location
            start_location_df['mean_of_transport'] = m_local
            start_location_df['total_costs_to_final_destination']\
                = target_infrastructure_local['total_costs_to_final_destination'].values

            if m_local not in ['Shipping', 'Road']:
                start_location_df['used_infrastructure'] = graph_id
                start_location_df['graph'] = graph_id

            if len(start_location_df.index) > 0:
                index = [(start_location, t, m_local) for t in target_infrastructure_local.index]
                index = pd.MultiIndex.from_tuples(index)
                start_location_df.index = index
            else:
                index = pd.MultiIndex(levels=[[], [], []], codes=[[], [], []])
                start_location_df.index = index

            start_location_df_list.append(start_location_df)

        if len(start_location_df_list) == 0:
            return pd.DataFrame(), benchmark_local, local_benchmarks_updated
        else:
            return pd.concat(start_location_df_list), benchmark_local, local_benchmarks_updated

    destination_continent = solution.get_destination_continent()

    commodity = solution.get_current_commodity_object()
    total_costs = solution.get_total_costs()
    final_destination = solution.get_destination()

    means_of_transport = data['Means_of_Transport']

    processed_targets_list = []
    for m_local in means_of_transport:

        transportation_costs_m_local = commodity.get_transportation_costs_specific_mean_of_transport(m_local)
        options_within_infrastructure_m_local \
            = options_within_infrastructure.loc[options_within_infrastructure['mean_of_transport'] == m_local].copy()

        if m_local == 'Road':
            if len(options_within_infrastructure_m_local.index) > 0:  # should normally only be destination --> check

                options_within_infrastructure_m_local['total_costs_to_final_destination'] = 0
                options_within_infrastructure_m_local['costs_to_destination'] = 0
                options_within_infrastructure_m_local['distance'] = 0
                options_within_infrastructure_m_local['mean_of_transport'] = 'Road'
                options_within_infrastructure_m_local['start_longitude'] = solution.get_current_location().x
                options_within_infrastructure_m_local['start_latitude'] = solution.get_current_location().y
                options_within_infrastructure_m_local['destination_longitude'] = solution.get_destination().x
                options_within_infrastructure_m_local['destination_latitude'] = solution.get_destination().y
                options_within_infrastructure_m_local['target_infrastructure'] = 'Destination'

                list_tuple_index = [('Destination', 'Destination', m_local)]
                new_index = pd.MultiIndex.from_tuples(list_tuple_index)
                options_within_infrastructure_m_local.index = new_index

                processed_targets_list.append(options_within_infrastructure_m_local)

        elif m_local == 'Shipping':
            target_infrastructure = data['Shipping']['ports'].copy()
            target_infrastructure['mean_of_transport'] = 'Shipping'

            infrastructure_distances = pd.DataFrame(data['Shipping']['Distances']['value'],
                                                    index=data['Shipping']['Distances']['index'],
                                                    columns=data['Shipping']['Distances']['columns'])

            # Only use ports which are on the same continent as the final destination
            target_infrastructure = _remove_ports_based_on_continent(target_infrastructure, destination_continent)
            target_infrastructure = _remove_options_based_on_used_infrastructure(solution, target_infrastructure)

            # Add some further information
            target_infrastructure.rename(columns={"longitude": "destination_longitude",
                                                  "latitude": "destination_latitude"},
                                         inplace=True)

            processed_targets, adjusted_benchmark, local_benchmarks = _assess_targets()
            if adjusted_benchmark < benchmark:
                benchmark = math.ceil(adjusted_benchmark * 10000) / 10000

            if processed_targets is not None:
                processed_targets_list.append(processed_targets)

        else:

            used_infrastructure = list(solution.get_used_infrastructure().values())

            for n_start in options_within_infrastructure_m_local.index:
                now = time.time()

                network = options_within_infrastructure_m_local.loc[n_start, 'graph']
                graph_id = network

                if graph_id in used_infrastructure:
                    continue

                graph_local = None
                target_infrastructure = None

                if m_local == 'Railroad':
                    graph_local = data['Railroad'][graph_id]['Graph']
                    target_infrastructure = data['Railroad'][graph_id]['GeoData'].copy()

                elif m_local in ['Pipeline_Gas', 'Pipeline_Gas_New']:
                    graph_local = data['Pipeline_Gas'][graph_id]['Graph']
                    target_infrastructure = data['Pipeline_Gas'][graph_id]['GeoData'].copy()

                elif m_local in ['Pipeline_Liquid', 'Pipeline_Liquid_New']:
                    graph_local = data['Pipeline_Gas'][graph_id]['Graph']
                    target_infrastructure = data['Pipeline_Liquid'][graph_id]['GeoData'].copy()

                # Apply Dijkstra to find shortest paths
                if False:
                    infrastructure_distances, paths = nx.single_source_dijkstra(graph_local, source=n_start)
                    infrastructure_distances = pd.DataFrame(infrastructure_distances, index=[n_start])
                elif False:
                    infrastructure_distances = data[m_local][graph_id]['Distances']['value']  # numpy array

                    infrastructure_distances = pd.DataFrame(data[m_local][graph_id]['Distances']['value'],
                                                            index=data[m_local][graph_id]['Distances']['index'],
                                                            columns=data[m_local][graph_id]['Distances']['columns'])

                    infrastructure_distances = infrastructure_distances.loc[[n_start]]
                elif False:
                    if False:
                        infrastructure_distances = data[m_local][graph_id]['Distances']
                    else:
                        infrastructure_distances = graph_data_local

                else:

                    path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'
                    infrastructure_distances\
                        = pd.read_hdf(path_data + '/inner_infrastructure_distances/' + n_start + '.h5', mode='r',
                                      title=graph_id, dtype=np.float16)
                    infrastructure_distances = infrastructure_distances.transpose()

                # Remove options which have been used already by solution
                if False:
                    now = time.time()
                    target_infrastructure = _remove_options_based_on_used_infrastructure(solution, target_infrastructure)
                    print(time.time() - now)

                # if no targets are left after removing, don't process empty targets
                if not target_infrastructure.index.tolist():
                    continue

                # Add some further information
                target_infrastructure.rename(columns={"longitude": "destination_longitude",
                                                      "latitude": "destination_latitude"},
                                             inplace=True)

                processed_targets, adjusted_benchmark, local_benchmarks = _assess_targets()

                if adjusted_benchmark < benchmark:
                    benchmark = math.ceil(adjusted_benchmark * 10000) / 10000

                if processed_targets is not None:
                    processed_targets_list.append(processed_targets)

    if len(processed_targets_list) > 0:
        reduced_df = pd.concat(processed_targets_list)
    else:
        reduced_df = pd.DataFrame()

    return reduced_df, benchmark


def _assess_options_outside_tolerance(data, solution, options_outside_tolerance, benchmark, configuration):

    location = solution.get_current_location()
    commodity = solution.get_current_commodity_object()

    total_costs = solution.get_total_costs()

    means_of_transport = data['Means_of_Transport'] + ['New_Pipeline_Gas', 'New_Pipeline_Liquid']

    # Add some further information
    options_outside_tolerance.rename(columns={"longitude": "destination_longitude",
                                              "latitude": "destination_latitude"},
                                     inplace=True)

    options_outside_tolerance['start_longitude'] = location.x
    options_outside_tolerance['start_latitude'] = location.y
    # options_outside_tolerance['target_infrastructure'] = options_outside_tolerance.index.tolist()

    df_to_concat = []
    for m_local in means_of_transport:

        print(m_local)

        m_local_df \
            = options_outside_tolerance[options_outside_tolerance['mean_of_transport']
                                        == m_local].copy()

        print('m local 1: ' + str(len(m_local_df.index)))

        if len(m_local_df.index) == 0:
            # m_local_df is empty
            continue

        # Calculate total costs of options outside tolerance based on direct path
        # and remove all which have higher costs than benchmark
        m_local_df['direct_distance_costs'] \
            = total_costs + m_local_df['direct_distance'] / 1000 \
            * commodity.get_transportation_costs_specific_mean_of_transport(m_local)

        # separate options based on check_benchmark column
        m_local_df = m_local_df[m_local_df['direct_distance_costs'] <= benchmark].copy()

        print('m local 2: ' + str(len(m_local_df.index)))

        # Check the distance from each option to the final destination
        # and calculate the minimal costs to the final destination
        # If minimal costs are higher than benchmark, remove these options
        m_local_df \
            = calculate_cheapest_option_to_final_destination(data, m_local_df, solution=solution)
        m_local_df['total_costs_to_final_destination'] \
            = m_local_df['costs_to_final_destination'] + m_local_df['direct_distance_costs']

        # separate options based on check_benchmark column
        m_local_df = m_local_df[m_local_df['total_costs_to_final_destination'] <= benchmark].copy()
        df_to_concat.append(m_local_df)

    options_outside_tolerance = pd.concat(df_to_concat)

    return options_outside_tolerance


def calculate_distance_known_infrastructure(data, configuration, solution, options_outside_tolerance,
                                            benchmark, local_benchmarks):

    location = solution.get_current_location()
    final_destination = solution.get_destination()

    commodity = solution.get_current_commodity_object()
    transportation_costs_road = commodity.get_transportation_costs_specific_mean_of_transport('Road')

    total_costs = solution.get_total_costs()

    # process options outside tolerance as some of them are transported via road
    road_transport_options = options_outside_tolerance[options_outside_tolerance['mean_of_transport']
                                                       == 'Road'].copy()
    road_transport_options['costs_to_destination'] = math.inf

    # No road options are routes which are based on road but new pipelines are assumed
    no_road_transport_options = options_outside_tolerance[options_outside_tolerance['mean_of_transport']
                                                          != 'Road'].copy()

    no_road_transport_options['distance'] = no_road_transport_options['direct_distance']
    for ind in no_road_transport_options.index:
        mean_of_transport = ind[2]
        transportation_costs = commodity.get_transportation_costs_specific_mean_of_transport(mean_of_transport)
        no_road_transport_options.loc[ind, 'costs_to_destination'] = \
            no_road_transport_options.loc[ind, 'direct_distance'] * transportation_costs / 1000
        no_road_transport_options.loc[ind, 'total_costs_to_destination'] \
            = total_costs + no_road_transport_options.loc[ind, 'direct_distance'] * transportation_costs / 1000

    # road options need road distances. The all distances road data contains all precalculated values and those
    # which have been calculated within the algorithm.

    # todo: check if everything works fine
    # means that infrastructure exists at current location. If also options outside tolerance exist, use
    # infrastructure at current position with each option in options outside index and use all distances
    # matrix to get road distance
    if configuration['use_OSRM']:
        # use precalculated distance matrix

        all_distances_road = data['all_distances_road']
        indexes_to_calculate_road_distance = []
        if len(road_transport_options.index) > 0:

            first_index_column = road_transport_options.index.get_level_values(0).tolist()
            unique_first_index = list(set(first_index_column))

            for ind in unique_first_index:

                road_transport_options_ind = road_transport_options.loc[[ind]]
                road_transport_options_ind_index = road_transport_options_ind.index
                second_index_column = road_transport_options_ind.index.get_level_values(1).tolist()

                if ind in all_distances_road.index:

                    available_data = list(set(second_index_column).intersection(all_distances_road.columns))
                    road_transport_options.loc[road_transport_options_ind_index, 'distance']\
                        = all_distances_road.loc[ind, available_data]
                    road_transport_options.loc[road_transport_options_ind_index, 'total_costs_to_destination']\
                        = total_costs\
                        + road_transport_options.loc[road_transport_options_ind_index, 'distance']\
                        * transportation_costs_road / 1000

                    not_available_data = list(set(second_index_column) - set(all_distances_road.columns))

                else:
                    not_available_data = second_index_column

                if not_available_data:
                    road_transport_options_second_ind = road_transport_options_ind.loc[:, not_available_data, :]
                    indexes_to_calculate_road_distance += road_transport_options_second_ind.index.tolist()

        road_transport_options_to_calculate = road_transport_options.loc[indexes_to_calculate_road_distance]
        road_transport_options.drop(indexes_to_calculate_road_distance, inplace=True)

        potential_destinations = pd.concat([road_transport_options, no_road_transport_options])

    else:
        # use direct distance * road multiplier to calculate road distances
        if len(road_transport_options.index) > 0:
            road_transport_options['distance']\
                = road_transport_options['direct_distance'] * configuration['no_road_multiplier']
            road_transport_options['costs_to_destination'] \
                = road_transport_options['distance'] * transportation_costs_road / 1000
            road_transport_options['total_costs_to_destination'] \
                = total_costs + road_transport_options['distance'] * transportation_costs_road / 1000

            potential_destinations = pd.concat([road_transport_options, no_road_transport_options])
        else:
            potential_destinations = no_road_transport_options

        road_transport_options_to_calculate = pd.DataFrame()

    if False:  # todo: adjust in configuration
        potential_destinations_check = potential_destinations[potential_destinations['check_benchmark']].copy()
        potential_destinations_no_checking = potential_destinations[~potential_destinations['check_benchmark']].copy()

        potential_destinations_check = potential_destinations_check[potential_destinations_check['total_costs_to_destination']
                                                                    <= benchmark].copy()
        # potential_destinations = potential_destinations[~potential_destinations.index.duplicated(keep='first')].copy()

        potential_destinations = pd.concat([potential_destinations_check, potential_destinations_no_checking])
    else:
        potential_destinations = \
            potential_destinations[potential_destinations['total_costs_to_destination'] <= benchmark].copy()

    if len(potential_destinations.index) > 0:
        # sort first s.t. cheapest (better) option is at top (necessary when removing duplicates)
        potential_destinations.sort_values(['total_costs_to_destination'], inplace=True)

        # add local benchmark columns for comparison. Set to infinity (which stays if no local benchmark available)
        potential_destinations['local_benchmark'] = math.inf

        # Change potential destination index s.t. it corresponds to local benchmark dictionary
        old_index = potential_destinations.index
        local_benchmark_index = [(ind[1], commodity.get_name()) for ind in potential_destinations.index]
        potential_destinations.index = local_benchmark_index

        # duplicates have to be removed as the set function below removes all duplicates automatically
        # and then "benchmarks" would be longer than potential destinations. This is no problem as the cheaper
        # solution would set the local benchmark anyways and the more expensive solution would be removed
        potential_destinations = potential_destinations[~potential_destinations.index.duplicated(keep='first')].copy()
        local_benchmark_index = potential_destinations.index

        # adjust length of old index if some have been removed. Cut of the end of the index --> possible because not
        # sorted differently
        old_index = old_index[:len(potential_destinations.index)]

        available_local_benchmarks = list(local_benchmarks.keys())

        # Check which potential destination has local benchmark and attach local benchmark
        common_local_benchmarks = list(set(available_local_benchmarks).intersection(set(local_benchmark_index)))
        if common_local_benchmarks:
            benchmarks = [local_benchmarks[k]['total_costs'] for k in common_local_benchmarks]
            potential_destinations.loc[common_local_benchmarks, 'local_benchmark'] = benchmarks

        # set old index
        potential_destinations.index = old_index

        if False:  # todo: adjust in configuration
            potential_destinations_check = potential_destinations[potential_destinations['check_benchmark']].copy()
            potential_destinations_no_checking = potential_destinations[~potential_destinations['check_benchmark']].copy()

            # remove all potential destinations based on local benchmark
            potential_destinations_check = potential_destinations_check[potential_destinations_check['total_costs_to_destination']
                                                            <= potential_destinations_check['local_benchmark']].copy()

            potential_destinations = pd.concat([potential_destinations_check, potential_destinations_no_checking])
        else:
            # remove all potential destinations based on local benchmark
            potential_destinations = \
                potential_destinations[potential_destinations['total_costs_to_destination'] <= potential_destinations['local_benchmark']].copy()

    return potential_destinations, road_transport_options_to_calculate


def compare_to_local_benchmarks(solution, potential_destinations,
                                name_infrastructure_column, name_costs_column,
                                local_benchmarks):
    commodity = solution.get_current_commodity_object()

    available_benchmarks = local_benchmarks['index'].values.tolist()

    new_benchmarks = [(inf, commodity.get_name()) for inf in potential_destinations[name_infrastructure_column]]

    # index to check --> these are only unique index
    common_index = list(set(new_benchmarks).intersection(set(available_benchmarks)))
    common_index.sort(key=lambda x: x[0])

    affected_infrastructure = [ind[0] for ind in common_index]

    local_benchmarks_to_check = local_benchmarks[local_benchmarks['index'].isin(common_index)].copy()
    local_benchmarks_to_check.sort_values(['index'], inplace=True)

    affected_destinations\
        = potential_destinations[potential_destinations[name_infrastructure_column].isin(affected_infrastructure)].copy()
    affected_destinations.sort_index(inplace=True)

    not_affected_destinations\
        = potential_destinations[~potential_destinations[name_infrastructure_column].isin(affected_infrastructure)].copy()

    affected_destinations['costs_local_benchmark'] = local_benchmarks_to_check.loc[:, 'total_costs']

    # todo: check again if sorted correctly
    # print(affected_destinations.index)
    # print(common_index)
    # print(local_benchmarks_to_check['index'])

    affected_destinations\
        = affected_destinations[affected_destinations[name_costs_column]
                                <= affected_destinations['costs_local_benchmark']].copy()

    potential_destinations = pd.concat([affected_destinations, not_affected_destinations])

    return potential_destinations
