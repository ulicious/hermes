import time
import math
import gc
import yaml

import pandas as pd

from shapely.geometry import Point

from methods_benchmark import check_if_benchmark_possible
from methods_routing import process_out_tolerance_branches, process_in_tolerance_branches_high_memory,\
    process_in_tolerance_branches_low_memory, get_complete_infrastructure
from methods_algorithm import postprocessing_branches, create_branches_based_on_commodities_at_start,\
    check_for_inaccessibility_and_at_destination, prepare_commodities
from script_benchmark import calculate_benchmark
from methods_geographic import get_continent_from_location
from methods_conversion import apply_conversion
from methods_cost_approximations import calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure
from data_processing._8_attach_conversion_costs_and_efficiency_to_locations import attach_conversion_costs_and_efficiency_to_locations

import logging
logging.getLogger().setLevel(logging.INFO)


def run_algorithm(args):

    """
    Script for calling the different methods to process branches

    @param args: list with the input data for the process: location_index, location_data, dictionary with common data
    dictionary with necessary paths, dictionary with configuration
    @return:
    """

    # get parameters from input
    location_index, location_data, data, config_file, configuration = args
    location_data = location_data.copy().loc[[location_index], :]
    location_data.index = ['Start']

    start_time = time.time()

    data = data.copy()

    print_information = configuration['print_runtime_information']

    # Load location specific parameters
    starting_location = Point([location_data.at['Start', 'longitude'], location_data.at['Start', 'latitude']])
    starting_continent = location_data.at['Start', 'continent_start']

    destination_location = Point(config_file['destination_location'])
    destination_continent = config_file['destination_continent']

    data['k'] = location_index

    # adjust data with new information
    data['start'] = {'location': starting_location,
                     'continent': starting_continent}

    # get all infrastructure options and check access to infrastructure
    complete_infrastructure = get_complete_infrastructure(data)
    complete_infrastructure = check_if_benchmark_possible(data, configuration, complete_infrastructure)

    # adjust minimal distances by checking if distance to destination is minimal distance
    minimal_distances = data['minimal_distances']
    minimal_distances['distance_to_destination'] = complete_infrastructure.loc[minimal_distances.index, 'distance_to_destination']
    to_destination_lower = minimal_distances[minimal_distances['minimal_distance'] >= minimal_distances['distance_to_destination']].index
    minimal_distances.loc[to_destination_lower, 'minimal_distance'] = minimal_distances.loc[to_destination_lower, 'distance_to_destination']
    minimal_distances.loc[to_destination_lower, 'closest_node'] = 'Destination'
    minimal_distances = minimal_distances.drop(['distance_to_destination'], axis=1)

    data['minimal_distances'] = minimal_distances

    # attach conversion costs and efficiencies to start
    path_techno_economic_data = config_file['paths']['project_folder'] + config_file['paths']['raw_data']
    yaml_file = open(path_techno_economic_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    conversions_location_data \
        = attach_conversion_costs_and_efficiency_to_locations(location_data, config_file, techno_economic_data_conversion)
    conversion_costs_and_efficiencies = pd.concat([data['conversion_costs_and_efficiencies'], conversions_location_data])
    data['conversion_costs_and_efficiencies'] = conversion_costs_and_efficiencies

    # add commodities
    commodities, commodity_names = prepare_commodities(config_file, location_data, data)

    for c in commodities:
        data['commodities']['commodity_objects'][c.get_name()] = c

    # load final commodities
    final_commodities = data['commodities']['final_commodities']

    # create branches based on commodities
    branches, branch_number = create_branches_based_on_commodities_at_start(data)

    if not check_for_inaccessibility_and_at_destination(data, configuration, complete_infrastructure, location_index, branches):
        return None
    
    # create empty local benchmark dataframe
    local_benchmarks = pd.DataFrame(columns=['comparison_index', 'current_total_costs',
                                             'current_node', 'current_commodity'])

    # calculate benchmarks
    benchmark = calculate_benchmark(data, configuration, complete_infrastructure)
    initial_benchmark_costs = benchmark

    # remove initial branches if they exceed benchmark
    branches = branches[branches['current_total_costs'] <= benchmark]

    # Start iterations. While loop runs as long as branches dataframe
    final_solution = None
    iteration = 0
    while not branches.empty:  # while loop runs as long as branches to process exist

        len_start_branches = len(branches.index)
        total_time = time.time()

        benchmark_old = benchmark

        """ Iterate through branches and build new branches based on conversion of commodities """
        time_conversion = time.time()
        if iteration > 0:
            # costs of other energy carriers at start is calculated when creating the start_destination_combinations
            # Therefore, no first conversion needed

            all_locations = data['conversion_costs_and_efficiencies']

            no_conversion_possible_locations = all_locations[all_locations['conversion_possible']].index.tolist()
            no_conversion_possible_branches = branches[branches['current_node'].isin(no_conversion_possible_locations)]

            conversion_possible_locations = [i for i in all_locations.index if i not in no_conversion_possible_locations]
            conversion_possible_branches = branches[branches['current_node'].isin(conversion_possible_locations)]

            branches, potential_final_solution, branch_number, benchmark, local_benchmarks = \
                apply_conversion(conversion_possible_branches, configuration, data, branch_number,
                                 benchmark, local_benchmarks, iteration, start_time)

            if potential_final_solution is not None:
                final_solution = potential_final_solution

            # Conversion is applied twice as we always go the route of conversion
            # from commodity X to H2 and from H2 to commodity Y
            # todo: first conversion is mostly from a energy carrier to hydrogen. Check if hydrogen was already used
            #  at location --> if this hydrogen is cheaper than after the conversion, then no second conversion is needed

            if not branches.empty:
                branches, potential_final_solution, branch_number, benchmark, local_benchmarks = \
                    apply_conversion(branches, configuration, data, branch_number,
                                     benchmark, local_benchmarks, iteration, start_time)

            if potential_final_solution is not None:
                final_solution = potential_final_solution

            branches = pd.concat([branches, no_conversion_possible_branches])

        time_conversion = time.time() - time_conversion
        len_conversion_branches = len(branches.index)

        """ Handle memory issues """
        # n = 0
        # delta_benchmark = benchmark - branches['current_total_costs'].min()
        # last_memory = math.inf
        # while True:
        #
        #     # if delta benchmark is small, options will be very limited --> process immediately
        #     if delta_benchmark < 20:
        #         break
        #
        #     free_memory = psutil.virtual_memory().available / (1024 ** 3)
        #     if free_memory < 200:
        #
        #         # if branches has waited 10 times, we break for loop to avoid stuck code
        #         if (free_memory > 150) & (n > 9):
        #             break
        #
        #         # if only few branches exist, we can process them
        #         if (free_memory > 150) & (len(branches.index) > 1000):
        #             break
        #
        #         # check last memory --> if very similar to current memory, might be stuck --> break
        #         if abs(last_memory - free_memory) < 2.5:
        #             break
        #
        #         if free_memory < 25:
        #             print('free memory at: ' + str(free_memory) + ' | ns: ' + str(n))
        #
        #         if n > 100:
        #             print(k)
        #
        #         time.sleep(30)
        #         n += 1
        #
        #         last_memory = free_memory
        #     else:
        #         break

        """ Start routing """
        # Now, the routing starts. The tendency is that branches which are already closer to the destination
        # might reach the destination faster and result in an update of the benchmark and
        # termination of some branches

        time_routing = time.time()
        old_branches = branches.copy()
        if not branches.empty:

            # add information to branches
            current_commodities = branches['current_commodity_object'].tolist()

            branches['road_transportation_costs'] = math.inf
            branches['new_transportation_costs'] = math.inf
            for commodity_object in set(current_commodities):
                commodity_object_branches \
                    = branches[branches['current_commodity_object'] == commodity_object].index

                for mot in ['Road', 'Shipping', 'Pipeline_Liquid', 'Pipeline_Gas']:
                    branches.loc[commodity_object_branches, mot + '_applicable'] \
                        = commodity_object.get_transportation_options_specific_mean_of_transport(mot)

                if commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                    branches.loc[commodity_object_branches, 'new_transportation_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Gas')
                elif commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                    branches.loc[commodity_object_branches, 'new_transportation_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Liquid')

                if commodity_object.get_transportation_options_specific_mean_of_transport('Road'):
                    branches.loc[commodity_object_branches, 'road_transportation_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Road')

                if commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                    branches.loc[commodity_object_branches, 'Pipeline_Gas_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Pipeline_Gas')

                if commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                    branches.loc[commodity_object_branches, 'Pipeline_Liquid_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Pipeline_Liquid')

                elif commodity_object.get_transportation_options_specific_mean_of_transport('Shipping'):
                    branches.loc[commodity_object_branches, 'Shipping_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Shipping')

            # we have two kind of options now:
            # 1 all branches which used new infrastructure / road previously are now at an infrastructure
            # --> use infrastructure
            # 2 all other options search for next infrastructure and use road / new pipeline
            in_infrastructure_branches \
                = branches[branches['current_transport_mean'].isin(['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])]

            out_infrastructure_branches \
                = branches[~branches['current_transport_mean'].isin(['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])]
            out_infrastructure_branches \
                = out_infrastructure_branches[out_infrastructure_branches['Road_applicable']
                                              | out_infrastructure_branches['Pipeline_Gas_applicable']
                                              | out_infrastructure_branches['Pipeline_Liquid_applicable']]

            # todo: check if code works properly and if not, check if this code is necessary. Else, remove
            # # Some branches are actually within tolerance to infrastructure and could use the infrastructure at this
            # # point. However, as it might not be the turn for in infrastructure transportation, we take these branches
            # out_infrastructure_in_tolerance_branches = pd.DataFrame()
            # if iteration > 0:
            #     minimal_distances_for_branches = minimal_distances.loc[out_infrastructure_branches['current_node'].tolist(), 'minimal_distance']
            #     minimal_distances_for_branches.index = out_infrastructure_branches.index
            #     out_infrastructure_branches['minimal_distance'] = minimal_distances_for_branches
            #
            #     already_in_tolerance_branches = out_infrastructure_branches[out_infrastructure_branches['minimal_distance'] <= configuration['tolerance_distance']].index
            #     not_in_tolerance_branches = out_infrastructure_branches[out_infrastructure_branches['minimal_distance'] > configuration['tolerance_distance']].index
            #
            #     out_infrastructure_branches = out_infrastructure_branches.loc[not_in_tolerance_branches, :]
            #     out_infrastructure_in_tolerance_branches = out_infrastructure_branches.loc[already_in_tolerance_branches, :]

            # there is a high chance that several branches are at the same infrastructure (pipeline or harbours).
            # it might be useful to check the branch which is at the infrastructure and has the lowest cost because
            # the chances that this branch will set the lowest local benchmark are high
            if not in_infrastructure_branches.empty:
                pipeline_gas_infrastructure = [i for i in in_infrastructure_branches.index
                                               if 'PG' in in_infrastructure_branches.at[i, 'current_node']]
                in_infrastructure_branches.loc[pipeline_gas_infrastructure, 'current_transport_mean'] = 'Pipeline_Gas'
                nodes = in_infrastructure_branches.loc[pipeline_gas_infrastructure, 'current_node'].values.tolist()

                pipeline_liquid_infrastructure = [i for i in in_infrastructure_branches.index
                                                  if 'PL' in in_infrastructure_branches.at[i, 'current_node']]
                in_infrastructure_branches.loc[pipeline_liquid_infrastructure, 'current_transport_mean'] = 'Pipeline_Liquid'
                nodes += in_infrastructure_branches.loc[pipeline_liquid_infrastructure, 'current_node'].values.tolist()

                combined_index = pipeline_gas_infrastructure + pipeline_liquid_infrastructure
                in_infrastructure_branches.loc[combined_index, 'graph'] \
                    = complete_infrastructure.loc[nodes, 'graph'].values.tolist()

                shipping_infrastructure = [i for i in in_infrastructure_branches.index
                                           if 'H' in in_infrastructure_branches.at[i, 'current_node']]
                in_infrastructure_branches.loc[shipping_infrastructure, 'current_transport_mean'] = 'Shipping'

                # based on the applicability and the necessary transport mean, we can already remove several branches
                # --> e.g. if transport mean is Pipeline_Gas but branch is not Pipeline_Gas_Applicable
                in_infrastructure_branches\
                    = in_infrastructure_branches[(in_infrastructure_branches['current_transport_mean'] == 'Pipeline_Gas')
                                             & (in_infrastructure_branches['Pipeline_Gas_applicable']) |
                                             (in_infrastructure_branches['current_transport_mean'] == 'Pipeline_Liquid')
                                             & (in_infrastructure_branches['Pipeline_Liquid_applicable']) |
                                             (in_infrastructure_branches['current_transport_mean'] == 'Shipping')
                                             & (in_infrastructure_branches['Shipping_applicable'])]

                lowest_cost_branches = []
                graphs = in_infrastructure_branches['graph'].unique()
                for g in graphs:
                    if isinstance(g, str):
                        if ('PG' in g) | ('PL' in g):  # todo: wenn nicht pipeline dann sollte graph None sein
                            g_branches = in_infrastructure_branches[in_infrastructure_branches['graph'] == g]
                            for c in g_branches['current_commodity'].unique():
                                # append the 5 cheapest branches to graph
                                lowest_cost_branches += g_branches[g_branches['current_commodity'] == c]['current_total_costs'].nsmallest(5).index.tolist()

                preselection = in_infrastructure_branches.loc[lowest_cost_branches, :].copy()

                if configuration['use_low_memory']:
                    preselection = process_in_tolerance_branches_low_memory(data, preselection,
                                                                            complete_infrastructure,
                                                                            benchmark, configuration,
                                                                            with_assessment=False)
                else:
                    preselection = process_in_tolerance_branches_high_memory(data, preselection,
                                                                             complete_infrastructure,
                                                                             benchmark, configuration,
                                                                             with_assessment=False)

                if not preselection.empty:
                    # compare results of preselection with other branches. The idea about preselection is that
                    # the branch which has the lowest cost to a pipeline network will set the price for the whole
                    # pipeline network as transportation within networks are cheaper than between networks.
                    # costs to node 1 (to network) + costs node 1 to node 2 (in network) < direct costs to node 2 (to network)

                    # increase current_total_costs minimally because it might lead to floating point problems
                    preselection['current_total_costs'] = preselection['current_total_costs'] * 1.00001

                    preselection.index = ['Z' + str(i) for i in range(len(preselection.index))]

                    preselection['comparison_index'] = [preselection.at[ind, 'current_node']
                                                        + '-' + preselection.at[ind, 'current_commodity']
                                                        for ind in preselection.index]

                    in_infrastructure_branches = pd.concat([in_infrastructure_branches, preselection])

                    # remove duplicates and keep only cheapest
                    in_infrastructure_branches.sort_values(['current_total_costs'], inplace=True)

                    in_infrastructure_branches = \
                        in_infrastructure_branches.drop_duplicates(subset=['comparison_index'], keep='first')

                    index_to_drop = [i for i in in_infrastructure_branches.index if 'Z' in i]
                    in_infrastructure_branches.drop(index_to_drop, inplace=True)

            if not out_infrastructure_branches.empty:

                # processing out tolerance branches can be quite memory expensive as all options will be considered
                # But some options can be removed. For example, if one branch cannot be converted to a
                # commodity which is transportable via oil pipeline because the conversion costs would exceed the benchmark,
                # than this branch does not need to look at the oil pipeline infrastructure

                # first we need for each branch the minimal costs of the conversion to any commodity which is
                # transportable via pipeline gas or pipeline liquid
                if out_infrastructure_branches['current_node'].tolist() == ['Start']:
                    # all branches at start -> no conversion at start possible -> if commodity is not transportable
                    # via pipeline than we just set pipeline costs to infinity. Else pipeline costs = current costs

                    out_infrastructure_branches['min_costs_pipeline_gas'] = math.inf
                    out_infrastructure_branches['min_costs_pipeline_liquid'] = math.inf

                    for c in data['commodities']['commodity_objects'].keys():
                        c_object = data['commodities']['commodity_objects'][c]
                        if c_object.get_transportation_options()['Pipeline_Gas']:
                            valid_branches_gas \
                                = out_infrastructure_branches[out_infrastructure_branches['current_commodity'] == c_object.get_name()].index
                            out_infrastructure_branches.loc[valid_branches_gas, 'min_costs_pipeline_gas'] \
                                = out_infrastructure_branches.loc[valid_branches_gas, 'current_total_costs']

                        if c_object.get_transportation_options()['Pipeline_Liquid']:
                            valid_branches_gas \
                                = out_infrastructure_branches[out_infrastructure_branches['current_commodity'] == c_object.get_name()].index
                            out_infrastructure_branches.loc[valid_branches_gas, 'min_costs_pipeline_liquid'] \
                                = out_infrastructure_branches.loc[valid_branches_gas, 'current_total_costs']

                else:
                    min_costs_pipeline_gas, min_costs_pipeline_liquid \
                        = calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure(data,
                                                                                            out_infrastructure_branches,
                                                                                            cost_column_name='current_total_costs')
                    out_infrastructure_branches['min_costs_pipeline_gas'] = min_costs_pipeline_gas
                    out_infrastructure_branches['min_costs_pipeline_liquid'] = min_costs_pipeline_liquid

                # now check which branch can potentially use pipelines. If a branch cannot use a pipeline type,
                # we don't have to assess the pipeline type for this branch
                no_pipeline_gas_branches \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] > benchmark)
                                              & (out_infrastructure_branches[
                                                     'min_costs_pipeline_liquid'] <= benchmark)].index.tolist()

                no_pipeline_liquid_branches \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] <= benchmark)
                                              & (out_infrastructure_branches[
                                                     'min_costs_pipeline_liquid'] > benchmark)].index.tolist()
                only_shipping_branches \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] > benchmark)
                                              & (out_infrastructure_branches[
                                                     'min_costs_pipeline_liquid'] > benchmark)].index.tolist()

                no_limitation \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] <= benchmark)
                                              & (out_infrastructure_branches['min_costs_pipeline_liquid'] <= benchmark)].index.tolist()

                # process all branches twice:
                # 1: use minimal distances as road distances are quite expensive and transporting to the closest
                # infrastructure can already be too expensive. Branches where it is too expensive are removed
                # 2: afterwards, don't use minimal distances and process all left branches with the complete
                # infrastructure
                if no_pipeline_gas_branches:
                    outside_options_no_gas \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[no_pipeline_gas_branches],
                                                         configuration, iteration, data, benchmark,
                                                         limitation='no_pipeline_gas', use_minimal_distance=True)

                    if not outside_options_no_gas.empty:
                        options_to_consider = outside_options_no_gas['previous_branch'].unique().tolist()

                        outside_options_no_gas \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmark,
                                                             limitation='no_pipeline_gas')
                    else:
                        outside_options_no_gas = pd.DataFrame()
                else:
                    outside_options_no_gas = pd.DataFrame()

                if no_pipeline_liquid_branches:
                    outside_options_no_liquid \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[no_pipeline_liquid_branches],
                                                         configuration, iteration, data, benchmark,
                                                         limitation='no_pipeline_liquid', use_minimal_distance=True)

                    if not outside_options_no_liquid.empty:
                        options_to_consider = outside_options_no_liquid['previous_branch'].unique().tolist()

                        outside_options_no_liquid \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmark,
                                                             limitation='no_pipeline_liquid')
                    else:
                        outside_options_no_liquid = pd.DataFrame()

                else:
                    outside_options_no_liquid = pd.DataFrame()

                if only_shipping_branches:
                    outside_options_only_shipping \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[only_shipping_branches],
                                                         configuration, iteration, data, benchmark,
                                                         limitation='no_pipelines', use_minimal_distance=True)

                    if not outside_options_only_shipping.empty:
                        options_to_consider = outside_options_only_shipping['previous_branch'].unique().tolist()

                        outside_options_only_shipping \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmark,
                                                             limitation='no_pipelines')

                    else:
                        outside_options_only_shipping = pd.DataFrame()
                else:
                    outside_options_only_shipping = pd.DataFrame()

                if no_limitation:
                    outside_options_no_limitation \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[no_limitation],
                                                         configuration, iteration, data, benchmark,
                                                         use_minimal_distance=True)

                    if not outside_options_no_limitation.empty:
                        options_to_consider = outside_options_no_limitation['previous_branch'].unique().tolist()

                        outside_options_no_limitation \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmark)
                    else:
                        outside_options_no_limitation = pd.DataFrame()
                else:
                    outside_options_no_limitation = pd.DataFrame()

                outside_options = pd.concat([outside_options_no_gas, outside_options_no_liquid,
                                             outside_options_only_shipping, outside_options_no_limitation])

            else:
                outside_options = pd.DataFrame()

            if not in_infrastructure_branches.empty:
                if not configuration['use_low_memory']:
                    in_tolerance_options \
                        = process_in_tolerance_branches_high_memory(data, in_infrastructure_branches,
                                                                    complete_infrastructure, benchmark, configuration)
                else:
                    in_tolerance_options \
                        = process_in_tolerance_branches_low_memory(data, in_infrastructure_branches,
                                                                   complete_infrastructure,
                                                                   benchmark, configuration)

                if not outside_options.empty:
                    branches = pd.concat([in_tolerance_options, outside_options], ignore_index=True)
                else:
                    branches = in_tolerance_options
            else:
                if not outside_options.empty:
                    branches = outside_options.reset_index()
                else:
                    branches = pd.DataFrame()

        if not branches.empty:

            # if 'PG_Node_5176' in branches.index.tolist():
            #     print(branches.loc['PG_Node_5176'])
            #     print('')

            branches['current_conversion_costs'] = 0

            branches.sort_values(['current_total_costs'], inplace=True)

            # branches['comparison_index'] = [branches.at[ind, 'current_node'] + '-'
            #                                  + branches.at[ind, 'current_commodity']
            #                                  for ind in branches.index]
            branches = branches.groupby('comparison_index').first().reset_index()

            # Reset the index if needed
            branches.reset_index(drop=True, inplace=True)

        # process all options
        if not branches.empty:

            # use local benchmark to remove branches
            # todo genau nachprüfen was hier passiert weil local benchmark und branches unterschiedlich groß sein können
            #  und deshalb die frage ist was mit merge passiert
            merged_df = pd.merge(branches, local_benchmarks, on='comparison_index',
                                 suffixes=('_branch', '_benchmark'))

            # Filter rows where the costs in branches are not higher than local_benchmarks
            filtered_df \
                = merged_df[merged_df['current_total_costs_branch'] > merged_df['current_total_costs_benchmark']]

            # Get the indices of the rows to be removed from df1
            indices_to_remove = filtered_df['comparison_index']

            # Remove rows from df1
            branches = branches[~branches['comparison_index'].isin(indices_to_remove)]

            # add remaining branches to local benchmark
            new_benchmarks = branches[['comparison_index', 'current_total_costs', 'current_commodity', 'current_node']]

            # remove duplicates and keep only cheapest
            local_benchmarks = pd.concat([local_benchmarks, new_benchmarks])
            local_benchmarks.sort_values(['current_total_costs'], inplace=True)
            local_benchmarks = local_benchmarks.drop_duplicates(subset=['comparison_index'], keep='first')

            # adjust index
            branches['branch_index'] = ['S' + str(branch_number + i) for i in range(len(branches.index))]
            branches.index = branches['branch_index'].tolist()
            branch_number += len(branches.index)

            # update information in dataframe
            branches['conversion_costs'] = 0

            branches['longitude_latitude'] = [(branches.at[i, 'longitude'], branches.at[i, 'latitude'])
                                               for i in branches.index]
            branches['current_continent'] = branches['longitude_latitude'].apply(get_continent_from_location)
            # todo previous transportation and conversion costs + sanity check: does everything work as intended

            branches.drop(['minimal_total_costs', 'longitude_latitude'], axis=1, inplace=True)

            branches = postprocessing_branches(branches, old_branches)

            # check if branches are at destination
            at_destination = branches[branches['distance_to_final_destination']
                                      <= configuration['to_final_destination_tolerance']]
            if not at_destination.empty:
                at_destination_and_correct_commodity \
                    = at_destination[at_destination['current_commodity'].isin(final_commodities)]

                at_destination = branches[
                    branches['distance_to_final_destination'] <= configuration['to_final_destination_tolerance']]
                if not at_destination_and_correct_commodity.empty:

                    at_destination_and_lower_benchmark_and_correct_commodity = \
                        at_destination_and_correct_commodity[at_destination_and_correct_commodity['current_total_costs']
                                                             <= benchmark]

                    if not at_destination_and_lower_benchmark_and_correct_commodity.empty:

                        min_benchmark_costs \
                            = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].min()

                        benchmark = min_benchmark_costs

                        final_solution_index \
                            = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].idxmin()

                        final_solution = branches.loc[final_solution_index, :].copy()
                        final_solution.loc['solving_time'] = time.time() - start_time

                    # remove all branches which are at final destination with correct commodity
                    branches.drop(at_destination_and_correct_commodity.index, inplace=True)

            # check again all branches because benchmark might has changed
            branches = branches[branches['current_total_costs'] <= benchmark]

        time_routing = time.time() - time_routing
        total_time = time.time() - total_time
        time_since_start = time.time() - start_time

        if print_information:

            if final_solution is not None:
                found = ' (found)'
            else:
                found = ' (not found)'

            print(str(location_index) + '-' + str(iteration) + ': Benchmark: ' + str(round(benchmark, 2)) + found +
                  ' | Time conversion: ' + str(round(time_conversion, 2)) + ' s' +
                  ' | Solutions after conversion: ' + str(len_conversion_branches) +
                  ' (' + str(len_start_branches) + ')' +
                  ' | Time routing: ' + str(round(time_routing, 2)) + ' s' +
                  ' | Solutions after routing: ' + str(len(branches.index)) +
                  ' | Time iteration: ' + str(round(total_time, 2)) + ' s' +
                  ' | Time since start: ' + str(round(time_since_start / 60, 2)) + ' m')

            len_routing_solutions_before = len(branches.index)

        iteration += 1

    # store solution in csv
    if final_solution is not None:
        if not final_solution.empty:
            final_solution.loc['status'] = 'complete'
            final_solution.to_csv(configuration['path_results'] + 'location_results/' + str(location_index) + '_final_solution.csv')
        else:
            benchmark = 'Not existing'  # todo adjust
    else:
        benchmark = 'Not existing'  # todo adjust

    print(str(location_index) + ': finished in ' + str(math.ceil((time.time() - start_time) / 60)) + ' minutes. Benchmark was ' +
          str(initial_benchmark_costs) + '. Solution is ' + str(benchmark))

    local_vars = list(locals().items())
    for var, obj in local_vars:
        if var != 'final_solution':
            del obj

    gc.collect()

    return None
