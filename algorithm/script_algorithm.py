import itertools
import time
import math
import gc

import pandas as pd

from shapely.geometry import Point

from algorithm.methods_benchmark import check_if_benchmark_possible
from algorithm.methods_routing import process_out_tolerance_branches, process_in_tolerance_branches_high_memory,\
    process_in_tolerance_branches_low_memory, get_complete_infrastructure, create_branches_from_in_tolerance_locations
from algorithm.methods_algorithm import postprocessing_branches, create_branches_based_on_commodities_at_start,\
    check_for_inaccessibility_and_at_destination, prepare_commodities, assess_for_benchmark, compare_to_local_benchmark,\
    drop_branch_comparison_columns, remove_duplicate_branches, update_branch_comparison_index
from algorithm.script_benchmark import calculate_benchmark
from algorithm.methods_geographic import update_branch_continents
from algorithm.methods_conversion import apply_conversion
from algorithm.methods_cost_approximations import calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure
from data_processing.helpers_attach_costs import attach_conversion_costs_and_efficiency_to_infrastructure, calculate_conversion_costs_and_efficiencies_for_all_combinations
from data_processing.configuration import load_technology_data
from algorithm.tracking import AlgorithmTracker, branch_count, print_benchmark_branches, track_benchmark_removal

import logging
logging.getLogger().setLevel(logging.INFO)


def _finalize_routing_branches(branches, old_branches, local_benchmarks, branch_number, data, configuration,
                               benchmark, benchmarks, benchmark_locations, final_commodities, final_solution,
                               complete_infrastructure):

    if branches.empty:
        return drop_branch_comparison_columns(branches), local_benchmarks, branch_number, final_solution, benchmark, benchmarks, benchmark_locations

    tracker = data.get('tracker')
    iteration = data.get('current_iteration')
    method = '_finalize_routing_branches'
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='input', before=branch_count(branches),
                      runtime_s=0.0,
                      details={'local_benchmarks': branch_count(local_benchmarks)})

    branches = branches.copy()
    branches['current_conversion_costs'] = 0
    time_deduplicate = time.perf_counter()
    branches = update_branch_comparison_index(branches, old_branches)
    branches.sort_values(['current_total_costs'], inplace=True)
    before_dedup = branch_count(branches)
    branches_before_dedup = branches
    branches = branches.drop_duplicates(subset=['comparison_index'], keep='first').reset_index(drop=True)
    track_benchmark_removal(data, configuration, branches_before_dedup, branches,
                            iteration=iteration, phase='routing_finalize', method=method,
                            code='drop_duplicates(comparison_index)')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='deduplicate_comparison_index', before=before_dedup,
                      after=branch_count(branches),
                      removed=before_dedup - branch_count(branches),
                      runtime_s=time.perf_counter() - time_deduplicate)

    if not local_benchmarks.empty:
        time_local_benchmark = time.perf_counter()
        before_local_benchmark = branch_count(branches)
        merged_df = pd.merge(branches, local_benchmarks, on='comparison_index',
                             suffixes=('_branch', '_benchmark'))
        filtered_df = merged_df[merged_df['current_total_costs_branch'] > merged_df['current_total_costs_benchmark']]
        indices_to_remove = filtered_df['comparison_index']
        branches_before_local_benchmark = branches
        branches = branches[~branches['comparison_index'].isin(indices_to_remove)]
        track_benchmark_removal(data, configuration, branches_before_local_benchmark, branches,
                                iteration=iteration, phase='routing_finalize', method=method,
                                code='filter_local_benchmark')
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                          event='filter_local_benchmark', before=before_local_benchmark,
                          after=branch_count(branches),
                          removed=before_local_benchmark - branch_count(branches),
                          runtime_s=time.perf_counter() - time_local_benchmark)

    if branches.empty:
        return drop_branch_comparison_columns(branches), local_benchmarks, branch_number, final_solution, benchmark, benchmarks, benchmark_locations

    time_update_local_benchmarks = time.perf_counter()
    before_local_benchmarks = branch_count(local_benchmarks)
    before_branch_count = branch_count(branches)
    new_benchmarks = branches[['comparison_index', 'current_total_costs', 'current_commodity', 'current_node']]
    local_benchmarks = pd.concat([local_benchmarks, new_benchmarks])
    local_benchmarks.sort_values(['current_total_costs'], inplace=True)
    local_benchmarks = local_benchmarks.drop_duplicates(subset=['comparison_index'], keep='first')
    branches = drop_branch_comparison_columns(branches)
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='update_local_benchmarks',
                      before=before_local_benchmarks, after=branch_count(local_benchmarks),
                      created=branch_count(local_benchmarks) - before_local_benchmarks,
                      runtime_s=time.perf_counter() - time_update_local_benchmarks,
                      details={'branches_used': before_branch_count})

    time_reindex_branches = time.perf_counter()
    before_reindex = branch_count(branches)
    branches['branch_index'] = ['S' + str(branch_number + i) for i in range(len(branches.index))]
    branches.index = branches['branch_index'].tolist()
    branch_number += len(branches.index)

    branches['conversion_costs'] = 0
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='reindex_branches',
                      before=before_reindex, after=branch_count(branches),
                      runtime_s=time.perf_counter() - time_reindex_branches,
                      details={'branch_number': branch_number})

    time_update_continents = time.perf_counter()
    before_update_continents = branch_count(branches)
    branches = update_branch_continents(branches, complete_infrastructure, world=data['world'])
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='update_branch_continents',
                      before=before_update_continents, after=branch_count(branches),
                      runtime_s=time.perf_counter() - time_update_continents)

    time_drop_columns = time.perf_counter()
    before_drop_columns = branch_count(branches)
    drop_columns = [c for c in ['minimal_total_costs', 'minimal_commodity'] if c in branches.columns]
    if drop_columns:
        branches.drop(drop_columns, axis=1, inplace=True)
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='drop_temporary_columns',
                      before=before_drop_columns, after=branch_count(branches),
                      runtime_s=time.perf_counter() - time_drop_columns,
                      details={'columns': drop_columns})

    time_postprocessing = time.perf_counter()
    before_postprocessing = branch_count(branches)
    branches_before_postprocessing = branches
    branches = postprocessing_branches(branches, old_branches)
    track_benchmark_removal(data, configuration, branches_before_postprocessing, branches,
                            iteration=iteration, phase='routing_finalize', method=method,
                            code='postprocessing_branches')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='postprocessing_branches',
                      before=before_postprocessing, after=branch_count(branches),
                      runtime_s=time.perf_counter() - time_postprocessing)

    time_assess_for_benchmark = time.perf_counter()
    branches_before_assess = branches
    final_solution, benchmark, benchmarks, benchmark_locations, branches \
        = assess_for_benchmark(data, configuration, benchmark, benchmarks, benchmark_locations, final_commodities, branches,
                               final_solution, complete_infrastructure)
    track_benchmark_removal(data, configuration, branches_before_assess, branches,
                            iteration=iteration, phase='routing_finalize', method=method,
                            code='assess_for_benchmark')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='after_assess_for_benchmark', after=branch_count(branches),
                      runtime_s=time.perf_counter() - time_assess_for_benchmark,
                      details={'benchmark': benchmark, 'final_solution_exists': final_solution is not None})

    if not branches.empty:
        time_final_benchmark = time.perf_counter()
        before_final_benchmark = branch_count(branches)
        branches['benchmark'] = branches['current_commodity'].map(benchmarks)
        branches_before_final_benchmark = branches
        branches = branches[branches['current_total_costs'] <= branches['benchmark']]
        track_benchmark_removal(data, configuration, branches_before_final_benchmark, branches,
                                iteration=iteration, phase='routing_finalize', method=method,
                                code='filter_updated_benchmark')
        if tracker is not None:
            tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                          event='filter_updated_benchmark', before=before_final_benchmark,
                          after=branch_count(branches),
                          removed=before_final_benchmark - branch_count(branches),
                          runtime_s=time.perf_counter() - time_final_benchmark)

    if tracker is not None:
        tracker.event(iteration=iteration, phase='routing_finalize', method=method,
                      event='output', after=branch_count(branches), runtime_s=0.0)

    return branches, local_benchmarks, branch_number, final_solution, benchmark, benchmarks, benchmark_locations


def _prepare_infrastructure_branches_for_routing(infrastructure_branches, complete_infrastructure, configuration, data,
                                                 benchmarks):

    if infrastructure_branches.empty:
        return pd.DataFrame()

    prepared_branches = infrastructure_branches.copy()
    if 'graph' not in prepared_branches.columns:
        prepared_branches['graph'] = None
    for mot in ['Road', 'Shipping', 'Pipeline_Gas', 'Pipeline_Liquid']:
        column_name = mot + '_applicable'
        if column_name not in prepared_branches.columns:
            prepared_branches[column_name] \
                = prepared_branches['current_commodity_object'].apply(
                    lambda commodity_object: commodity_object.get_transportation_options_specific_mean_of_transport(mot)
                )

    pipeline_gas_infrastructure = [i for i in prepared_branches.index
                                   if 'PG' in prepared_branches.at[i, 'current_node']]
    prepared_branches.loc[pipeline_gas_infrastructure, 'current_transport_mean'] = 'Pipeline_Gas'
    nodes = prepared_branches.loc[pipeline_gas_infrastructure, 'current_node'].values.tolist()

    pipeline_liquid_infrastructure = [i for i in prepared_branches.index
                                      if 'PL' in prepared_branches.at[i, 'current_node']]
    prepared_branches.loc[pipeline_liquid_infrastructure, 'current_transport_mean'] = 'Pipeline_Liquid'
    nodes += prepared_branches.loc[pipeline_liquid_infrastructure, 'current_node'].values.tolist()

    combined_index = pipeline_gas_infrastructure + pipeline_liquid_infrastructure
    if combined_index:
        prepared_branches.loc[combined_index, 'graph'] \
            = complete_infrastructure.loc[nodes, 'graph'].values.tolist()

    shipping_infrastructure = [i for i in prepared_branches.index
                               if 'H' in prepared_branches.at[i, 'current_node']]
    prepared_branches.loc[shipping_infrastructure, 'current_transport_mean'] = 'Shipping'

    prepared_branches \
        = prepared_branches[(prepared_branches['current_transport_mean'] == 'Pipeline_Gas')
                            & (prepared_branches['Pipeline_Gas_applicable']) |
                            (prepared_branches['current_transport_mean'] == 'Pipeline_Liquid')
                            & (prepared_branches['Pipeline_Liquid_applicable']) |
                            (prepared_branches['current_transport_mean'] == 'Shipping')
                            & (prepared_branches['Shipping_applicable'])]

    if prepared_branches.empty:
        return pd.DataFrame()

    lowest_cost_branches = []
    graphs = prepared_branches['graph'].dropna().unique()
    for g in graphs:
        if isinstance(g, str) and (('PG' in g) or ('PL' in g)):
            g_branches = prepared_branches[prepared_branches['graph'] == g]
            for c in g_branches['current_commodity'].unique():
                lowest_cost_branches += g_branches[g_branches['current_commodity'] == c]['current_total_costs'].nsmallest(5).index.tolist()

    preselection = prepared_branches.loc[lowest_cost_branches, :].copy() if lowest_cost_branches else pd.DataFrame()

    if not preselection.empty:
        if configuration['use_low_memory']:
            preselection = process_in_tolerance_branches_low_memory(data, preselection,
                                                                    complete_infrastructure,
                                                                    benchmarks, configuration,
                                                                    with_assessment=False)
        else:
            preselection = process_in_tolerance_branches_high_memory(data, preselection,
                                                                     complete_infrastructure,
                                                                     benchmarks, configuration,
                                                                     with_assessment=False)

        if not preselection.empty:
            preselection['current_total_costs'] = preselection['current_total_costs'] * 1.00001
            preselection.index = ['Z' + str(i) for i in range(len(preselection.index))]

            prepared_branches = pd.concat([prepared_branches, preselection])
            prepared_branches.sort_values(['current_total_costs'], inplace=True)
            prepared_branches = remove_duplicate_branches(prepared_branches)

            index_to_drop = [i for i in prepared_branches.index if 'Z' in str(i)]
            if index_to_drop:
                prepared_branches.drop(index_to_drop, inplace=True)

    return prepared_branches


def run_algorithm(args):

    """
    Script for calling the different methods to process branches

    @param list args: list with the input data for the process: location_index, location_data,
     dictionary with common data dictionary with necessary paths, dictionary with configuration
    """

    # get parameters from input
    location_index, location_data, data, config_file, configuration = args
    location_data = location_data.copy().loc[[location_index], :]
    location_data.index = ['Start']

    print(str(location_index) + ': Start Processing')

    start_time = time.time()
    tracker = AlgorithmTracker(location_index, configuration['path_results'])
    tracker.event(phase='location', method='run_algorithm', event='start')

    data = data.copy()
    data['location_index'] = location_index
    data['tracker'] = tracker

    print_information = configuration['print_runtime_information']

    # Load location specific parameters
    starting_location = Point([location_data.at['Start', 'longitude'], location_data.at['Start', 'latitude']])
    starting_continent = location_data.at['Start', 'continent_start']

    data['k'] = location_index

    # adjust data with new information
    data['start'] = {'location': starting_location,
                     'continent': starting_continent}

    # get all infrastructure options and check access to infrastructure
    complete_infrastructure = get_complete_infrastructure(data, configuration)

    if configuration['destination_type'] == 'country':
        infrastructure_at_destination = complete_infrastructure.loc[data['destination']['infrastructure'], :].copy()

        if infrastructure_at_destination.empty:
            print(str(data['k']) + ': Infrastructure at destination is empty. Adjust considered infrastructure in process raw data script')
            return None

        data['destination']['infrastructure'] = infrastructure_at_destination

    complete_infrastructure = check_if_benchmark_possible(data, configuration, complete_infrastructure)

    # adjust minimal distances by checking if distance to destination is minimal distance
    minimal_distances = data['minimal_distances']

    minimal_distances['distance_to_destination'] = complete_infrastructure.loc[minimal_distances.index, 'distance_to_destination']
    to_destination_lower = minimal_distances[minimal_distances['minimal_distance'] >= minimal_distances['distance_to_destination']].index
    minimal_distances.loc[to_destination_lower, 'minimal_distance'] = minimal_distances.loc[to_destination_lower, 'distance_to_destination']

    if configuration['destination_type'] == 'location':
        minimal_distances.loc[to_destination_lower, 'closest_node'] = 'Destination'

    minimal_distances = minimal_distances.drop(['distance_to_destination'], axis=1)

    data['minimal_distances'] = minimal_distances

    # attach conversion costs and efficiencies to start
    techno_economic_data_conversion, _ = load_technology_data(config_file)

    conversions_location_data \
        = attach_conversion_costs_and_efficiency_to_infrastructure(location_data, config_file,
                                                                   techno_economic_data_conversion, with_tqdm=False)
    conversion_costs_and_efficiency \
        = calculate_conversion_costs_and_efficiencies_for_all_combinations(config_file, conversions_location_data,
                                                                           techno_economic_data_conversion)
    conversion_costs_and_efficiencies = pd.concat([data['conversion_costs_and_efficiencies'], conversions_location_data])
    data['conversion_costs_and_efficiencies'] = conversion_costs_and_efficiencies

    # add commodities
    commodities, commodity_names = prepare_commodities(config_file, location_data, data)

    data['commodities']['all_commodities'] = commodity_names

    for c in commodities:
        data['commodities']['commodity_objects'][c.get_name()] = c

    # load final commodities
    final_commodities = data['commodities']['final_commodities']

    # create branches based on commodities
    time_create_start_branches = time.perf_counter()
    branches, branch_number = create_branches_based_on_commodities_at_start(data)
    new_approach_branches_created = 0
    tracker.event(phase='initialization', method='create_branches_based_on_commodities_at_start',
                  event='created_start_branches', created=branch_count(branches),
                  after=branch_count(branches), runtime_s=time.perf_counter() - time_create_start_branches,
                  details={'branch_number': branch_number})

    if not check_for_inaccessibility_and_at_destination(data, configuration, complete_infrastructure, location_index, branches):
        tracker.event(phase='location', method='run_algorithm', event='stop_inaccessible',
                      runtime_s=time.time() - start_time)
        return None
    
    # create empty local benchmark dataframe
    local_benchmarks = pd.DataFrame(columns=['comparison_index', 'current_total_costs',
                                             'current_node', 'current_commodity'])

    # calculate benchmarks
    time_benchmark = time.time()
    benchmark, benchmarks, benchmark_locations, benchmark_info = calculate_benchmark(data, configuration, complete_infrastructure)
    data['benchmark_info'] = benchmark_info
    tracker.event(phase='benchmark', method='calculate_benchmark', event='runtime',
                  runtime_s=time.time() - time_benchmark,
                  details={'benchmark': benchmark})
    if math.isinf(benchmark):
        print(str(data['k']) + ': Not able to calculate benchmark')
        tracker.event(phase='location', method='run_algorithm', event='stop_no_benchmark',
                      runtime_s=time.time() - start_time)
        return None

    cumulative_benchmark_costs = []
    total = 0
    for i in benchmark_info[4]:
        total += i
        cumulative_benchmark_costs.append(total)

    initial_benchmark_costs = benchmark

    # remove initial branches if they exceed benchmark
    # to compare the different commodities, the benchmark is adjusted by the fuel price
    branches['benchmark'] = branches['current_commodity'].map(benchmarks)
    branches_before_initial_benchmark = branches
    conversion_branches = branches[branches['current_total_costs'] <= branches['benchmark']]
    track_benchmark_removal(data, configuration, branches_before_initial_benchmark, conversion_branches,
                            phase='initialization', method='run_algorithm',
                            code='initial_filter_current_total_vs_benchmark')

    print(str(location_index) + ': Benchmark calculated after [s]: ' + str(time.time() - start_time))

    # Start iterations. While loop runs as long as branches dataframe
    final_solution = None
    iteration = 0
    while not branches.empty:  # while loop runs as long as branches to process exist

        data['current_iteration'] = iteration
        len_start_branches = len(branches.index)
        time_iteration = time.time()
        tracker.event(iteration=iteration, phase='iteration', method='run_algorithm',
                      event='start', before=len_start_branches,
                      runtime_s=0.0,
                      details={'benchmark': benchmark, 'branch_number': branch_number})

        benchmark_old = benchmark

        """ Iterate through branches and build new branches based on conversion of commodities """
        time_conversion = time.time()
        if iteration > 0:
            # costs of other energy carriers at start is calculated when creating the start_destination_combinations
            # Therefore, no first conversion needed

            all_locations = data['conversion_costs_and_efficiencies']

            if False in all_locations['conversion_possible'].tolist():
                no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
            else:
                no_conversion_possible_locations = []

            no_conversion_possible_branches = branches[branches['current_node'].isin(no_conversion_possible_locations)]

            time_split_conversion = time.perf_counter()
            conversion_possible_locations = [i for i in all_locations.index if i not in no_conversion_possible_locations]
            conversion_possible_branches = branches[branches['current_node'].isin(conversion_possible_locations)]
            tracker.event(iteration=iteration, phase='conversion', method='run_algorithm',
                          event='split_conversion_possible', before=branch_count(branches),
                          after=branch_count(conversion_possible_branches),
                          runtime_s=time.perf_counter() - time_split_conversion,
                          details={'no_conversion_possible_branches': branch_count(no_conversion_possible_branches)})

            branches, potential_final_solution, branch_number, benchmark, benchmarks, benchmark_locations, local_benchmarks = \
                apply_conversion(conversion_possible_branches, configuration, data, branch_number,
                                 benchmark, benchmarks, benchmark_locations, local_benchmarks, iteration,
                                 complete_infrastructure)

            if potential_final_solution is not None:
                final_solution = potential_final_solution

            # branches, branch_number, local_benchmarks = compare_to_local_benchmark(data, branch_number, branches,
            #                                                                        local_benchmarks)

            # Conversion is applied twice as we always go the route of conversion
            # from commodity X to H2 and from H2 to commodity Y
            # todo (for further increase of speed): comparison to local benchmarks

            # if not branches.empty:
            #     branches, potential_final_solution, branch_number, benchmark, benchmarks, benchmark_locations, local_benchmarks = \
            #         apply_conversion(branches, configuration, data, branch_number,
            #                          benchmark, benchmarks, benchmark_locations, local_benchmarks, iteration,
            #                          complete_infrastructure)

            # if potential_final_solution is not None:
            #     final_solution = potential_final_solution

            # merge all processed and not processed branches
            before_merge_no_conversion = branch_count(branches)
            time_merge_no_conversion = time.perf_counter()
            branches = pd.concat([branches, no_conversion_possible_branches])
            tracker.event(iteration=iteration, phase='conversion', method='run_algorithm',
                          event='merge_no_conversion_possible', before=before_merge_no_conversion,
                          after=branch_count(branches),
                          created=branch_count(branches) - before_merge_no_conversion,
                          runtime_s=time.perf_counter() - time_merge_no_conversion)

        # check if benchmark solution is still in branches
        if not branches.empty and configuration['print_benchmark_info']:
            print_benchmark_branches(branches, benchmark_info, cumulative_benchmark_costs, label='Conversion')

        time_conversion = time.time() - time_conversion
        tracker.event(iteration=iteration, phase='conversion', method='run_algorithm',
                      event='runtime', before=len_start_branches,
                      after=branch_count(branches), runtime_s=time_conversion)
        time_since_start = time.time() - start_time
        time_iteration_start = time.time() - time_iteration
        len_conversion_branches = len(branches.index)

        if print_information:

            if final_solution is not None:
                found = ' (found)'
            else:
                found = ' (not found)'

            print(str(location_index) + '-' + str(iteration) + ': Benchmark: ' + str(round(benchmark, 2)) + found +
                  ' | Time conversion: ' + str(round(time_conversion, 2)) + ' s' +
                  ' | Solutions after conversion: ' + str(len_conversion_branches) +
                  ' (' + str(len_start_branches) + ')' +
                  ' | Time iteration: ' + str(round(time_iteration_start, 2)) + ' s' +
                  ' | Time since start: ' + str(round(time_since_start / 60, 2)) + ' m')

            len_routing_solutions_before = len(branches.index)

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
        new_approach_branches_current_iteration = 0
        if not branches.empty:

            # add information to branches
            current_commodities = branches['current_commodity_object'].tolist()

            branches['road_transportation_costs'] = math.inf
            branches['new_transportation_costs'] = math.inf
            for commodity_object in set(current_commodities):
                commodity_object_branches \
                    = branches[branches['current_commodity_object'] == commodity_object].index

                # attach boolean if available transport option
                for mot in config_file['available_transport_means']:
                    branches.loc[commodity_object_branches, mot + '_applicable'] \
                        = commodity_object.get_transportation_options_specific_mean_of_transport(mot)

                    commodity_object.get_transportation_costs_specific_mean_of_transport(mot)

                # attach transportation costs
                if True:
                    if commodity_object.get_transportation_options_specific_mean_of_transport('New_Pipeline_Gas'):
                        branches.loc[commodity_object_branches, 'new_transportation_costs'] \
                            = commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Gas')
                    elif commodity_object.get_transportation_options_specific_mean_of_transport('New_Pipeline_Liquid'):
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
            time_split_routing = time.perf_counter()
            in_infrastructure_branches \
                = branches[branches['current_transport_mean'].isin(['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])]

            out_infrastructure_branches \
                = branches[~branches['current_transport_mean'].isin(['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])]
            out_infrastructure_branches \
                = out_infrastructure_branches[out_infrastructure_branches['Road_applicable']
                                              | out_infrastructure_branches['Pipeline_Gas_applicable']
                                              | out_infrastructure_branches['Pipeline_Liquid_applicable']]
            tracker.event(iteration=iteration, phase='routing', method='run_algorithm',
                          event='split_routing_inputs', before=branch_count(branches),
                          runtime_s=time.perf_counter() - time_split_routing,
                          details={'in_infrastructure_branches': branch_count(in_infrastructure_branches),
                                   'out_infrastructure_branches': branch_count(out_infrastructure_branches)})

            new_approach_branches_current_iteration = 0
            in_tolerance_options = pd.DataFrame()
            if not in_infrastructure_branches.empty:
                pending_in_branches = in_infrastructure_branches.copy()
                collected_in_tolerance_options = []

                while not pending_in_branches.empty:
                    prepared_infrastructure_branches \
                        = _prepare_infrastructure_branches_for_routing(pending_in_branches,
                                                                      complete_infrastructure,
                                                                      configuration,
                                                                      data,
                                                                      benchmarks)

                    if prepared_infrastructure_branches.empty:
                        break

                    if not configuration['use_low_memory']:
                        current_in_tolerance_options \
                            = process_in_tolerance_branches_high_memory(data, prepared_infrastructure_branches,
                                                                        complete_infrastructure, benchmarks,
                                                                        configuration)
                    else:
                        current_in_tolerance_options \
                            = process_in_tolerance_branches_low_memory(data, prepared_infrastructure_branches,
                                                                       complete_infrastructure,
                                                                       benchmarks, configuration)

                    current_in_tolerance_options, local_benchmarks, branch_number, final_solution, benchmark, \
                        benchmarks, benchmark_locations \
                        = _finalize_routing_branches(current_in_tolerance_options,
                                                    prepared_infrastructure_branches,
                                                    local_benchmarks,
                                                    branch_number,
                                                    data,
                                                    configuration,
                                                    benchmark,
                                                    benchmarks,
                                                    benchmark_locations,
                                                    final_commodities,
                                                    final_solution,
                                                    complete_infrastructure)

                    if current_in_tolerance_options.empty:
                        break

                    collected_in_tolerance_options.append(current_in_tolerance_options)

                    temporary_zero_distance_branches \
                        = create_branches_from_in_tolerance_locations(data,
                                                                      current_in_tolerance_options,
                                                                      complete_infrastructure,
                                                                      benchmarks,
                                                                      configuration)

                    temporary_zero_distance_branches, local_benchmarks, branch_number, final_solution, benchmark, \
                        benchmarks, benchmark_locations \
                        = _finalize_routing_branches(temporary_zero_distance_branches,
                                                    current_in_tolerance_options,
                                                    local_benchmarks,
                                                    branch_number,
                                                    data,
                                                    configuration,
                                                    benchmark,
                                                    benchmarks,
                                                    benchmark_locations,
                                                    final_commodities,
                                                    final_solution,
                                                    complete_infrastructure)

                    if temporary_zero_distance_branches.empty:
                        break

                    new_approach_branches_current_iteration += len(temporary_zero_distance_branches.index)
                    new_approach_branches_created += len(temporary_zero_distance_branches.index)
                    pending_in_branches = temporary_zero_distance_branches.copy()

                if collected_in_tolerance_options:
                    in_tolerance_options = pd.concat(collected_in_tolerance_options, ignore_index=False)

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
                out_infrastructure_branches['benchmark'] = out_infrastructure_branches['current_commodity'].map(benchmarks)

                no_pipeline_gas_branches \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] > out_infrastructure_branches['benchmark'])
                                              & (out_infrastructure_branches['min_costs_pipeline_liquid'] <= out_infrastructure_branches['benchmark'])].index.tolist()

                no_pipeline_liquid_branches \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] <= out_infrastructure_branches['benchmark'])
                                              & (out_infrastructure_branches['min_costs_pipeline_liquid'] > out_infrastructure_branches['benchmark'])].index.tolist()

                only_shipping_branches \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] > out_infrastructure_branches['benchmark'])
                                              & (out_infrastructure_branches['min_costs_pipeline_liquid'] > out_infrastructure_branches['benchmark'])].index.tolist()

                no_limitation \
                    = out_infrastructure_branches[(out_infrastructure_branches['min_costs_pipeline_gas'] <= out_infrastructure_branches['benchmark'])
                                              & (out_infrastructure_branches['min_costs_pipeline_liquid'] <= out_infrastructure_branches['benchmark'])].index.tolist()

                # covered_branches = no_pipeline_gas_branches + no_pipeline_liquid_branches + only_shipping_branches + no_limitation
                only_tolerance_branches = [i for i in out_infrastructure_branches.index if i not in no_limitation]

                # process all branches twice:
                # 1: use minimal distances as road distances are quite expensive and transporting to the closest
                # infrastructure can already be too expensive. Branches where it is too expensive are removed
                # 2: afterwards, don't use minimal distances and process all left branches with the complete
                # infrastructure
                if no_pipeline_gas_branches:
                    outside_options_no_gas \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[no_pipeline_gas_branches],
                                                         configuration, iteration, data, benchmarks,
                                                         limitation='no_pipeline_gas', use_minimal_distance=True)

                    if not outside_options_no_gas.empty:
                        options_to_consider = outside_options_no_gas['previous_branch'].unique().tolist()

                        outside_options_no_gas \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmarks,
                                                             limitation='no_pipeline_gas')
                    else:
                        outside_options_no_gas = pd.DataFrame()
                else:
                    outside_options_no_gas = pd.DataFrame()

                if no_pipeline_liquid_branches:
                    outside_options_no_liquid \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[no_pipeline_liquid_branches],
                                                         configuration, iteration, data, benchmarks,
                                                         limitation='no_pipeline_liquid', use_minimal_distance=True)

                    if not outside_options_no_liquid.empty:
                        options_to_consider = outside_options_no_liquid['previous_branch'].unique().tolist()

                        outside_options_no_liquid \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmarks,
                                                             limitation='no_pipeline_liquid')
                    else:
                        outside_options_no_liquid = pd.DataFrame()

                else:
                    outside_options_no_liquid = pd.DataFrame()

                if only_shipping_branches:
                    outside_options_only_shipping \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[only_shipping_branches],
                                                         configuration, iteration, data, benchmarks,
                                                         limitation='no_pipelines', use_minimal_distance=True)

                    if not outside_options_only_shipping.empty:
                        options_to_consider = outside_options_only_shipping['previous_branch'].unique().tolist()

                        outside_options_only_shipping \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmarks,
                                                             limitation='no_pipelines')

                    else:
                        outside_options_only_shipping = pd.DataFrame()
                else:
                    outside_options_only_shipping = pd.DataFrame()

                if no_limitation:
                    outside_options_no_limitation \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[no_limitation],
                                                         configuration, iteration, data, benchmarks,
                                                         use_minimal_distance=True)

                    if not outside_options_no_limitation.empty:
                        options_to_consider = outside_options_no_limitation['previous_branch'].unique().tolist()

                        outside_options_no_limitation \
                            = process_out_tolerance_branches(complete_infrastructure,
                                                             out_infrastructure_branches.loc[options_to_consider],
                                                             configuration, iteration, data, benchmarks)
                    else:
                        outside_options_no_limitation = pd.DataFrame()
                else:
                    outside_options_no_limitation = pd.DataFrame()

                # always process all options without limitation if they only aim at in tolerance solutions
                outside_options_in_tolerance \
                    = process_out_tolerance_branches(complete_infrastructure,
                                                     out_infrastructure_branches.loc[only_tolerance_branches],
                                                     configuration, iteration, data, benchmarks,
                                                     use_minimal_distance=True, limitation='only_in_tolerance')

                if not outside_options_in_tolerance.empty:
                    options_to_consider = outside_options_in_tolerance['previous_branch'].unique().tolist()

                    outside_options_in_tolerance \
                        = process_out_tolerance_branches(complete_infrastructure,
                                                         out_infrastructure_branches.loc[options_to_consider],
                                                         configuration, iteration, data, benchmarks,
                                                         limitation='only_in_tolerance')
                else:
                    outside_options_in_tolerance = pd.DataFrame()

                time_combine_outside_options = time.perf_counter()
                outside_options = pd.concat([outside_options_no_gas, outside_options_no_liquid,
                                             outside_options_only_shipping, outside_options_no_limitation,
                                             outside_options_in_tolerance])
                tracker.event(iteration=iteration, phase='routing', method='run_algorithm',
                              event='out_routing_options_combined',
                              after=branch_count(outside_options),
                              runtime_s=time.perf_counter() - time_combine_outside_options,
                              details={'no_gas': branch_count(outside_options_no_gas),
                                       'no_liquid': branch_count(outside_options_no_liquid),
                                       'only_shipping': branch_count(outside_options_only_shipping),
                                       'no_limitation': branch_count(outside_options_no_limitation),
                                       'only_in_tolerance': branch_count(outside_options_in_tolerance)})

            else:
                outside_options = pd.DataFrame()

            processed_outside_options = pd.DataFrame()
            if not outside_options.empty:
                processed_outside_options, local_benchmarks, branch_number, final_solution, benchmark, benchmarks, \
                    benchmark_locations \
                    = _finalize_routing_branches(outside_options,
                                                old_branches,
                                                local_benchmarks,
                                                branch_number,
                                                data,
                                                configuration,
                                                benchmark,
                                                benchmarks,
                                                benchmark_locations,
                                                final_commodities,
                                                final_solution,
                                                complete_infrastructure)

            routing_results = [df for df in [in_tolerance_options, processed_outside_options] if not df.empty]
            if routing_results:
                before_combine = sum(branch_count(df) for df in routing_results)
                time_combine_routing = time.perf_counter()
                branches = pd.concat(routing_results, ignore_index=False)
                branches.sort_values(['current_total_costs'], inplace=True)
                branches_before_routing_dedup = branches
                branches = remove_duplicate_branches(branches)
                track_benchmark_removal(data, configuration, branches_before_routing_dedup, branches,
                                        iteration=iteration, phase='routing', method='run_algorithm',
                                        code='combine_in_and_out_routing/remove_duplicate_branches')
                tracker.event(iteration=iteration, phase='routing', method='run_algorithm',
                              event='combine_in_and_out_routing', before=before_combine,
                              after=branch_count(branches), removed=before_combine - branch_count(branches),
                              runtime_s=time.perf_counter() - time_combine_routing)
            else:
                branches = pd.DataFrame()
                track_benchmark_removal(data, configuration, old_branches, branches,
                                        iteration=iteration, phase='routing', method='run_algorithm',
                                        code='routing_results_empty',
                                        details={'in_tolerance_options': branch_count(in_tolerance_options),
                                                 'processed_outside_options': branch_count(processed_outside_options)})

        if False and not branches.empty:

            # if 'PG_Node_5176' in branches.index.tolist():
            #     print(branches.loc['PG_Node_5176'])
            #     print('')

            branches['current_conversion_costs'] = 0

            branches.sort_values(['current_total_costs'], inplace=True)

            # branches['comparison_index'] = [branches.at[ind, 'current_node'] + '-'
            #                                  + branches.at[ind, 'current_commodity']
            #                                  for ind in branches.index]
            branches = remove_duplicate_branches(branches, reset_index=True)

        # process all options
        if False and not branches.empty:

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

            branches = update_branch_continents(branches, complete_infrastructure, world=data['world'])

            branches.drop(['minimal_total_costs', 'minimal_commodity'],
                          axis=1, inplace=True, errors='ignore')

            branches = postprocessing_branches(branches, old_branches)

            final_solution, benchmark, benchmarks, benchmark_locations, branches \
                = assess_for_benchmark(data, configuration, benchmark, benchmarks, benchmark_locations, final_commodities, branches,
                                       final_solution, complete_infrastructure)

            # check again all branches because benchmark might has changed
            # to compare the different commodities, the benchmark is adjusted by the strike price
            if not branches.empty:
                branches['benchmark'] = branches['current_commodity'].map(benchmarks)
                branches = branches[branches['current_total_costs'] <= branches['benchmark']]

        if not branches.empty and configuration['print_benchmark_info']:
            print_benchmark_branches(branches, benchmark_info, cumulative_benchmark_costs, label='Routing')

        time_routing_start = time.time() - time_routing
        tracker.event(iteration=iteration, phase='routing', method='run_algorithm',
                      event='runtime', after=branch_count(branches),
                      runtime_s=time_routing_start,
                      details={'branch_number': branch_number,
                               'new_approach_branches_created': new_approach_branches_created,
                               'new_approach_branches_current_iteration': new_approach_branches_current_iteration})
        time_iteration_start = time.time() - time_iteration
        time_since_start = time.time() - start_time
        tracker.event(iteration=iteration, phase='iteration', method='run_algorithm',
                      event='end', before=len_start_branches, after=branch_count(branches),
                      runtime_s=time_iteration_start,
                      details={'benchmark': benchmark, 'benchmark_old': benchmark_old})

        # # save branches
        # branches.to_csv(
        #     configuration['path_results'] + 'assessment_current_run/' + str(location_index) + '_' + str(iteration) + '.csv')

        if print_information:

            if final_solution is not None:
                found = ' (found)'
            else:
                found = ' (not found)'

            print(str(location_index) + '-' + str(iteration) + ': Benchmark: ' + str(round(benchmark, 2)) + found +
                  ' | Time conversion: ' + str(round(time_conversion, 2)) + ' s' +
                  ' | Solutions after conversion: ' + str(len_conversion_branches) +
                  ' (' + str(len_start_branches) + ')' +
                  ' | Time routing: ' + str(round(time_routing_start, 2)) + ' s' +
                  ' | Solutions after routing: ' + str(len(branches.index)) +
                  ' | Total branches created: ' + str(branch_number) +
                  ' | New approach branches: ' + str(new_approach_branches_created) +
                  ' (' + str(new_approach_branches_current_iteration) + ' this iteration)' +
                  ' | Time iteration: ' + str(round(time_iteration_start, 2)) + ' s' +
                  ' | Time since start: ' + str(round(time_since_start / 60, 2)) + ' m')

            len_routing_solutions_before = len(branches.index)

        iteration += 1

    # store solution in csv
    if final_solution is not None:
        if not final_solution.empty:
            final_solution.loc['current_node'] = 'Destination'
            final_solution.loc['destination'] = data['destination']['location']
            final_solution.loc['solving_time'] = time.time() - start_time
            final_solution.loc['status'] = 'complete'
            final_solution.to_csv(configuration['path_results'] + 'location_results/' + str(location_index) + '_final_solution.csv')
        else:
            benchmark = 'Not existing'  # todo adjust
    else:
        benchmark = 'Not existing'  # todo adjust

    print(str(location_index) + ': finished in ' + str(math.ceil((time.time() - start_time) / 60)) + ' minutes. Benchmark was ' +
          str(initial_benchmark_costs) + '. Solution is ' + str(benchmark) +
          '. Total branches created: ' + str(branch_number) +
          '. New approach branches: ' + str(new_approach_branches_created))
    tracker.event(phase='location', method='run_algorithm', event='end',
                  runtime_s=time.time() - start_time,
                  details={'initial_benchmark': initial_benchmark_costs,
                           'solution': benchmark,
                           'total_branches_created': branch_number,
                           'new_approach_branches_created': new_approach_branches_created,
                           'final_solution_exists': final_solution is not None})

    local_vars = list(locals().items())
    for var, obj in local_vars:
        if var != 'final_solution':
            del obj

    gc.collect()

    return None
