import time

import pandas as pd

from algorithm.methods_algorithm import create_new_branches_based_on_conversion, postprocessing_branches, assess_for_benchmark, \
    drop_branch_comparison_columns, remove_duplicate_branches, update_branch_comparison_index
from algorithm.methods_geographic import calc_distance_list_to_single, calc_distance_list_to_list
from algorithm.methods_cost_approximations import calculate_cheapest_option_to_final_destination
from algorithm.tracking import branch_count, get_tracker, track_benchmark_removal


def apply_conversion(branches, configuration, data, branch_number, benchmark, benchmarks, benchmark_locations,
                     local_benchmarks, iteration, complete_infrastructure):

    """
    Script for conversion of current branches

    @param pandas.DataFrame branches: dataframe with current branches
    @param dict configuration: dictionary with configuration
    @param dict data: dictionary with common data
    @param int branch_number: current branch number
    @param float benchmark: current benchmark
    @param dict benchmarks: current benchmarks
    @param pandas.DataFrame local_benchmarks: dataframe with local benchmarks at nodes and ports
    @param int iteration: current iteration

    @return:
    - update branches dataframe
    - final solution if one branch is at destination with right commodity
    - updated branch number
    - updated benchmark
    - updated local benchmarks
    """

    final_commodities = data['commodities']['final_commodities']
    final_solution = None
    tracker = get_tracker(data)
    method = 'apply_conversion'
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='input', before=branch_count(branches),
                      runtime_s=0.0,
                      details={'benchmark': benchmark, 'branch_number': branch_number})

    # save current branches as old branches to allow comparison
    old_branches = branches.copy()

    # branches with distance 0 from previous iteration will not be conversed
    time_split = time.perf_counter()
    no_conversion_branches = branches[branches['current_distance'] == 0].copy()

    # others will
    conversion_branches = branches[branches['current_distance'] > 0].copy()
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='split_distance_zero', before=branch_count(branches),
                      runtime_s=time.perf_counter() - time_split,
                      details={'no_conversion_branches': branch_count(no_conversion_branches),
                               'conversion_input_branches': branch_count(conversion_branches)})
    before_create_conversion = branch_count(conversion_branches)
    time_create_conversion = time.perf_counter()
    conversion_branches, branch_number \
        = create_new_branches_based_on_conversion(conversion_branches, data, branch_number, benchmarks)
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='create_new_branches_based_on_conversion',
                      before=before_create_conversion, after=branch_count(conversion_branches),
                      created=branch_count(conversion_branches),
                      runtime_s=time.perf_counter() - time_create_conversion,
                      details={'branch_number': branch_number})

    # assess for benchmark
    conversion_branches['benchmark'] = conversion_branches['current_commodity'].map(benchmarks)
    before_current_benchmark = branch_count(conversion_branches)
    time_current_benchmark = time.perf_counter()
    conversion_before_current_benchmark = conversion_branches
    conversion_branches = conversion_branches[conversion_branches['current_total_costs'] <= conversion_branches['benchmark']]
    track_benchmark_removal(data, configuration, conversion_before_current_benchmark, conversion_branches,
                            iteration=iteration, phase='conversion', method=method,
                            code='filter_current_total_vs_benchmark')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='filter_current_total_vs_benchmark',
                      before=before_current_benchmark, after=branch_count(conversion_branches),
                      removed=before_current_benchmark - branch_count(conversion_branches),
                      runtime_s=time.perf_counter() - time_current_benchmark)

    final_destination = data['destination']['location']

    # conversion_branches['distance_to_final_destination'] \
    #     = calc_distance_list_to_single(conversion_branches['latitude'], conversion_branches['longitude'],
    #                                    final_destination.y, final_destination.x)

    if configuration['destination_type'] == 'location':
        conversion_branches['distance_to_final_destination'] \
            = calc_distance_list_to_single(conversion_branches['latitude'], conversion_branches['longitude'],
                                           final_destination.y, final_destination.x)
    else:
        # destination is polygon -> each infrastructure has different closest point to destination
        infrastructure_in_destination = data['destination']['infrastructure']
        distances = calc_distance_list_to_list(conversion_branches['latitude'], conversion_branches['longitude'],
                                               infrastructure_in_destination['latitude'], infrastructure_in_destination['longitude'])

        distances = pd.DataFrame(distances, index=infrastructure_in_destination.index, columns=conversion_branches.index).transpose()

        conversion_branches.loc[conversion_branches.index, 'distance_to_final_destination'] = distances.min('columns')

    in_destination_tolerance \
        = conversion_branches[conversion_branches['distance_to_final_destination']
                              <= configuration['to_final_destination_tolerance']].index
    conversion_branches.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

    # get costs for all options outside tolerance

    min_values, min_commodities = calculate_cheapest_option_to_final_destination(data, conversion_branches, benchmarks,
                                                                                 'current_total_costs')

    conversion_branches['minimal_total_costs'] = min_values
    conversion_branches['minimal_commodity'] = min_commodities

    # throws out options too expensive
    # to compare the different commodities, the benchmark is adjusted by the fuel price
    conversion_branches['benchmark'] = conversion_branches['minimal_commodity'].map(benchmarks)
    before_minimal_benchmark = branch_count(conversion_branches)
    time_minimal_benchmark = time.perf_counter()
    conversion_before_minimal_benchmark = conversion_branches
    conversion_branches = conversion_branches[conversion_branches['minimal_total_costs'] <= conversion_branches['benchmark']]
    track_benchmark_removal(data, configuration, conversion_before_minimal_benchmark, conversion_branches,
                            iteration=iteration, phase='conversion', method=method,
                            code='filter_minimal_total_vs_benchmark')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='filter_minimal_total_vs_benchmark',
                      before=before_minimal_benchmark, after=branch_count(conversion_branches),
                      removed=before_minimal_benchmark - branch_count(conversion_branches),
                      runtime_s=time.perf_counter() - time_minimal_benchmark)

    # remove duplicates
    time_deduplicate = time.perf_counter()
    conversion_branches.sort_values(['current_total_costs'], inplace=True)
    before_dedup = branch_count(conversion_branches)
    conversion_before_dedup = conversion_branches
    conversion_branches = remove_duplicate_branches(conversion_branches)
    track_benchmark_removal(data, configuration, conversion_before_dedup, conversion_branches,
                            iteration=iteration, phase='conversion', method=method,
                            code='deduplicate_comparison_index')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='deduplicate_comparison_index',
                      before=before_dedup, after=branch_count(conversion_branches),
                      removed=before_dedup - branch_count(conversion_branches),
                      runtime_s=time.perf_counter() - time_deduplicate)

    if iteration > 0 and not conversion_branches.empty:
        # assessment via local benchmarks makes only sense as soon as iteration has moved at least once

        # use local benchmark to remove branches
        time_local_benchmark = time.perf_counter()
        conversion_branches = update_branch_comparison_index(conversion_branches)
        before_local_benchmark = branch_count(conversion_branches)
        if not local_benchmarks.empty:
            merged_df = pd.merge(conversion_branches, local_benchmarks, on='comparison_index',
                                 suffixes=('_branch', '_benchmark'))

            # Filter rows where the costs in branches are higher than local_benchmarks
            filtered_df \
                = merged_df[merged_df['current_total_costs_branch'] > merged_df['current_total_costs_benchmark']]

            # Get the indices of the rows to be removed from df1
            indices_to_remove = filtered_df['comparison_index']

            # Remove rows from df1
            conversion_before_local_benchmark = conversion_branches
            conversion_branches = conversion_branches[~conversion_branches['comparison_index'].isin(indices_to_remove)]
            track_benchmark_removal(data, configuration, conversion_before_local_benchmark, conversion_branches,
                                    iteration=iteration, phase='conversion', method=method,
                                    code='filter_local_benchmark')
        if tracker is not None:
            tracker.event(iteration=iteration, phase='conversion', method=method,
                          event='filter_local_benchmark',
                          before=before_local_benchmark, after=branch_count(conversion_branches),
                          removed=before_local_benchmark - branch_count(conversion_branches),
                          runtime_s=time.perf_counter() - time_local_benchmark)

        # add remaining branches to local benchmark
        if not conversion_branches.empty:
            conversion_branches = update_branch_comparison_index(conversion_branches)
            new_benchmarks = conversion_branches[['comparison_index', 'current_total_costs',
                                                  'current_commodity', 'current_node']]

            # remove duplicates and keep only cheapest
            local_benchmarks = pd.concat([local_benchmarks, new_benchmarks])
            local_benchmarks.sort_values(['current_total_costs'], inplace=True)
            local_benchmarks = local_benchmarks.drop_duplicates(subset=['comparison_index'], keep='first')
        conversion_branches = drop_branch_comparison_columns(conversion_branches)

    if no_conversion_branches.empty & conversion_branches.empty:
        return pd.DataFrame(), final_solution, branch_number, benchmark, benchmarks, benchmark_locations, local_benchmarks

    conversion_branches = postprocessing_branches(conversion_branches, old_branches)

    branches = pd.concat([no_conversion_branches, conversion_branches])

    time_final_dedup = time.perf_counter()
    branches.sort_values(['current_total_costs'], inplace=True)
    before_final_dedup = branch_count(branches)
    branches_before_final_dedup = branches
    branches = remove_duplicate_branches(branches)
    track_benchmark_removal(data, configuration, branches_before_final_dedup, branches,
                            iteration=iteration, phase='conversion', method=method,
                            code='deduplicate_output')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='deduplicate_output',
                      before=before_final_dedup, after=branch_count(branches),
                      removed=before_final_dedup - branch_count(branches),
                      runtime_s=time.perf_counter() - time_final_dedup)

    # check if branches are at destination
    time_assess_for_benchmark = time.perf_counter()
    branches_before_assess = branches
    final_solution, benchmark, benchmarks, benchmark_locations, branches \
        = assess_for_benchmark(data, configuration, benchmark, benchmarks, benchmark_locations, final_commodities, branches,
                               final_solution, complete_infrastructure)
    track_benchmark_removal(data, configuration, branches_before_assess, branches,
                            iteration=iteration, phase='conversion', method=method,
                            code='assess_for_benchmark')
    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='after_assess_for_benchmark', after=branch_count(branches),
                      runtime_s=time.perf_counter() - time_assess_for_benchmark,
                      details={'benchmark': benchmark, 'final_solution_exists': final_solution is not None})

    # check again all branches because benchmark might have changed
    # to compare the different commodities, the benchmark is adjusted by the fuel price
    if not branches.empty:
        before_final_filter = branch_count(branches)
        time_final_filter = time.perf_counter()
        branches['benchmark'] = branches['current_commodity'].map(benchmarks)
        branches_before_final_filter = branches
        branches = branches[branches['current_total_costs'] <= branches['benchmark']]
        track_benchmark_removal(data, configuration, branches_before_final_filter, branches,
                                iteration=iteration, phase='conversion', method=method,
                                code='filter_updated_benchmark')
        if tracker is not None:
            tracker.event(iteration=iteration, phase='conversion', method=method,
                          event='filter_updated_benchmark',
                          before=before_final_filter, after=branch_count(branches),
                          removed=before_final_filter - branch_count(branches),
                          runtime_s=time.perf_counter() - time_final_filter)

    if tracker is not None:
        tracker.event(iteration=iteration, phase='conversion', method=method,
                      event='output', after=branch_count(branches),
                      runtime_s=0.0,
                      details={'branch_number': branch_number})

    return branches, final_solution, branch_number, benchmark, benchmarks, benchmark_locations, local_benchmarks
