import time

import pandas as pd

from algorithm.methods_algorithm import create_new_branches_based_on_conversion, postprocessing_branches
from algorithm.methods_geographic import calc_distance_list_to_single, calc_distance_list_to_list
from algorithm.methods_cost_approximations import calculate_cheapest_option_to_final_destination


def apply_conversion(branches, configuration, data, branch_number, benchmark, local_benchmarks, iteration, start_time):

    """
    Script for conversion of current branches

    @param pandas.DataFrame branches: dataframe with current branches
    @param dict configuration: dictionary with configuration
    @param dict data: dictionary with common data
    @param int branch_number: current branch number
    @param float benchmark: current benchmark
    @param pandas.DataFrame local_benchmarks: dataframe with local benchmarks at nodes and ports
    @param int iteration: current iteration
    @param float start_time: seconds since start of processing of current start location

    @return:
    - update branches dataframe
    - final solution if one branch is at destination with right commodity
    - updated branch number
    - updated benchmark
    - updated local benchmarks
    """

    final_commodities = data['commodities']['final_commodities']
    final_solution = None

    # save current branches as old branches to allow comparison
    old_branches = branches.copy()

    # branches with distance 0 from previous iteration will not be conversed
    no_conversion_branches = branches[branches['current_distance'] == 0].copy()

    # others will
    conversion_branches = branches[branches['current_distance'] > 0].copy()
    conversion_branches, branch_number \
        = create_new_branches_based_on_conversion(conversion_branches, data, branch_number, benchmark)

    # assess newly created branches
    conversion_branches = conversion_branches[conversion_branches['current_total_costs'] <= benchmark]

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
    conversion_branches['minimal_total_costs_to_final_destination'] \
        = calculate_cheapest_option_to_final_destination(data, conversion_branches,
                                                         benchmark, 'current_total_costs')

    # throws out options to expensive
    conversion_branches \
        = conversion_branches[conversion_branches['minimal_total_costs_to_final_destination'] <= benchmark]

    # remove duplicates
    conversion_branches['comparison_index'] = conversion_branches.apply(
        lambda row: f"{row['current_node']}-{row['current_commodity']}",
        axis=1)
    conversion_branches.sort_values(['current_total_costs'], inplace=True)
    conversion_branches = conversion_branches.drop_duplicates(subset=['comparison_index'], keep='first')

    if iteration > 0:
        # assessment via local benchmarks makes only sense as soon as iteration has moved at least once

        # use local benchmark to remove branches
        merged_df = pd.merge(conversion_branches, local_benchmarks, on='comparison_index',
                             suffixes=('_branch', '_benchmark'))

        # Filter rows where the costs in branches are higher than local_benchmarks
        filtered_df \
            = merged_df[merged_df['current_total_costs_branch'] > merged_df['current_total_costs_benchmark']]

        # Get the indices of the rows to be removed from df1
        indices_to_remove = filtered_df['comparison_index']

        # Remove rows from df1
        branches = branches[~branches['comparison_index'].isin(indices_to_remove)]

        # add remaining branches to local benchmark
        new_benchmarks = branches[['comparison_index', 'current_total_costs',
                                   'current_commodity', 'current_node']]

        # remove duplicates and keep only cheapest
        local_benchmarks = pd.concat([local_benchmarks, new_benchmarks])
        local_benchmarks.sort_values(['current_total_costs'], inplace=True)
        local_benchmarks = local_benchmarks.drop_duplicates(subset=['comparison_index'], keep='first')

    if no_conversion_branches.empty & conversion_branches.empty:
        return pd.DataFrame(), final_solution, branch_number, benchmark, local_benchmarks

    conversion_branches = postprocessing_branches(conversion_branches, old_branches)

    branches = pd.concat([no_conversion_branches, conversion_branches])

    branches.sort_values(['current_total_costs'], inplace=True)
    branches = branches.drop_duplicates(subset=['comparison_index'], keep='first')

    # check if branches are at destination
    at_destination = branches[
        branches['distance_to_final_destination'] <= configuration['to_final_destination_tolerance']]
    if not at_destination.empty:

        # check if final commodity
        at_destination_and_correct_commodity = at_destination[
            at_destination['current_commodity'].isin(final_commodities)]
        if not at_destination_and_correct_commodity.empty:

            # check if lower than benchmark
            at_destination_and_lower_benchmark_and_correct_commodity = \
                at_destination_and_correct_commodity[
                    at_destination_and_correct_commodity['current_total_costs'] <= benchmark]
            if not at_destination_and_lower_benchmark_and_correct_commodity.empty:

                # update final solution
                min_benchmark_costs \
                    = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].min()
                benchmark = min_benchmark_costs
                final_solution_index \
                    = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].idxmin()
                final_solution = branches.loc[final_solution_index, :].copy()
                final_solution.loc['solving_time'] = time.time() - start_time

                # set current node to destination
                final_solution['current_node'] = 'Destination'

            # remove all branches which are at final destination with correct commodity
            branches.drop(at_destination_and_correct_commodity.index, inplace=True)

    # check again all branches because benchmark might have changed
    branches = branches[branches['current_total_costs'] <= benchmark]

    return branches, final_solution, branch_number, benchmark, local_benchmarks
