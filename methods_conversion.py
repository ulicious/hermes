import pandas as pd

from methods_algorithm import create_new_solutions_based_on_conversion, postprocessing_solutions
from _helpers import calc_distance_list_to_single,  calculate_cheapest_option_to_final_destination


def apply_conversion(solutions, configuration, data, solution_number, benchmark, local_benchmarks, iteration):

    final_commodities = data['Commodities']['final']
    final_solution = None

    # save current solutions as old solutions to allow comparison
    old_solutions = solutions.copy()

    # solutions with distance 0 from previous iteration will not be conversed
    no_conversion_solutions = solutions[solutions['current_distance'] == 0].copy()

    # others will
    conversion_solutions = solutions[solutions['current_distance'] > 0].copy()
    conversion_solutions, solution_number \
        = create_new_solutions_based_on_conversion(conversion_solutions, data, solution_number, benchmark)

    # assess newly created solutions
    conversion_solutions = conversion_solutions[conversion_solutions['current_total_costs'] <= benchmark]

    final_destination = data['destination']['location']

    conversion_solutions['distance_to_final_destination'] \
        = calc_distance_list_to_single(conversion_solutions['latitude'], conversion_solutions['longitude'],
                                       final_destination.y, final_destination.x)

    in_destination_tolerance \
        = conversion_solutions[conversion_solutions['distance_to_final_destination']
                               <= configuration['to_final_destination_tolerance']].index
    conversion_solutions.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

    # get costs for all options outside tolerance
    conversion_solutions['minimal_total_costs_to_final_destination'] \
        = calculate_cheapest_option_to_final_destination(data, conversion_solutions,
                                                         configuration, benchmark,
                                                         'current_total_costs')

    # throws out options to expensive
    conversion_solutions \
        = conversion_solutions[conversion_solutions['minimal_total_costs_to_final_destination'] <= benchmark]

    # remove duplicates
    conversion_solutions['comparison_index'] = conversion_solutions.apply(
        lambda row: f"{row['current_node']}-{row['current_commodity']}",
        axis=1)
    conversion_solutions.sort_values(['current_total_costs'], inplace=True)
    conversion_solutions = conversion_solutions.drop_duplicates(subset=['comparison_index'], keep='first')

    if iteration > 0:
        # assessment via local benchmarks makes only sense as soon as iteration has moved at least once

        # use local benchmark to remove solutions
        merged_df = pd.merge(conversion_solutions, local_benchmarks, on='comparison_index',
                             suffixes=('_solution', '_benchmark'))

        # Filter rows where the costs in solutions are higher than local_benchmarks
        filtered_df \
            = merged_df[merged_df['current_total_costs_solution'] > merged_df['current_total_costs_benchmark']]

        # Get the indices of the rows to be removed from df1
        indices_to_remove = filtered_df['comparison_index']

        # Remove rows from df1
        solutions = solutions[~solutions['comparison_index'].isin(indices_to_remove)]

        # add remaining solutions to local benchmark
        new_benchmarks = solutions[['comparison_index', 'current_total_costs',
                                    'current_commodity', 'current_node']]

        # remove duplicates and keep only cheapest
        local_benchmarks = pd.concat([local_benchmarks, new_benchmarks])
        local_benchmarks.sort_values(['current_total_costs'], inplace=True)
        local_benchmarks = local_benchmarks.drop_duplicates(subset=['comparison_index'], keep='first')

    conversion_solutions = postprocessing_solutions(conversion_solutions, old_solutions)

    solutions = pd.concat([no_conversion_solutions, conversion_solutions])

    solutions.sort_values(['current_total_costs'], inplace=True)
    solutions = solutions.drop_duplicates(subset=['comparison_index'], keep='first')

    # check if solutions are at destination
    at_destination = solutions[
        solutions['distance_to_final_destination'] <= configuration['to_final_destination_tolerance']]
    if not at_destination.empty:
        at_destination_and_correct_commodity = at_destination[
            at_destination['current_commodity'].isin(final_commodities)]
        if not at_destination_and_correct_commodity.empty:
            at_destination_and_lower_benchmark_and_correct_commodity = \
                at_destination_and_correct_commodity[
                    at_destination_and_correct_commodity['current_total_costs'] <= benchmark]
            if not at_destination_and_lower_benchmark_and_correct_commodity.empty:
                min_benchmark_costs \
                    = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].min()
                benchmark = min_benchmark_costs
                final_solution_index \
                    = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].idxmin()
                final_solution = solutions.loc[final_solution_index, :].copy()
                final_solution.loc['status'] = 'intermediate'

            # remove all solutions which are at final destination with correct commodity
            solutions.drop(at_destination_and_correct_commodity.index, inplace=True)

        # drop all solutions at destination as further transportation is not necessary
        at_destination = set(at_destination.index) - set(at_destination_and_correct_commodity.index)
        solutions.drop(at_destination, inplace=True)

    # check again all solutions because benchmark might have changed
    solutions = solutions[solutions['current_total_costs'] <= benchmark]

    return solutions, final_solution, solution_number, benchmark, local_benchmarks
