import time
import math

import gc

import pandas as pd
from shapely.geometry import Point

from methods_benchmark import check_if_benchmark_possible

from methods_routing import process_out_tolerance_solutions, process_in_tolerance_solutions, get_complete_infrastructure

from process_input_data import attach_new_ports

from methods_algorithm import create_new_solutions_based_on_conversion, postprocessing_solutions,\
    create_solutions_based_on_commodities_at_start, check_for_inaccessibility_and_at_destination,\
    adjust_production_costs_of_commodities

from script_benchmark import calculate_benchmark

from _helpers import calc_distance_list_to_single, get_continent_from_location, \
    calculate_cheapest_option_to_final_destination, calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure

import psutil

import logging
logging.getLogger().setLevel(logging.INFO)


def run_algorithm(args):

    # get parameters from input
    k, num_cores_local, location_data, data, configuration, print_information = args
    location_data = location_data.loc[k, :]

    start_time = time.time()

    # Load location specific parameters
    starting_location = Point([location_data.at['start_lon'], location_data.at['start_lat']])
    starting_continent = location_data.at['continent_start']
    destination_location = Point([location_data.at['destination_lon'], location_data.at['destination_lat']])
    destination_continent = location_data.at['continent_destination']

    data['k'] = k

    data = adjust_production_costs_of_commodities(location_data, data)

    # adjust data with new information
    data['start'] = {'location': starting_location,
                     'continent': starting_continent}

    # todo: diskutieren ob neue Häfen installiert werden sollen. Wenn ja, wie und wann??
    # data = attach_new_ports(data, configuration, starting_continent, starting_location, destination_continent,
    #                         destination_location)

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

    # load final commodities
    final_commodities = data['commodities']['final_commodities']

    # create solutions based on commodities
    solutions, solution_number = create_solutions_based_on_commodities_at_start(data)

    if not check_for_inaccessibility_and_at_destination(data, configuration, complete_infrastructure, location_data, k,
                                                        solutions):
        return None
    
    # create empty local benchmark dataframe
    local_benchmarks = pd.DataFrame(columns=['comparison_index', 'current_total_costs',
                                             'current_node', 'current_commodity'])

    # calculate benchmarks
    benchmark = calculate_benchmark(data, configuration, complete_infrastructure)
    initial_benchmark_costs = benchmark

    # remove initial solutions if they exceed benchmark
    solutions = solutions[solutions['current_total_costs'] <= benchmark]

    # Start iterations. While loop runs as long as solutions dataframe
    final_solution = None
    iteration = 0
    while not solutions.empty:

        len_start_solutions = len(solutions.index)
        total_time = time.time()

        benchmark_old = benchmark

        """ Iterate through solutions and build new solutions based on conversion of commodities """
        time_conversion = time.time()
        if configuration['allow_first_iteration_conversion'] | (iteration > 0):

            old_solutions = solutions.copy()

            # solutions with distance 0 from previous iteration will not be conversed
            no_conversion_solutions = solutions[solutions['current_distance'] == 0].copy()

            # others will
            conversion_solutions = solutions[solutions['current_distance'] > 0].copy()
            conversion_solutions, solution_number \
                = create_new_solutions_based_on_conversion(conversion_solutions, data, solution_number, benchmark)

            if False: #not conversion_solutions.empty:
                PG_Node_71643 = conversion_solutions[conversion_solutions['current_node'] == 'PG_Node_71643']
                print(PG_Node_71643[['current_total_costs', 'current_commodity']])

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

            if False: #not conversion_solutions.empty:
                PG_Node_71643 = conversion_solutions[conversion_solutions['current_node'] == 'PG_Node_71643']
                print(PG_Node_71643[['minimal_total_costs_to_final_destination', 'current_commodity']])

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
                        min_benchmark_costs\
                            = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].min()
                        benchmark = min_benchmark_costs
                        final_solution_index\
                            = at_destination_and_lower_benchmark_and_correct_commodity['current_total_costs'].idxmin()
                        final_solution = solutions.loc[final_solution_index, :].copy()
                        final_solution.loc['status'] = 'intermediate'
                        # final_solution.to_csv(path_csvs + str(k) + '_final_solution.csv')

                    # remove all solutions which are at final destination with correct commodity
                    solutions.drop(at_destination_and_correct_commodity.index, inplace=True)

                # drop all solutions at destination as further transportation is not necessary
                at_destination = set(at_destination.index) - set(at_destination_and_correct_commodity.index)
                solutions.drop(at_destination, inplace=True)

            # check again all solutions because benchmark might has changed
            solutions = solutions[solutions['current_total_costs'] <= benchmark]

        time_conversion = time.time() - time_conversion
        time_between = time.time()

        len_conversion_solutions = len(solutions.index)

        # continue process only if sufficient memory is available
        n = 0
        delta_benchmark = benchmark - solutions['current_total_costs'].min()
        last_memory = math.inf
        while True:

            # if delta benchmark is small, options will be very limited --> process immediately
            if delta_benchmark < 20:
                break

            free_memory = psutil.virtual_memory().available / (1024 ** 3)
            if free_memory < 200:

                # if solutions has waited 10 times, we break for loop to avoid stuck code
                if (free_memory > 150) & (n > 9):
                    break

                # if only few solutions exist, we can process them
                if (free_memory > 150) & (len(solutions.index) > 1000):
                    break

                # check last memory --> if very similar to current memory, might be stuck --> break
                if abs(last_memory - free_memory) < 2.5:
                    break

                if free_memory < 25:
                    print('free memory at: ' + str(free_memory) + ' | ns: ' + str(n))

                if n > 100:
                    print(k)

                time.sleep(30)
                n += 1

                last_memory = free_memory
            else:
                break

        """ Start routing """
        # Now, the routing starts. The tendency is that solutions which are already closer to the destination
        # might reach the destination faster and result in an update of the benchmark and
        # termination of some solutions

        time_routing = time.time()
        old_solutions = solutions.copy()
        if not solutions.empty:

            # add information to solutions
            current_commodities = solutions['current_commodity_object'].tolist()

            solutions['road_transportation_costs'] = math.inf
            solutions['new_transportation_costs'] = math.inf
            for commodity_object in set(current_commodities):
                commodity_object_solutions \
                    = solutions[solutions['current_commodity_object'] == commodity_object].index

                for mot in ['Road', 'Shipping', 'Pipeline_Liquid', 'Pipeline_Gas']:
                    solutions.loc[commodity_object_solutions, mot + '_applicable'] \
                        = commodity_object.get_transportation_options_specific_mean_of_transport(mot)

                if commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                    solutions.loc[commodity_object_solutions, 'new_transportation_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Gas')
                elif commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                    solutions.loc[commodity_object_solutions, 'new_transportation_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Liquid')

                if commodity_object.get_transportation_options_specific_mean_of_transport('Road'):
                    solutions.loc[commodity_object_solutions, 'road_transportation_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Road')

                if commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                    solutions.loc[commodity_object_solutions, 'Pipeline_Gas_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Pipeline_Gas')

                if commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                    solutions.loc[commodity_object_solutions, 'Pipeline_Liquid_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Pipeline_Liquid')

                elif commodity_object.get_transportation_options_specific_mean_of_transport('Shipping'):
                    solutions.loc[commodity_object_solutions, 'Shipping_costs'] \
                        = commodity_object.get_transportation_costs_specific_mean_of_transport('Shipping')

            # Solutions will be transported via Road even though commodity cannot be transported via road if
            # distance is below tolerance --> then no "real" transportation takes place (distance = 0)
            # if no "real" transportation takes place, then we can ignore the circumstance that commodity is not
            # transportable via road
            if iteration > 0:
                not_processable_solutions = solutions[solutions['Road_applicable'] == False]
                if not not_processable_solutions.empty:
                    not_processable_solutions['minimal_distance'] \
                        = minimal_distances.loc[not_processable_solutions['current_node'].tolist(), 'minimal_distance']
                    not_processable_solutions \
                        = not_processable_solutions[not_processable_solutions['minimal_distance']
                                                    <= configuration['tolerance_distance']].index
                    solutions.loc[not_processable_solutions, 'Road_applicable'] = True
                    solutions.loc[not_processable_solutions, 'road_transportation_costs'] = 0

            # we have two kind of options now:
            # 1 all solutions which used new infrastructure / road previously are now at a infrastructure
            # --> use infrastructure
            # 2 all other options search for next infrastructure
            in_tolerance_solutions = solutions[solutions['current_transport_mean'].isin(['Road',
                                                                                         'New_Pipeline_Gas',
                                                                                         'New_Pipeline_Liquid'])]

            out_tolerance_solutions = solutions[~solutions['current_transport_mean'].isin(['Road',
                                                                                           'New_Pipeline_Gas',
                                                                                           'New_Pipeline_Liquid'])]
            out_tolerance_solutions \
                = out_tolerance_solutions[out_tolerance_solutions['Road_applicable']
                                          | out_tolerance_solutions['Pipeline_Gas_applicable']
                                          | out_tolerance_solutions['Pipeline_Liquid_applicable']]

            # there is a high chance that several solutions are at the same infrastructure (pipeline or harbours).
            # it might be useful to check the solution which is at the infrastructure and has the lowest cost an.
            if not in_tolerance_solutions.empty:
                pipeline_gas_infrastructure = [i for i in in_tolerance_solutions.index
                                               if 'PG' in in_tolerance_solutions.at[i, 'current_node']]
                in_tolerance_solutions.loc[pipeline_gas_infrastructure, 'current_transport_mean'] = 'Pipeline_Gas'
                nodes = in_tolerance_solutions.loc[pipeline_gas_infrastructure, 'current_node'].values.tolist()

                pipeline_liquid_infrastructure = [i for i in in_tolerance_solutions.index
                                                  if 'PL' in in_tolerance_solutions.at[i, 'current_node']]
                in_tolerance_solutions.loc[pipeline_liquid_infrastructure, 'current_transport_mean'] = 'Pipeline_Liquid'
                nodes += in_tolerance_solutions.loc[pipeline_liquid_infrastructure, 'current_node'].values.tolist()

                combined_index = pipeline_gas_infrastructure + pipeline_liquid_infrastructure
                in_tolerance_solutions.loc[combined_index, 'graph'] \
                    = complete_infrastructure.loc[nodes, 'graph'].values.tolist()

                shipping_infrastructure = [i for i in in_tolerance_solutions.index
                                           if 'H' in in_tolerance_solutions.at[i, 'current_node']]
                in_tolerance_solutions.loc[shipping_infrastructure, 'current_transport_mean'] = 'Shipping'

                # based on the applicability and the necessary transport mean, we can already remove several solutions
                # --> e.g. if transport mean is Pipeline_Gas but solution is not Pipeline_Gas_Applicable
                in_tolerance_solutions\
                    = in_tolerance_solutions[(in_tolerance_solutions['current_transport_mean'] == 'Pipeline_Gas')
                                             & (in_tolerance_solutions['Pipeline_Gas_applicable']) |
                                             (in_tolerance_solutions['current_transport_mean'] == 'Pipeline_Liquid')
                                             & (in_tolerance_solutions['Pipeline_Liquid_applicable']) |
                                             (in_tolerance_solutions['current_transport_mean'] == 'Shipping')
                                             & (in_tolerance_solutions['Shipping_applicable'])]

                lowest_cost_solutions = []
                graphs = in_tolerance_solutions['graph'].unique()
                for g in graphs:
                    if isinstance(g, str):
                        if ('PG' in g) | ('PL' in g):  # todo: wenn nicht pipeline dann sollte graph None sein
                            g_solutions = in_tolerance_solutions[in_tolerance_solutions['graph'] == g]
                            for c in g_solutions['current_commodity'].unique():
                                lowest_cost_solutions.append(g_solutions[g_solutions['current_commodity'] == c]
                                                             ['current_total_costs'].idxmin())

                preselection = in_tolerance_solutions.loc[lowest_cost_solutions, :].copy()
                preselection = process_in_tolerance_solutions(data, preselection, complete_infrastructure,
                                                              local_benchmarks, benchmark, configuration, k,
                                                              with_assessment=False)

                if not preselection.empty:
                    # increase current_total_costs minimally because it might lead to floating point problems
                    preselection['current_total_costs'] = preselection['current_total_costs'] * 1.00001

                    preselection.index = ['Z' + str(i) for i in range(len(preselection.index))]

                    preselection['comparison_index'] = [preselection.at[ind, 'current_node']
                                                        + '-' + preselection.at[ind, 'current_commodity']
                                                        for ind in preselection.index]

                    in_tolerance_solutions = pd.concat([in_tolerance_solutions, preselection])

                    # remove duplicates and keep only cheapest
                    in_tolerance_solutions.sort_values(['current_total_costs'], inplace=True)

                    in_tolerance_solutions =\
                        in_tolerance_solutions.drop_duplicates(subset=['comparison_index'], keep='first')

                    index_to_drop = [i for i in in_tolerance_solutions.index if 'Z' in i]
                    in_tolerance_solutions.drop(index_to_drop, inplace=True)

            if not out_tolerance_solutions.empty:

                # todo: man könnte auch noch alle pipelines entfernen, welche auf einem anderen Kontinent sind

                if False:# not out_tolerance_solutions.empty:
                    PG_Node_71643 = out_tolerance_solutions[out_tolerance_solutions['current_node'] == 'PG_Node_71643']
                    print(PG_Node_71643['taken_routes'].tolist())

                # processing out tolerance solutions can be quite memory expensive as all options will be considered
                # But some options can be removed. For example, if one solution cannot be converted to a
                # commodity which is transportable via oil pipeline because the conversion costs would exceed the benchmark,
                # than this solutions does not need to look at the oil pipeline infrastructure

                # first we need for each solution the minimal costs of the conversion to any commodity which is
                # transportable via pipeline gas or pipeline liquid
                min_costs_pipeline_gas, min_costs_pipeline_liquid \
                    = calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure(data,
                                                                                        out_tolerance_solutions,
                                                                                        cost_column_name='current_total_costs')
                out_tolerance_solutions['min_costs_pipeline_gas'] = min_costs_pipeline_gas
                out_tolerance_solutions['min_costs_pipeline_liquid'] = min_costs_pipeline_liquid

                no_pipeline_gas_solutions \
                    = out_tolerance_solutions[(out_tolerance_solutions['min_costs_pipeline_gas'] > benchmark)
                                              & (out_tolerance_solutions[
                                                     'min_costs_pipeline_liquid'] <= benchmark)].index.tolist()

                no_pipeline_liquid_solutions \
                    = out_tolerance_solutions[(out_tolerance_solutions['min_costs_pipeline_gas'] <= benchmark)
                                              & (out_tolerance_solutions[
                                                     'min_costs_pipeline_liquid'] > benchmark)].index.tolist()
                only_shipping_solutions \
                    = out_tolerance_solutions[(out_tolerance_solutions['min_costs_pipeline_gas'] > benchmark)
                                              & (out_tolerance_solutions[
                                                     'min_costs_pipeline_liquid'] > benchmark)].index.tolist()

                no_limitation \
                    = out_tolerance_solutions[(out_tolerance_solutions['min_costs_pipeline_gas'] <= benchmark)
                                              & (out_tolerance_solutions['min_costs_pipeline_liquid'] <= benchmark)].index.tolist()

                if no_pipeline_gas_solutions:
                    outside_options_no_gas \
                        = process_out_tolerance_solutions(complete_infrastructure,
                                                          out_tolerance_solutions.loc[no_pipeline_gas_solutions],
                                                          configuration, iteration, data, benchmark, local_benchmarks,
                                                          limitation='no_pipeline_liquid', use_minimal_distance=True)

                    if not outside_options_no_gas.empty:
                        options_to_consider = outside_options_no_gas['previous_solution'].unique().tolist()

                        outside_options_no_gas \
                            = process_out_tolerance_solutions(complete_infrastructure,
                                                              out_tolerance_solutions.loc[no_pipeline_gas_solutions],
                                                              configuration, iteration, data, benchmark, local_benchmarks,
                                                              limitation='no_pipeline_gas')
                    else:
                        outside_options_no_gas = pd.DataFrame()
                else:
                    outside_options_no_gas = pd.DataFrame()

                if no_pipeline_liquid_solutions:
                    outside_options_no_liquid \
                        = process_out_tolerance_solutions(complete_infrastructure,
                                                          out_tolerance_solutions.loc[no_pipeline_liquid_solutions],
                                                          configuration, iteration, data, benchmark, local_benchmarks,
                                                          limitation='no_pipeline_liquid', use_minimal_distance=True)

                    if not outside_options_no_liquid.empty:
                        options_to_consider = outside_options_no_liquid['previous_solution'].unique().tolist()

                        outside_options_no_liquid \
                            = process_out_tolerance_solutions(complete_infrastructure,
                                                              out_tolerance_solutions.loc[options_to_consider],
                                                              configuration, iteration, data, benchmark, local_benchmarks,
                                                              limitation='no_pipeline_liquid')
                    else:
                        outside_options_no_liquid = pd.DataFrame()

                    # outside_options_no_liquid \
                    #     = process_out_tolerance_solutions(complete_infrastructure,
                    #                                       out_tolerance_solutions.loc[no_pipeline_liquid_solutions],
                    #                                       configuration, iteration, data, benchmark, local_benchmarks,
                    #                                       limitation='no_pipeline_liquid')
                else:
                    outside_options_no_liquid = pd.DataFrame()

                if only_shipping_solutions:
                    outside_options_only_shipping \
                        = process_out_tolerance_solutions(complete_infrastructure,
                                                          out_tolerance_solutions.loc[only_shipping_solutions],
                                                          configuration, iteration, data, benchmark, local_benchmarks,
                                                          limitation='no_pipeline_liquid', use_minimal_distance=True)

                    if not outside_options_only_shipping.empty:
                        options_to_consider = outside_options_only_shipping['previous_solution'].unique().tolist()

                        outside_options_only_shipping \
                            = process_out_tolerance_solutions(complete_infrastructure,
                                                              out_tolerance_solutions.loc[options_to_consider],
                                                              configuration, iteration, data, benchmark, local_benchmarks,
                                                              limitation='no_pipelines')

                    else:
                        outside_options_only_shipping = pd.DataFrame()
                else:
                    outside_options_only_shipping = pd.DataFrame()

                if no_limitation:
                    outside_options_no_limitation \
                        = process_out_tolerance_solutions(complete_infrastructure,
                                                          out_tolerance_solutions.loc[no_limitation],
                                                          configuration, iteration, data, benchmark, local_benchmarks,
                                                          limitation='no_pipeline_liquid', use_minimal_distance=True)

                    if not outside_options_no_limitation.empty:
                        options_to_consider = outside_options_no_limitation['previous_solution'].unique().tolist()

                        outside_options_no_limitation \
                            = process_out_tolerance_solutions(complete_infrastructure,
                                                              out_tolerance_solutions.loc[options_to_consider],
                                                              configuration, iteration, data, benchmark, local_benchmarks)
                    else:
                        outside_options_no_limitation = pd.DataFrame()
                else:
                    outside_options_no_limitation = pd.DataFrame()

                outside_options = pd.concat([outside_options_no_gas, outside_options_no_liquid,
                                             outside_options_only_shipping, outside_options_no_limitation])

                if False:# not outside_options.empty:
                    PG_Node_22373 = outside_options[outside_options['current_node'] == 'PG_Node_22373']
                    print(PG_Node_22373['current_distance'])

                    H492 = outside_options[outside_options['current_node'] == 'H492']
                    print(H492)

                    # todo: problem: Methan Gas ist nicht transportierbar auf der Straße. Wenn aber
                    #  die Distanz = 0 ist, dann sollte das ignoriert werden

            else:
                outside_options = pd.DataFrame()

            if not in_tolerance_solutions.empty:
                in_tolerance_options \
                    = process_in_tolerance_solutions(data, in_tolerance_solutions, complete_infrastructure,
                                                     local_benchmarks, benchmark, configuration, k)

                if False: #not in_tolerance_options.empty:
                    PG_Node_71643 = in_tolerance_options[in_tolerance_options['current_node'] == 'PG_Node_22373']
                    print(PG_Node_71643)

                    H828 = in_tolerance_options[in_tolerance_options['current_node'] == 'H828']
                    print(H828)

                if not outside_options.empty:
                    solutions = pd.concat([in_tolerance_options, outside_options], ignore_index=True)
                else:
                    solutions = in_tolerance_options
            else:
                if not outside_options.empty:
                    solutions = outside_options.reset_index()
                else:
                    solutions = pd.DataFrame()

        if not solutions.empty:

            solutions['current_conversion_costs'] = 0

            solutions.sort_values(['current_total_costs'], inplace=True)

            solutions['comparison_index'] = [solutions.at[ind, 'current_node'] + '-'
                                               + solutions.at[ind, 'current_commodity']
                                               for ind in solutions.index]
            solutions = solutions.groupby('comparison_index').first().reset_index()

            # Reset the index if needed
            solutions.reset_index(drop=True, inplace=True)

        # process all options
        if not solutions.empty:

            # use local benchmark to remove solutions
            # todo genau nachprüfen was hier passiert weil local benchmark und solutions unterschiedlich groß sein können
            #  und deshalb die frage ist was mit merge passiert
            merged_df = pd.merge(solutions, local_benchmarks, on='comparison_index',
                                 suffixes=('_solution', '_benchmark'))

            # Filter rows where the costs in solutions are not higher than local_benchmarks
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

            # adjust index
            solutions['solution_index'] = ['S' + str(solution_number + i) for i in range(len(solutions.index))]
            solutions.index = solutions['solution_index'].tolist()
            solution_number += len(solutions.index)

            # update information in dataframe
            solutions['conversion_costs'] = 0

            solutions['longitude_latitude'] = [(solutions.at[i, 'longitude'], solutions.at[i, 'latitude'])
                                               for i in solutions.index]
            solutions['current_continent'] = solutions['longitude_latitude'].apply(get_continent_from_location)
            # todo previous transportation and conversion costs + sanity check: does everything work as intended

            solutions.drop(['minimal_total_costs_to_final_destination', 'longitude_latitude'],
                           axis=1, inplace=True)

            solutions = postprocessing_solutions(solutions, old_solutions)

            # check if solutions are at destination
            at_destination = solutions[solutions['distance_to_final_destination']
                                       <= configuration['to_final_destination_tolerance']]
            if not at_destination.empty:
                at_destination_and_correct_commodity \
                    = at_destination[at_destination['current_commodity'].isin(final_commodities)]

                at_destination = solutions[
                    solutions['distance_to_final_destination'] <= configuration['to_final_destination_tolerance']]
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

                        final_solution = solutions.loc[final_solution_index, :].copy()

                    # remove all solutions which are at final destination with correct commodity
                    solutions.drop(at_destination_and_correct_commodity.index, inplace=True)

            # check again all solutions because benchmark might has changed
            solutions = solutions[solutions['current_total_costs'] <= benchmark]

        time_routing = time.time() - time_routing
        total_time = time.time() - total_time
        time_since_start = time.time() - start_time

        if print_information:

            if final_solution is not None:
                found = ' (found)'
            else:
                found = ' (not found)'

            print(str(k) + '-' + str(iteration) + ': Benchmark: ' + str(round(benchmark, 2)) + found +
                  ' | Time conversion: ' + str(round(time_conversion, 2)) + ' s' +
                  ' | Solutions after conversion: ' + str(len_conversion_solutions) +
                  ' (' + str(len_start_solutions) + ')' +
                  ' | Time routing: ' + str(round(time_routing, 2)) + ' s' +
                  ' | Solutions after routing: ' + str(len(solutions.index)) +
                  ' | Time iteration: ' + str(round(total_time, 2)) + ' s' +
                  ' | Time since start: ' + str(round(time_since_start / 60, 2)) + ' m')

            len_routing_solutions_before = len(solutions.index)

        iteration += 1

    # store solution in csv
    if final_solution is not None:
        if not final_solution.empty:
            final_solution.loc['status'] = 'complete'
            final_solution.to_csv(configuration['path_results'] + str(k) + '_final_solution.csv')
        else:
            benchmark = 'Not existing'  # todo adjust
    else:
        benchmark = 'Not existing'  # todo adjust

    print(str(k) + ': finished in ' + str(math.ceil((time.time() - start_time) / 60)) + ' minutes. Benchmark was ' +
          str(initial_benchmark_costs) + '. Solution is ' + str(benchmark))

    local_vars = list(locals().items())
    for var, obj in local_vars:
        if var != 'final_solution':
            del obj

    gc.collect()

    return None
