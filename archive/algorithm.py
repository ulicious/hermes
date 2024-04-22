import time
import math

import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm
from shapely.geometry import Point

from methods_checking import check_total_costs_of_solutions, sort_solutions_by_distance_to_destination, remove_solution_duplications
from algorithm.methods_benchmark import find_shipping_benchmark_solution

from algorithm.object_commodity import create_commodity_objects
from archive.object_solution import Solution, create_solution_dataframe
from algorithm.process_input_data import process_network_data, attach_new_ports
from algorithm.methods_algorithm import create_solutions_from_conversion, create_solutions_from_routing, process_road_options_without_route, \
    process_solutions_reaching_end, update_benchmark_based_on_historic_paths, update_graph_data_based_on_parts, \
    create_graph_from_graph_data, update_local_benchmark_based_on_graph
from algorithm.methods_geographic import calc_distance_list_to_single

import logging
logging.getLogger().setLevel(logging.INFO)


def start_algorithm(configuration, colors, location_data,
                    commodity_conversion_data, commodity_conversion_efficiency_data,
                    commodity_transportation_data, commodity_transportation_efficiency_data,
                    pipeline_gas_geodata, pipeline_gas_graphs,
                    pipeline_liquid_geodata, pipeline_liquid_graphs,
                    railroad_geodata, railroad_graphs,
                    ports, all_distances_inner_infrastructure, all_distances_road, coastlines):

    def run_algorithm(k, num_cores_local, historic_most_cost_effective_routes_local,
                      graph_data_local, graph_connector_data_local):

        start_time = time.time()

        data_k = data.copy()

        # Load basic information
        starting_location = Point([location_data.loc[k, 'start_lon'], location_data.loc[k, 'start_lat']])
        starting_continent = location_data.loc[k, 'continent_start']
        destination_location = Point([location_data.loc[k, 'destination_lon'], location_data.loc[k, 'destination_lat']])
        destination_continent = location_data.loc[k, 'continent_destination']
        final_commodity = location_data.loc[k, 'target_commodity']

        all_infrastructure_local = data_k['All_Infrastructure'].copy()
        all_infrastructure_local['distance_to_destination'] \
            = calc_distance_list_to_single(all_infrastructure_local['latitude'], all_infrastructure_local['longitude'],
                                           destination_location.y, destination_location.x)

        data_k = attach_new_ports(data_k, configuration, starting_continent, starting_location, destination_continent,
                                  destination_location,  coastlines)

        # Commodities at start depend on data given. Get production costs of they exist at start
        commodities_at_start = [i for i in location_data.columns
                                if i not in ['start_lon', 'start_lat', 'destination_lat', 'destination_lon',
                                             'target_commodity', 'country_start', 'continent_start',
                                             'country_destination', 'continent_destination']
                                if i != 'N/A']

        production_costs = {}
        for c in commodities_at_start:
            production_costs[c] = float(location_data.loc[k, c])

        # Based on conversion and transportation parameters, commodity objects are created. These hold all information
        # on the commodity. Does also include those not available at start (!= commodity_at_start)
        commodities, commodity_names, commodity_names_to_commodity, means_of_transport \
            = create_commodity_objects(production_costs,
                                       commodity_conversion_data, commodity_conversion_efficiency_data,
                                       commodity_transportation_data, commodity_transportation_efficiency_data)

        # The data dictionary holds common information/data/parameter which apply for all solutions.
        # todo: check if these are necessary
        data_k['Means_of_Transport'] = means_of_transport
        data_k['Start_End_Combination'] = k

        data_k['Commodities'] = {}
        for c in commodities:
            data_k['Commodities'][c.get_name()] = c

        scenario_count = 0  # used to have unique scenario number for each scenario
        new_node_count = 0  # used to have unique node number for new nodes

        # holds the solutions of the iterations. Gets filled with new solutions and solutions are
        # removed if finished or to expensive. If empty, algorithm stops
        solutions = []

        # Iteration counter
        iteration = 0

        # Set final commodity
        if final_commodity == 'None':
            final_commodity = commodity_names
        else:
            final_commodity = [final_commodity]

        # Create initial solutions based on producible commodities at start
        benchmark = None
        for c in commodities:

            if c.get_production_costs() is None:
                continue

            # Set initial data --> iteration is 0
            iteration_data = {'location': {iteration: starting_location},
                              'continent': {iteration: starting_continent},
                              'commodity': {iteration: c},
                              'commodity_name': {iteration: c.get_name()},
                              'used_node': {iteration: None},
                              'used_infrastructure': {iteration: []},
                              'used_transport_mean': {iteration: None},
                              'total_costs': {iteration: c.get_production_costs()},
                              'transportation_costs': {iteration: 0},
                              'conversion_costs': {iteration: c.get_production_costs()},
                              'length': {iteration: 0},
                              'solution': {iteration: None},
                              'solution_name': {iteration: 'S' + str(scenario_count)}}

            # create solutions
            s = Solution(name='S' + str(scenario_count),
                         destination=destination_location,
                         destination_continent=destination_continent,
                         final_commodity=final_commodity,
                         iteration_data=iteration_data,
                         iterations=[iteration],
                         total_cost=c.get_production_costs(),
                         total_production_costs=c.get_production_costs())

            solutions.append(s)

            # Get benchmark solution based on cheapest transportation costs  # todo adjust: find nearest approach
            if c.get_name() == 'Ammonia':
                benchmark_solution = find_shipping_benchmark_solution(s, data_k, final_commodity, starting_continent,
                                                                      destination_continent, configuration, coastlines)

                benchmark = benchmark_solution.get_total_costs()

                # print(benchmark_solution.get_iteration_data())

                # solution_df = create_solution_dataframe(benchmark_solution)
                # solution_df.to_csv(path_csvs + str(k) + '_benchmark_solution.csv')

            scenario_count += 1

        if print_information:
            print(str(k) + ': Benchmark is: ' + str(benchmark))

        # Remove solutions which are already higher than benchmark
        solutions = check_total_costs_of_solutions(solutions, benchmark)

        # All solutions are stored in a dictionary. They can be used to analyze and plot the solution development
        solutions_per_iteration = {str(iteration) + 'T': solutions}

        final_solution = None  # Is the solutions which has the lowest cost and has arrived at destination with right commodity
        local_benchmarks_dict = {}

        # stores all solutions which are redundant --> solutions which have been a local benchmark but are not anymore
        solutions_to_remove = []

        historic_most_cost_effective_parts_local_copy = {} # todo:describe

        # Start iterations. While loop runs as long as solutions list is not empty
        while len(solutions) > 0:

            solutions_reaching_end = []
            new_historic_paths = {}

            # create and fill graph
            if graph_data_local:
                graph_data_local_copy = graph_data_local.copy()
                graph_connector_data_local_copy = graph_connector_data_local.copy()
                graphs = create_graph_from_graph_data(graph_data_local_copy)

                local_benchmarks_dict = update_local_benchmark_based_on_graph(solutions, local_benchmarks_dict, graphs,
                                                                              graph_connector_data_local_copy)

            # print(str(k) + '-' + str(iteration) + ': Number solutions: ' + str(len(solutions)))

            benchmark_old = benchmark
            len_old_solutions = len(solutions)
            benchmark, solutions = update_benchmark_based_on_historic_paths(benchmark,
                                                                            historic_most_cost_effective_routes_local,
                                                                            solutions)
            if print_information:
                if benchmark_old != benchmark:
                    print(str(k) + '-' + str(iteration) + ': Benchmark was updated to ' + str(
                        benchmark) + ' (from ' + str(benchmark_old) + ')')

                if len_old_solutions != len(solutions):
                    print(str(len_old_solutions - len(solutions)) + ' solutions have been removed')

            solutions = check_total_costs_of_solutions(solutions.copy(), benchmark)

            """ Iterate through solutions and build new solutions based on conversion of commodities """
            if iteration > 0:
                # conversion in first iteration is not allowed as they should produce the commodity directly
                iteration_conversion = str(iteration) + 'C'
                solutions_per_iteration[iteration_conversion] = []

                logging.info('Start solution creation based on conversion')
                new_solutions = []
                for s in solutions:

                    # Check if current solution has used a redundant solution at some point. Remove if so
                    s_solution_path = list(s.get_solution_names().values())
                    if s_solution_path:
                        if solutions_to_remove:
                            if bool(set(s_solution_path) & set(solutions_to_remove)):
                                continue

                    # As an update of the benchmark might occur within this loop, check all solutions for benchmark
                    if s.get_total_costs() > benchmark:
                        continue

                    s.add_iteration(iteration_conversion)

                    # Create new solutions based on conversion
                    new_commodity_solutions, benchmark, final_solution, scenario_count, solutions_per_iteration, \
                        local_benchmarks_dict, solutions_to_remove, solutions_reaching_end, historic_path_costs \
                        = create_solutions_from_conversion(s, scenario_count, commodities, benchmark, final_solution,
                                                           configuration, iteration_conversion, solutions_per_iteration,
                                                           local_benchmarks_dict, solutions_to_remove, solutions_reaching_end)
                    new_solutions += new_commodity_solutions

                    for key in [*historic_path_costs.keys()]:
                        if key not in [*new_historic_paths.keys()]:
                            new_historic_paths[key] = historic_path_costs[key]

                # remove solutions which are at same location with same commodity. Only keep cheapest
                # Such duplicate solutions exist as process is iteratively
                solutions, solutions_to_remove = remove_solution_duplications(new_solutions.copy(), solutions_to_remove)

                # Throw out solutions which exceed benchmark as benchmark is updated iterative
                # (solutions from earlier iterations might be accessed by older (and higher) benchmark
                solutions = check_total_costs_of_solutions(solutions, benchmark)

            """ Start routing """
            logging.info('Start solution creation based on transportation')

            # Now, the routing starts. The tendency is that solutions which are already closer to the destination
            # might reach the destination faster and result in an update of the benchmark and
            # termination of some solutions
            solutions = sort_solutions_by_distance_to_destination(solutions, destination_location)

            iteration_transportation = str(iteration) + 'T'
            solutions_per_iteration[iteration_transportation] = []

            new_solutions = []
            road_options_to_process = []
            solution_to_road_option = {}
            solution_to_road_option_list = []
            j = 0
            length_solutions = len(solutions)
            # print(length_solutions)

            benchmark_old = benchmark
            len_old_solutions = len(solutions)
            historic_most_cost_effective_routes_local_copy = historic_most_cost_effective_routes_local.copy()
            benchmark, solutions = update_benchmark_based_on_historic_paths(benchmark,
                                                                            historic_most_cost_effective_routes_local_copy,
                                                                            solutions)
            if print_information:
                if benchmark_old != benchmark:
                    print(str(k) + '-' + str(iteration) + ': Benchmark was updated to ' + str(benchmark) + ' (from ' + str(benchmark_old) + ')')

                if len_old_solutions != len(solutions):
                    print(str(len_old_solutions - len(solutions)) + ' solutions have been removed')

            solutions = check_total_costs_of_solutions(solutions.copy(), benchmark)

            times = {'time_to_split': 0,
                     'time_asses_in_tolerance': 0,
                     'time_asses_outside_tolerance': 0,
                     'time_calculating_distances': 0,
                     'time_concat_solutions': 0,
                     'processing_solutions': 0,
                     'find_potential_destinations': 0,
                     'get_all_options': 0,
                     'process_networks': 0,
                     'attaching_new_nodes': 0,
                     'check_closest_point': 0,
                     'time_concat_options': 0}

            time_remove_solutions = 0
            time_process_solutions = 0

            total_time_start = time.time()
            for s in solutions:

                # Check if the current solution is not cheapest anymore regarding taken path
                now = time.time()
                s_solution_path = list(s.get_solution_names().values())
                if s_solution_path:
                    if solutions_to_remove:
                        if bool(set(s_solution_path) & set(solutions_to_remove)):
                            # if any step in the solution appears in the list obsolete solutions, don't process solution
                            continue
                time_remove_solutions += time.time() - now

                # As an update of the benchmark might occur within this loop, check all solutions for benchmark
                if s.get_total_costs() <= benchmark:

                    s.add_iteration(iteration_transportation)

                    now = time.time()
                    # Start solution creation based on routing
                    new_routing_solutions, benchmark, final_solution, scenario_count, new_node_count,\
                        local_benchmarks_dict, solutions_to_remove, road_transport_options_to_calculate,\
                        historic_path_costs, solutions_reaching_end, times \
                        = create_solutions_from_routing(data_k, s, scenario_count, new_node_count, benchmark,
                                                        final_solution, configuration, iteration_transportation,
                                                        solutions_per_iteration, local_benchmarks_dict,
                                                        solutions_to_remove,
                                                        historic_most_cost_effective_parts_local_copy, iteration,
                                                        solutions_reaching_end, times)

                    for key in [*historic_path_costs.keys()]:
                        if key not in [*new_historic_paths.keys()]:
                            new_historic_paths[key] = historic_path_costs[key]

                    time_process_solutions += time.time() - now

                    for i in range(len(road_transport_options_to_calculate.index)):
                        solution_to_road_option_list.append(j+i)
                        solution_to_road_option[j+i] = s

                    j += len(road_transport_options_to_calculate.index)

                    road_options_to_process.append(road_transport_options_to_calculate)

                    if new_routing_solutions:
                        new_solutions += new_routing_solutions

                time_processing_solutions = time.time() - total_time_start

                # remove solutions which are at same location with same commodity. Only keep cheapest
                # Such duplicate solutions exist as process is iterative
                solutions, solutions_to_remove = remove_solution_duplications(new_solutions.copy(),
                                                                              solutions_to_remove)

                # Throw out solutions which exceed benchmark as benchmark is updated iteratively
                # (solutions from earlier iterations might be accessed by older (and higher) benchmark
                solutions = check_total_costs_of_solutions(solutions, benchmark)

            if print_information:
                print(str(k) + '-' + str(iteration) + ': Time routing: ' + str(
                    time_processing_solutions / 60) + ' minutes for ' + str(length_solutions) + ' solutions. ' +
                      'Average: ' + str(time_processing_solutions / length_solutions) + ' s/solution')

            if False:

                print('time_processing: ' + str(time_process_solutions))

                print('Total time: ' + str(sum([*times.values()])) + '\n'
                      + '. Time to split: ' + str(times['time_to_split']) + '\n'
                      + '. Time to asses in tolerance: ' + str(times['time_asses_in_tolerance']) + '\n'
                      + '. Time to asses out tolerance: ' + str(times['time_asses_outside_tolerance']) + '\n'
                      + '. Time to calculate distances: ' + str(times['time_calculating_distances']) + '\n'
                      + '. Time to concat solutions: ' + str(times['time_concat_options']) + '\n'
                      + '. Time to process_networks: ' + str(times['process_networks']) + '\n'
                      + '. Time to get options: ' + str(times['get_all_options']))

            if solutions_reaching_end:

                # add successful solutions to shared variable
                historic_most_cost_effective_routes_local \
                    = process_solutions_reaching_end(solutions_reaching_end, historic_most_cost_effective_routes_local,
                                                     final_solution)

            if new_historic_paths:
                graph_data_local, graph_connector_data_local \
                    = update_graph_data_based_on_parts(new_historic_paths,
                                                       graph_data_local, graph_connector_data_local)

            """ process all road options which have no route yet """
            if False:
                logging.info('Process potential destinations needing OSRM data')
                now = time.time()
                road_options_to_process = pd.concat(road_options_to_process)
                if len(road_options_to_process.index) > 0:
                    road_options_to_process['solution_index'] = solution_to_road_option_list
                    final_solution, new_solutions, local_benchmarks, solutions_to_remove = \
                        process_road_options_without_route(data, road_options_to_process, solution_to_road_option, benchmark,
                                                           local_benchmarks, scenario_count, solutions_per_iteration,
                                                           iteration_transportation, configuration, final_solution,
                                                           solutions_to_remove)

                    new_solutions = check_total_costs_of_solutions(new_solutions.copy(), benchmark)
                    solutions += new_solutions
                print('time for OSRM: ' + str((time.time() - now) / 60) + ' minutes')

            iteration += 1

        # store solution in csv
        if final_solution is not None:
            solution_df = create_solution_dataframe(final_solution)
            solution_df.to_csv(path_csvs + str(k) + '_final_solution.csv')

            final_solution_costs = final_solution.get_total_costs()
        else:
            final_solution_costs = 'Not existing'

        print(str(k) + ': finished in ' + str(math.ceil((time.time() - start_time) / 60)) + ' minutes. Benchmark was ' +
              str(benchmark_solution.get_total_costs()) + '. Solution is ' + str(final_solution_costs))

    # process input data
    # location_data = get_start_destination_combinations(location_data)

    # sort the starting locations by their distance to the final destination
    min_index = location_data.index[0]
    distances_from_start_to_destination = calc_distance_list_to_single(location_data['start_lat'],
                                                                       location_data['start_lon'],
                                                                       location_data.loc[min_index, 'destination_lat'],
                                                                       location_data.loc[min_index, 'destination_lon'])

    location_data['distance_to_final_destination'] = distances_from_start_to_destination
    location_data = location_data.sort_values(by=['distance_to_final_destination'])
    location_data.index = range(len(location_data.index))
    # location_data = location_data.loc[[18], :]
    # location_data = location_data.iloc[0:119]

    # todo: return useful data for analysis

    path_plots = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/graphs/'
    path_csvs = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/csvs/'

    data = {'Shipping': {'ports': ports}}
    data['all_distances_inner_infrastructure'] = all_distances_inner_infrastructure
    data['all_distances_road'] = all_distances_road
    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_geodata,
                                pipeline_gas_graphs)
    # data = process_network_data(data, 'Railroad', railroad_geodata, railroad_graphs)
    data['Pipeline_Liquid'] = {}
    data['Railroad'] = {}

    all_infrastructure = pd.concat((ports[['latitude', 'longitude']], pipeline_gas_geodata[['latitude', 'longitude']]))

    data['All_Infrastructure'] = all_infrastructure
    data['Coastline'] = coastlines

    print_information = True

    manager = Manager()
    historic_most_cost_effective_routes = manager.dict()
    historic_most_cost_effective_parts = manager.dict()

    graph_data = manager.dict()
    graph_connector_data = manager.dict()

    num_cores = min(120, multiprocessing.cpu_count() - 1)
    if False:
        inputs = tqdm(location_data.index.tolist())
        Parallel(n_jobs=num_cores)(delayed(run_algorithm)(inp, num_cores,
                                                          historic_most_cost_effective_routes,
                                                          graph_data, graph_connector_data) for inp in inputs)

    # print(results)
