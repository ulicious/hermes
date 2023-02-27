
from methods_checking import check_total_costs_of_solutions, sort_solutions_by_distance_to_destination, get_unique_solutions
from methods_routing import find_benchmark_solution

from object_commodity import create_commodity_objects
from object_solution import Solution
from load_input_data import get_start_destination_combinations
from process_input_data import process_network_data
from methods_plotting import plot_solution_path
from methods_algorithm import create_conversion_solutions, iterate_through_means_of_transport

import logging
import time
logging.getLogger().setLevel(logging.INFO)


def start_algorithm(configuration, location_data, commodity_conversion_data, commodity_transportation_data,
                    pipeline_geodata, pipeline_graphs,
                    railroad_geodata, railroad_graphs,
                    ports):

    # process input data
    start_destination_combinations, final_commodity_list, production_costs_list,\
        country_start_list, continent_start_list, country_destination_list, continent_destination_list\
        = get_start_destination_combinations(location_data)

    # todo: use real data
    pipeline_liquid_networks = pipeline_gas_networks = railroad_networks = {}
    # ports = pd.DataFrame()

    if True:

        pipeline_gas_networks = process_network_data(pipeline_geodata, pipeline_graphs)
        railroad_networks = process_network_data(railroad_geodata, railroad_graphs)

        for j, start_destination_combination in enumerate(start_destination_combinations):

            starting_location = start_destination_combination[0]
            destination_location = start_destination_combination[1]
            destination_continent = continent_destination_list[j]
            final_commodity = final_commodity_list[j]
            production_costs = production_costs_list[j]

            commodities, commodity_names, commodity_names_to_commodity, means_of_transport \
                = create_commodity_objects(production_costs, commodity_conversion_data, commodity_transportation_data)

            # Create initial solutions
            solutions = []
            benchmark = None
            final_solutions = {}
            final_solution = None
            i = 0
            c_num = 0
            for c in commodities:

                s = Solution(name=str(i) + '_' + str(c_num), current_location=starting_location,
                             current_commodity=c.get_name(), current_commodity_object=c,
                             destination=destination_location, final_commodity=final_commodity,
                             total_cost=c.get_production_costs(), ports=ports.copy(),
                             pipeline_gas_networks=pipeline_gas_networks.copy(),
                             pipeline_liquid_networks=pipeline_liquid_networks.copy(),
                             railroad_networks=railroad_networks.copy())

                solutions.append(s)

                # Get benchmark solution based on most expensive commodity production costs
                if c.get_production_costs() == max(production_costs_list[j].values()):
                    benchmark_solution = find_benchmark_solution(s, ports)
                    final_solutions[i] = benchmark_solution
                    final_solution = benchmark_solution
                    benchmark = benchmark_solution.get_total_costs()

                c_num += 1

            solutions = check_total_costs_of_solutions(solutions, benchmark)
            solutions_per_iteration = {}
            all_solutions = []

            i += 1
            while len(solutions) > 0:
                final_solutions[i] = final_solutions[i-1]

                print('Number solutions: ' + str(len(solutions)))
                solutions_per_iteration[i] = solutions

                for s in solutions:
                    s.set_name(s.get_name() + '_' + str(i))

                """ Iterate through solutions and build new solutions based on conversion of commodities """
                logging.info('Start solution creation based on conversion')
                new_solutions = []
                for s in solutions:

                    # As an update of the benchmark might occur within this loop, check all solutions for benchmark
                    # Because all following solutions will also have higher costs
                    if s.get_total_costs() > benchmark:
                        continue

                    new_commodity_solutions, benchmark, final_solution, c_num\
                        = create_conversion_solutions(s, commodities, benchmark, final_solution, c_num, configuration)
                    new_solutions += new_commodity_solutions

                len_solutions_conversion = len(new_solutions) - len(new_solutions)
                solutions = new_solutions.copy()

                logging.info('%s solutions have been created from conversion', len_solutions_conversion)

                """ Start routing """
                logging.info('Start solution creation based on transportation')
                r_num = 0

                # Now, the routing starts. The tendency is that solutions which are already closer to the destination
                # might reach the destination faster and result in an update of the benchmark and
                # termination of some solutions
                solutions = sort_solutions_by_distance_to_destination(solutions, destination_location)

                new_solutions = []
                for s in solutions:
                    # As an update of the benchmark might occur within this loop, check all solutions for benchmark
                    # Because all following solutions will also have higher costs
                    if s.get_total_costs() > benchmark:
                        continue

                    new_routing_solutions, benchmark, final_solution, r_num\
                        = iterate_through_means_of_transport(s, means_of_transport, destination_continent,
                                                             destination_location, ports, benchmark, final_solution,
                                                             r_num, configuration)

                    if new_routing_solutions:
                        new_solutions += new_routing_solutions

                solutions = new_solutions.copy()
                len_solutions_after = len(solutions) - len_solutions_conversion

                logging.info('%s solution(s) have been created from transportation', len_solutions_after)
                logging.info('Assess and remove solution(s)')

                len_before = len(solutions)
                time_before = time.time()

                solutions = check_total_costs_of_solutions(solutions, benchmark)
                solutions = get_unique_solutions(all_solutions, solutions, means_of_transport, commodity_names)

                len_deleted = len_before - len(solutions)
                time_deleted = round(time.time() - time_before, 5)
                average_time = len_deleted / time_deleted
                logging.info('%s solution(s) have been removed', len_deleted)
                logging.info('Deletion took %s second (%s solutions per second)', time_deleted, average_time)

                all_solutions += solutions
                final_solutions[i] = final_solution

                i += 1

            plot_solution_path(solutions_per_iteration, final_solutions)
