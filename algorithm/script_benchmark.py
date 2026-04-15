import itertools
import math

import numpy as np
import pandas as pd

from collections import defaultdict

from algorithm.methods_benchmark import find_shipping_benchmark_solution, find_pipeline_shipping_solution, find_pipeline_solution


def calculate_benchmark(data, configuration, complete_infrastructure):

    """
    Method runs script which uses different routes to create a valid benchmark solution.
    Possible to print all information on different routes if set in configuration

    @param dict data: dictionary with common data
    @param dict configuration: dictionary with configuration
    @param pandas.DataFrame() complete_infrastructure: all nodes, ports and destination

    @return: value of route with minimal costs
    """

    # todo: derive benchmark solution in clear structure
    print_benchmark_info = configuration['print_benchmark_info']
    min_value_overall = math.inf
    min_value_info = None

    benchmarks = defaultdict(lambda: math.inf)
    benchmark_locations = defaultdict(None)
    benchmark_infos = defaultdict(None)

    commodities = [*data['commodities']['commodity_objects'].keys()]
    commodity_combinations = itertools.combinations_with_replacement(commodities, 2)
    for combination in commodity_combinations:

        commodity_1 = data['commodities']['commodity_objects'][combination[0]]
        commodity_2 = data['commodities']['commodity_objects'][combination[1]]

        conversion_into_each_other_possible = False
        if (commodity_1.get_conversion_options()[commodity_2.get_name()]) & (commodity_2.get_conversion_options()[commodity_1.get_name()]):
            conversion_into_each_other_possible = True
        elif commodity_1 == commodity_2:
            conversion_into_each_other_possible = True

        if conversion_into_each_other_possible:
            # pipeline solutions always depend on conversion. If commodity 1 is not convertible into commodity 2 or
            # both commodities are the same, then we don't use these benchmarks

            if False: # commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_1.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (commodity_1.get_name() != 'Hydrogen_Gas'):
                    if commodity_2.get_transportation_options_specific_mean_of_transport('Shipping'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                            = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_1,
                                                              commodity_2, pipeline_type='Pipeline_Gas')

                        if used_commodities:
                            commodity = used_commodities[-1]
                            if min_value < benchmarks[commodity]:
                                benchmarks[commodity] = math.ceil(min_value)
                                benchmark_locations[commodity] = used_nodes[-2]
                                benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means, used_nodes, distances, costs)

                            if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                                benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                                benchmark_locations[commodity_at_destination] = used_nodes[-2]

                            min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                            if min_value < min_value_overall:
                                min_value_overall = min_value

                                min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if False: # commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_2.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (commodity_2.get_name() != 'Hydrogen_Gas'):
                    if commodity_1.get_transportation_options_specific_mean_of_transport('Shipping'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                            = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_2,
                                                              commodity_1, pipeline_type='Pipeline_Gas')

                        if used_commodities:
                            commodity = used_commodities[-1]
                            if min_value < benchmarks[commodity]:
                                benchmarks[commodity] = math.ceil(min_value)
                                benchmark_locations[commodity] = used_nodes[-2]
                                benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means,
                                                              used_nodes, distances, costs)

                            if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                                benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                                benchmark_locations[commodity_at_destination] = used_nodes[-2]

                            min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                            if min_value < min_value_overall:
                                min_value_overall = min_value

                                min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if False: #commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_2.get_transportation_options_specific_mean_of_transport('Shipping'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                        = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_1,
                                                          commodity_2, pipeline_type='Pipeline_Liquid')

                    if used_commodities:
                        commodity = used_commodities[-1]
                        if min_value < benchmarks[commodity]:
                            benchmarks[commodity] = math.ceil(min_value)
                            benchmark_locations[commodity] = used_nodes[-2]
                            benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means, used_nodes,
                                                          distances, costs)

                        if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                            benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                            benchmark_locations[commodity_at_destination] = used_nodes[-2]

                        min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if False: # commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_1.get_transportation_options_specific_mean_of_transport('Shipping'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                        = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_2,
                                                          commodity_1, pipeline_type='Pipeline_Liquid')

                    if used_commodities:
                        commodity = used_commodities[-1]
                        if min_value < benchmarks[commodity]:
                            benchmarks[commodity] = math.ceil(min_value)
                            benchmark_locations[commodity] = used_nodes[-2]
                            benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means, used_nodes,
                                                          distances, costs)

                        if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                            benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                            benchmark_locations[commodity_at_destination] = used_nodes[-2]

                        min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_1.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (
                        commodity_1.get_name() != 'Hydrogen_Gas'):
                    if commodity_2.get_transportation_options_specific_mean_of_transport('Road'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                            = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_1,
                                                     commodity_2, pipeline_type='Pipeline_Gas')

                        if used_commodities:
                            commodity = used_commodities[-1]
                            if min_value < benchmarks[commodity]:
                                benchmarks[commodity] = math.ceil(min_value)
                                benchmark_locations[commodity] = used_nodes[-2]
                                benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means,
                                                              used_nodes, distances, costs)

                            if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                                benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                                benchmark_locations[commodity_at_destination] = used_nodes[-2]

                            min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                            if min_value < min_value_overall:
                                min_value_overall = min_value

                                min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_2.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (
                        commodity_2.get_name() != 'Hydrogen_Gas'):
                    if commodity_1.get_transportation_options_specific_mean_of_transport('Road'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                            = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_2,
                                                     commodity_1,
                                                     pipeline_type='Pipeline_Gas')

                        if used_commodities:
                            commodity = used_commodities[-1]
                            if min_value < benchmarks[commodity]:
                                benchmarks[commodity] = math.ceil(min_value)
                                benchmark_locations[commodity] = used_nodes[-2]
                                benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means,
                                                              used_nodes, distances, costs)

                            if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                                benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                                benchmark_locations[commodity_at_destination] = used_nodes[-2]

                            min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                            if min_value < min_value_overall:
                                min_value_overall = min_value

                                min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_2.get_transportation_options_specific_mean_of_transport('Road'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                        = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_1,
                                                 commodity_2, pipeline_type='Pipeline_Liquid')

                    if used_commodities:
                        commodity = used_commodities[-1]
                        if min_value < benchmarks[commodity]:
                            benchmarks[commodity] = math.ceil(min_value)
                            benchmark_locations[commodity] = used_nodes[-2]
                            benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means, used_nodes,
                                                          distances, costs)

                        if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                            benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                            benchmark_locations[commodity_at_destination] = used_nodes[-2]

                        min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_1.get_transportation_options_specific_mean_of_transport('Road'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                        = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_2,
                                                 commodity_1, pipeline_type='Pipeline_Liquid')

                    if used_commodities:
                        commodity = used_commodities[-1]
                        if min_value < benchmarks[commodity]:
                            benchmarks[commodity] = math.ceil(min_value)
                            benchmark_locations[commodity] = used_nodes[-2]
                            benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means, used_nodes,
                                                          distances, costs)

                        if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                            benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                            benchmark_locations[commodity_at_destination] = used_nodes[-2]

                        min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

        if ((commodity_1 == commodity_1) & commodity_1.get_transportation_options_specific_mean_of_transport('Shipping')
                & commodity_1.get_transportation_options_specific_mean_of_transport('Road')):
            min_value, used_commodities, used_transport_means, used_nodes, distances, costs, costs_com2_at_destination, commodity_at_destination \
                = find_shipping_benchmark_solution(data, configuration, complete_infrastructure, commodity_1)

            if used_commodities:
                commodity = used_commodities[-1]
                if min_value < benchmarks[commodity]:
                    benchmarks[commodity] = math.ceil(min_value)
                    benchmark_locations[commodity] = used_nodes[-2]
                    benchmark_infos[commodity] = (min_value, used_commodities, used_transport_means, used_nodes,
                                                  distances, costs)

                if costs_com2_at_destination < benchmarks[commodity_at_destination]:
                    benchmarks[commodity_at_destination] = math.ceil(costs_com2_at_destination)
                    benchmark_locations[commodity_at_destination] = used_nodes[-2]

                min_value -= data['commodities']['strike_prices'][used_commodities[-1]]

                if min_value < min_value_overall:
                    min_value_overall = min_value

                    min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

    # # to make sure every commodity has a benchmark, missing ones are added based on hydrogen gas conversion
    # hydrogen_gas = data['commodities']['commodity_objects']['Hydrogen_Gas']
    # for c in data['commodities']['all_commodities']:
    #     if c not in benchmarks.keys():
    #         if hydrogen_gas.get_conversion_options()[c]:
    #             conversion_costs = hydrogen_gas.get_conversion_costs_specific_commodity('Destination', c)
    #             conversion_losses = hydrogen_gas.get_conversion_efficiency_specific_commodity('Destination', c)
    #
    #             benchmarks[c] = (benchmarks['Hydrogen_Gas'] + conversion_costs) / conversion_losses

    # # in case of two conversions necessary, run again over commodities
    # for c in data['commodities']['all_commodities']:
    #     if c in benchmarks.keys():
    #         continue
    #
    #     lowest_costs = math.inf
    #
    #     for c_start in data['commodities']['all_commodities']:
    #         c_start_object = data['commodities']['commodity_objects'][c_start]
    #         if c_start_object.get_conversion_options()[c]:
    #             conversion_costs = c_start_object.get_conversion_costs_specific_commodity('Destination', c)
    #             conversion_losses = c_start_object.get_conversion_efficiency_specific_commodity('Destination', c)
    #             costs = (benchmarks[c_start] + conversion_costs) / conversion_losses
    #
    #             if costs < lowest_costs:
    #                 benchmarks[c] = costs
    #                 lowest_costs = costs

    # some benchmarks are quite high which leaves a lot of options open. Adjust benchmarks based on the benchmark of other comoodities
    final_commodities = data['commodities']['final_commodities']
    old_benchmarks = {}

    while old_benchmarks != benchmarks:
        old_benchmarks = benchmarks.copy()
        benchmarks_keys = [*benchmarks.keys()]

        for commodity_target in data['commodities']['commodity_objects'].keys():
            commodity_benchmark = -math.inf
            commodity_location = None

            for commodity_start in benchmarks_keys:

                if commodity_start == commodity_target:
                    continue

                if (commodity_start in final_commodities) & (benchmarks[commodity_start] != math.inf) & (commodity_target in final_commodities):
                    # if target commodity is final commodity, then it's always possible to change between commodities

                    c_start_object = data['commodities']['commodity_objects'][commodity_start]
                    if c_start_object.get_conversion_options()[commodity_target]:

                        conversion_costs = c_start_object.get_conversion_costs().loc[benchmark_locations[commodity_start], commodity_target]
                        conversion_efficiency = c_start_object.get_conversion_efficiencies().loc[benchmark_locations[commodity_start], commodity_target]

                        costs = (benchmarks[commodity_start] + conversion_costs) / conversion_efficiency

                        # if new and old benchmark is infinite then overwrite with most expensive costs
                        if math.isinf(costs) & math.isinf(benchmarks[commodity_target]):
                            # if new costs are infinite as well, take most expensive, non-infinite costs
                            conversion_costs = c_start_object.get_conversion_costs().loc[:, commodity_target]
                            conversion_efficiency = c_start_object.get_conversion_efficiencies().loc[:, commodity_target]

                            costs = (benchmarks[commodity_start] + conversion_costs) / conversion_efficiency
                            costs.replace([math.inf, -math.inf], np.nan, inplace=True)
                            costs.dropna(inplace=True)

                            if costs.max() < benchmarks[commodity_target]:  # only replace if cheaper
                                benchmarks[commodity_target] = math.ceil(costs.max())
                                benchmark_locations[commodity_target] = costs.idxmax()

                        else:
                            if costs < benchmarks[commodity_target]:  # only replace if cheaper
                                benchmarks[commodity_target] = math.ceil(costs)
                                benchmark_locations[commodity_target] = benchmark_locations[commodity_start]

                elif (commodity_start in final_commodities) & (not math.isinf(benchmarks[commodity_start])) & (commodity_target not in final_commodities):
                    # if target commodity is not final commodity, upper bound is more difficult to address
                    # since different locations and fuel prices might favor more expensive commodities and latter conversion
                    # In this case, iterate over final commodities and check maximal costs the target commodity can have
                    # to allow a conversion to the final commodities

                    c_target_object = data['commodities']['commodity_objects'][commodity_target]
                    if c_target_object.get_conversion_options()[commodity_start]:

                        conversion_costs = c_target_object.get_conversion_costs().loc[:, commodity_start]
                        conversion_costs = conversion_costs.replace([np.inf, -np.inf], np.nan).dropna()

                        conversion_efficiency = c_target_object.get_conversion_efficiencies().loc[conversion_costs.index, commodity_start]

                        costs = benchmarks[commodity_start] * conversion_efficiency - conversion_costs
                        costs = pd.to_numeric(costs, errors="coerce")

                        if costs.max() > commodity_benchmark:  # only replace if more expensive
                            commodity_benchmark = math.ceil(costs.max())
                            commodity_location = costs.idxmax()

            if commodity_target not in final_commodities:
                benchmarks[commodity_target] = commodity_benchmark
                benchmark_locations[commodity_target] = commodity_location

    # commodities which are not convertable to other commodities will be adjusted based on their fuel price and the benchmark
    for commodity in benchmarks.keys():
        if commodity in final_commodities:
            c_object = data['commodities']['commodity_objects'][commodity]
            c_fuel_price = data['commodities']['strike_prices'][commodity]
            if not any(c_object.get_conversion_options().values()):
                benchmarks[commodity] = min(math.ceil(min_value_overall + c_fuel_price), benchmarks[commodity])

    if not math.isinf(min_value_overall):
        benchmark = math.ceil(min_value_overall)

        if print_benchmark_info:
            print('Location: ' + str(data['location_index']))
            print(benchmark)
            for i in min_value_info:
                print(i)
            print(dict(benchmarks))

    else:
        benchmark = math.inf

    return benchmark, benchmarks, benchmark_locations, min_value_info
