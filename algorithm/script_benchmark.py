import itertools
import math

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

    print_benchmark_info = configuration['print_benchmark_info']
    min_value_overall = math.inf
    min_value_info = None

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

            if commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_1.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (commodity_1.get_name() != 'Hydrogen_Gas'):
                    if commodity_2.get_transportation_options_specific_mean_of_transport('Shipping'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                            = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_1,
                                                              commodity_2, pipeline_type='Pipeline_Gas')

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_2.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (commodity_2.get_name() != 'Hydrogen_Gas'):
                    if commodity_1.get_transportation_options_specific_mean_of_transport('Shipping'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                            = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_2,
                                                              commodity_1, pipeline_type='Pipeline_Gas')

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_2.get_transportation_options_specific_mean_of_transport('Shipping'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                        = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_1,
                                                          commodity_2, pipeline_type='Pipeline_Liquid')

                    if min_value < min_value_overall:
                        min_value_overall = min_value

                        min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_1.get_transportation_options_specific_mean_of_transport('Shipping'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                        = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, commodity_2,
                                                          commodity_1, pipeline_type='Pipeline_Liquid')

                    if min_value < min_value_overall:
                        min_value_overall = min_value

                        min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_1.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (
                        commodity_1.get_name() != 'Hydrogen_Gas'):
                    if commodity_2.get_transportation_options_specific_mean_of_transport('Road'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                            = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_1,
                                                     commodity_2, pipeline_type='Pipeline_Gas')

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                if ((commodity_2.get_name() == 'Hydrogen_Gas') & configuration['H2_ready_infrastructure']) | (
                        commodity_2.get_name() != 'Hydrogen_Gas'):
                    if commodity_1.get_transportation_options_specific_mean_of_transport('Road'):

                        min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                            = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_2,
                                                     commodity_1,
                                                     pipeline_type='Pipeline_Gas')

                        if min_value < min_value_overall:
                            min_value_overall = min_value

                            min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_1.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_2.get_transportation_options_specific_mean_of_transport('Road'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                        = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_1,
                                                 commodity_2, pipeline_type='Pipeline_Liquid')

                    if min_value < min_value_overall:
                        min_value_overall = min_value

                        min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

            if commodity_2.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid'):
                if commodity_1.get_transportation_options_specific_mean_of_transport('Road'):

                    min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                        = find_pipeline_solution(data, configuration, complete_infrastructure, commodity_2,
                                                 commodity_1, pipeline_type='Pipeline_Liquid')

                    if min_value < min_value_overall:
                        min_value_overall = min_value

                        min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

        if ((commodity_1 == commodity_1) & commodity_1.get_transportation_options_specific_mean_of_transport('Shipping')
                & commodity_1.get_transportation_options_specific_mean_of_transport('Road')):
            min_value, used_commodities, used_transport_means, used_nodes, distances, costs \
                = find_shipping_benchmark_solution(data, configuration, complete_infrastructure, commodity_1)

            if min_value < min_value_overall:
                min_value_overall = min_value

                min_value_info = [used_commodities, used_transport_means, used_nodes, distances, costs]

    if not math.isinf(min_value_overall):
        benchmark = math.ceil(min_value_overall)
    else:
        benchmark = math.inf

    if print_benchmark_info:
        print(benchmark)
        for i in min_value_info:
            print(i)

    return benchmark
