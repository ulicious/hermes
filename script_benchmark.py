import math

from methods_benchmark import find_shipping_benchmark_solution, find_road_benchmark_solution, \
    find_pipeline_shipping_solution, find_pipeline_solution


def calculate_benchmark(data, configuration, complete_infrastructure):
    pipeline_commodity = data['commodities']['commodity_objects']['Methane_Gas']
    shipping_commodity = data['commodities']['commodity_objects']['Methane_Liquid']

    min_value_1, used_commodities, used_transport_means, used_nodes, distances, costs \
        = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, pipeline_commodity,
                                          shipping_commodity)

    print_benchmark_info = False
    if print_benchmark_info:
        print(min_value_1)
        print(used_commodities)
        print(used_transport_means)
        print(used_nodes)
        print(distances)
        print(costs)

    if configuration['H2_ready_infrastructure']:
        pipeline_commodity = data['commodities']['commodity_objects']['Hydrogen_Gas']
        shipping_commodity = data['commodities']['commodity_objects']['Ammonia']

        min_value_2, used_commodities, used_transport_means, used_nodes, distances, costs \
            = find_pipeline_shipping_solution(data, configuration, complete_infrastructure, pipeline_commodity,
                                              shipping_commodity)

        if print_benchmark_info:
            print(min_value_2)
            print(used_commodities)
            print(used_transport_means)
            print(used_nodes)
            print(distances)
            print(costs)
    else:
        min_value_2 = math.inf

    if configuration['H2_ready_infrastructure']:
        pipeline_commodity = data['commodities']['commodity_objects']['Hydrogen_Gas']
        road_commodity = data['commodities']['commodity_objects']['DBT']

        min_value_3, used_commodities, used_transport_means, used_nodes, distances, costs \
            = find_pipeline_solution(data, configuration, complete_infrastructure, pipeline_commodity, road_commodity)
        # todo: pipeline type als option geben

        if print_benchmark_info:
            print(min_value_3)
            print(used_commodities)
            print(used_transport_means)
            print(used_nodes)
            print(distances)
            print(costs)

    else:
        min_value_3 = math.inf

    shipping_commodity = data['commodities']['commodity_objects']['Ammonia']
    min_value_4, used_commodities, used_transport_means, used_nodes, distances, costs \
        = find_shipping_benchmark_solution(data, configuration, complete_infrastructure, shipping_commodity)

    if print_benchmark_info:
        print(min_value_4)
        print(used_commodities)
        print(used_transport_means)
        print(used_nodes)
        print(distances)
        print(costs)

    shipping_commodity = data['commodities']['commodity_objects']['Methanol']
    min_value_5, used_commodities, used_transport_means, used_nodes, distances, costs \
        = find_shipping_benchmark_solution(data, configuration, complete_infrastructure, shipping_commodity)

    if print_benchmark_info:
        print(min_value_5)
        print(used_commodities)
        print(used_transport_means)
        print(used_nodes)
        print(distances)
        print(costs)

    shipping_commodity = data['commodities']['commodity_objects']['DBT']
    min_value_6, used_commodities, used_transport_means, used_nodes, distances, costs \
        = find_shipping_benchmark_solution(data, configuration, complete_infrastructure, shipping_commodity)

    if print_benchmark_info:
        print(min_value_6)
        print(used_commodities)
        print(used_transport_means)
        print(used_nodes)
        print(distances)
        print(costs)

    benchmark = math.ceil(min(min_value_1, min_value_2, min_value_3, min_value_4, min_value_5, min_value_6))

    return benchmark
