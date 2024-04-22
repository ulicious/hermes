
def find_road_benchmark_solution(s, data, final_commodity, configuration):

    """


    @param s:
    @param data:
    @param final_commodity:
    @param configuration:
    @return:
    """

    s_new = deepcopy(s)
    s_location = s_new.get_current_location()
    s_destination = s_new.get_destination()
    s_commodity = s_new.get_current_commodity_object()

    direct_distance = calc_distance_single_to_single(s_location.y, s_location.x,
                                                     s_destination.y, s_destination.x)

    if direct_distance <= configuration['max_length_road'] / configuration['no_road_multiplier']:

        s_new = create_new_solution_from_routing_result(s_new, 1, s_commodity, 'Road',
                                                        s_destination,
                                                        direct_distance * configuration['no_road_multiplier'],
                                                        used_node=None,
                                                        iteration=2,
                                                        used_infrastructure=None)
    else:
        return None

    if s_commodity.get_name() not in final_commodity:
        commodities = data['commodities']['commodity_objects']

        cheapest_conversion = math.inf
        cheapest_conversion_commodity = None

        s_new_costs = s_new.get_total_costs()

        for c in [*commodities.keys()]:
            if s_commodity.get_conversion_options_specific_commodity(c):
                if c in final_commodity:

                    conversion_costs = s_commodity.get_conversion_costs_specific_commodity('Destination', c)
                    conversion_efficiency = s_commodity.get_conversion_efficiency_specific_commodity(c)

                    conversion_costs = (s_new_costs + conversion_costs) / conversion_efficiency

                    if conversion_costs < cheapest_conversion:
                        cheapest_conversion = conversion_costs
                        cheapest_conversion_commodity = commodities[c]

        if cheapest_conversion_commodity is None:  # todo: normally conversion should be possible
            return None

        s_new = create_new_solution_from_conversion_result(s_new, 3, cheapest_conversion_commodity, 4)

    return s_new
