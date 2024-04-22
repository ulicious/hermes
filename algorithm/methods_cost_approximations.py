import math

import pandas as pd


def calculate_cheapest_option_to_final_destination(data, branches, benchmark, cost_column_name):

    """
    The method iterates over all possible combinations of
    conversion - transportation based on direct distance to final destination - conversion
    to calculate the lowest possible cost to the final destination

    @param data: dictionary with general data
    @param branches: current branches which includes current commodity and location
    @param benchmark: current benchmark
    @param cost_column_name: name of the total cost column in options
    @return: options with 'costs to final destination' column representing minimal total costs possible
    """

    means_of_transport = data['transport_means']
    final_commodities = data['commodities']['final_commodities']

    # add information if conversion at node or closest infrastructure is possible
    all_locations = data['conversion_costs_and_efficiencies']

    branches['current_node_conversion'] = True
    no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    no_conversion_possible_branches = branches[branches['current_node'].isin(no_conversion_possible_locations)].index
    branches.loc[no_conversion_possible_branches, 'current_node_conversion'] = False

    columns = ['current_commodity', cost_column_name, 'distance_to_final_destination',
               'current_transport_mean', 'current_node', 'current_node_conversion']
    cheapest_options = pd.DataFrame(branches[columns], columns=columns)

    cheapest_options.index = range(len(branches.index))

    considered_commodities = cheapest_options['current_commodity'].unique()

    # approach uses 'continue'. Therefore, it might be possible that no columns are generated for some options
    # because they are too expensive. These will use basic costs and will be remove due to infinite costs
    created_columns = ['basic_costs']
    cheapest_options['basic_costs'] = math.inf

    for c_start in considered_commodities:

        c_start_df = cheapest_options[cheapest_options['current_commodity'] == c_start]

        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()

        first_location_conversion_possible = c_start_df[c_start_df['current_node_conversion']].index
        first_location_conversion_not_possible = c_start_df[~c_start_df['current_node_conversion']].index

        # c_start is converted into c_transported
        for c_transported in [*data['commodities']['commodity_objects'].keys()]:
            if c_start != c_transported:
                # calculate conversion costs from c_start to c_transported
                if c_start_conversion_options[c_transported]:

                    # get conversion costs and efficiency of locations where conversion is possible
                    conversion_costs_first \
                        = c_start_object.get_conversion_costs_specific_commodity(c_start_df.loc[first_location_conversion_possible, 'current_node'],
                                                                                 c_transported)
                    conversion_costs_first.index = first_location_conversion_possible

                    conversion_efficiency_first \
                        = c_start_object.get_conversion_efficiency_specific_commodity(c_start_df.loc[first_location_conversion_possible, 'current_node'],
                                                                                      c_transported)
                    conversion_efficiency_first.index = first_location_conversion_possible

                    c_start_df.loc[first_location_conversion_possible, c_transported + '_conversion_costs'] = \
                        (c_start_df[cost_column_name] + conversion_costs_first) / conversion_efficiency_first

                    c_start_df.loc[first_location_conversion_not_possible, c_transported + '_conversion_costs'] = math.inf

                else:
                    continue
            else:
                # also no conversion is possible and c_start = c_transported is transported
                c_start_df[c_transported + '_conversion_costs'] = c_start_df[cost_column_name]

            if c_start_df[c_transported + '_conversion_costs'].min() > benchmark:
                # if all conversion costs are already higher than benchmark no further calculations will be made
                # as benchmark is already violated
                continue

            c_transported_object = data['commodities']['commodity_objects'][c_transported]
            transportation_options = c_transported_object.get_transportation_options()
            c_transported_conversion_options = c_transported_object.get_conversion_options()

            for m in means_of_transport:
                if not transportation_options[m]:
                    # commodity not transportable via this option
                    continue

                else:
                    c_start_df[c_transported + '_transportation_costs_' + m] \
                        = c_transported_object.get_transportation_costs_specific_mean_of_transport(m) / 1000 \
                        * c_start_df['distance_to_final_destination']

                    # shipping is only applicable once. Therefore, shipping costs are set to infinity for all
                    # options which have used shipping before (see below)
                    if m == 'Shipping':
                        options_m = c_start_df[c_start_df['current_transport_mean'] == m].index
                        c_start_df.loc[options_m, c_transported + '_transportation_costs_' + m] = math.inf

                    # after transportation, conversion at destination to final commodity if necessary
                    for c_end in [*data['commodities']['commodity_objects'].keys()]:

                        name_column = 'costs_' + c_start + '_' + c_transported + '_' + m + '_' + c_end

                        if c_end in final_commodities:
                            if c_transported != c_end:
                                if c_transported_conversion_options[c_end]:

                                    conversion_costs_at_destination \
                                        = c_transported_object.get_conversion_costs_specific_commodity('Destination', c_end)

                                    conversion_efficiency_at_destination \
                                        = c_transported_object.get_conversion_efficiency_specific_commodity('Destination', c_end)

                                    cheapest_options[name_column] = \
                                        (c_start_df[c_transported + '_conversion_costs']
                                         + c_start_df[c_transported + '_transportation_costs_' + m]
                                         + conversion_costs_at_destination) / conversion_efficiency_at_destination

                                    created_columns.append(name_column)
                                else:
                                    continue
                            else:
                                cheapest_options.loc[c_start_df.index, name_column] \
                                    = c_start_df[c_transported + '_conversion_costs'] \
                                    + c_start_df[c_transported + '_transportation_costs_' + m]
                                created_columns.append(name_column)
                        else:
                            continue

    return cheapest_options[created_columns].min(axis=1).tolist()


def calculate_cheapest_option_to_closest_infrastructure(data, branches, configuration, benchmark, cost_column_name):

    """
    This method calculates the lowest transportation costs from different locations to their closest infrastructure
    conversion - transportation to the closest infrastructure via road or new infrastructure - conversion

    This method is applied after the in-tolerance-transportation took place. That means that next transportation
    will either be road or new pipeline. We can calculate the minimal costs to the next infrastructure by using
    the distance to the closest infrastructure of each location. As road and new pipelines are quite expensive
    transport means, we might exceed the benchmark quite fast and are able to remove several branches because if the
    transport to the closest infrastructure is already more expensive than the benchmark, so will be the transport
    to all other infrastructure nodes

    @param data: dictionary with general data
    @param branches: current branches which include current commodity and location
    @param configuration: dictionary with assumptions and settings
    @param benchmark: current benchmark for assessment
    @param cost_column_name: column of options where information on current costs is saved
    @return: DataFrame with 'costs to final destination' column for each branch
    """

    final_commodities = data['commodities']['final_commodities']
    max_length_new_segment = configuration['max_length_new_segment']
    max_length_road = configuration['max_length_road']
    no_road_multiplier = configuration['no_road_multiplier']

    # load minimal distances and add Destination with 0
    minimal_distances = data['minimal_distances']
    minimal_distances.loc['Destination', 'minimal_distances'] = 0

    # minimal distances are used to analyze if road transportation is necessary. Without minimal distances
    # we would have to assume that pipelines are possible which are quite cheap and almost no branches would be
    # removed.

    # add minimal distance and closest node
    branches['minimal_distance'] = minimal_distances.loc[branches['current_node'].tolist(), 'minimal_distance'].tolist()
    branches['closest_node'] = minimal_distances.loc[branches['current_node'].tolist(), 'closest_node'].tolist()

    # add information if conversion at current node or closest node is possible
    all_locations = data['conversion_costs_and_efficiencies']

    branches['current_node_conversion'] = True
    no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    no_conversion_possible_branches = branches[branches['current_node'].isin(no_conversion_possible_locations)].index
    branches.loc[no_conversion_possible_branches, 'current_node_conversion'] = False

    branches['closest_node_conversion'] = True
    no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    no_conversion_possible_branches = branches[branches['closest_node'].isin(no_conversion_possible_locations)].index
    branches.loc[no_conversion_possible_branches, 'closest_node_conversion'] = False

    # only keep necessary columns
    columns = ['current_commodity', cost_column_name, 'distance_to_final_destination', 'minimal_distance',
               'current_transport_mean', 'current_node', 'closest_node', 'current_node_conversion', 'closest_node_conversion']
    cheapest_options = pd.DataFrame(branches[columns], columns=columns)

    cheapest_options.index = range(len(branches.index))

    considered_commodities = cheapest_options['current_commodity'].unique()

    # if minimal distance is below tolerance distance, no further costs occur
    in_tolerance = cheapest_options[cheapest_options['minimal_distance']
                                    <= configuration['tolerance_distance']].index
    cheapest_options.loc[in_tolerance, 'minimal_distance'] = 0

    # approach uses 'continue'. Therefore, it might be possible that no columns are generated for some options
    # because they are too expensive. These will use basic costs and will be remove due to infinity costs
    created_columns = ['basic_costs']
    cheapest_options['basic_costs'] = math.inf

    for c_start in considered_commodities:

        c_start_df = cheapest_options[cheapest_options['current_commodity'] == c_start]
        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()

        # we need to assess which of the closest infrastructures can be reached
        possible = c_start_df[c_start_df['minimal_distance'] <= ((max_length_road + max_length_new_segment) / no_road_multiplier)].index
        possible_only_road = c_start_df[c_start_df['minimal_distance'] <= (max_length_road / no_road_multiplier)].index
        possible_only_new = c_start_df[c_start_df['minimal_distance'] <= (max_length_new_segment / no_road_multiplier)].index

        first_location_conversion_possible = c_start_df[c_start_df['current_node_conversion']].index
        first_location_conversion_not_possible = c_start_df[~c_start_df['current_node_conversion']].index

        second_location_conversion_possible = c_start_df[c_start_df['closest_node_conversion']].index
        second_location_conversion_not_possible = c_start_df[~c_start_df['closest_node_conversion']].index

        # c_start is converted into c_transported
        for c_transported in [*data['commodities']['commodity_objects'].keys()]:

            if c_start != c_transported:
                # calculate conversion costs from c_start to c_transported
                if c_start_conversion_options[c_transported]:

                    # get conversion costs and efficiency of locations where conversion is possible
                    conversion_costs_first \
                        = c_start_object.get_conversion_costs_specific_commodity(c_start_df.loc[first_location_conversion_possible, 'current_node'],
                                                                                 c_transported)
                    conversion_costs_first.index = first_location_conversion_possible

                    conversion_efficiency_first \
                        = c_start_object.get_conversion_efficiency_specific_commodity(c_start_df.loc[first_location_conversion_possible, 'current_node'],
                                                                                      c_transported)
                    conversion_efficiency_first.index = first_location_conversion_possible

                    # calculate conversion costs for locations where conversion is possible
                    c_start_df.loc[first_location_conversion_possible, c_transported + '_conversion_costs'] = \
                        (c_start_df.loc[first_location_conversion_possible, cost_column_name] + conversion_costs_first) / conversion_efficiency_first

                    # calculate conversion costs for locations where conversion is impossible
                    c_start_df.loc[first_location_conversion_not_possible, c_transported + '_conversion_costs'] = math.inf
                else:
                    continue
            else:
                # also no conversion is possible and c_start = c_transported is transported
                c_start_df[c_transported + '_conversion_costs'] = c_start_df[cost_column_name]

            if c_start_df[c_transported + '_conversion_costs'].min() > benchmark:
                # if all conversion costs are already higher than benchmark no further calculations will be made
                # as benchmark is already violated
                continue

            c_transported_object = data['commodities']['commodity_objects'][c_transported]
            transportation_options = c_transported_object.get_transportation_options()
            c_transported_conversion_options = c_transported_object.get_conversion_options()
            c_transported_transportation_costs = c_transported_object.get_transportation_costs()

            for m in ['New_Pipeline_Gas', 'New_Pipeline_Liquid', 'Road']:

                c_start_df[c_transported + '_transportation_costs_' + m] = math.inf

                if m != 'Road':
                    if transportation_options[m]:
                        # new possible
                        c_start_df['new_distance'] = c_start_df['minimal_distance'].apply(lambda x: min(max_length_new_segment / no_road_multiplier, x))

                        if transportation_options['Road']:
                            # road and new possible --> cover as much with new, rest with road
                            c_start_df.loc[possible, 'road_distance'] \
                                = (c_start_df.loc[possible, 'minimal_distance'] - c_start_df.loc[possible, 'new_distance']).apply(lambda x: min(max_length_road / no_road_multiplier, x))

                            c_start_df.loc[possible, c_transported + '_transportation_costs_' + m] \
                                = c_transported_transportation_costs['Road'] / 1000 * c_start_df.loc[possible, 'road_distance'] \
                                + c_transported_transportation_costs[m] / 1000 * c_start_df.loc[possible, 'new_distance'] * no_road_multiplier

                        else:
                            # no road possible, only new
                            c_start_df.loc[possible_only_new, c_transported + '_transportation_costs_' + m] \
                                = c_transported_transportation_costs[m] / 1000 * c_start_df.loc[possible_only_new, 'new_distance'] * no_road_multiplier

                    else:
                        # new not possible
                        if transportation_options['Road']:
                            # no new possible, only road --> use only road

                            c_start_df.loc[possible_only_road, 'road_distance'] = c_start_df.loc[possible_only_road, 'minimal_distance']
                            c_start_df.loc[possible_only_road, c_transported + '_transportation_costs_' + m] \
                                = c_start_df.loc[possible_only_road, 'road_distance'] / 1000 * c_transported_transportation_costs['Road'] * no_road_multiplier
                        else:
                            # not possible to transport at all
                            continue

                else:
                    if transportation_options['Road']:
                        c_start_df.loc[possible_only_road, 'road_distance'] = c_start_df.loc[possible_only_road, 'minimal_distance']
                        c_start_df.loc[possible_only_road, c_transported + '_transportation_costs_' + m] \
                            = c_start_df.loc[possible_only_road, 'road_distance'] / 1000 * c_transported_transportation_costs['Road'] * no_road_multiplier
                    else:
                        # not possible to transport at all
                        continue

                # on top to the transportation costs, we can calculate the minimal conversion costs to one of the
                # final commodity at the closest node
                for c_end in [*data['commodities']['commodity_objects'].keys()]:

                    name_column = 'costs_' + c_start + '_' + c_transported + '_' + m + '_' + c_end

                    if c_end in final_commodities:
                        if c_transported != c_end:
                            if c_transported_conversion_options[c_end]:

                                # get conversion costs and efficiency of locations where conversion is possible
                                conversion_costs_second \
                                    = c_transported_object.get_conversion_costs_specific_commodity(c_start_df.loc[second_location_conversion_possible, 'closest_node'],
                                                                                                   c_end)
                                conversion_costs_second.index = second_location_conversion_possible

                                conversion_efficiency_second \
                                    = c_transported_object.get_conversion_efficiency_specific_commodity(c_start_df.loc[second_location_conversion_possible, 'closest_node'],
                                                                                                        c_end)
                                conversion_efficiency_second.index = second_location_conversion_possible

                                # calculate conversion costs for locations where conversion is possible
                                cheapest_options.loc[second_location_conversion_possible, name_column] = \
                                    (c_start_df.loc[second_location_conversion_possible, c_transported + '_conversion_costs']
                                     + c_start_df.loc[second_location_conversion_possible, c_transported + '_transportation_costs_' + m]
                                     + conversion_costs_second) / conversion_efficiency_second

                                # if no conversion at location possible, we use minimal conversion costs and efficiency
                                min_conversion_costs = c_transported_object.get_minimal_conversion_costs(c_end)
                                min_conversion_efficiency = c_transported_object.get_minimal_conversion_efficiency(c_end)

                                cheapest_options.loc[second_location_conversion_not_possible, name_column] = \
                                    (c_start_df.loc[second_location_conversion_not_possible, c_transported + '_conversion_costs']
                                     + c_start_df.loc[second_location_conversion_not_possible, c_transported + '_transportation_costs_' + m]
                                     + min_conversion_costs) / min_conversion_efficiency

                                created_columns.append(name_column)
                            else:
                                continue
                        else:
                            cheapest_options.loc[c_start_df.index, name_column] \
                                = c_start_df[c_transported + '_conversion_costs'] \
                                + c_start_df[c_transported + '_transportation_costs_' + m]
                            created_columns.append(name_column)
                    else:
                        continue

    return cheapest_options[created_columns].min(axis=1).tolist()


def calculate_minimal_costs_conversion_for_oil_and_gas_infrastructure(data, branches, cost_column_name):

    """
    This method calculates the minimal conversion costs to commodities which are applicable for pipeline transportation.
    If minimal total costs (total costs before conversion + minimal conversion costs) are higher than benchmark,
    pipeline will not be used for branch as costs would exceed benchmark as soon as branch is at pipeline
    and conversion is done

    @param data: dictionary with general data
    @param branches: current branch which includes current commodity and location
    @param cost_column_name: name of column showing costs
    @return: dataframe with 'costs to final destination' column
    """

    # add information if conversion at node or closest infrastructure is possible
    all_locations = data['conversion_costs_and_efficiencies']

    branches['current_node_conversion'] = True
    no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    no_conversion_possible_branches = branches[branches['current_node'].isin(no_conversion_possible_locations)].index
    branches.loc[no_conversion_possible_branches, 'current_node_conversion'] = False

    # only keep necessary columns
    columns = ['current_commodity', cost_column_name, 'distance_to_final_destination',
               'current_transport_mean', 'current_node', 'current_node_conversion']
    cheapest_options = pd.DataFrame(branches[columns], columns=columns)

    cheapest_options.index = range(len(branches.index))

    considered_commodities = cheapest_options['current_commodity'].unique()

    # approach uses 'continue'. Therefore, it might be possible that no columns are generated for some options
    # because they are too expensive. These will use basic costs and will be remove due to infinity costs
    created_columns = ['basic_costs_Pipeline_Gas', 'basic_costs_Pipeline_Liquid']
    cheapest_options[created_columns] = math.inf

    for c_start in considered_commodities:
        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()
        c_start_df = cheapest_options[cheapest_options['current_commodity'] == c_start]

        location_conversion_possible = c_start_df[c_start_df['current_node_conversion']].index
        location_conversion_not_possible = c_start_df[~c_start_df['current_node_conversion']].index

        # if commodity is already transportable via pipelines, no additional conversion
        c_start_transportation_options = c_start_object.get_transportation_options()
        for m in ['Pipeline_Gas', 'Pipeline_Liquid']:
            if c_start_transportation_options[m]:
                cheapest_options.loc[c_start_df.index, c_start + '_' + m] = c_start_df[cost_column_name]

        for c_conversion in [*data['commodities']['commodity_objects'].keys()]:
            if c_start != c_conversion:
                if c_start_conversion_options[c_conversion]:

                    conversion_costs \
                        = c_start_object.get_conversion_costs_specific_commodity(c_start_df.loc[location_conversion_possible, 'current_node'], c_conversion)
                    conversion_costs.index = location_conversion_possible

                    conversion_efficiency \
                        = c_start_object.get_conversion_efficiency_specific_commodity(c_start_df.loc[location_conversion_possible, 'current_node'], c_conversion)
                    conversion_efficiency.index = location_conversion_possible

                    c_conversion_object = data['commodities']['commodity_objects'][c_conversion]
                    transportation_options = c_conversion_object.get_transportation_options()

                    for m in ['Pipeline_Gas', 'Pipeline_Liquid']:
                        if transportation_options[m]:

                            # calculate costs for branches with possible conversion locations
                            cheapest_options.loc[location_conversion_possible, c_conversion + '_' + m] = \
                                (c_start_df.loc[location_conversion_possible, cost_column_name] + conversion_costs) / conversion_efficiency

                            # set not possible conversion branches to infinity
                            cheapest_options.loc[location_conversion_not_possible, c_conversion + '_' + m] = math.inf

                            created_columns.append(c_conversion + '_' + m)

    pipeline_gas_columns = [c for c in cheapest_options.columns if 'Pipeline_Gas' in c]
    pipeline_gas_cheapest_options = cheapest_options[pipeline_gas_columns].min(axis=1).tolist()

    pipeline_liquid_columns = [c for c in cheapest_options.columns if 'Pipeline_Liquid' in c]
    pipeline_liquid_cheapest_options = cheapest_options[pipeline_liquid_columns].min(axis=1).tolist()

    return pipeline_gas_cheapest_options, pipeline_liquid_cheapest_options
