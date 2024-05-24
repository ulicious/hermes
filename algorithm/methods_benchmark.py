import math

import pandas as pd

from algorithm.methods_geographic import calc_distance_list_to_single, calc_distance_list_to_list, check_if_reachable_on_land


def check_if_benchmark_possible(data, configuration, complete_infrastructure):

    """
    Based in given starting location and configurations, this method assesses if a benchmark can be calculated.
    For example, if now land connection exists between the starting location and any infrastructure or distances
    are longer than set in configuration --> infrastructure = False

    @param dict data: dictionary with common data
    @param dict configuration: dictionary with configuration
    @param pandas.DataFrame complete_infrastructure: all infrastructure (ports, pipelines, destination)
    @return: complete_infrastructure but with information if reachable from start or destination
    """

    # to get a first upper cost limit, several approaches are used to calculate a valid solution
    # possible approaches are:
    # 1: transportation via road to port, shipping, and transportation to destination
    # 2: transportation to pipeline, transportation in pipeline, and transportation to destination
    # 3: road transportation all the way
    starting_location = data['start']['location']
    destination_location = data['destination']['location']

    max_length_road = configuration['max_length_road']
    max_length_new_segment = configuration['max_length_new_segment']
    no_road_multiplier = configuration['no_road_multiplier']

    to_destination_tolerance = configuration['to_final_destination_tolerance']

    coastlines = data['coastlines']

    # check if solution is possible
    # first, check if based on configuration infrastructure is reachable from start and destination
    complete_infrastructure['distance_to_start'] \
        = calc_distance_list_to_single(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
                                       starting_location.y, starting_location.x)
    complete_infrastructure['distance_to_destination'] \
        = calc_distance_list_to_single(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
                                       destination_location.y, destination_location.x)
    index_in_tolerance = complete_infrastructure[complete_infrastructure['distance_to_destination'] <= to_destination_tolerance].index
    complete_infrastructure.loc[index_in_tolerance, 'distance_to_destination'] = 0

    max_length = max(max_length_road, max_length_new_segment) / no_road_multiplier

    complete_infrastructure.sort_values(['distance_to_destination'], inplace=True)
    distance_to_destination = complete_infrastructure[complete_infrastructure['distance_to_destination'] <= max_length].index.tolist()
    distance_to_destination.remove('Destination')

    # we want at least 10 harbours within these options where we calculate the reachability
    complete_infrastructure.sort_values(['distance_to_start'], inplace=True)

    first_ten_harbours = []
    for n, i in enumerate(complete_infrastructure.index):
        if 'H' in i:
            first_ten_harbours.append(i)

            if len(first_ten_harbours) == 10:
                break

    first_index = ['Destination'] + first_ten_harbours
    new_index = first_index + [i for i in complete_infrastructure.index if i not in first_index]
    complete_infrastructure = complete_infrastructure.loc[new_index, :]

    distance_to_start = complete_infrastructure[complete_infrastructure['distance_to_start'] <= max_length].index

    complete_infrastructure['reachable_from_start'] = False
    complete_infrastructure['reachable_from_destination'] = False

    infrastructure_available = False
    if (len(distance_to_start) > 0) & (len(distance_to_destination) > 0):
        # true if from start and from destination infrastructure exists within set parameters
        # for road or new infrastructure
        infrastructure_available = True

    # just because infrastructure exists, it does not mean that it can be used (e.g. not reachable via transportation
    # due to obstacles like water)
    if infrastructure_available:
        complete_infrastructure['longitude_on_coastline'] = complete_infrastructure['longitude_on_coastline'].fillna(complete_infrastructure['longitude'])
        complete_infrastructure['latitude_on_coastline'] = complete_infrastructure['latitude_on_coastline'].fillna(complete_infrastructure['latitude'])

        reachable_from_start = check_if_reachable_on_land(starting_location,
                                                          complete_infrastructure.loc[distance_to_start[:1000], 'longitude_on_coastline'],
                                                          complete_infrastructure.loc[distance_to_start[:1000], 'latitude_on_coastline'],
                                                          coastlines,
                                                          get_only_availability=True)

        reachable_from_destination = check_if_reachable_on_land(destination_location,
                                                                complete_infrastructure.loc[distance_to_destination[:1000], 'longitude_on_coastline'],
                                                                complete_infrastructure.loc[distance_to_destination[:1000], 'latitude_on_coastline'],
                                                                coastlines,
                                                                get_only_availability=True)

        complete_infrastructure.loc[distance_to_start[:1000], 'reachable_from_start'] = reachable_from_start
        complete_infrastructure.loc[distance_to_destination[:1000], 'reachable_from_destination'] = reachable_from_destination

    return complete_infrastructure


def find_shipping_benchmark_solution(data, configuration, all_options, shipping_commodity):

    """
    Finds a valid shipping solution for a given commodity based on various parameters and configurations.
    Start -> Road / New Pipeline -> Ship -> Road / New Pipeline -> Destination

    Parameters:
    @param data: (dict) A dictionary containing data related to commodities and their transportation.
    @param configuration: (dict) A dictionary containing configuration parameters for the shipping process.
    @param all_options: (DataFrame) DataFrame containing all available shipping options.
    @param shipping_commodity: (object) An object representing the commodity to be shipped.

    @return:
    Tuple: A tuple containing the following elements:
        1. min_value (float): The minimum cost value for shipping the commodity.
        2. used_commodities (list): A list of used commodities in the shipping process.
        3. used_transport_means (list): A list of transportation means used in the shipping process.
        4. used_nodes (list): A list of nodes (locations) visited in the shipping process.
        5. travelled_distances (list): A list of distances travelled in the shipping process.
        6. costs (list): A list of costs associated with the shipping process.
    """

    used_commodities = [shipping_commodity.get_name()]
    used_nodes = ['Start']
    used_transport_means = []
    travelled_distances = []
    costs = [shipping_commodity.get_production_costs()]
    min_value = shipping_commodity.get_production_costs()

    in_tolerance_distance_option = configuration['tolerance_distance']
    in_tolerance_distance_destination = configuration['to_final_destination_tolerance']

    max_length_road = configuration['max_length_road']
    max_length_new_segment = configuration['max_length_new_segment']
    no_road_multiplier = configuration['no_road_multiplier']

    final_commodities = data['commodities']['final_commodities']

    shipping_options = all_options[all_options['reachable_from_start'] | all_options['reachable_from_destination']].copy()
    port_index = [i for i in shipping_options.index if 'H' in i]
    shipping_options = shipping_options.loc[port_index, :]

    shipping_distances = pd.read_csv(configuration['path_processed_data'] + 'inner_infrastructure_distances/port_distances.csv', index_col=0)

    if shipping_options.empty:
        return None

    transportation_options = shipping_commodity.get_transportation_options()
    transportation_costs = shipping_commodity.get_transportation_costs()

    conversion_options = shipping_commodity.get_conversion_options()
    conversion_costs = shipping_commodity.get_conversion_costs()
    conversion_losses = shipping_commodity.get_conversion_efficiencies()

    # calculate shipping_costs
    viable_start_ports = shipping_options[shipping_options['reachable_from_start']].index
    viable_end_ports = shipping_options[shipping_options['reachable_from_destination']].index
    all_viable_ports = list(set(viable_start_ports.tolist() + viable_end_ports.tolist()))

    # overwrite distances if they are within tolerances
    in_tolerance = shipping_options[shipping_options['distance_to_start'] <= in_tolerance_distance_option].index
    shipping_options.loc[in_tolerance, 'distance_to_start'] = 0

    in_tolerance = shipping_options[shipping_options['distance_to_destination'] <= in_tolerance_distance_destination].index
    shipping_options.loc[in_tolerance, 'distance_to_destination'] = 0

    transportation_costs_shipping = transportation_costs['Shipping']
    shipping_costs = shipping_distances.loc[all_viable_ports, all_viable_ports] / 1000 * transportation_costs_shipping

    # calculate transportation costs of first land transport
    shipping_options['road_transportation_costs_from_start'] = math.inf
    if transportation_options['Road']:

        valid_options \
            = shipping_options[shipping_options['reachable_from_start']
                               & (shipping_options['distance_to_start'] <= max_length_road / no_road_multiplier)].index

        shipping_options.loc[valid_options, 'road_transportation_costs_from_start'] \
            = shipping_options.loc[valid_options, 'distance_to_start'] / 1000 * transportation_costs['Road'] * no_road_multiplier

    shipping_options['pipeline_gas_transportation_costs_from_start'] = math.inf
    if transportation_options['Pipeline_Gas']:

        valid_options = \
            shipping_options[shipping_options['reachable_from_start'] &
                             (shipping_options['distance_to_start'] <= max_length_new_segment / no_road_multiplier)].index

        shipping_options.loc[valid_options, 'pipeline_gas_transportation_costs_from_start'] \
            = shipping_options.loc[valid_options, 'distance_to_start'] / 1000 * transportation_costs['New_Pipeline_Gas'] * no_road_multiplier

    shipping_options['pipeline_liquid_transportation_costs_from_start'] = math.inf
    if transportation_options['Pipeline_Liquid']:

        valid_options \
            = shipping_options[shipping_options['reachable_from_start'] &
                               (shipping_options['distance_to_start'] <= max_length_new_segment / no_road_multiplier)].index

        shipping_options.loc[valid_options, 'pipeline_liquid_transportation_costs_from_start'] \
            = shipping_options.loc[valid_options, 'distance_to_start'] / 1000 * transportation_costs['New_Pipeline_Liquid'] * no_road_multiplier

    # calculate costs of land transport from second port to destination
    shipping_options['road_transportation_costs_to_destination'] = math.inf
    if transportation_options['Road']:
        valid_options \
            = shipping_options[shipping_options['reachable_from_destination']
                               & (shipping_options['distance_to_destination'] <= max_length_road / no_road_multiplier)].index

        shipping_options.loc[valid_options, 'road_transportation_costs_to_destination'] \
            = shipping_options.loc[valid_options, 'distance_to_destination'] / 1000 * transportation_costs['Road'] * no_road_multiplier

    shipping_options['pipeline_gas_transportation_costs_to_destination'] = math.inf
    if transportation_options['Pipeline_Gas']:
        valid_options = \
            shipping_options[shipping_options['reachable_from_destination'] &
                             (shipping_options['distance_to_destination'] <= max_length_new_segment / no_road_multiplier)].index

        shipping_options.loc[valid_options, 'pipeline_gas_transportation_costs_to_destination'] \
            = shipping_options.loc[valid_options, 'distance_to_destination'] / 1000 * transportation_costs['New_Pipeline_Gas'] * no_road_multiplier

    shipping_options['pipeline_liquid_transportation_costs_to_destination'] = math.inf
    if transportation_options['Pipeline_Liquid']:
        valid_options \
            = shipping_options[shipping_options['reachable_from_destination'] &
                               (shipping_options['distance_to_destination'] <= max_length_new_segment / no_road_multiplier)].index

        shipping_options.loc[valid_options, 'pipeline_liquid_transportation_costs_to_destination'] \
            = shipping_options.loc[valid_options, 'distance_to_destination'] / 1000 * transportation_costs['New_Pipeline_Liquid'] * no_road_multiplier

    # calculate all combinations of first land, shipping and second land transport
    shipping_options = shipping_options.loc[shipping_costs.index, :]

    chosen_min_value = math.inf
    idx = None
    chosen_mot_1 = None
    chosen_mot_2 = None
    for mot_1 in ['road', 'pipeline_gas', 'pipeline_liquid']:
        for mot_2 in ['road', 'pipeline_gas', 'pipeline_liquid']:
            option_costs \
                = shipping_costs.add(shipping_options[mot_1 + '_transportation_costs_from_start']).transpose()\
                .add(shipping_options[mot_2 + '_transportation_costs_to_destination']).stack()

            option_min_value = option_costs.min()
            option_idx = option_costs.idxmin()

            if option_min_value < chosen_min_value:
                chosen_min_value = option_min_value
                idx = option_idx

                chosen_mot_1 = mot_1
                chosen_mot_2 = mot_2

    if idx is None:
        return math.inf, None, None, None, None, None

    min_value += chosen_min_value

    used_nodes.append(idx[0])
    used_nodes.append(idx[1])
    used_nodes.append('Destination')

    travelled_distances.append(shipping_options.at[idx[0], 'distance_to_start'])
    travelled_distances.append(shipping_distances.at[idx[0], idx[1]])
    travelled_distances.append(shipping_options.at[idx[1], 'distance_to_destination'])

    if chosen_mot_1 == 'road':
        used_transport_means.append('Road')
    elif chosen_mot_1 == 'pipeline_gas':
        used_transport_means.append('Pipeline_Gas')
    else:
        used_transport_means.append('Pipeline_Liquid')

    used_transport_means.append('Shipping')

    if chosen_mot_2 == 'road':
        used_transport_means.append('Road')
    elif chosen_mot_2 == 'pipeline_gas':
        used_transport_means.append('Pipeline_Gas')
    else:
        used_transport_means.append('Pipeline_Liquid')

    costs.append(shipping_options.at[idx[0], chosen_mot_1 + '_transportation_costs_from_start'])
    costs.append(shipping_costs.at[idx[0], idx[1]])
    costs.append(shipping_options.at[idx[1], chosen_mot_2 + '_transportation_costs_to_destination'])

    # check if shipped commodity is in final commodity. Use the cheapest conversion if not
    if shipping_commodity.get_name() not in final_commodities:
        commodities = data['commodities']['commodity_objects']

        cheapest_conversion = math.inf
        cheapest_conversion_commodity = None

        for c in final_commodities:
            if conversion_options[c]:

                conversion_costs_c = conversion_costs.at['Destination', c]
                conversion_efficiency = conversion_losses.at['Destination', c]

                conversion_costs_c = (min_value + conversion_costs_c) / conversion_efficiency

                if conversion_costs_c < cheapest_conversion:
                    cheapest_conversion = conversion_costs_c
                    cheapest_conversion_commodity = commodities[c].get_name()

        used_commodities.append(cheapest_conversion_commodity)
        costs.append(cheapest_conversion - min_value)
        min_value = cheapest_conversion

    return min_value, used_commodities, used_transport_means, used_nodes, travelled_distances, costs


def find_pipeline_shipping_solution(data, configuration, complete_infrastructure, pipeline_commodity, shipping_commodity):

    """
    Finds a valid pipeline + shipping solution for a given commodity based on various parameters and configurations.
    Start -> Road / New Pipeline -> Existing Pipeline -> Road / New Pipeline -> Ship -> Road / New Pipeline -> Destination

    @param data: dictionary with common data
    @param configuration: dictionary with configuration
    @param complete_infrastructure: dataframe containing all nodes and ports
    @param pipeline_commodity: commodity used for pipelines
    @param shipping_commodity: commodity used for shipping

    @return:
    A tuple containing the following elements:
        1. min_value (float): The minimum cost value for shipping the commodity.
        2. used_commodities (list): A list of used commodities in the shipping process.
        3. used_transport_means (list): A list of transportation means used in the shipping process.
        4. used_nodes (list): A list of nodes (locations) visited in the shipping process.
        5. travelled_distances (list): A list of distances travelled in the shipping process.
        6. costs (list): A list of costs associated with the shipping process.
    """

    path_data = configuration['path_processed_data']

    used_commodities = []
    used_nodes = ['Start']
    used_transport_means = []
    travelled_distances = []
    costs = []

    max_length_road = configuration['max_length_road']
    max_length_new_segment = configuration['max_length_new_segment']
    build_new_infrastructure = configuration['build_new_infrastructure']
    no_road_multiplier = configuration['no_road_multiplier']
    in_tolerance_distance_option = configuration['tolerance_distance']
    in_tolerance_distance_destination = configuration['to_final_destination_tolerance']

    final_commodities = data['commodities']['final_commodities']

    transportation_costs_liquid = shipping_commodity.get_transportation_costs()

    transportation_costs_gas = pipeline_commodity.get_transportation_costs()

    conversion_costs_liquid = shipping_commodity.get_conversion_costs()
    conversion_losses_liquid = shipping_commodity.get_conversion_efficiencies()

    conversion_costs_gas = pipeline_commodity.get_conversion_costs()
    conversion_losses_gas = pipeline_commodity.get_conversion_efficiencies()

    conversion_costs = data['conversion_costs_and_efficiencies']
    no_conversion_nodes = conversion_costs[~conversion_costs['conversion_possible']].index

    pipeline_options = complete_infrastructure[complete_infrastructure['reachable_from_start']].copy()
    pg_index = [i for i in pipeline_options.index if ('PG' in i) & (i not in no_conversion_nodes)]
    pipeline_options = pipeline_options.loc[pg_index, :]

    options_shipping = complete_infrastructure[complete_infrastructure['reachable_from_destination']].copy()
    port_index = [i for i in options_shipping.index if ('H' in i) & (i not in no_conversion_nodes)]
    options_shipping = options_shipping.loc[port_index, :]

    viable_start_nodes = pipeline_options[pipeline_options['reachable_from_start']].index

    distance_to_start = math.inf
    closest_node_to_start = None
    closest_graph_to_start = None
    pipeline_options = pipeline_options.loc[viable_start_nodes, :]
    for g in pipeline_options['graph'].unique():

        if g is None:
            continue

        g_options = pipeline_options[pipeline_options['graph'] == g]

        distance_closest_node_g_to_start = g_options['distance_to_start'].min()
        closest_node_to_start_g = g_options['distance_to_start'].idxmin()

        if distance_closest_node_g_to_start < distance_to_start:
            distance_to_start = distance_closest_node_g_to_start
            closest_graph_to_start = g
            closest_node_to_start = closest_node_to_start_g

    if closest_graph_to_start is None:
        return math.inf, None, None, None, None, None

    # get distance where transportation gas makes more sense than transportation liquid and conversion to gas
    distance_max_gas = 0
    if pipeline_commodity.get_transportation_options()['Road']:
        transportation_costs_gas_road = transportation_costs_gas['Road']
        transportation_costs_liquid_road = transportation_costs_liquid['Road']

        conversion_costs = conversion_costs_liquid.at[closest_node_to_start, pipeline_commodity.get_name()]
        conversion_losses = conversion_losses_liquid.at[closest_node_to_start, pipeline_commodity.get_name()]

        min_value_gas = pipeline_commodity.get_production_costs()
        min_value_liquid = shipping_commodity.get_production_costs()
        distance_max_gas = \
            (min_value_liquid + conversion_costs + min_value_gas * conversion_losses) \
            / (transportation_costs_gas_road * conversion_losses - transportation_costs_liquid_road)

    if distance_to_start <= in_tolerance_distance_option:
        # if distance to start is 0 because it is in tolerance to pipeline, use gas and not transportation costs
        distance_to_start = 0

        min_value = pipeline_commodity.get_production_costs()
        costs.append(min_value)
        used_commodities.append(pipeline_commodity.get_name())

    elif (distance_to_start * no_road_multiplier <= max_length_new_segment) & build_new_infrastructure:
        # if pipeline is within distance of new infrastructure, use new infrastructure
        min_value = pipeline_commodity.get_production_costs()
        costs.append(min_value)

        transportation_costs_option_gas \
            = distance_to_start * no_road_multiplier / 1000 * transportation_costs_gas['New_Pipeline_Gas']

        min_value += transportation_costs_option_gas
        costs.append(transportation_costs_option_gas)
        used_transport_means.append('New_Pipeline_Gas')
        used_commodities.append(pipeline_commodity.get_name())

    elif (distance_to_start * no_road_multiplier <= distance_max_gas) & pipeline_commodity.get_transportation_options()['Road']:
        # if distance to start is below the distance where it makes more sense to transport the gas commodity via road
        # then transporting in liquid form and converse it, use gas commodity and road
        min_value = pipeline_commodity.get_production_costs()
        costs.append(min_value)

        transportation_costs_option_gas \
            = distance_to_start * no_road_multiplier / 1000 * transportation_costs_gas['Road']

        min_value += transportation_costs_option_gas
        costs.append(transportation_costs_option_gas)
        used_transport_means.append('Road')
        used_commodities.append(pipeline_commodity.get_name())

    elif distance_to_start * no_road_multiplier > max(max_length_road, max_length_new_segment):
        # no first transportation possible
        return math.inf, None, None, None, None, None
    else:
        min_value = shipping_commodity.get_production_costs()
        costs.append(min_value)

        transportation_costs_option_liquid \
            = distance_to_start * no_road_multiplier / 1000 * transportation_costs_liquid['Road']

        min_value += transportation_costs_option_liquid
        costs.append(transportation_costs_option_liquid)
        used_transport_means.append('Road')
        used_commodities.append(shipping_commodity.get_name())

        # conversion to gas commodity for pipeline transportation
        min_value_before = min_value
        min_value = (min_value + conversion_costs_liquid.at[closest_node_to_start, pipeline_commodity.get_name()]) \
            / conversion_losses_liquid.at[closest_node_to_start, pipeline_commodity.get_name()]
        costs.append(min_value - min_value_before)
        used_commodities.append(pipeline_commodity.get_name())

    used_nodes.append(closest_node_to_start)
    travelled_distances.append(distance_to_start)

    # find ports
    geodata_start = data['Pipeline_Gas'][closest_graph_to_start]['NodeLocations'].copy()

    distance_to_port = calc_distance_list_to_list(geodata_start['latitude'], geodata_start['longitude'],
                                                  options_shipping['latitude'], options_shipping['longitude'])

    distance_to_port = pd.DataFrame(distance_to_port.transpose(), index=geodata_start.index,
                                    columns=options_shipping.index)

    distance_pipeline_to_port = distance_to_port.min().min()
    if distance_pipeline_to_port <= in_tolerance_distance_option:
        distance_pipeline_to_port = 0

    closest_node_first_to_second = distance_to_port.stack().idxmin()[0]
    closest_node_second_to_first = distance_to_port.stack().idxmin()[1]

    # second step: inner pipeline transportation
    infrastructure_distances \
        = pd.read_hdf(path_data + '/inner_infrastructure_distances/' + closest_node_to_start + '.h5', mode='r',
                      title=closest_node_to_start)
    infrastructure_distances = infrastructure_distances.transpose()
    distance = infrastructure_distances.at[closest_node_to_start, closest_node_first_to_second]

    min_value += distance / 1000 * transportation_costs_gas['Pipeline_Gas']
    costs.append(distance / 1000 * transportation_costs_gas['Pipeline_Gas'])
    travelled_distances.append(distance)
    used_transport_means.append('Pipeline_Gas')
    used_nodes.append(closest_node_first_to_second)

    # calculate distance between pipeline and ship
    if (distance_pipeline_to_port * no_road_multiplier <= max_length_new_segment) & build_new_infrastructure:

        # if new pipelines are possible, we can use the commodity as is (gas)
        transportation_costs_option_gas \
            = distance_pipeline_to_port * no_road_multiplier / 1000 * transportation_costs_gas['New_Pipeline_Gas']
        costs.append(transportation_costs_option_gas)

        min_value += transportation_costs_option_gas
        used_transport_means.append('New_Pipeline_Gas')

        # will be shipped afterwards therefore conversion necessary
        min_value_before = min_value
        min_value = (min_value + conversion_costs_gas.at[closest_node_second_to_first, shipping_commodity.get_name()]) \
            / conversion_losses_gas.at[closest_node_second_to_first, shipping_commodity.get_name()]
        costs.append(min_value - min_value_before)
        used_commodities.append(shipping_commodity.get_name())

    elif distance_pipeline_to_port * no_road_multiplier > max(max_length_road, max_length_new_segment):
        # no first transportation
        return math.inf, None, None, None, None, None
    else:

        # needs road transportation --> conversion to shipping commodity
        min_value_before = min_value
        min_value = (min_value + conversion_costs_gas.at[closest_node_second_to_first, shipping_commodity.get_name()]) \
            / conversion_losses_gas.at[closest_node_second_to_first, shipping_commodity.get_name()]
        costs.append(min_value - min_value_before)
        used_commodities.append(shipping_commodity.get_name())

        transportation_costs_option_liquid \
            = distance_pipeline_to_port * no_road_multiplier / 1000 * transportation_costs_liquid['Road']

        costs.append(transportation_costs_option_liquid)

        min_value += transportation_costs_option_liquid
        used_transport_means.append('Road')

    travelled_distances.append(distance_pipeline_to_port)
    used_nodes.append(closest_node_second_to_first)

    # now add shipping costs
    shipping_distances = pd.read_csv(configuration['path_processed_data'] + 'inner_infrastructure_distances/port_distances.csv', index_col=0)

    destination_port = options_shipping['distance_to_destination'].idxmin()
    to_destination_distance = options_shipping['distance_to_destination'].min()

    shipping_distance = shipping_distances.at[closest_node_second_to_first, destination_port]

    min_value += shipping_distance / 1000 * transportation_costs_liquid['Shipping']
    costs.append(shipping_distance / 1000 * transportation_costs_liquid['Shipping'])
    used_transport_means.append('Shipping')
    travelled_distances.append(shipping_distance)
    used_nodes.append(destination_port)

    if to_destination_distance <= in_tolerance_distance_destination:
        to_destination_distance = 0

    # calculate distance between ship and destination
    if (to_destination_distance * no_road_multiplier <= max_length_new_segment) & build_new_infrastructure:

        # was shipped with shipping commodity and needs to be conversed to pipeline commodity to be
        # transported via pipeline
        min_value_before = min_value
        min_value = (min_value + conversion_costs_liquid.at['Destination', pipeline_commodity.get_name()]) \
            / conversion_losses_liquid.at['Destination', pipeline_commodity.get_name()]
        costs.append(min_value - min_value_before)
        used_commodities.append(pipeline_commodity.get_name())

        transportation_costs_option_gas \
            = to_destination_distance * no_road_multiplier / 1000 * transportation_costs_gas['New_Pipeline_Gas']
        costs.append(transportation_costs_option_gas)

        min_value += transportation_costs_option_gas
        used_transport_means.append('New_Pipeline_Gas')

    elif to_destination_distance * no_road_multiplier > max(max_length_road, max_length_new_segment):
        # no first transportation
        return math.inf, None, None, None, None, None
    else:
        transportation_costs_option_liquid \
            = to_destination_distance * no_road_multiplier / 1000 * transportation_costs_liquid['Road']
        costs.append(transportation_costs_option_liquid)

        min_value += transportation_costs_option_liquid
        used_transport_means.append('Road')

    travelled_distances.append(to_destination_distance)
    used_nodes.append('Destination')

    if used_commodities[-1] not in final_commodities:
        cheapest_conversion = math.inf
        cheapest_conversion_commodity = None

        commodity_object = data['commodities']['commodity_objects'][used_commodities[-1]]

        min_value_before = min_value

        for c in final_commodities:
            if c in commodity_object.get_conversion_options():

                conversion_costs = commodity_object.get_conversion_costs_specific_commodity('Destination', c)
                conversion_efficiency = commodity_object.get_conversion_efficiency_specific_commodity('Destination', c)

                conversion_costs = (min_value + conversion_costs) / conversion_efficiency

                if conversion_costs < cheapest_conversion:
                    cheapest_conversion = conversion_costs
                    cheapest_conversion_commodity = c

        min_value = cheapest_conversion
        costs.append(min_value - min_value_before)
        used_commodities.append(cheapest_conversion_commodity)

    return min_value, used_commodities, used_transport_means, used_nodes, travelled_distances, costs


def find_pipeline_solution(data, configuration, complete_infrastructure, pipeline_commodity, road_commodity):

    """
    Finds a valid pipeline solution for a given commodity based on various parameters and configurations.
    Start -> Road / New Pipeline -> Existing Pipeline -> Road / New Pipeline -> Existing Pipeline -> Road / New Pipeline -> Destination

    Important: This method uses several conversion. Therefore, nodes at pipelines are removed if conversion
    at node is not possible

    @param data: dictionary with common data
    @param configuration: dictionary with configuration
    @param complete_infrastructure: dataframe containing all nodes and ports
    @param pipeline_commodity: commodity used for pipelines
    @param road_commodity: commodity used for road transport

    @return:
    A tuple containing the following elements:
        1. min_value (float): The minimum cost value for shipping the commodity.
        2. used_commodities (list): A list of used commodities in the shipping process.
        3. used_transport_means (list): A list of transportation means used in the shipping process.
        4. used_nodes (list): A list of nodes (locations) visited in the shipping process.
        5. travelled_distances (list): A list of distances travelled in the shipping process.
        6. costs (list): A list of costs associated with the shipping process.
    """

    path_data = configuration['path_processed_data']

    used_commodities = []
    used_nodes = ['Start']
    used_transport_means = []
    travelled_distances = []
    costs = []

    max_length_road = configuration['max_length_road']
    max_length_new_segment = configuration['max_length_new_segment']
    build_new_infrastructure = configuration['build_new_infrastructure']
    no_road_multiplier = configuration['no_road_multiplier']
    in_tolerance_distance = configuration['to_final_destination_tolerance']

    final_commodities = data['commodities']['final_commodities']

    conversion_costs = data['conversion_costs_and_efficiencies']
    no_conversion_nodes = conversion_costs[~conversion_costs['conversion_possible']].index

    transportation_costs_liquid = road_commodity.get_transportation_costs()

    transportation_costs_gas = pipeline_commodity.get_transportation_costs()

    conversion_costs_liquid = road_commodity.get_conversion_costs()
    conversion_losses_liquid = road_commodity.get_conversion_efficiencies()

    conversion_costs_gas = pipeline_commodity.get_conversion_costs()
    conversion_losses_gas = pipeline_commodity.get_conversion_efficiencies()

    pipeline_options_from_start = complete_infrastructure[complete_infrastructure['reachable_from_start']].copy()
    pg_index = [i for i in pipeline_options_from_start.index if ('PG' in i) & (i not in no_conversion_nodes)]
    pipeline_options_from_start = pipeline_options_from_start.loc[pg_index, :]

    pipeline_options_to_destination = complete_infrastructure[complete_infrastructure['reachable_from_destination']].copy()
    pg_index = [i for i in pipeline_options_to_destination.index if ('PG' in i) & (i not in no_conversion_nodes)]
    pipeline_options_to_destination = pipeline_options_to_destination.loc[pg_index, :]

    g_start_decided = None
    g_destination_decided = None
    node_start_g_start = None
    node_end_g_start = None

    node_start_g_destination = None
    node_end_g_destination = None

    distance_start_to_pipeline = math.inf
    distance_between_pipelines = math.inf
    distance_pipeline_to_destination = math.inf

    min_road_length = math.inf
    for g_start in pipeline_options_from_start['graph'].unique():

        if g_start is None:
            continue

        if g_start in pipeline_options_to_destination['graph'].unique():
            g_start_decided = g_start
            g_destination_decided = g_start

            g_options_start = pipeline_options_from_start[pipeline_options_from_start['graph'] == g_start]

            distance_closest_node_g_to_start = g_options_start['distance_to_start'].min()
            closest_node_to_start_g = g_options_start['distance_to_start'].idxmin()

            g_options_destination = pipeline_options_to_destination[pipeline_options_to_destination['graph'] == g_start]

            distance_closest_node_g_to_destination = g_options_destination['distance_to_destination'].min()
            closest_node_to_destination_g = g_options_destination['distance_to_destination'].idxmin()

            node_start_g_start = closest_node_to_start_g
            node_start_g_destination = closest_node_to_destination_g

            distance_start_to_pipeline = distance_closest_node_g_to_start
            distance_between_pipelines = 0
            distance_pipeline_to_destination = distance_closest_node_g_to_destination

            break

        else:
            g_options_start = pipeline_options_from_start[pipeline_options_from_start['graph'] == g_start]

            distance_closest_node_g_to_start = g_options_start['distance_to_start'].min()
            closest_node_to_start_g = g_options_start['distance_to_start'].idxmin()

            for g_destination in pipeline_options_to_destination['graph'].unique():

                if g_destination is None:
                    continue

                g_options_destination = pipeline_options_to_destination[pipeline_options_to_destination['graph'] == g_destination]

                distance_closest_node_g_to_destination = g_options_destination['distance_to_destination'].min()
                closest_node_to_destination_g = g_options_destination['distance_to_destination'].idxmin()

                distance_between_pipelines = calc_distance_list_to_list(g_options_start['latitude'],
                                                                        g_options_start['longitude'],
                                                                        g_options_destination['latitude'],
                                                                        g_options_destination['longitude'])

                distance_between_pipelines = pd.DataFrame(distance_between_pipelines.transpose(),
                                                          index=g_options_start.index,
                                                          columns=g_options_destination.index).stack()

                nodes_between_pipelines = distance_between_pipelines.idxmin()
                distance_between_pipelines = distance_between_pipelines.min()

                if distance_closest_node_g_to_start + distance_between_pipelines + distance_closest_node_g_to_destination < min_road_length:
                    min_road_length = distance_closest_node_g_to_start + distance_between_pipelines + distance_closest_node_g_to_destination

                    g_start_decided = g_start
                    g_destination_decided = g_destination

                    node_start_g_start = closest_node_to_start_g
                    node_end_g_start = nodes_between_pipelines[0]

                    node_start_g_destination = nodes_between_pipelines[1]
                    node_end_g_destination = closest_node_to_destination_g

                    distance_start_to_pipeline = distance_closest_node_g_to_start
                    distance_between_pipelines = distance_between_pipelines
                    distance_pipeline_to_destination = distance_closest_node_g_to_destination

    if (g_start_decided is None) | (g_destination_decided is None):
        return math.inf, None, None, None, None, None

    if distance_start_to_pipeline <= in_tolerance_distance:
        distance_start_to_pipeline = 0

    if (distance_start_to_pipeline * no_road_multiplier <= max_length_new_segment) & build_new_infrastructure:
        min_value = pipeline_commodity.get_production_costs()
        costs.append(min_value)

        transportation_costs_option_gas \
            = distance_start_to_pipeline * no_road_multiplier / 1000 * transportation_costs_gas['New_Pipeline_Gas']

        min_value += transportation_costs_option_gas
        costs.append(transportation_costs_option_gas)
        used_transport_means.append('New_Pipeline_Gas')
        used_commodities.append(pipeline_commodity.get_name())

    elif distance_start_to_pipeline * no_road_multiplier > max(max_length_road, max_length_new_segment):
        # no first transportation possible
        return math.inf, None, None, None, None, None
    else:
        min_value = road_commodity.get_production_costs()
        costs.append(min_value)

        transportation_costs_option_liquid \
            = distance_start_to_pipeline * no_road_multiplier / 1000 * transportation_costs_liquid['Road']

        min_value += transportation_costs_option_liquid
        costs.append(transportation_costs_option_liquid)
        used_transport_means.append('Road')
        used_commodities.append(road_commodity.get_name())

        # conversion to gas commodity for pipeline transportation
        min_value_before = min_value
        min_value = (min_value + conversion_costs_liquid.at[node_start_g_start, pipeline_commodity.get_name()]) \
            / conversion_losses_liquid.at[node_start_g_start, pipeline_commodity.get_name()]
        costs.append(min_value - min_value_before)
        used_commodities.append(pipeline_commodity.get_name())

    used_nodes.append(node_start_g_start)
    travelled_distances.append(distance_start_to_pipeline * no_road_multiplier)

    # second step: inner pipeline transportation
    if g_start_decided == g_destination_decided:
        infrastructure_distances \
            = pd.read_hdf(path_data + '/inner_infrastructure_distances/' + node_start_g_start + '.h5', mode='r',
                          title=node_start_g_start)
        infrastructure_distances = infrastructure_distances.transpose()
        distance = infrastructure_distances.at[node_start_g_start, node_start_g_destination]

        min_value += distance / 1000 * transportation_costs_gas['Pipeline_Gas']
        costs.append(distance / 1000 * transportation_costs_gas['Pipeline_Gas'])
        travelled_distances.append(distance)
        used_transport_means.append('Pipeline_Gas')
        used_nodes.append(node_start_g_destination)

    else:
        infrastructure_distances \
            = pd.read_hdf(path_data + '/inner_infrastructure_distances/' + node_start_g_start + '.h5', mode='r',
                          title=node_start_g_start)
        infrastructure_distances = infrastructure_distances.transpose()
        distance = infrastructure_distances.at[node_start_g_start, node_end_g_start]

        min_value += distance / 1000 * transportation_costs_gas['Pipeline_Gas']
        costs.append(distance / 1000 * transportation_costs_gas['Pipeline_Gas'])
        travelled_distances.append(distance)
        used_transport_means.append('Pipeline_Gas')
        used_nodes.append(node_start_g_start)

        # calculate distance between pipelines
        if distance_between_pipelines < in_tolerance_distance:
            distance_between_pipelines = 0

        if (distance_between_pipelines * no_road_multiplier <= max_length_new_segment) & build_new_infrastructure:

            # if new pipelines are possible, we can use the commodity as is (gas)
            transportation_costs_option_gas \
                = distance_between_pipelines * no_road_multiplier / 1000 * transportation_costs_gas['New_Pipeline_Gas']
            costs.append(transportation_costs_option_gas)

            min_value += transportation_costs_option_gas
            used_transport_means.append('New_Pipeline_Gas')

        elif distance_between_pipelines * no_road_multiplier > max(max_length_road, max_length_new_segment):
            # no first transportation
            return math.inf, None, None, None, None, None
        else:

            # needs road transportation --> conversion to road commodity
            min_value_before = min_value
            min_value = (min_value + conversion_costs_gas.at[node_end_g_start, road_commodity.get_name()]) \
                / conversion_losses_gas.at[node_end_g_start, road_commodity.get_name()]
            costs.append(min_value - min_value_before)
            used_commodities.append(road_commodity.get_name())

            transportation_costs_option_liquid \
                = distance_between_pipelines * no_road_multiplier / 1000 * transportation_costs_liquid['Road']

            costs.append(transportation_costs_option_liquid)

            min_value += transportation_costs_option_liquid
            used_transport_means.append('Road')

            # conversion to gas as afterwards pipeline transportation again
            min_value_before = min_value
            min_value = (min_value + conversion_costs_liquid.at[node_end_g_start, pipeline_commodity.get_name()]) \
                / conversion_losses_liquid.at[node_end_g_start, pipeline_commodity.get_name()]
            costs.append(min_value - min_value_before)
            used_commodities.append(pipeline_commodity.get_name())

        travelled_distances.append(distance_between_pipelines * no_road_multiplier)
        used_nodes.append(node_end_g_start)

        # second pipeline
        infrastructure_distances \
            = pd.read_hdf(path_data + '/inner_infrastructure_distances/' + node_start_g_destination + '.h5', mode='r',
                          title=node_start_g_destination)
        infrastructure_distances = infrastructure_distances.transpose()
        distance = infrastructure_distances.at[node_start_g_destination, node_end_g_destination]

        min_value += distance / 1000 * transportation_costs_gas['Pipeline_Gas']
        costs.append(distance / 1000 * transportation_costs_gas['Pipeline_Gas'])
        travelled_distances.append(distance)
        used_transport_means.append('Pipeline_Gas')
        used_nodes.append(node_start_g_start)

    # calculate distance between ship and destination
    if distance_pipeline_to_destination <= in_tolerance_distance:
        distance_pipeline_to_destination = 0

    if (distance_pipeline_to_destination * no_road_multiplier <= max_length_new_segment) & build_new_infrastructure:

        transportation_costs_option_gas \
            = distance_pipeline_to_destination * no_road_multiplier / 1000 * transportation_costs_gas['New_Pipeline_Gas']
        costs.append(transportation_costs_option_gas)

        min_value += transportation_costs_option_gas
        used_transport_means.append('New_Pipeline_Gas')

    elif distance_pipeline_to_destination * no_road_multiplier > max(max_length_road, max_length_new_segment):
        # no first transportation
        return math.inf, None, None, None, None, None
    else:
        # needs road transportation --> conversion to road commodity
        min_value_before = min_value
        min_value = (min_value + conversion_costs_gas.at['Destination', road_commodity.get_name()]) \
            / conversion_losses_gas.at['Destination', road_commodity.get_name()]
        costs.append(min_value - min_value_before)
        used_commodities.append(road_commodity.get_name())

        # transportation via road
        transportation_costs_option_liquid \
            = distance_pipeline_to_destination * no_road_multiplier / 1000 * transportation_costs_liquid['Road']
        costs.append(transportation_costs_option_liquid)

        min_value += transportation_costs_option_liquid
        used_transport_means.append('Road')

    travelled_distances.append(distance_pipeline_to_destination * no_road_multiplier)

    used_nodes.append('Destination')

    if used_commodities[-1] not in final_commodities:
        cheapest_conversion = math.inf
        cheapest_conversion_commodity = None

        commodity_object = data['commodities']['commodity_objects'][used_commodities[-1]]

        min_value_before = min_value

        for c in final_commodities:
            if commodity_object.get_conversion_options()[c]:

                conversion_costs = commodity_object.get_conversion_costs_specific_commodity('Destination', c)
                conversion_efficiency = commodity_object.get_conversion_efficiency_specific_commodity('Destination', c)

                conversion_costs = (min_value + conversion_costs) / conversion_efficiency

                if conversion_costs < cheapest_conversion:
                    cheapest_conversion = conversion_costs
                    cheapest_conversion_commodity = c

        min_value = cheapest_conversion
        costs.append(min_value - min_value_before)
        used_commodities.append(cheapest_conversion_commodity)

    return min_value, used_commodities, used_transport_means, used_nodes, travelled_distances, costs
