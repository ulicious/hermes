from _helpers import calc_distance
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from copy import deepcopy

from methods_checking import check_total_costs_of_solutions
from methods_routing import find_locations_within_tolerance
from methods_road_transport import find_routes_road_transportation
from methods_networks import find_shortest_path_in_existing_graph, attach_new_node_to_graph
from methods_shipping import find_searoute, remove_and_sort_ports

from object_solution import create_new_solution_from_routing_result, create_new_solution_from_conversion_result
from _helpers import calc_distance_lists


def create_conversion_solutions(solution, commodities, benchmark, final_solution, c_num, configuration):
    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()

    new_solutions = []
    for c in commodities:
        if s_commodity.get_name() != c.get_name():
            if s_commodity.get_conversion_options_specific_commodity(c.get_name()):

                s_new = create_new_solution_from_conversion_result(solution, c, c_num)

                # Don't add solutions which have already higher costs than benchmark
                if s_new.get_total_costs() <= benchmark:
                    new_solutions.append(s_new)

                    # Check if solutions has arrived in destination and has right target commodity
                    # If so, update benchmark and remove solutions
                    if s_new.check_if_in_destination(final_destination,
                                                     configuration['to_final_destination_tolerance']) \
                            & (s_new.get_current_commodity() == final_commodity):
                        benchmark = s_new.get_total_costs()
                        final_solution = s_new
                        new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)

                    c_num += 1

        else:  # keep original solution when solution commodity = c
            new_solutions.append(solution)

    return new_solutions, benchmark, final_solution, c_num


def iterate_through_means_of_transport(solution, means_of_transport,
                                       destination_continent, destination_location, ports,
                                       benchmark, final_solution, r_num, configuration):
    s_commodity = solution.get_current_commodity_object()

    # Reachable locations are those which are within a certain tolerance, meaning, that we don't have
    # to do another road step but just assume that we are already there. Gap is closed with
    # transportation with current mean of transport
    locations_within_tolerance = find_locations_within_tolerance(solution, configuration['tolerance_distance'])

    new_solutions = []
    for mean_of_transport in means_of_transport:

        if mean_of_transport == 'Road':

            # To avoid circles, repeating road transportation is not allowed if the solution has used road
            # transportation beforehand. The algorithm will find the direct path of possible.
            if solution.get_used_transport_means():
                if solution.get_last_used_transport_means() == 'Road':
                    continue

            road_and_direct_transportation_solutions, benchmark, final_solution, r_num \
                = find_routes_based_on_road_and_direct_transportation(solution, benchmark,
                                                                      final_solution, r_num, configuration)

            if road_and_direct_transportation_solutions:
                new_solutions += road_and_direct_transportation_solutions

        else:

            reachable_locations_for_mean_of_transport = locations_within_tolerance[mean_of_transport]

            if (reachable_locations_for_mean_of_transport is not None) \
                    & s_commodity.get_transportation_options_specific_mean_of_transport(mean_of_transport):

                if mean_of_transport in ['Railroad', 'Pipeline_Gas', 'Pipeline_Liquid']:

                    new_network_solutions, benchmark, final_solution, r_num\
                        = find_routes_through_network(solution, mean_of_transport,
                                                      reachable_locations_for_mean_of_transport,
                                                      benchmark, final_solution, r_num, configuration)

                    if new_network_solutions:
                        new_solutions += new_network_solutions

                else:  # Shipping

                    new_shipping_solutions, benchmark, final_solution, r_num \
                        = find_routes_based_on_shipping(solution, mean_of_transport,
                                                        reachable_locations_for_mean_of_transport,
                                                        ports, destination_continent, destination_location,
                                                        benchmark, final_solution, r_num, configuration)

                    if new_shipping_solutions:
                        new_solutions += new_shipping_solutions

    return new_solutions, benchmark, final_solution, r_num


def find_routes_based_on_road_and_direct_transportation(solution, benchmark, final_solution, r_num, configuration):
    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()

    new_solutions = []

    # Find routes to existing infrastructure (ports, feed-in and railroad stations)
    target_data = find_routes_road_transportation(solution, benchmark,
                                                  configuration)
    for target in target_data:

        target_system = target[0]
        target_point = target[1]
        road_distance_to_target = target[2]
        linestring_to_target = target[3]

        s_new = create_new_solution_from_routing_result(solution, s_commodity, 'Road',
                                                        target_point, road_distance_to_target,
                                                        linestring_to_target,
                                                        r_num)

        if s_new.get_total_costs() <= benchmark:
            new_solutions.append(s_new)

            # Check if solution has arrived in destination and has right commodity
            # If so, update benchmark and remove solutions
            if s_new.check_if_in_destination(final_destination,
                                             configuration['to_final_destination_tolerance']) \
                    & (s_new.get_current_commodity() == final_commodity):
                benchmark = s_new.get_total_costs()
                final_solution = s_new
                new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)

            r_num += 1

        if target_system in ['Pipeline_Gas', 'Pipeline_Liquid', 'Railroad']:
            # If distance to network (pipeline or railroad) is larger than the tolerance, a new
            # segment can be installed. This new segment is not added to the graph of the
            # network but instead, directly installed

            # todo: As the target location is decided based on road transport,
            #  only targets are possible that are reachable by street
            #  Could be extended to any location (e.g. underwater pipelines)

            target_system += '_New'

            if solution.get_current_commodity_object().get_transportation_options_specific_mean_of_transport(
                    target_system):

                solutions_from_segments, benchmark, final_solution, r_num \
                    = build_new_network_segments(solution, target_system, target_point,
                                                 road_distance_to_target, linestring_to_target,
                                                 benchmark, final_solution,
                                                 r_num, configuration)

                new_solutions += solutions_from_segments

    return new_solutions, benchmark, final_solution, r_num


def build_new_network_segments(solution, target_system, target_point, road_distance_to_target,
                               road_path_linestring_to_target, benchmark, final_solution, r_num, configuration):

    """
    Method builds new segment from current location to network if distance is higher than tolerance. Furthermore, new
    solutions are implemented, which are located at the target location and which have used the new segment

    :param solution: Current solution
    :param target_system: E.g. gas pipeline, liquid pipeline, railroad
    :param target_point: Destination in iteration
    :param road_distance_to_target
    :param road_path_linestring_to_target: Linestring which describes the path of road transportation
    :param benchmark: benchmark of algorithm
    :param final_solution: current solution which sets benchmark
    :param r_num: current ID
    :param configuration
    :return: solutions which use new segments
    """

    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()
    origin_location = solution.get_current_location()
    direct_distance_to_target = calc_distance(origin_location.y, origin_location.x,
                                              target_point.y, target_point.x)

    new_solutions = []

    # Get configurations
    max_length_new_segment = configuration[target_system]['max_length_new_segment']
    follow_existing_roads = configuration[target_system]['follow_existing_roads']
    use_direct_path = configuration[target_system]['use_direct_path']

    if follow_existing_roads:  # uses the road path as new segment path
        if road_distance_to_target <= max_length_new_segment:
            s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                            target_system,
                                                            target_point,
                                                            road_distance_to_target,
                                                            road_path_linestring_to_target,
                                                            r_num)

            if s_new.get_total_costs() <= benchmark:
                new_solutions.append(s_new)

                # Check if solutions has arrived in destination and has right target commodity
                # If so, update benchmark and remove solutions
                if s_new.check_if_in_destination(final_destination,
                                                 configuration['to_final_destination_tolerance']) \
                        & (s_new.get_current_commodity() == final_commodity):
                    benchmark = s_new.get_total_costs()
                    final_solution = s_new
                    new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)

                r_num += 1

    if use_direct_path:  # uses as beeline path as new segment path
        if direct_distance_to_target <= max_length_new_segment:
            linestring_to_target = LineString([origin_location, target_point])

            s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                            target_system,
                                                            target_point,
                                                            direct_distance_to_target,
                                                            linestring_to_target,
                                                            r_num)

            if s_new.get_total_costs() <= benchmark:
                new_solutions.append(s_new)

                # Check if solutions has arrived in destination and has right target commodity
                # If so, update benchmark and remove solutions
                if s_new.check_if_in_destination(final_destination,
                                                 configuration['to_final_destination_tolerance']) \
                        & (s_new.get_current_commodity() == final_commodity):
                    benchmark = s_new.get_total_costs()
                    final_solution = s_new
                    new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)

                r_num += 1

    return new_solutions, benchmark, final_solution, r_num


def find_routes_through_network(solution, mean_of_transport, reachable_locations_for_mean_of_transport,
                                benchmark, final_solution, r_num, configuration):

    """
    Iterates through reachable locations, identifies corresponding network and finds shortest path through network using
    Dijkstra

    :param solution: Solution Object
    :param mean_of_transport: Mean of Transport
    :param reachable_locations_for_mean_of_transport: List with all reachable locations
    :param benchmark: Current Benchmark
    :param final_solution: Current final solution setting benchmark
    :param r_num: ID of routing
    :param configuration:
    :return: new solutions, updated benchmark, final solution and r_num
    """

    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()
    current_location = solution.get_current_location()

    new_solutions = []
    for n_start in reachable_locations_for_mean_of_transport.index:
        graph_id = reachable_locations_for_mean_of_transport.loc[n_start, 'graph']

        if mean_of_transport == 'Railroad':
            graph = solution.get_railroad_networks()[graph_id]['Graph']
            graph_data = solution.get_railroad_networks()[graph_id]['GraphData']
            graph_object = solution.get_railroad_networks()[graph_id]['GraphObject']
            geodata = solution.get_railroad_networks()[graph_id]['GeoData']

        elif mean_of_transport in ['Pipeline_Gas', 'Pipeline_Gas_New']:
            graph = solution.get_pipeline_gas_networks()[graph_id]['Graph']
            graph_data = solution.get_pipeline_gas_networks()[graph_id]['GraphData']
            graph_object = solution.get_pipeline_gas_networks()[graph_id]['GraphObject']
            geodata = solution.get_pipeline_gas_networks()[graph_id]['GeoData']
        else:
            graph = solution.get_pipeline_liquid_networks()[graph_id]['Graph']
            graph_data = solution.get_pipeline_liquid_networks()[graph_id]['GraphData']
            graph_object = solution.get_pipeline_liquid_networks()[graph_id]['GraphObject']
            geodata = solution.get_pipeline_liquid_networks()[graph_id]['GeoData']

        # Check if n_start is already in graph or is new node based on shortest distance
        # Update geodata, graph_data and graph_object if new node
        if n_start not in geodata.index:

            new_node_in_network = nearest_points(graph_object, current_location)[0]
            new_node_in_network = Point([round(new_node_in_network.x, 5), round(new_node_in_network.y, 5)])
            geodata, graph, graph_data, graph_object = \
                attach_new_node_to_graph(geodata.copy(), deepcopy(graph), graph_data.copy(), deepcopy(graph_object),
                                         graph_id, new_node_in_network)

        target_nodes_geodata = geodata[geodata['graph'] == graph_id]

        # Calculate beeline distance and costs to current location. Drop if costs are higher than benchmark
        target_nodes_geodata['distance'] = calc_distance_lists(target_nodes_geodata['latitude'],
                                                               target_nodes_geodata['longitude'],
                                                               current_location.y, current_location.x)
        target_nodes_geodata.sort_values(['distance'], inplace=True)
        target_nodes_geodata['costs'] = target_nodes_geodata['distance'] / 1000 \
            * s_commodity.get_transportation_costs_specific_mean_of_transport(mean_of_transport) \
            + solution.get_total_costs()

        target_nodes = target_nodes_geodata[target_nodes_geodata['costs'] <= benchmark].index

        # Iterate through nodes which have lower beeline costs

        for n_target in target_nodes:

            if n_start != n_target:

                lat_start = geodata.loc[n_start, 'latitude']
                lon_start = geodata.loc[n_start, 'longitude']

                # Check if line has to be added if station is not exact
                # on current location (due to tolerance_distance)
                if ((lat_start != current_location.y)
                        & (lon_start != current_location.x)):

                    distance_to_bridge_tolerance = calc_distance(lat_start, lon_start,
                                                                 current_location.y, current_location.x)

                    if distance_to_bridge_tolerance > 0:
                        stations_and_routes = find_shortest_path_in_existing_graph(
                            geodata, graph, graph_data, n_start, n_target, current_location,
                            distance_to_bridge_tolerance)

                    else:
                        stations_and_routes = find_shortest_path_in_existing_graph(
                            geodata, graph, graph_data, n_start, n_target)

                    target_point = stations_and_routes[0]
                    distance_to_target = stations_and_routes[1]
                    line_to_target = stations_and_routes[2]

                    s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                                    mean_of_transport,
                                                                    target_point,
                                                                    distance_to_target,
                                                                    line_to_target, r_num)

                    # Remove the graph from the new solution
                    # as graph cannot be used twice from same or following solutions
                    if mean_of_transport == 'Railroad':
                        s_new.remove_railroad_network(graph_id)
                    elif mean_of_transport in ['Pipeline_Gas']:
                        s_new.remove_pipeline_gas_network(graph_id)
                    else:
                        s_new.remove_pipeline_liquid_network(graph_id)

                    # Don't add solutions which have already higher costs than benchmark
                    if s_new.get_total_costs() < benchmark:
                        new_solutions.append(s_new)

                        # Check if solutions has arrived in destination
                        # and has right target commodity
                        # If so, update benchmark and remove solutions
                        if s_new.check_if_in_destination(final_destination,
                                                         configuration['to_final_destination_tolerance']) \
                                & (s_new.get_current_commodity() == final_commodity):
                            benchmark = s_new.get_total_costs()
                            new_solutions = check_total_costs_of_solutions(new_solutions,
                                                                           benchmark)

                            final_solution = s_new

                        r_num += 1

    return new_solutions, benchmark, final_solution, r_num


def find_routes_based_on_shipping(solution, mean_of_transport, reachable_locations_for_mean_of_transport,
                                  ports, destination_continent, destination_location,
                                  benchmark, final_solution, r_num, configuration):

    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()

    current_costs = solution.get_total_costs()
    transportation_costs \
        = s_commodity.get_transportation_costs_specific_mean_of_transport(mean_of_transport)

    ports_of_solution = solution.get_ports().copy()
    ports_of_solution = remove_and_sort_ports(ports_of_solution.copy(),
                                              destination_continent,
                                              destination_location,
                                              transportation_costs,
                                              current_costs, benchmark)

    new_solutions = []
    for port_start in reachable_locations_for_mean_of_transport.index:
        for port_target in ports_of_solution.index:

            if port_start == port_target:
                continue

            current_location = solution.get_current_location()

            lat_start = ports.loc[port_start, 'latitude']
            lon_start = ports.loc[port_start, 'longitude']

            # Check if line has to be added if port is not exact on current location
            # (due to tolerance_distance)
            if ((lat_start != current_location.y)
                    & (lon_start != current_location.x)):
                distance = calc_distance(lat_start, lon_start,
                                         current_location.y,
                                         current_location.x)

                ports_and_routes = find_searoute((ports.loc[port_start, 'longitude'],
                                                  ports.loc[port_start, 'latitude']),
                                                 ((ports.loc[port_target, 'longitude'],
                                                   ports.loc[port_target, 'latitude'])),
                                                 current_location, distance)
            else:
                ports_and_routes = find_searoute((ports.loc[port_start, 'longitude'],
                                                  ports.loc[port_start, 'latitude']),
                                                 ((ports.loc[port_target, 'longitude'],
                                                   ports.loc[port_target, 'latitude'])))

            target_point = ports_and_routes[0]
            distance_to_target = ports_and_routes[1]
            line_to_target = ports_and_routes[2]

            s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                            mean_of_transport,
                                                            target_point,
                                                            distance_to_target,
                                                            line_to_target, r_num)

            # Don't add solutions which have already higher costs than benchmark
            if s_new.get_total_costs() < benchmark:
                new_solutions.append(s_new)

                # Check if solutions has arrived in destination
                # & has right target commodity
                # If so, update benchmark and remove solutions
                if s_new.check_if_in_destination(final_destination,
                                                 configuration['to_final_destination_tolerance']) \
                        & (s_new.get_current_commodity() == final_commodity):
                    benchmark = s_new.get_total_costs()
                    new_solutions = check_total_costs_of_solutions(new_solutions,
                                                                   benchmark)

                    final_solution = s_new

                r_num += 1

            s_new.remove_port(port_start)

    return new_solutions, benchmark, final_solution, r_num

