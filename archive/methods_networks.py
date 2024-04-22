from shapely.geometry import Point, LineString
import geopandas as gpd
import math

from methods_geographic import calc_distance_single_to_single
from object_solution import create_new_solution_from_routing_result


def attach_new_node_to_graph(geodata, graph, graph_data, graph_object, graph_name,
                             new_node_in_network, new_node_count):

    # Find position of new node and add information to geodata
    new_node_in_network_index = 'New_Build_' + str(new_node_count) + '_' + str(len(geodata.index) + 1)
    geodata.loc[new_node_in_network_index, 'latitude'] = round(new_node_in_network.y, 4)
    geodata.loc[new_node_in_network_index, 'longitude'] = round(new_node_in_network.x, 4)
    geodata.loc[new_node_in_network_index, 'graph'] = graph_name
    geodata.loc[new_node_in_network_index, 'considered'] = True

    new_node_count += 1

    # Identify LineString where new node will be placed
    left_df = gpd.GeoDataFrame(geometry=[new_node_in_network])
    right_df = gpd.GeoDataFrame(geometry=[graph_object]).explode(ignore_index=True)
    df_n = gpd.sjoin_nearest(left_df, right_df).merge(right_df, left_on="index_right",
                                                      right_index=True)

    affected_line = df_n['geometry_y'].values[0]  # this LineString will be divided
    starting_node_x = round(affected_line.coords.xy[0][0], 4)
    starting_node_y = round(affected_line.coords.xy[1][0], 4)

    ending_node_x = round(affected_line.coords.xy[0][-1], 4)
    ending_node_y = round(affected_line.coords.xy[1][-1], 4)

    # Check which nodes are at the ends of the affected line
    starting_node_index = geodata[(geodata['longitude'] == starting_node_x) &
                                  (geodata['latitude'] == starting_node_y)].index
    starting_node_index = starting_node_index[0]

    ending_node_index = geodata[(geodata['longitude'] == ending_node_x) &
                                (geodata['latitude'] == ending_node_y)].index
    ending_node_index = ending_node_index[0]

    # Separate LineString into first and last part
    first_part_line = []
    last_part_line = []

    c_before = None
    affected_line_segment_reached = False
    for c in affected_line.coords:

        if c_before is not None:
            c_now = Point([round(c[0], 4), round(c[1], 4)])
            distance = round(LineString([c_before, c_now]).distance(new_node_in_network), 3)

            if distance == 0:
                affected_line_segment_reached = True

            if not affected_line_segment_reached:
                first_part_line.append((round(c[0], 4), round(c[1], 4)))
            else:
                last_part_line.append((round(c[0], 4), round(c[1], 4)))
        else:
            first_part_line.append((round(c[0], 4), round(c[1], 4)))

        c_before = Point([round(c[0], 4), round(c[1], 4)])

    first_part_line.append((round(new_node_in_network.x, 4), round(new_node_in_network.y, 4)))
    last_part_line = [(round(new_node_in_network.x, 4), round(new_node_in_network.y, 4))] + last_part_line

    # Calculate distances of both parts
    c_before = None
    distance_first_line = 0
    for c in first_part_line:
        if c_before is not None:
            distance_first_line += calc_distance_single_to_single(c[1], c[0], c_before[1], c_before[0])
        c_before = c

    c_before = None
    distance_last_line = 0
    for c in last_part_line:
        if c_before is not None:
            distance_last_line += calc_distance_single_to_single(c[1], c[0], c_before[1], c_before[0])
        c_before = c

    # Add first part as new edge to graph & update graph data
    graph.add_edge(starting_node_index, new_node_in_network_index, weight=distance_first_line)

    first_part_linestring = LineString(first_part_line)

    if len(graph_data.index) == 0:
        new_edge = 0
    else:
        new_edge = len(graph_data.index)

    # new_edge = max_edge + 1
    graph_data.loc[new_edge, 'graph'] = graph_name
    graph_data.loc[new_edge, 'node_start'] = starting_node_index
    graph_data.loc[new_edge, 'node_end'] = new_node_in_network_index
    graph_data.loc[new_edge, 'costs'] = distance_first_line
    graph_data.loc[new_edge, 'line'] = first_part_linestring

    # Add last part as new edge to graph & update graph data
    graph.add_edge(ending_node_index, new_node_in_network_index, weight=distance_last_line)

    last_part_linestring = LineString(last_part_line)

    if len(graph_data.index) == 0:
        new_edge = 0
    else:
        new_edge = len(graph_data.index)

    # new_edge = max_edge + 1
    graph_data.loc[new_edge, 'graph'] = graph_name
    graph_data.loc[new_edge, 'node_start'] = new_node_in_network_index
    graph_data.loc[new_edge, 'node_end'] = ending_node_index
    graph_data.loc[new_edge, 'costs'] = distance_last_line
    graph_data.loc[new_edge, 'line'] = last_part_linestring

    return geodata, graph, graph_data, graph_object, new_node_count


def process_target_nodes(n_target, solution, s_commodity, mean_of_transport, geodata, graph_data, graph_id,
                         current_location, n_start, lengths, paths, benchmark, iteration):

    if n_start != n_target:

        lat_start = geodata.loc[n_start, 'latitude']
        lon_start = geodata.loc[n_start, 'longitude']

        # Check if line has to be added if station is not exact
        # on current location (due to tolerance_distance)
        if ((lat_start != current_location.y)
                & (lon_start != current_location.x)):

            distance_real_start_to_n_start = calc_distance_single_to_single(lat_start, lon_start,
                                                                            current_location.y, current_location.x)

            distance_in_graph = lengths[n_target]

            target_point = Point([geodata.loc[n_target, 'longitude'],
                                  geodata.loc[n_target, 'latitude']])
            distance_to_target = distance_in_graph

            s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                            mean_of_transport,
                                                            target_point,
                                                            distance_to_target,
                                                            iteration)

            # Remove the graph from the new solution
            # as graph cannot be used twice from same or following solutions
            if mean_of_transport == 'Railroad':
                s_new.add_used_railroad_network(graph_id)
            elif mean_of_transport in ['Pipeline_Gas']:
                s_new.add_used_pipeline_gas_network(graph_id)
            else:
                s_new.add_used_pipeline_liquid_network(graph_id)

            # Don't add solutions which have already higher costs than benchmark
            if s_new.get_total_costs() < benchmark:
                return s_new
            else:
                return None

def build_new_network_segments(solution, target_system, target_point, road_distance_to_target,
                               benchmark, final_solution, configuration, iteration):

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
    direct_distance_to_target = calc_distance_single_to_single(origin_location.y, origin_location.x,
                                                               target_point.y, target_point.x)

    new_solutions = []

    # Get configurations
    max_length_new_segment = configuration[target_system]['max_length_new_segment']
    follow_existing_roads = configuration[target_system]['follow_existing_roads']
    use_direct_path = configuration[target_system]['use_direct_path']

    if follow_existing_roads:  # uses the road path as new segment path
        if road_distance_to_target <= max_length_new_segment:

            # todo: the line string of the road path is deleted. How to see the road path?

            s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                            target_system,
                                                            target_point,
                                                            road_distance_to_target,
                                                            iteration)

            # Check if new solution is less expensive than benchmark
            if s_new.get_total_costs() <= benchmark:
                # Check if solution has arrived in destination and has right commodity
                if s_new.check_if_in_destination(final_destination,
                                                 configuration['to_final_destination_tolerance']) \
                        & (s_new.get_commodity_name() == final_commodity):
                    benchmark = s_new.get_total_costs()
                    final_solution = s_new
                    new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)
                else:
                    # if new solutions has not arrived at destination but has lower total costs than the benchmark,
                    # it will be further processed
                    new_solutions.append(s_new)

    if use_direct_path:  # uses as beeline path as new segment path
        if direct_distance_to_target <= max_length_new_segment:

            s_new = create_new_solution_from_routing_result(solution, s_commodity,
                                                            target_system,
                                                            target_point,
                                                            direct_distance_to_target,
                                                            iteration)

            # todo: move to own method as it is used several times in different parts of the code
            # Check if new solution is less expensive than benchmark
            if s_new.get_total_costs() <= benchmark:
                # Check if solution has arrived in destination and has right commodity
                if s_new.check_if_in_destination(final_destination,
                                                 configuration['to_final_destination_tolerance']) \
                        & (s_new.get_commodity_name() == final_commodity):
                    benchmark = s_new.get_total_costs()
                    final_solution = s_new
                    new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)
                else:
                    # if new solutions has not arrived at destination but has lower total costs than the benchmark,
                    # it will be further processed
                    new_solutions.append(s_new)

    return new_solutions, benchmark, final_solution