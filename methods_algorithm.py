import time

import networkx as nx
import numpy as np
from shapely.geometry import Point

from script_finding_potential_destinations import find_potential_destinations
from object_solution import create_new_solution_from_routing_result, create_new_solution_from_conversion_result, \
    process_new_solution
from methods_road_transport import get_road_distances_between_options
from _helpers import check_if_last_was_new_segment
from methods_checking import remove_solution_duplications


def create_solutions_from_conversion(solution, scenario_count, commodities, benchmark, final_solution, configuration,
                                     iteration, solutions_per_iteration, local_benchmarks, solutions_to_remove,
                                     solutions_reaching_end):
    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()

    historic_paths = {}

    new_solutions = []
    for c in commodities:
        if s_commodity.get_name() != c.get_name():
            if s_commodity.get_conversion_options_specific_commodity(c.get_name()):

                used_node = list(solution.get_used_nodes().values())[-1]

                s_new = create_new_solution_from_conversion_result(solution, scenario_count, c, iteration)
                # solutions_per_iteration[iteration].append(s_new)
                scenario_count += 1

                current_commodity = s_new.get_current_commodity_object()

                # process the left solutions
                final_solution, new_solutions, local_benchmarks, solutions_to_remove, benchmark, solutions_reaching_end \
                    = process_new_solution(s_new, new_solutions, final_solution,
                                           benchmark, local_benchmarks, solutions_to_remove,
                                           final_destination, final_commodity,
                                           current_commodity, used_node,
                                           configuration, solutions_reaching_end)

        else:  # keep original solution when solution commodity = c
            s_new = create_new_solution_from_conversion_result(solution, scenario_count, c, iteration)
            new_solutions.append(s_new)
            scenario_count += 1

            solutions_per_iteration[iteration].append(s_new)

    return new_solutions, benchmark, final_solution, scenario_count, solutions_per_iteration, local_benchmarks, \
        solutions_to_remove, solutions_reaching_end, historic_paths


def create_solutions_from_historic_paths(solution, scenario_count, new_node_count, benchmark, final_solution,
                                         configuration, iteration, local_benchmarks,
                                         solutions_to_remove, historic_path_costs, solutions_reaching_end):

    # todo: Jede solution, die am Ziel ankommt, auswerten und analysieren, ob nachfolgende solutions Kosten reißen (umgekehrt von local benchmark)
    # todo: Datenbanken mit Informationen zu Lösungen global verfügbar machen (umgekehrte Benchmark)
    # todo: basierend auf umgekehrte Benchmark globale Benchmark neuer Start-Standorte berechnen --> nicht nur einfachste Lösung

    new_solutions = []
    used_infrastructure = [element for inner_list in solution.get_used_infrastructure().values() for element in
                           inner_list]
    s_location = solution.get_current_location()
    current_commodity = solution.get_current_commodity_object()
    current_costs = solution.get_total_costs()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()

    affected_keys = [k for k in [*historic_path_costs.keys()] if k[1] == s_location]

    for k in affected_keys:
        for target_infrastructure in [*historic_path_costs[k].keys()]:
            target_commodity = target_infrastructure[0]
            target_system = target_infrastructure[1]
            target_node = target_infrastructure[2]

            if target_commodity == current_commodity.get_name():
                if target_node not in used_infrastructure:
                    costs_of_segment = historic_path_costs[k][target_infrastructure]['costs']
                    distance_to_target = historic_path_costs[k][target_infrastructure]['distance']
                    target_point = historic_path_costs[k][target_infrastructure]['target_location']
                    used_infrastructure = historic_path_costs[k][target_infrastructure]['used_infrastructure']

                    solution.add_used_infrastructure(iteration, used_infrastructure)

                    if costs_of_segment + current_costs <= benchmark:
                        s_new = create_new_solution_from_routing_result(solution, scenario_count, current_commodity, target_system,
                                                                        target_point, distance_to_target, used_infrastructure,
                                                                        target_infrastructure, iteration)

                        final_solution, new_solutions, local_benchmarks, solutions_to_remove, benchmark, solutions_reaching_end \
                            = process_new_solution(s_new, new_solutions, final_solution,
                                                   benchmark, local_benchmarks, solutions_to_remove,
                                                   final_destination, final_commodity,
                                                   current_commodity, target_infrastructure,
                                                   configuration, solutions_reaching_end)

    return solution, new_solutions, benchmark, final_solution, scenario_count, new_node_count, \
        local_benchmarks, solutions_to_remove, solutions_reaching_end


def create_solutions_from_routing(data, solution, scenario_count, new_node_count, benchmark, final_solution,
                                  configuration, iteration_transportation, solutions_per_iteration, local_benchmarks,
                                  solutions_to_remove, iteration, solutions_reaching_end):

    historic_path_costs = {}

    # Apply approach to filter potential destinations from all destinations
    potential_destinations, road_transport_options_to_calculate, new_node_count, data \
        = find_potential_destinations(data, configuration, solution, benchmark, local_benchmarks, new_node_count,
                                      iteration_transportation)

    # Convert potential destinations into new solutions
    s_commodity = solution.get_current_commodity_object()
    final_destination = solution.get_destination()
    final_commodity = solution.get_final_commodity()

    if False:

        # todo: potential improvement: implement bulk creation of new solutions and their assessment -> might reduce
        #  time further if possible

        s_new = deepcopy(solution)

        # fast way to implement new solutions
        important_columns = ['mean_of_transport', 'distance', 'used_infrastructure']
        target_locations = potential_destinations[["destination_longitude", "destination_latitude"]].apply(Point, axis=1)
        scenarios = ['S' + str(scenario_count+i) for i in range(len(potential_destinations.index))]
        used_nodes = potential_destinations.index.get_level_values(1)
        old_solutions = [s_new for i in range(len(potential_destinations.index))]

        for m in data['Means_of_Transport']:

            transport_costs = s_commodity.get_transportation_costs_specific_mean_of_transport(m)

            sub_pd = potential_destinations[potential_destinations['mean_of_transport'] == m].index
            potential_destinations.loc[sub_pd, 'costs']\
                = transport_costs * potential_destinations.loc[sub_pd, 'distance'] / 1000

        country, continent = get_country_and_continent_from_location(target_location.x, target_location.y)
        s_new.add_continent(iteration_transportation, continent)

        scenario_count += len(potential_destinations.index)

    else:

        new_solutions = []
        for ind in potential_destinations.index:

            # The total_costs_to_final_destination holds as destinations do not change. But the benchmark changes.
            # To avoid processing destination above benchmark, check regularly
            if potential_destinations.loc[ind, 'total_costs_to_final_destination'] > benchmark:
                continue

            target_system = potential_destinations.loc[ind, 'mean_of_transport']
            target_point = Point([potential_destinations.loc[ind, 'destination_longitude'],
                                  potential_destinations.loc[ind, 'destination_latitude']])
            distance_to_target = potential_destinations.loc[ind, 'distance']
            delta_costs = potential_destinations.loc[ind, 'costs_to_destination']

            mean_of_transport = ind[2]
            used_infrastructure = []
            if mean_of_transport == 'Shipping':
                # in case of shipping we need start and end harbor
                used_infrastructure = [potential_destinations.loc[ind, 'used_infrastructure'], ind[1]]
            elif mean_of_transport != 'Road':
                isinstance([potential_destinations.loc[ind, 'graph']], str)

                if isinstance([potential_destinations.loc[ind, 'graph']], str):
                    # some new pipelines are not connected to a graph. Therefore also no graph id
                    used_infrastructure = [potential_destinations.loc[ind, 'graph']]

            used_node = ind[1]

            # add new path to historic path costs dictionary

            # new infrastructure and the utilization of roads might be limited to no consecutive utilization
            # --> only new pipelines or only roads
            # this is necessary to avoid solutions which build parallel networks to existing networks
            # Therefore, we might not store such solutions to the global historic paths as we cannot assure that
            # new pipeline infrastructure or road utilization applies to all solutions
            if iteration >= 1:

                start = (solution.get_current_location(), solution.get_current_commodity_name())
                end = (target_point, solution.get_current_commodity_name(), target_system)

                if start not in list(historic_path_costs.keys()):
                    historic_path_costs[start] = {end: delta_costs}
                else:
                    if end not in list(historic_path_costs[start].keys()):
                        historic_path_costs[start] = {end: delta_costs}
                    else:
                        if delta_costs < historic_path_costs[start][end]:
                            historic_path_costs[start][end] = delta_costs

            s_new = create_new_solution_from_routing_result(solution, scenario_count, s_commodity, target_system,
                                                            target_point, distance_to_target, used_infrastructure,
                                                            used_node, iteration_transportation)
            scenario_count += 1

            # solutions_per_iteration[iteration].append(s_new)

            final_solution, new_solutions, local_benchmarks, solutions_to_remove, benchmark, solutions_reaching_end \
                = process_new_solution(s_new, new_solutions, final_solution,
                                       benchmark, local_benchmarks, solutions_to_remove,
                                       final_destination, final_commodity,
                                       s_commodity, used_node,
                                       configuration, solutions_reaching_end)

    return new_solutions, benchmark, final_solution, scenario_count, new_node_count, \
        local_benchmarks, solutions_to_remove, road_transport_options_to_calculate, historic_path_costs, \
        solutions_reaching_end


def process_road_options_without_route(data, road_options_without_route, solution_to_road_option, benchmark,
                                       local_benchmarks, scenario_count, solutions_per_iteration, iteration,
                                       configuration, final_solution, solutions_to_remove):

    options_to_process = road_options_without_route[~road_options_without_route.index.duplicated(keep='first')]

    starts = {}
    starts_locations = []
    destinations = {}
    destinations_locations = []
    start_destination_combination = []
    start_destination_combination_index = []
    position_start = 0
    position_destination = 0
    for ind in options_to_process.index:

        if ind[0] not in starts:
            starts[ind[0]] = {'position': position_start,
                              'index': ind[0],
                              'coordinates': (options_to_process.loc[ind, 'start_longitude'],
                                              options_to_process.loc[ind, 'start_latitude'])}

            starts_locations.append(Point([options_to_process.loc[ind, 'start_longitude'],
                                           options_to_process.loc[ind, 'start_latitude']]))

            chosen_start = position_start
            position_start += 1

        else:
            chosen_start = starts[ind[0]]['position']

        if ind[1] not in destinations:

            destinations[ind[1]] = {'position': position_destination,
                                    'index': ind[1],
                                    'coordinates': (options_to_process.loc[ind, 'destination_longitude'],
                                                    options_to_process.loc[ind, 'destination_latitude'])}

            destinations_locations.append(Point([options_to_process.loc[ind, 'destination_longitude'],
                                                 options_to_process.loc[ind, 'destination_latitude']]))

            chosen_destination = position_destination
            position_destination += 1

        else:
            chosen_destination = destinations[ind[1]]['position']

        if (chosen_start, chosen_destination) not in start_destination_combination:
            start_destination_combination.append((chosen_start, chosen_destination))
            start_destination_combination_index.append(ind)

    if False:
        distances = get_road_distances_between_options(starts_locations, destinations_locations,
                                                       step_size=450)
    else:
        distances = options_to_process.loc[:, 'direct_distance'] * 1.5

    for i, combination in enumerate(start_destination_combination):
        start = combination[0]
        destination = combination[1] + position_start
        index_combination = start_destination_combination_index[i]

        distance_list = distances[start]
        distance_combination = distance_list[destination]

        road_options_without_route.loc[index_combination, 'distance'] = distance_combination

    # Drop all rows with costs higher than benchmark
    idx = np.ones(len(road_options_without_route.index), dtype=bool)
    costs_to_destination_list = []
    for i in range(len(road_options_without_route.index)):

        solution_index = road_options_without_route.iloc[i]['solution_index']
        solution = solution_to_road_option[solution_index]
        total_costs = solution.get_total_costs()

        transported_commodity = road_options_without_route.iloc[i]['transported_commodity']
        transported_commodity_object = data['Commodities'][transported_commodity]
        transportation_costs_road = transported_commodity_object.get_transportation_costs_specific_mean_of_transport('Road')

        costs_to_destination \
            = total_costs + road_options_without_route.iloc[i]['distance'] * transportation_costs_road / 1000

        if costs_to_destination > benchmark:
            idx[i] = False
        else:
            costs_to_destination_list.append(costs_to_destination)

    road_options_without_route = road_options_without_route.iloc[idx].copy()
    road_options_without_route['costs_to_destination'] = costs_to_destination_list

    # Based on the true costs, assess solution and throw out if other was cheaper
    idx = np.ones(len(road_options_without_route.index), dtype=bool)
    for i in range(len(road_options_without_route.index)):
        transported_commodity = road_options_without_route.iloc[i]['transported_commodity']
        direct_distance_costs = road_options_without_route.iloc[i]['costs_to_destination']

        infrastructure = road_options_without_route.index[i][1]

        sub_df\
            = local_benchmarks[(local_benchmarks['infrastructure'] == infrastructure) &
                               (local_benchmarks['commodity'] == transported_commodity)]
        sub_df_index = sub_df[sub_df['total_costs'] < direct_distance_costs].index

        if len(sub_df_index) > 0:
            idx[i] = False

    road_options_without_route = road_options_without_route.iloc[idx]

    # Create solutions from rest of road options
    new_solutions = []
    for i in range(len(road_options_without_route.index)):

        # The total_costs_to_final_destination holds as destinations do not change. But the benchmark changes. To avoid
        # processing destination above benchmark, check regularly
        if road_options_without_route.iloc[i]['total_costs_to_final_destination'] > benchmark:
            continue

        target_system = road_options_without_route.iloc[i]['mean_of_transport']
        target_point = Point([road_options_without_route.iloc[i]['destination_longitude'],
                              road_options_without_route.iloc[i]['destination_latitude']])
        distance_to_target = road_options_without_route.iloc[i]['distance']

        used_infrastructure = road_options_without_route.iloc[i]['used_infrastructure']
        used_node = road_options_without_route.index[i][1]

        solution_index = road_options_without_route.iloc[i]['solution_index']
        solution = solution_to_road_option[solution_index]
        s_commodity = solution.get_current_commodity_object()
        final_destination = solution.get_destination()
        final_commodity = solution.get_final_commodity()

        s_new = create_new_solution_from_routing_result(solution, scenario_count, s_commodity, target_system,
                                                        target_point, distance_to_target, used_infrastructure,
                                                        used_node, iteration)
        scenario_count += 1

        # solutions_per_iteration[iteration].append(s_new)

        final_solution, new_solutions, local_benchmarks, solutions_to_remove \
            = process_new_solution(s_new, new_solutions, final_solution,
                                   benchmark, local_benchmarks, solutions_to_remove,
                                   final_destination, final_commodity,
                                   s_commodity, used_node,
                                   configuration)

    if False:

        for key in [*road_distances_to_calculate.keys()]:
            indexes_to_analyze = road_distances_to_calculate[key]

            distances = {}

            if len(indexes_to_analyze) > 5:  # bulk calculation of road distances
                road_transport_options.loc[indexes_to_analyze, 'distance'] \
                    = get_road_distance_to_options(configuration, location,
                                                   road_transport_options.loc[indexes_to_analyze, :])

                for ind in indexes_to_analyze:

                    if road_transport_options.loc[ind, 'distance'] is None:
                        road_transport_options.drop([ind], inplace=True)
                        indexes_to_analyze.remove(ind)

                    else:

                        road_transport_options.loc[ind, 'costs_to_destination'] \
                            = road_transport_options.loc[ind, 'distance'] * transportation_costs_road / 1000

                        if ind[0] != 'Start':  # Start does vary so don't store
                            if ind[0] not in [*distances.keys()]:
                                distances[ind[0]] = {}
                            distances[ind[0]][ind[1]] = road_transport_options.loc[
                                ind, 'distance']  # todo: might be dropped therefor error

                        if road_transport_options.loc[ind, 'costs_to_destination'] > benchmark:
                            road_transport_options.drop([ind], inplace=True)

            # Update the all distances dataframe with newly calculated road distances
            if key != 'Start':  # Start does vary so don't store
                distances = pd.DataFrame.from_dict(distances)
                not_in_index = []
                for ind_distances in distances.index:
                    if ind_distances not in all_distances_road.index:
                        not_in_index.append(str(ind_distances))

                new_distances = pd.DataFrame(math.nan, index=not_in_index, columns=not_in_index)

                if distances.columns[0] not in all_distances_road.index:
                    new_distances.loc[distances.columns[0], :] = math.nan
                    new_distances.loc[:, distances.columns[0]] = math.nan

                all_distances_road = pd.concat([all_distances_road, new_distances])

                all_distances_road.loc[distances.columns[0], distances.index] = distances[distances.columns[0]].tolist()
                all_distances_road.loc[distances.index, distances.columns[0]] = distances[distances.columns[0]].tolist()

        # overwrite data with adjusted all distances road
        data['all_distances_road'] = all_distances_road

    return final_solution, new_solutions, local_benchmarks, solutions_to_remove


def process_solutions_reaching_end(solutions_reaching_end, costs_to_final_destination, final_solution):
    for s in solutions_reaching_end:

        iterations = s.get_iterations()

        iteration_data = s.get_iteration_data()

        total_costs_to_current_location = ''
        for num, iteration in enumerate(reversed(iterations[2:])):

            key = None
            if iteration[1] == 'C':
                commodity_before_conversion = iteration_data['commodity'][iterations[num-2]]
                commodity_after_conversion = iteration_data['commodity'][iteration]

                if commodity_before_conversion.get_name() != commodity_after_conversion.get_name():

                    conversion_costs \
                        = commodity_before_conversion.get_conversion_costs_specific_commodity(commodity_after_conversion.get_name())
                    loss_of_product\
                        = commodity_before_conversion.get_conversion_loss_of_educt_specific_commodity(commodity_after_conversion.get_name())

                    if total_costs_to_current_location == '':
                        total_costs_to_current_location = '((x' + '+' + str(
                            conversion_costs) + ')/' + str(loss_of_product) + ')'
                    else:
                        total_costs_to_current_location = '((' + total_costs_to_current_location + '+' + str(conversion_costs) + ')/' + str(loss_of_product) + ')'

                    location = iteration_data['location'][iteration]  # location BEFORE transportation
                    key = (location, commodity_before_conversion.get_name())

            elif iteration[1] == 'T':

                transportation_costs_at_iteration = iteration_data['transportation_costs'][iteration]

                if total_costs_to_current_location == '':
                    total_costs_to_current_location = '(x' + '+' + str(transportation_costs_at_iteration) + ')'
                else:
                    total_costs_to_current_location = '(' + total_costs_to_current_location + '+' + str(
                        transportation_costs_at_iteration) + ')'

                location = iteration_data['location'][iterations[num+1]]
                commodity = iteration_data['commodity'][iteration].get_name()

                key = (location, commodity)

            is_final = False
            if s == final_solution:
                is_final = True

            if key is not None:
                if key in [*costs_to_final_destination.keys()]:

                    # check if cost_addition <-> is_final combination already in costs_to_final_destination dictionary
                    add = True
                    for i, c in enumerate(costs_to_final_destination[key]['costs']):
                        if (c == total_costs_to_current_location) & (is_final == costs_to_final_destination[key]['final'][i]):
                            add = False
                            break

                    if add:
                        if is_final:
                            # always add final solutions to first positions
                            # -- > they are cheaper as no other solution is better
                            costs_to_final_destination[key]['costs'].insert(0, total_costs_to_current_location)
                            costs_to_final_destination[key]['final'].insert(0, is_final)
                        else:
                            costs_to_final_destination[key]['costs'].append(total_costs_to_current_location)
                            costs_to_final_destination[key]['final'].append(is_final)
                else:
                    costs_to_final_destination[key] = {'costs': [total_costs_to_current_location],
                                                       'final': [is_final]}

    return costs_to_final_destination


def update_graph_data_based_on_parts(new_historic_parts, graph_data, graph_connector_data):

    for start in new_historic_parts:
        if start not in list(graph_data.keys()):
            graph_data[start] = new_historic_parts[start]
        else:
            for end in list(new_historic_parts[start].keys()):
                if end not in list(graph_data[start].keys()):
                    graph_data[start][end] = new_historic_parts[start][end]
                else:
                    if new_historic_parts[start][end] < graph_data[start][end]:
                        graph_data[start][end] = new_historic_parts[start][end]

    return graph_data, graph_connector_data


def create_graph_from_graph_data(graph_data):

    if False:

        graphs = {}
        for commodity in list(graph_data.keys()):
            graphs[commodity] = nx.Graph()

            node_combinations = []
            for i, ind in enumerate(graph_data[commodity].index):
                start_node = ind[0]
                end_node = ind[1]
                costs = graph_data[commodity].iloc[i].values[0]

                if ((start_node, end_node) not in node_combinations) & ((end_node, start_node) not in node_combinations):
                    graphs[commodity].add_edge(start_node, end_node, weight=costs)

                    node_combinations.append((start_node, end_node))
                    node_combinations.append((end_node, start_node))

    else:
        graphs = {}
        for commodity, data in graph_data.items():
            graphs[commodity] = nx.Graph()
            edge_set = set()  # Store edge combinations in a set

            for i, row in data.iterrows():
                start_node, end_node, costs = i[0], i[1], row[0]

                if (start_node, end_node) not in edge_set and (end_node, start_node) not in edge_set:
                    graphs[commodity].add_edge(start_node, end_node, weight=costs)
                    edge_set.add((start_node, end_node))
                    edge_set.add((end_node, start_node))

        graph_dict = {}
        for commodity, data in graph_data.items():

            for i, row in data.iterrows():
                start_node, end_node, costs = i[0], i[1], row[0]
                key = (start_node, commodity)

                if key in list(graph_dict.keys()):
                    if end_node in list(graph_dict[key].keys()):
                        if costs < graph_dict[key][end_node]:
                            graph_dict[key][end_node] = costs
                    else:
                        graph_dict[key][end_node] = costs
                else:
                    graph_dict[key] = {end_node: costs}

    return graphs, graph_dict


def update_local_benchmark_based_on_graph(solutions, graph_dict, local_benchmark, global_benchmark):

    import pandas as pd
    import shapely

    obsolete_solutions = []

    final_destination = solutions[0].get_destination()

    routing_costs = pd.DataFrame(graph_dict).transpose()

    # change columns to get format (location, commodity). Before it was (location, commodity, transport mean) &
    # reformulate to string (to allow groupby)
    routing_costs.columns = [c[0].wkt + '*' + c[1] for c in routing_costs.columns]

    # get minimum per column
    routing_costs = routing_costs.groupby(routing_costs.columns, axis=1).min()

    # reformulate it back to (point, commodity)
    routing_costs.columns = [(shapely.wkt.loads(c.split('*')[0]), c.split('*')[1]) for c in routing_costs.columns]

    # get costs of current solutions
    costs_per_solution = {}
    for s in solutions:
        current_commodity = s.get_current_commodity_name()
        current_location = s.get_current_location()
        current_total_costs = s.get_total_costs()

        costs_per_solution[(current_location, current_commodity)] = current_total_costs

    common_solutions = list(set(costs_per_solution.keys()).intersection(routing_costs.index.tolist()))
    routing_costs = routing_costs.loc[common_solutions, :]

    # add total costs
    total_costs = [costs_per_solution[location] for location in routing_costs.index]
    routing_costs = routing_costs.add(total_costs, axis='index')

    # remove columns which only consist of nan values --> these locations are not connected to graph
    routing_costs.dropna(axis='columns', how='all', inplace=True)

    # update benchmark --> if destination is in graph, then we are able to calculate a new benchmark
    if final_destination in routing_costs.columns:
        min_value_at_final_destination = routing_costs[final_destination].min()
        if min_value_at_final_destination < global_benchmark:
            global_benchmark = min_value_at_final_destination
            print('updated benchmark')

    # get common targets
    common_targets = list(set(routing_costs.columns).intersection(set(list(local_benchmark.keys()))))

    if common_targets:  # only if common targets exist

        # get all target locations which the local benchmark and the routing costs have in common
        routing_costs_common = routing_costs[common_targets]

        # get values of local_benchmark based on routing costs columns
        values_benchmark = {key: local_benchmark[key]['total_costs']
                            for key in common_targets}
        solutions_benchmark = {key: local_benchmark[key]['solution']
                               for key in common_targets}

        # create dataframes of local benchmark, and concat them to routing so they are comparable
        df_local_benchmark = pd.DataFrame(values_benchmark, index=['local_benchmark'])
        df_local_benchmark.columns = common_targets
        comparison = pd.concat([df_local_benchmark, routing_costs_common])

        # compare local benchmark and routing with each other --> get minimal value for each location
        comparison_only_min_values = comparison.idxmin()

        # get only such columns where local benchmark is not lower. Only such have to be considered
        index_lower_than_benchmark \
            = comparison_only_min_values[comparison_only_min_values != 'local_benchmark']

        dict_updated_values \
            = {ind: {'total_costs': routing_costs_common.at[index_lower_than_benchmark.at[ind], ind],
                     'solution': None}
               for ind in index_lower_than_benchmark.index}

        # update local benchmark with the new values
        local_benchmark.update(dict_updated_values)

        # with the updated benchmark, some solutions might be obsolete as they achieve only higher costs
        obsolete_solutions += [solutions_benchmark[key]
                               for key in index_lower_than_benchmark.index
                               if solutions_benchmark[key] is not None]

    # not common keys are keys, which are not in local benchmark but in routing costs
    not_common_targets = list(set(routing_costs.columns) - set(list(local_benchmark.keys())))

    if not_common_targets:  # only if not common targets exist
        # print(not_common_targets)

        # get not common routing costs
        routing_costs_not_common = routing_costs[not_common_targets]

        # get start location / end location combinations with minimal costs
        comparison_only_min_values = routing_costs_not_common.idxmin()

        # create new dictionary with all new values
        try:
            dict_new_values \
                = {ind: {'total_costs': routing_costs_not_common.at[comparison_only_min_values.at[ind], ind],
                         'solution': None}
                   for ind in comparison_only_min_values.index}
        except:
            print(routing_costs_not_common)
            print(comparison_only_min_values)
            for ind in comparison_only_min_values.index:
                print(ind)
                print(comparison_only_min_values.at[ind])
                print(routing_costs_not_common.at[comparison_only_min_values.at[ind], ind])
            dict_new_values \
                = {ind: {'total_costs': routing_costs_not_common.at[comparison_only_min_values.at[ind], ind],
                         'solution': None}
                   for ind in comparison_only_min_values.index}

        # attach created dictionary to local benchmark
        local_benchmark.update(dict_new_values)

    return local_benchmark, obsolete_solutions


def update_benchmark_based_on_historic_paths(benchmark, historic_paths, solutions):

    import math
    from py_expression_eval import Parser

    parser = Parser()

    for s in solutions:

        current_location = s.get_current_location()
        current_commodity = s.get_current_commodity_name()
        current_total_costs = s.get_total_costs()

        key = (current_location, current_commodity)

        if (current_location, current_commodity) in [*historic_paths.keys()]:

            costs = historic_paths[key]['costs']
            final = historic_paths[key]['final']

            for i, cost_addition in enumerate(costs):

                is_final = final[i]

                expr = parser.parse(cost_addition.replace('x', str(current_total_costs)))
                total_costs = expr.evaluate({})

                if math.ceil(total_costs*100)/100 <= benchmark:
                    # update benchmark based on projection of costs
                    # --> allow some tolerance as it is not a real solution
                    benchmark = math.ceil(total_costs*100)/100

    return benchmark, solutions
