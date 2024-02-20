import math

import pandas as pd

from _helpers import calc_distance_single_to_single, calc_distance_list_to_single


def adjust_production_costs_of_commodities(location_data, data):

    # load commodities, means of transport etc. already here
    # Commodities at start depend on data given. Get production costs of they exist at start
    commodities_at_start = [i for i in location_data.index
                            if i not in ['start_lon', 'start_lat', 'destination_lat', 'destination_lon',
                                         'target_commodity', 'country_start', 'continent_start',
                                         'country_destination', 'continent_destination',
                                         'distance_to_final_destination']
                            if i != 'N/A']

    # set production costs based on location data
    for c in commodities_at_start:
        c_object = data['commodities']['commodity_objects'][c]
        c_object.set_production_costs(float(location_data.at[c]))

    for c in data['commodities']['commodity_objects'].keys():
        c_object = data['commodities']['commodity_objects'][c]
        if c_object.get_production_costs() is None:
            lowest_costs = math.inf
            for c_at_start in commodities_at_start:
                c_at_start = data['commodities']['commodity_objects'][c_at_start]
                if c_at_start.get_conversion_options_specific_commodity(c_object.get_name()):
                    conversion_costs_to_c = c_at_start.get_conversion_costs_specific_commodity(c_object.get_name())
                    conversion_loss_of_educt = c_at_start.get_conversion_loss_of_educt_specific_commodity(c_object.get_name())
                    lowest_costs_commodity \
                        = (c_at_start.get_production_costs() + conversion_costs_to_c) / conversion_loss_of_educt

                    if lowest_costs_commodity < lowest_costs:
                        lowest_costs = lowest_costs_commodity

            c_object.set_production_costs(lowest_costs)
            data['commodities']['commodity_objects'][c] = c_object

    return data


def create_solutions_based_on_commodities_at_start(data):
    solutions = pd.DataFrame(columns=['starting_latitude', 'starting_longitude', 'previous_solution',
                                        'latitude', 'longitude', 'current_commodity',
                                        'all_previous_commodities', 'current_commodity_object', 'current_total_costs',
                                        'all_previous_total_costs', 'current_transportation_costs',
                                        'all_previous_transportation_costs', 'current_conversion_costs',
                                        'all_previous_conversion_costs', 'all_previous_solutions',
                                        'current_transport_mean',
                                        'all_previous_transport_means', 'current_node',
                                        'all_previous_nodes', 'current_infrastructure',
                                        'all_previous_infrastructure', 'current_distance', 'all_previous_distances',
                                        'current_continent', 'distance_to_final_destination', 'solution_index',
                                        'comparison_index'])

    starting_location = data['start']['location']
    starting_continent = data['start']['continent']
    destination_location = data['destination']['location']
    commodities = data['commodities']['commodity_objects']

    distance_to_final_destination = calc_distance_single_to_single(starting_location.y, starting_location.x,
                                                                   destination_location.y, destination_location.x)

    comparison_index = []
    solution_number = 0
    for c in commodities.keys():
        c_object = commodities[c]

        solution_index = 'S' + str(solution_number)
        solution_number += 1

        solutions.loc[solution_index, 'starting_latitude'] = starting_location.y
        solutions.loc[solution_index, 'starting_longitude'] = starting_location.x
        solutions.loc[solution_index, 'latitude'] = starting_location.y
        solutions.loc[solution_index, 'longitude'] = starting_location.x
        solutions.loc[solution_index, 'previous_solution'] = None
        solutions.loc[solution_index, 'current_commodity'] = c_object.get_name()
        solutions.loc[solution_index, 'current_commodity_object'] = c_object
        solutions.loc[solution_index, 'current_continent'] = starting_continent
        solutions.loc[solution_index, 'current_total_costs'] = c_object.get_production_costs()
        solutions.loc[solution_index, 'current_transportation_costs'] = 0
        solutions.loc[solution_index, 'current_conversion_costs'] = 0
        solutions.loc[solution_index, 'current_transport_mean'] = None
        solutions.loc[solution_index, 'current_infrastructure'] = None
        solutions.loc[solution_index, 'current_node'] = 'Start'
        solutions.loc[solution_index, 'current_distance'] = 0
        solutions.loc[solution_index, 'solution_index'] = solution_index
        solutions.loc[solution_index, 'all_previous_commodities'] = [c_object.get_name()]
        solutions.loc[solution_index, 'all_previous_total_costs'] = [c_object.get_production_costs()]
        solutions.loc[solution_index, 'taken_routes'] = [c_object.get_name()]

        comparison_index.append(('Start', c_object.get_name()))

    solutions['comparison_index'] = comparison_index

    solutions['all_previous_solutions'] = [[s] for s in solutions.index]
    solutions['all_previous_transportation_costs'] = [[] for s in solutions.index]
    solutions['all_previous_conversion_costs'] = [[] for s in solutions.index]
    solutions['all_previous_transport_means'] = [[None] for s in solutions.index]
    solutions['all_previous_infrastructure'] = [[] for s in solutions.index]
    solutions['all_previous_nodes'] = [['Start'] for s in solutions.index]
    solutions['all_previous_distances'] = [[0] for s in solutions.index]
    solutions['distance_to_final_destination'] = [distance_to_final_destination for s in solutions.index]

    return solutions, solution_number


def check_for_inaccessibility_and_at_destination(data, configuration, complete_infrastructure, location_data, k,
                                                 solutions):

    continue_processing = True

    starting_location = data['start']['location']
    destination_location = data['destination']['location']
    final_commodities = data['commodities']['final_commodities']

    # first, check if based on configuration infrastructure is reachable from start and destination
    complete_infrastructure['distance_to_start'] \
        = calc_distance_list_to_single(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
                                       starting_location.y, starting_location.x)
    complete_infrastructure['distance_to_destination'] \
        = calc_distance_list_to_single(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
                                       destination_location.y, destination_location.x)

    max_length = max(configuration['max_length_road'],
                     configuration['max_length_new_segment']) / configuration['no_road_multiplier']

    distance_to_start = complete_infrastructure[complete_infrastructure['distance_to_start']
                                                <= max_length].index

    distance_to_destination = complete_infrastructure[complete_infrastructure['distance_to_destination']
                                                      <= max_length].index

    distance_to_destination.drop(['Destination'])

    if (len(distance_to_start) == 0) & (len(distance_to_destination) == 0):
        print(str(k) + ': Parameters limit the access to infrastructure')

        result = pd.Series(['no benchmark', starting_location.y, starting_location.x],
                           index=['status', 'latitude', 'longitude'])
        result.to_csv(configuration['path_results'] + str(k) + '_no_benchmark.csv')
        continue_processing = False

    reachable_from_start = complete_infrastructure[complete_infrastructure['reachable_from_start']].index
    reachable_from_destination = complete_infrastructure[complete_infrastructure['reachable_from_destination']].index

    if (len(reachable_from_start) == 0) | (len(reachable_from_destination) == 0):
        print(str(k) + ': No infrastructure on same land mass as start or destination')

        result = pd.Series(['no benchmark', starting_location.y, starting_location.x],
                           index=['status', 'latitude', 'longitude'])
        result.to_csv(configuration['path_results'] + str(k) + '_no_benchmark.csv')
        continue_processing = False

    # if location is already at destination --> return cheapest solution if right commodity
    location_data.at['distance_to_final_destination'] \
        = calc_distance_single_to_single(location_data.at['start_lat'],
                                         location_data.at['start_lon'],
                                         complete_infrastructure.at['Destination', 'latitude'],
                                         complete_infrastructure.at['Destination', 'longitude'])

    if location_data.at['distance_to_final_destination'] < configuration['to_final_destination_tolerance']:
        cheapest_option = math.inf
        chosen_solution = None
        for s in solutions:
            if solutions.at[s, 'current_commodity'] in final_commodities:
                if solutions.at[s, 'current_total_costs'] < cheapest_option:
                    cheapest_option = solutions.at[s, 'current_total_costs']
                    chosen_solution = solutions.loc[s, :].copy()

        chosen_solution.at['status'] = 'complete'
        chosen_solution.to_csv(configuration['path_results'] + str(k) + '_final_solution.csv')
        print(str(k) + ' is already in tolerance to destination')
        continue_processing = False

    return continue_processing


def create_new_solutions_based_on_conversion(solutions, data, solution_number, benchmark):

    index = []

    total_costs = []
    all_previous_total_costs = []

    current_conversion_costs = []
    all_previous_conversion_costs = []

    all_previous_transportation_costs = []

    current_commodity = []
    previous_commodity = []
    all_previous_commodities = []
    current_commodity_object = []

    starting_latitude = []
    starting_longitude = []
    longitude = []
    latitude = []

    current_infrastructure = []
    all_previous_infrastructure = []

    current_transport_mean = []
    all_previous_transport_means = []

    current_node = []
    all_previous_nodes = []

    all_previous_solutions = []

    current_distance = []
    all_previous_distances = []

    continent = []
    distance_to_final_destination = []

    previous_solutions = []

    taken_route = []

    for c_start in solutions['current_commodity'].unique():

        c_start_df = solutions[solutions['current_commodity'] == c_start]

        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()

        # # if all conversions are higher than benchmark, ignore all conversions
        # c_start_conversion_costs = c_start_object.get_conversion_costs()
        # if all(i > benchmark for i in c_start_conversion_costs.values() if type(i) != str):
        #     continue

        for c_transported in [*data['commodities']['commodity_objects'].keys()]:
            c_transported_object = data['commodities']['commodity_objects'][c_transported]
            if c_start != c_transported:
                if c_start_conversion_options[c_transported]:

                    # if conversions costs are already higher than benchmark, ignore conversion
                    if c_start_object.get_conversion_costs_specific_commodity(c_transported) > benchmark:
                        continue

                    len_index = len(c_start_df.index)

                    costs = \
                        (c_start_df['current_total_costs']
                         + c_start_object.get_conversion_costs_specific_commodity(c_transported)) \
                        / c_start_object.get_conversion_loss_of_educt_specific_commodity(c_transported) \

                    total_costs += costs.tolist()
                    previous_commodity += [c_transported] * len_index

                    costs = costs - c_start_df['current_total_costs']
                    current_conversion_costs += costs.tolist()

                    taken_route += [(c_start, c_transported)] * len_index
                else:
                    continue
            else:
                len_index = len(c_start_df.index)

                total_costs += c_start_df['current_total_costs'].tolist()
                current_conversion_costs += [0] * len_index

                taken_route += [(c_start, c_start)] * len_index

            len_index = len(c_start_df.index)

            all_previous_solutions += c_start_df['all_previous_solutions'].tolist()

            current_commodity += [c_transported] * len_index
            current_commodity_object += [c_transported_object] * len_index
            all_previous_commodities += c_start_df['all_previous_commodities'].tolist()

            continent += c_start_df['current_continent'].values.tolist()

            starting_latitude += c_start_df['starting_latitude'].values.tolist()
            starting_longitude += c_start_df['starting_longitude'].values.tolist()
            latitude += c_start_df['latitude'].values.tolist()
            longitude += c_start_df['longitude'].values.tolist()
            distance_to_final_destination += c_start_df['distance_to_final_destination'].values.tolist()

            current_infrastructure += c_start_df['current_infrastructure'].values.tolist()
            all_previous_infrastructure += c_start_df['all_previous_infrastructure'].values.tolist()

            current_transport_mean += c_start_df['current_transport_mean'].values.tolist()
            all_previous_transport_means += c_start_df['all_previous_transport_means'].values.tolist()

            current_node += c_start_df['current_node'].values.tolist()
            all_previous_nodes += c_start_df['all_previous_nodes'].values.tolist()

            current_distance += [0] * len_index
            all_previous_distances += c_start_df['all_previous_distances'].values.tolist()

            all_previous_transportation_costs += c_start_df['all_previous_transportation_costs'].values.tolist()
            all_previous_conversion_costs += c_start_df['all_previous_conversion_costs'].values.tolist()
            all_previous_total_costs += c_start_df['all_previous_total_costs'].values.tolist()

            previous_solutions += c_start_df.index.values.tolist()

    current_transportation_costs = [0 for i in range(len(total_costs))]
    comparison_index = [(current_node[n], current_commodity[n]) for n in range(len(current_node))]
    index += ['S' + str(solution_number + i) for i in range(len(total_costs))]
    solution_number += len(total_costs)

    solutions_dict = {'latitude': latitude,
                      'longitude': longitude,

                      'current_commodity': current_commodity,
                      'current_commodity_object': current_commodity_object,

                      'current_total_costs': total_costs,

                      'current_transportation_costs': current_transportation_costs,

                      'current_conversion_costs': current_conversion_costs,

                      'current_transport_mean': current_transport_mean,

                      'current_node': current_node,

                      'current_infrastructure': current_infrastructure,

                      'current_distance': current_distance,

                      'current_continent': continent,
                      'distance_to_final_destination': distance_to_final_destination,

                      'solution_index': index,
                      'comparison_index': comparison_index,

                      'previous_solution': previous_solutions,

                      'taken_route': taken_route}

    solutions = pd.DataFrame(solutions_dict, index=index)

    return solutions, solution_number


def postprocessing_solutions(solutions, old_solutions):
    # moves all current parameters to previous

    if 'all_previous_infrastructure' in solutions.columns:
        solutions = solutions.drop(columns=['all_previous_infrastructure'])

    columns_to_keep = ['all_previous_transport_means', 'all_previous_infrastructure',
                       'all_previous_nodes', 'all_previous_solutions',
                       'all_previous_distances', 'all_previous_transportation_costs', 'all_previous_conversion_costs',
                       'all_previous_total_costs', 'all_previous_commodities', 'solution_index',
                       'starting_latitude', 'starting_longitude', 'taken_routes']
    old_solutions = old_solutions[columns_to_keep]

    solutions = pd.merge(solutions, old_solutions, left_on='previous_solution', right_on='solution_index', how='left')
    solutions.rename(columns={'solution_index_x': 'solution_index'}, inplace=True)
    solutions.index = solutions['solution_index'].tolist()

    solutions['all_previous_transport_means'] \
        = solutions.apply(lambda row: row['all_previous_transport_means'] + [row['current_transport_mean']], axis=1)

    solutions['all_previous_infrastructure'] \
        = solutions.apply(lambda row: row['all_previous_infrastructure'] + [row['current_infrastructure']], axis=1)

    solutions['all_previous_nodes'] \
        = solutions.apply(lambda row: row['all_previous_nodes'] + [row['current_node']], axis=1)

    solutions['all_previous_solutions'] \
        = solutions.apply(lambda row: row['all_previous_solutions'] + [row['solution_index']], axis=1)

    solutions['all_previous_distances'] \
        = solutions.apply(lambda row: row['all_previous_distances'] + [row['current_distance']], axis=1)

    solutions['all_previous_transportation_costs'] \
        = solutions.apply(lambda row: row['all_previous_transportation_costs'] + [row['current_transportation_costs']], axis=1)

    solutions['all_previous_conversion_costs'] \
        = solutions.apply(lambda row: row['all_previous_conversion_costs'] + [row['current_conversion_costs']], axis=1)

    solutions['all_previous_total_costs'] \
        = solutions.apply(lambda row: row['all_previous_total_costs'] + [row['current_total_costs']], axis=1)

    solutions['previous_commodity'] = solutions['current_commodity']
    solutions['all_previous_commodities'] \
        = solutions.apply(lambda row: row['all_previous_commodities'] + [row['current_commodity']], axis=1)

    solutions['taken_routes'] \
        = solutions.apply(lambda row: row['taken_routes'] + [row['taken_route']], axis=1)

    return solutions


def apply_local_benchmark(solutions, local_benchmarks, solutions_to_remove, update_local_benchmark=False):

    # update existing benchmarks
    solutions['old_index'] = solutions.index
    solutions.index = solutions['comparison_index']
    common_index = solutions.index.intersection(local_benchmarks.index)

    solution_df_subset = solutions.loc[common_index, :].copy()
    local_benchmarks_dict_subset = local_benchmarks.loc[common_index, :]

    # find all places where solution is more expensive than local benchmark --> remove solutions
    solutions_higher_benchmark = solution_df_subset[
        solution_df_subset['current_total_costs'] > local_benchmarks_dict_subset['total_costs']].index
    solutions.drop(index=solutions_higher_benchmark, inplace=True)

    if update_local_benchmark:
        # find all places where solution is cheaper than local benchmark --> update local benchmark
        solutions_lower_benchmark = solution_df_subset[
            solution_df_subset['current_total_costs'] < local_benchmarks_dict_subset['total_costs']].index

        solutions_to_remove += local_benchmarks.loc[solutions_lower_benchmark, 'solution'].tolist()

        local_benchmarks.loc[common_index, 'total_costs'] \
            = solutions.loc[solutions_lower_benchmark, 'current_total_costs']
        local_benchmarks.loc[common_index, 'solution'] \
            = solutions.loc[solutions_lower_benchmark, 'solution_index']

        # add new benchmarks
        new_benchmarks = solutions.index.difference(local_benchmarks.index)
        new_benchmarks_df = pd.DataFrame(
            {'total_costs': solutions.loc[new_benchmarks, 'current_total_costs'],
             'solution': solutions.loc[new_benchmarks, 'solution_index']},
            index=new_benchmarks)
        local_benchmarks = pd.concat([local_benchmarks, new_benchmarks_df])

    solutions.index = solutions['old_index']
    solutions.drop(columns=['old_index'], inplace=True)

    return solutions, local_benchmarks, solutions_to_remove


