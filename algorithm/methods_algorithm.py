import math
import os
import yaml

import pandas as pd
import numpy as np

from shapely.geometry import Point
from shapely.ops import nearest_points

from algorithm.methods_geographic import calc_distance_single_to_single, calc_distance_list_to_single
from algorithm.object_commodity import create_commodity_objects


def prepare_commodities(config_file, location_data, data):
    """
    This method loads the techno economic data and calculates conversion costs and conversion efficiencies for each
    location and the transportation costs for each transport mean

    @param dict config_file: dictionary with all configuration
    @param pandas.DataFrame location_data: location specific levelized costs
    @param dict data: dictionary with all common data
    @return: commodity objects and list with commodity names
    """
    path_data = config_file['project_folder_path'] + 'raw_data/'
    yaml_file = open(path_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    yaml_file = open(path_data + 'techno_economic_data_transportation.yaml')
    techno_economic_data_transportation = yaml.load(yaml_file, Loader=yaml.FullLoader)

    conversion_costs_and_efficiencies = data['conversion_costs_and_efficiencies']

    # remove all commodities which do not have conversions and are not final commodities -> dead end
    config_file = config_file.copy()
    config_file['available_commodity'] = [
        commodity for commodity in config_file['available_commodity']
        if (commodity in config_file['target_commodity'])
        or techno_economic_data_conversion[commodity]['potential_conversions']
    ]

    # get commodities and associated data
    commodities, commodity_names \
        = create_commodity_objects(location_data, conversion_costs_and_efficiencies, techno_economic_data_conversion,
                                   techno_economic_data_transportation, config_file)

    return commodities, commodity_names


def create_branches_based_on_commodities_at_start(data):
    """
    Based on commodities, calculate starting branches for each commodity

    @param dict data: dictionary containing commodities
    @return: dataframe with starting branches and integer representing current branch number
    """

    branches = pd.DataFrame(columns=['starting_latitude', 'starting_longitude', 'previous_branch',
                                     'latitude', 'longitude', 'current_commodity',
                                     'all_previous_commodities', 'current_commodity_object', 'current_total_costs',
                                     'all_previous_total_costs', 'current_transportation_costs',
                                     'all_previous_transportation_costs', 'current_conversion_costs',
                                     'all_previous_conversion_costs', 'all_previous_branches',
                                     'current_transport_mean',
                                     'all_previous_transport_means', 'current_node',
                                     'all_previous_nodes', 'current_infrastructure',
                                     'all_previous_infrastructure', 'current_distance', 'all_previous_distances',
                                     'current_continent', 'distance_to_final_destination', 'branch_index',
                                     'comparison_index', 'taken_routes'])

    branches['taken_routes'] = branches['taken_routes'].astype(object)

    starting_location = data['start']['location']
    starting_continent = data['start']['continent']
    destination_location = data['destination']['location']
    commodities = data['commodities']['commodity_objects']

    if isinstance(destination_location, Point):
        distance_to_final_destination = calc_distance_single_to_single(starting_location.y, starting_location.x,
                                                                       destination_location.y, destination_location.x)
    else:
        # destination is polygon -> each infrastructure has different closest point to destination
        infrastructure_in_destination = data['destination']['infrastructure']

        distance_to_final_destination \
            = calc_distance_list_to_single(infrastructure_in_destination['latitude'],
                                           infrastructure_in_destination['longitude'],
                                           starting_location.y, starting_location.x)

        distance_to_final_destination = distance_to_final_destination.min()

    comparison_index = []
    branch_number = 0
    for c in commodities.keys():
        c_object = commodities[c]

        branch_index = 'S' + str(branch_number)
        branch_number += 1

        import numpy as np

        branches.loc[branch_index, 'starting_latitude'] = starting_location.y
        branches.loc[branch_index, 'starting_longitude'] = starting_location.x
        branches.loc[branch_index, 'latitude'] = starting_location.y
        branches.loc[branch_index, 'longitude'] = starting_location.x
        branches.loc[branch_index, 'previous_branch'] = None
        branches.loc[branch_index, 'current_commodity'] = c_object.get_name()
        branches.loc[branch_index, 'current_commodity_object'] = c_object
        branches.loc[branch_index, 'current_continent'] = starting_continent
        branches.loc[branch_index, 'current_total_costs'] = c_object.get_production_costs()
        branches.loc[branch_index, 'current_transportation_costs'] = 0
        branches.loc[branch_index, 'current_conversion_costs'] = 0
        branches.loc[branch_index, 'current_transport_mean'] = None
        branches.loc[branch_index, 'current_infrastructure'] = None
        branches.loc[branch_index, 'current_node'] = 'Start'
        branches.loc[branch_index, 'current_distance'] = 0
        branches.loc[branch_index, 'branch_index'] = branch_index
        branches.loc[branch_index, 'all_previous_commodities'] = [c_object.get_name()]
        branches.loc[branch_index, 'all_previous_total_costs'] = [c_object.get_production_costs()]
        branches.at[branch_index, 'taken_routes'] = [(c_object.get_name(), c_object.get_starting_efficiency())]
        branches.loc[branch_index, 'total_efficiency'] = c_object.get_starting_efficiency()
        branches.loc[branch_index, 'destination'] = destination_location

        comparison_index.append(('Start', c_object.get_name()))

    branches['comparison_index'] = comparison_index

    branches['all_previous_branches'] = [[b] for b in branches.index]
    branches['all_previous_transportation_costs'] = [[] for b in branches.index]
    branches['all_previous_conversion_costs'] = [[] for b in branches.index]
    branches['all_previous_transport_means'] = [[None] for b in branches.index]
    branches['all_previous_infrastructure'] = [[] for b in branches.index]
    branches['all_previous_nodes'] = [['Start'] for b in branches.index]
    branches['all_previous_distances'] = [[0] for b in branches.index]
    branches['distance_to_final_destination'] = [distance_to_final_destination for b in branches.index]

    return branches, branch_number


def check_for_inaccessibility_and_at_destination(data, configuration, complete_infrastructure, location_integer,
                                                 branches):
    """
    Method to assess if we should go into the branch process

    will return False if:
    - start location is on a landmass where there is no infrastructure --> will save empty dataframe for branch
    - start location is on a landmass where there is infrastructure but is not reachable as parameters (for example,
    max road distance to low) --> will save empty dataframe for branch
    - start location is within tolerance to destination --> check if right commodity and save dataframe with branch

    will return True else

    @param dict data: dictionary with common data
    @param dict configuration: dictionary with configuration
    @param pandas.DataFrame complete_infrastructure: dataframe with all infrastructure (ports, pipelines, destination)
    @param int location_integer: identification of current starting location
    @param pandas.DataFrame branches: dataframe with branches --> if at destination, we only need to convert the branches
    @return: boolean if we need to continue to process the branch
    """

    continue_processing = True

    starting_location = data['start']['location']
    destination_location = data['destination']['location']
    final_commodities = data['commodities']['final_commodities']

    # first, check if based on configuration infrastructure is reachable from start and destination
    max_length = max(configuration['max_length_road'],
                     configuration['max_length_new_segment']) / configuration['no_road_multiplier']

    distance_to_start = complete_infrastructure[complete_infrastructure['distance_to_start']
                                                <= max_length].index

    distance_to_destination = complete_infrastructure[complete_infrastructure['distance_to_destination']
                                                      <= max_length].index

    if 'Destination' in distance_to_destination:
        distance_to_destination.drop(['Destination'])

    if (len(distance_to_start) == 0) & (len(distance_to_destination) == 0):
        print(str(location_integer) + ': Parameters limit the access to infrastructure')

        result = pd.Series(['no benchmark', starting_location.y, starting_location.x],
                           index=['status', 'latitude', 'longitude'])
        result.to_csv(configuration['path_results'] + str(location_integer) + '_no_benchmark.csv')
        continue_processing = False

    reachable_from_start = complete_infrastructure[complete_infrastructure['reachable_from_start']].index
    reachable_from_destination = complete_infrastructure[complete_infrastructure['reachable_from_destination']].index

    if continue_processing & ((len(reachable_from_start) == 0) | (len(reachable_from_destination) == 0)):
        print(str(location_integer) + ': No infrastructure on same land mass as start or destination')

        result = pd.Series(['no benchmark', starting_location.y, starting_location.x],
                           index=['status', 'latitude', 'longitude'])
        result.to_csv(configuration['path_results'] + 'location_results/' + str(location_integer) + '_no_benchmark.csv')
        continue_processing = False

    # if location is already at destination --> return cheapest branch if right commodity
    if isinstance(destination_location, Point):
        min_distance_to_destination \
            = calc_distance_single_to_single(starting_location.y, starting_location.x,
                                             destination_location.y, destination_location.x)
    else:
        if starting_location.within(destination_location):
            min_distance_to_destination = 0
        else:
            # destination is polygon -> each infrastructure has different closest point to destination
            infrastructure_in_destination = data['destination']['infrastructure']

            complete_infrastructure.loc[infrastructure_in_destination.index, 'distance_to_destination'] = 0

            distances_to_destination \
                = calc_distance_list_to_single(infrastructure_in_destination['latitude'],
                                               infrastructure_in_destination['longitude'],
                                               starting_location.y, starting_location.x)

            min_distance_to_destination = distances_to_destination.min()

    if continue_processing & (min_distance_to_destination < configuration['to_final_destination_tolerance']):
        cheapest_option = math.inf
        chosen_branch = None
        for s in branches.index:
            if branches.at[s, 'current_commodity'] in final_commodities:
                if branches.at[s, 'current_total_costs'] < cheapest_option:
                    cheapest_option = branches.at[s, 'current_total_costs']
                    chosen_branch = branches.loc[s, :].copy()

        chosen_branch.at['status'] = 'complete'
        chosen_branch.at['solving_time'] = 0
        chosen_branch.to_csv(configuration['path_results'] + 'location_results/' + str(location_integer) + '_final_solution.csv')
        print(str(location_integer) + ' is already in tolerance to destination')
        continue_processing = False

    # check if production costs are math.inf --> no potential at location
    cheapest_option = math.inf
    for s in branches.index:
        if branches.at[s, 'current_total_costs'] < cheapest_option:
            cheapest_option = branches.at[s, 'current_total_costs']

    if continue_processing & (cheapest_option == math.inf):
        print(str(location_integer) + ' has no production potential')
        result = pd.Series(['no potential', starting_location.y, starting_location.x],
                           index=['status', 'latitude', 'longitude'])
        result.to_csv(configuration['path_results'] + 'location_results/' + str(location_integer) + '_no_potential.csv')
        continue_processing = False

    return continue_processing


def create_new_branches_based_on_conversion(branches, data, branch_number, benchmarks):
    """
    Iterates through all commodities and creates new branches based on conversions from current branches

    @param pandas.DataFrame branches: dataframe with current branches
    @param dict data: dictionary with common data
    @param int branch_number: current branch number
    @param dict benchmarks: current benchmarks
    @return: dataframe with new branches based on conversion and new branch number
    """

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

    all_previous_branches = []

    current_distance = []
    all_previous_distances = []

    continent = []
    distance_to_final_destination = []

    all_destinations = []

    previous_branches = []

    taken_route = []

    efficiencies = []

    # conversion from c_start to c_end
    for c_start in branches['current_commodity'].unique():

        c_start_df = branches[branches['current_commodity'] == c_start]

        c_start_object = data['commodities']['commodity_objects'][c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()

        for c_end in [*data['commodities']['commodity_objects'].keys()]:
            c_transported_object = data['commodities']['commodity_objects'][c_end]
            if c_start != c_end:
                if c_start_conversion_options[c_end]:

                    # if min conversions costs are already higher than benchmark, ignore conversion
                    min_conversion_costs \
                        = c_start_object.get_conversion_costs_specific_commodity(c_start_df['current_node'],
                                                                                 c_end).min()

                    if min_conversion_costs > benchmarks[c_end]:
                        continue

                    len_index = len(c_start_df.index)

                    # get conversion costs and conversion efficiency for all locations
                    conversion_costs = c_start_object.get_conversion_costs_specific_commodity(
                        c_start_df['current_node'], c_end)
                    conversion_costs.index = c_start_df.index

                    conversion_efficiency = c_start_object.get_conversion_efficiency_specific_commodity(
                        c_start_df['current_node'], c_end)
                    conversion_efficiency.index = c_start_df.index

                    # calculate costs
                    costs = (c_start_df['current_total_costs'] + conversion_costs) / conversion_efficiency

                    total_costs += costs.tolist()
                    previous_commodity += [c_end] * len_index

                    costs = costs - c_start_df['current_total_costs']
                    current_conversion_costs += costs.tolist()

                    efficiencies += [conversion_efficiency.loc[i] * c_start_df.loc[i, 'total_efficiency'] for i in c_start_df.index]

                    taken_route += [(c_start, c_end, conversion_efficiency.loc[i]) for i in conversion_efficiency.index]
                else:
                    continue
            else:
                # if c_start is c_end, don't apply conversion and use old branches as new branches without adding costs
                len_index = len(c_start_df.index)

                total_costs += c_start_df['current_total_costs'].tolist()
                current_conversion_costs += [0] * len_index

                taken_route += [(c_start, c_start, 1)] * len_index

                efficiencies += c_start_df['total_efficiency'].tolist()

            # get other necessary information for later processing
            len_index = len(c_start_df.index)

            all_previous_branches += c_start_df['all_previous_branches'].tolist()

            current_commodity += [c_end] * len_index
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

            all_destinations += c_start_df['destination'].values.tolist()

            previous_branches += c_start_df.index.values.tolist()

    # convert all data to dictionary and process dictionary to new branch dataframe
    current_transportation_costs = [0 for i in range(len(total_costs))]
    comparison_index = [(current_node[n], current_commodity[n]) for n in range(len(current_node))]
    index += ['S' + str(branch_number + i) for i in range(len(total_costs))]
    branch_number += len(total_costs)

    branches_dict = {'latitude': latitude,
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

                     'branch_index': index,
                     'comparison_index': comparison_index,

                     'previous_branch': previous_branches,

                     'taken_route': taken_route,

                     'total_efficiency': efficiencies,

                     'destination': all_destinations}

    branches = pd.DataFrame(branches_dict, index=index)

    return branches, branch_number


def postprocessing_branches(branches, old_branches):
    """
    Past information is stored as lists (e.g., list with previously visited nodes). Add current information to these
    lists

    @param pandas.DataFrame branches: dataframe with new branches
    @param pandas.DataFrame old_branches: dataframe with old branches
    @return: complete dataframe with new branches and all information
    """

    if 'all_previous_infrastructure' in branches.columns:
        branches = branches.drop(columns=['all_previous_infrastructure'])

    columns_to_keep = ['all_previous_transport_means', 'all_previous_infrastructure',
                       'all_previous_nodes', 'all_previous_branches',
                       'all_previous_distances', 'all_previous_transportation_costs', 'all_previous_conversion_costs',
                       'all_previous_total_costs', 'all_previous_commodities', 'branch_index',
                       'starting_latitude', 'starting_longitude', 'taken_routes', 'total_efficiency', 'destination']
    old_branches = old_branches[columns_to_keep]

    branches = pd.merge(branches, old_branches, left_on='previous_branch', right_on='branch_index', how='left')
    branches.rename(columns={'branch_index_x': 'branch_index'}, inplace=True)
    branches.index = branches['branch_index'].tolist()

    branches['all_previous_transport_means'] \
        = branches.apply(lambda row: row['all_previous_transport_means'] + [row['current_transport_mean']], axis=1)

    branches['all_previous_infrastructure'] \
        = branches.apply(lambda row: row['all_previous_infrastructure'] + [row['current_infrastructure']], axis=1)

    branches['all_previous_nodes'] \
        = branches.apply(lambda row: row['all_previous_nodes'] + [row['current_node']], axis=1)

    branches['all_previous_branches'] \
        = branches.apply(lambda row: row['all_previous_branches'] + [row['branch_index']], axis=1)

    branches['all_previous_distances'] \
        = branches.apply(lambda row: row['all_previous_distances'] + [row['current_distance']], axis=1)

    branches['all_previous_transportation_costs'] \
        = branches.apply(lambda row: row['all_previous_transportation_costs'] + [row['current_transportation_costs']],
                         axis=1)

    branches['all_previous_conversion_costs'] \
        = branches.apply(lambda row: row['all_previous_conversion_costs'] + [row['current_conversion_costs']], axis=1)

    branches['all_previous_total_costs'] \
        = branches.apply(lambda row: row['all_previous_total_costs'] + [row['current_total_costs']], axis=1)

    branches['previous_commodity'] = branches['current_commodity']
    branches['all_previous_commodities'] \
        = branches.apply(lambda row: row['all_previous_commodities'] + [row['current_commodity']], axis=1)

    branches['taken_routes'] \
        = branches.apply(lambda row: row['taken_routes'] + [row['taken_route']], axis=1)

    branches.rename(columns={'total_efficiency_x': 'total_efficiency'}, inplace=True)

    branches.rename(columns={'destination_y': 'destination'}, inplace=True)

    columns_to_keep = []
    for c in branches.columns:
        if '_y' not in c:
            columns_to_keep.append(c)
    branches = branches[columns_to_keep]

    return branches


def apply_local_benchmark(branches, local_benchmarks, branches_to_remove, update_local_benchmark=False):
    """
    Since branches develop independently of each other, it might be possible that two branches end up at the same
    node with the same commodity but at different iterations. The local benchmarks dictionary contains this information.
    New branches will be removed if they visit a node which has been visited by another branch and the previous branch
    is cheaper. If it is the other way around (the new branch is cheaper), we can remove all branches which are a
    predecessor of the old branch.

    Additionally, local benchmark is updated if new branches visit nodes which have not been visited before

    @param branches: dataframe with branches
    @param local_benchmarks: dataframe with information on previously visited nodes and costs
    @param branches_to_remove: indicates which branches should be removed as predecessors are more expensive
    @param update_local_benchmark: boolean if local benchmark should be updated
    @return: dataframe with reduced branches, dataframe with updated local benchmarks, lists with branches to remove
    """

    # update existing benchmarks
    branches['old_index'] = branches.index
    branches.index = branches['comparison_index']
    common_index = branches.index.intersection(local_benchmarks.index)

    branch_df_subset = branches.loc[common_index, :].copy()
    local_benchmarks_dict_subset = local_benchmarks.loc[common_index, :]

    # find all places where branch is more expensive than local benchmark --> remove branches
    branches_higher_benchmark = branch_df_subset[
        branch_df_subset['current_total_costs'] > local_benchmarks_dict_subset['total_costs']].index
    branches.drop(index=branches_higher_benchmark, inplace=True)

    if update_local_benchmark:
        # find all places where branch is cheaper than local benchmark --> update local benchmark
        branches_lower_benchmark = branch_df_subset[
            branch_df_subset['current_total_costs'] < local_benchmarks_dict_subset['total_costs']].index

        branches_to_remove += local_benchmarks.loc[branches_lower_benchmark, 'branch'].tolist()

        local_benchmarks.loc[common_index, 'total_costs'] \
            = branches.loc[branches_lower_benchmark, 'current_total_costs']
        local_benchmarks.loc[common_index, 'branch'] \
            = branches.loc[branches_lower_benchmark, 'branch_index']

        # add new benchmarks
        new_benchmarks = branches.index.difference(local_benchmarks.index)
        new_benchmarks_df = pd.DataFrame(
            {'total_costs': branches.loc[new_benchmarks, 'current_total_costs'],
             'branch': branches.loc[new_benchmarks, 'branch_index']},
            index=new_benchmarks)
        local_benchmarks = pd.concat([local_benchmarks, new_benchmarks_df])

    # todo: branches to remove is currently not used

    branches.index = branches['old_index']
    branches.drop(columns=['old_index'], inplace=True)

    return branches, local_benchmarks, branches_to_remove


def assess_for_benchmark(data, configuration, benchmark, benchmarks, benchmark_locations, final_commodities, branches,
                         final_solution, complete_infrastructure):
    # check if branches are at destination
    at_destination = branches[branches['distance_to_final_destination']
                              <= configuration['to_final_destination_tolerance']]

    # branches at destination update the benchmarks of each commodity
    for commodity in at_destination['current_commodity'].unique():
        if commodity in final_commodities:
            com_ind = at_destination[at_destination['current_commodity'] == commodity].index
            com_df = at_destination.loc[com_ind, :]

            idx_min_costs = com_df['current_total_costs'].idxmax()
            min_costs = com_df.at[idx_min_costs, 'current_total_costs']
            min_costs_location = com_df.at[idx_min_costs, 'current_node']

            # max because all branches are at destination but conversion might be necessary which could change cheapest branch
            if min_costs < benchmarks[commodity]:
                benchmarks[commodity] = math.ceil(min_costs)
                benchmark_locations[commodity] = min_costs_location

    # now check for at destination and right commodity to
    if not at_destination.empty:
        at_destination_and_correct_commodity \
            = at_destination[at_destination['current_commodity'].isin(final_commodities)]

        if not at_destination_and_correct_commodity.empty:

            # assess for benchmark
            at_destination_and_correct_commodity['benchmark'] = at_destination_and_correct_commodity[
                'current_commodity'].map(benchmarks)
            at_destination_and_lower_benchmark_and_correct_commodity = at_destination_and_correct_commodity[
                at_destination_and_correct_commodity['current_total_costs'] <=
                at_destination_and_correct_commodity['benchmark']]

            if not at_destination_and_lower_benchmark_and_correct_commodity.empty:

                for commodity in at_destination_and_lower_benchmark_and_correct_commodity['current_commodity'].unique():
                    com_ind \
                        = at_destination_and_lower_benchmark_and_correct_commodity[
                        at_destination_and_correct_commodity['current_commodity'] == commodity].index
                    com_df = at_destination_and_lower_benchmark_and_correct_commodity.loc[com_ind, :]
                    min_costs = com_df['current_total_costs'].max()
                    benchmarks[commodity] = math.ceil(min_costs)

                    min_costs = com_df['current_total_costs'].min()
                    fuel_price = data['commodities']['strike_prices'][commodity]
                    if min_costs - fuel_price < benchmark:
                        benchmark = min_costs - fuel_price
                        min_costs_index = com_df['current_total_costs'].idxmin()

                        final_solution = branches.loc[min_costs_index, :].copy()

            # check which of the branches at destination and with right commodity can be removed and which one should be kept since
            # conversion at another infrastructure might be possible
            index_to_remove = at_destination_and_correct_commodity.index
            at_destination_locations = complete_infrastructure[complete_infrastructure['distance_to_destination'] <= configuration['to_final_destination_tolerance']].index

            for commodity_start in at_destination_and_correct_commodity['current_commodity'].unique():
                for commodity_target in data['commodities']['final_commodities']:
                    com_ind = at_destination_and_correct_commodity[at_destination_and_correct_commodity['current_commodity'] == commodity_start].index
                    com_df = at_destination_and_correct_commodity.loc[com_ind, :]

                    c_start_object = data['commodities']['commodity_objects'][commodity_start]
                    if c_start_object.get_conversion_options()[commodity_target]:
                        conversion_costs = c_start_object.get_conversion_costs().loc[at_destination_locations, commodity_target]
                        conversion_efficiency = c_start_object.get_conversion_efficiencies().loc[at_destination_locations, commodity_target]

                        row2 = conversion_costs.values
                        row3 = conversion_efficiency.values

                        conversion_costs = pd.DataFrame(np.tile(row2, (len(com_df.index), 1)), index=com_df.index, columns=conversion_costs.index)
                        conversion_efficiency = pd.DataFrame(np.tile(row3, (len(com_df.index), 1)), index=com_df.index, columns=conversion_efficiency.index)

                        costs = conversion_costs.add(com_df['current_total_costs'], axis=0).div(conversion_efficiency)

                        row_min = costs.min(axis=1) - data['commodities']['strike_prices'][commodity_target]
                        idx = row_min[row_min < benchmark].index

                        index_to_remove = [i for i in index_to_remove if i not in idx]  # as soon as one index achieves costs below benchmark --> keep

            # remove all branches which are at final destination with correct commodity
            branches.drop(index_to_remove, inplace=True)

            if final_solution is not None:
                at_destination = branches[branches['distance_to_final_destination'] <= configuration['to_final_destination_tolerance']]
                if not at_destination.empty:
                    at_destination_and_correct_commodity = at_destination[at_destination['current_commodity'].isin(final_commodities)]

                    if not at_destination_and_correct_commodity.empty:

                        # based on current benchmark, we can calculate for each branch if they can achieve this benchmark at all
                        index_to_remove = at_destination_and_correct_commodity.index
                        at_destination_locations \
                            = complete_infrastructure[complete_infrastructure['distance_to_destination'] <= configuration[ 'to_final_destination_tolerance']].index

                        for commodity_start in at_destination_and_correct_commodity['current_commodity'].unique():
                            commodity_target = final_solution.at['current_commodity']
                            com_ind = at_destination_and_correct_commodity[at_destination_and_correct_commodity['current_commodity'] == commodity_start].index
                            com_df = at_destination_and_correct_commodity.loc[com_ind, :]

                            c_start_object = data['commodities']['commodity_objects'][commodity_start]
                            if c_start_object.get_conversion_options()[commodity_target]:
                                conversion_costs = c_start_object.get_conversion_costs().loc[at_destination_locations, commodity_target]
                                conversion_costs = conversion_costs.replace([np.inf, -np.inf], np.nan).dropna()

                                conversion_efficiency = c_start_object.get_conversion_efficiencies().loc[conversion_costs.index, commodity_target]

                                row2 = conversion_costs.values
                                row3 = conversion_efficiency.values

                                conversion_costs = pd.DataFrame(np.tile(row2, (len(com_df.index), 1)), index=com_df.index,
                                                                columns=conversion_costs.index)
                                conversion_efficiency = pd.DataFrame(np.tile(row3, (len(com_df.index), 1)), index=com_df.index,
                                                                     columns=conversion_efficiency.index)

                                costs = benchmarks[commodity_target] / conversion_efficiency - conversion_costs

                                row_min = costs.min(axis=1)
                                idx = row_min[row_min < benchmarks[commodity_start]].index

                                index_to_remove = [i for i in index_to_remove if i not in idx]  # as soon as one index achieves costs below benchmark --> keep

                        # remove all branches which are at final destination with correct commodity
                        branches.drop(index_to_remove, inplace=True)

    # similar to the benchmark script, compare benchmarks with each other
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

                if (commodity_start in final_commodities) & (not math.isinf(benchmarks[commodity_start])) & (commodity_target in final_commodities):
                    # if target commodity is final commodity, then it's always possible to change between commodities

                    c_start_object = data['commodities']['commodity_objects'][commodity_start]
                    if c_start_object.get_conversion_options()[commodity_target]:

                        conversion_costs = c_start_object.get_conversion_costs().loc[benchmark_locations[commodity_start], commodity_target]
                        conversion_efficiency = c_start_object.get_conversion_efficiencies().loc[benchmark_locations[commodity_start], commodity_target]

                        costs = (benchmarks[commodity_start] + conversion_costs) / conversion_efficiency

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

    # benchmarks might have updated --> remove branches if higher than benchmarks
    branches['benchmark'] = branches['current_commodity'].map(benchmarks)
    branches = branches[branches['current_total_costs'] <= branches['benchmark']]

    return final_solution, benchmark, benchmarks, benchmark_locations, branches


