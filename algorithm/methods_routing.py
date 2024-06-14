import math
import gc

import pandas as pd
import numpy as np
import networkx as nx

from algorithm.methods_geographic import calc_distance_list_to_single, calc_distance_list_to_list
from algorithm.methods_cost_approximations import calculate_cheapest_option_to_closest_infrastructure, \
    calculate_cheapest_option_to_final_destination

import warnings
warnings.filterwarnings("ignore")


def get_complete_infrastructure(data):

    """
    Method to collect all ports, nodes and destination in one single dataframe

    @param dict data: dictionary with common data

    @return: pandas.DataFrame with all nodes, ports and destination
    """

    complete_infrastructure = pd.DataFrame()
    infrastructure_to_concat = []
    final_destination = data['destination']['location']
    for m in ['Road', 'Shipping', 'Pipeline_Gas', 'Pipeline_Liquid']:

        if m == 'Road':
            # Check final destination and add to option outside tolerance if applicable
            complete_infrastructure.loc['Destination', 'latitude'] = final_destination.y
            complete_infrastructure.loc['Destination', 'longitude'] = final_destination.x
            complete_infrastructure.loc['Destination', 'current_transport_mean'] = m
            complete_infrastructure.loc['Destination', 'graph'] = None
            complete_infrastructure.loc['Destination', 'continent'] = data['destination']['continent']

            continue

        # get all complete_infrastructure of current mean of transport
        if m == 'Shipping':

            # get all complete_infrastructure of current mean of transport
            shipping_infrastructure = data[m]['ports']
            shipping_infrastructure['current_transport_mean'] = m
            shipping_infrastructure['graph'] = None

            infrastructure_to_concat.append(shipping_infrastructure)

        else:
            networks = data[m].keys()
            for n in networks:
                network_infrastructure = data[m][n]['NodeLocations'].copy()
                network_infrastructure['current_transport_mean'] = m
                infrastructure_to_concat.append(network_infrastructure)

    complete_infrastructure = pd.concat([complete_infrastructure] + infrastructure_to_concat)

    # create common infrastructure column
    complete_infrastructure['infrastructure'] = complete_infrastructure.index
    graph_df = complete_infrastructure[complete_infrastructure['graph'].apply(lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) if isinstance(x, (list, float)) else False)]
    complete_infrastructure.loc[graph_df.index, 'infrastructure'] = complete_infrastructure.loc[graph_df.index, 'infrastructure']

    return complete_infrastructure


def process_out_tolerance_branches(complete_infrastructure, branches, configuration, iteration, data, benchmark,
                                   use_minimal_distance=False, limitation=None):

    """
    Method to assess potential transportation destinations via road or new pipelines based on current branches
    and all available infrastructure.

    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param pandas.DataFrame branches: DataFrame with current branches
    @param dict configuration: dictionary with configuration
    @param int iteration: current iteration
    @param dict data: dictionary with common data
    @param float benchmark: current benchmark
    @param bool use_minimal_distance: (optional) boolean to set if minimal distances are used to assess locations
    @param str limitation: (optional) determines which infrastructure can actually be used

    @return: pandas.DataFrame with new branches
    """

    tolerance_distance = configuration['tolerance_distance']
    max_length_new_segment = configuration['max_length_new_segment']
    max_length_road = configuration['max_length_road']
    no_road_multiplier = configuration['no_road_multiplier']
    build_new_infrastructure = configuration['build_new_infrastructure']

    if iteration == 0:
        # if iteration is 0, we don't make any preselection as we have only a very limited amount of branches
        # and calculating distances for these few branches is possible without long computation times

        # only use options which are actually reachable from start
        complete_infrastructure = complete_infrastructure[complete_infrastructure['reachable_from_start']]

        if limitation == 'no_pipeline_gas':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PG' not in i]
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'no_pipeline_liquid':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PL' not in i]
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        elif limitation == 'no_pipelines':
            reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i]
            distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                   complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                   branches['latitude'], branches['longitude'])

        else:  # don't limit infrastructure at all
            reduced_infrastructure_index = complete_infrastructure.index
            distances = calc_distance_list_to_list(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
                                                   branches['latitude'], branches['longitude'])

        road_transportation_costs = {}

        new_transportation_costs = {}

        branches_to_keep_road = []
        branches_to_keep_new = []

        # for each branch, assess if new pipelines or road is applicable based on current commodity
        road_distances = pd.DataFrame(distances.transpose(), index=reduced_infrastructure_index, columns=branches.index)
        for visited_infrastructure in branches.index:
            current_commodity_object = branches.at[visited_infrastructure, 'current_commodity_object']

            pipeline_applicable \
                = current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas') \
                | current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid')

            # check which transport mean was used in previous iteration
            was_not_road = branches.at[visited_infrastructure, 'current_transport_mean'] != 'Road'
            was_not_new = branches.at[visited_infrastructure, 'current_transport_mean'] not in ['New_Pipeline_Gas', 'New_Pipeline_Liquid']

            # we cannot use road or pipeline twice in a row
            road_applicable = current_commodity_object.get_transportation_options_specific_mean_of_transport('Road')
            road_applicable = road_applicable & was_not_road & was_not_new

            # check which infrastructure can be transported via road
            if road_applicable:
                # branches where last one was not road or new & commodity can be transported via road
                road_transportation_costs[visited_infrastructure]\
                    = current_commodity_object.get_transportation_costs_specific_mean_of_transport('Road')
                branches_to_keep_road.append(visited_infrastructure)
            else:
                # branches where the above does not allow new road but as infrastructure is within tolerance, we can
                # ignore transport mean as in this case we assume that we are already there
                in_tolerance_options = road_distances[visited_infrastructure]
                in_tolerance_options = in_tolerance_options[in_tolerance_options <= configuration['tolerance_distance']].index
                road_distances[visited_infrastructure] = math.nan
                road_distances.loc[in_tolerance_options, visited_infrastructure] = 0

                road_transportation_costs[visited_infrastructure] = 0
                branches_to_keep_road.append(visited_infrastructure)

            # we cannot use new pipeline or road pipeline twice in a row
            pipeline_applicable = pipeline_applicable & was_not_new & was_not_road

            # check if new pipelines are allowed and if so, which branches can use them
            if pipeline_applicable & build_new_infrastructure:
                if current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                    # gas pipeline
                    new_transportation_costs[visited_infrastructure]\
                        = current_commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Gas')
                else:
                    # oil pipeline
                    new_transportation_costs[visited_infrastructure]\
                        = current_commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Liquid')
                branches_to_keep_new.append(visited_infrastructure)

        # for road options, only keep branches where road is applicable
        road_distances = road_distances[branches_to_keep_road]
        road_transportation_costs = pd.Series(road_transportation_costs.values(), index=road_transportation_costs.keys())

        # remove road options where distance is above allowed configuration  # todo: hier müssen alle transporte von gleicher zur gleichen nfrasktrutur entfernt werden
        road_options = road_distances[road_distances <= max_length_road / no_road_multiplier]
        road_options = road_options.transpose().stack().dropna().reset_index()
        road_options.columns = ['previous_branch', 'current_node', 'current_distance']

        # for new pipeline options, only keep branches where new pipelines are allowed and are applicable
        if build_new_infrastructure:
            new_infrastructure_distances = pd.DataFrame(distances.transpose(), index=reduced_infrastructure_index,
                                                        columns=branches.index)

            new_infrastructure_distances = new_infrastructure_distances[branches_to_keep_new]
            new_transportation_costs = pd.Series(new_transportation_costs.values(), index=new_transportation_costs.keys())

            new_infrastructure_options \
                = new_infrastructure_distances[new_infrastructure_distances <= max_length_new_segment / no_road_multiplier]
            new_infrastructure_options = new_infrastructure_options.transpose().stack().dropna().reset_index()
            new_infrastructure_options.columns = ['previous_branch', 'current_node', 'current_distance']
        else:
            new_infrastructure_options = pd.DataFrame()

    else:
        # if iteration is not 0, there might be a large number of branches. Therefore, we need to preselect
        # potential branches

        road_transportation_costs = branches['road_transportation_costs']

        new_transportation_costs = branches['new_transportation_costs']

        minimal_distances = data['minimal_distances']

        all_road_distances = []
        all_new_distances = []

        if not use_minimal_distance:
            # if we don't use minimal distances, we have to assess all locations

            # if several branches with different commodities are at same location, we still need distances for this
            # location only once. Therefore, remove duplicates
            branches_no_duplicates = branches.drop_duplicates(subset=['current_node'], keep='first')

            if limitation == 'no_pipeline_gas':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PG' not in i]
                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            elif limitation == 'no_pipeline_liquid':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'PL' not in i]
                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            elif limitation == 'no_pipelines':
                reduced_infrastructure_index = [i for i in complete_infrastructure.index if 'H' in i] + ['Destination']
                distances = calc_distance_list_to_list(complete_infrastructure.loc[reduced_infrastructure_index, 'latitude'],
                                                       complete_infrastructure.loc[reduced_infrastructure_index, 'longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            else:
                reduced_infrastructure_index = complete_infrastructure.index
                distances = calc_distance_list_to_list(complete_infrastructure['latitude'], complete_infrastructure['longitude'],
                                                       branches_no_duplicates['latitude'],
                                                       branches_no_duplicates['longitude'])

            distances = pd.DataFrame(distances.transpose(),
                                     index=reduced_infrastructure_index,
                                     columns=branches_no_duplicates['current_node'])

        else:
            # Check distance not for all infrastructure but just closest one to assess, if the closest
            # infrastructure is already too expensive. This approach helps to remove branches

            branches_no_duplicates = branches.drop_duplicates(subset=['current_node'], keep='first')

            branches_no_duplicates['minimal_distance'] \
                = minimal_distances.loc[branches_no_duplicates['current_node'].tolist(), 'minimal_distance'].tolist()
            branches_no_duplicates['closest_node'] \
                = minimal_distances.loc[branches_no_duplicates['current_node'].tolist(), 'closest_node'].tolist()

            distances \
                = branches_no_duplicates.set_index([branches_no_duplicates['current_node'],
                                                    branches_no_duplicates['closest_node']])['minimal_distance'].unstack().transpose()

        # Instead of iterating over all visited infrastructure for all branches, we iterate over each visited
        # infrastructure once and check which branch has been there
        def flatten_list(nested_list):  # get single list of all previously visited nodes and ports
            flattened_list = []
            for element in nested_list:
                if isinstance(element, list):
                    flattened_list.extend(flatten_list(element))
                else:
                    flattened_list.append(element)
            return flattened_list
        all_previous_infrastructure = list(set(flatten_list(branches['all_previous_infrastructure'].tolist())))

        # for each already visited node or port, get all branches which have been there already
        branches_to_remove_based_on_visited_infrastructure = {}
        for visited_infrastructure in all_previous_infrastructure:
            if visited_infrastructure is not None:

                # check which branch has the visited infrastructure in all_previous_infrastructure
                affected_branches = branches[branches['all_previous_infrastructure'].apply(lambda x: visited_infrastructure in x)].index

                if 'PG' in visited_infrastructure:
                    branches_to_remove_based_on_visited_infrastructure[visited_infrastructure] \
                        = {'nodes': data['Pipeline_Gas'][visited_infrastructure]['NodeLocations'].index.tolist(),
                           'branches': affected_branches}

                elif 'PL' in visited_infrastructure:
                    branches_to_remove_based_on_visited_infrastructure[visited_infrastructure] \
                        = {'nodes': data['Pipeline_Liquid'][visited_infrastructure]['NodeLocations'].index.tolist(),
                           'branches': affected_branches}

                else:
                    branches_to_remove_based_on_visited_infrastructure[visited_infrastructure] \
                        = {'nodes': [visited_infrastructure],
                           'branches': affected_branches}

        # iterate over all commodities. Necessary to look at each commodity to check if applicable for road or
        # new pipeline and to get costs of transport
        for c in branches['current_commodity'].unique():
            c_branches = branches[branches['current_commodity'] == c]
            if c_branches.empty:
                continue

            # c_distance is the complete infrastructure (except limitations). We need to make it smaller
            c_distances = distances.copy()
            commodity_object = data['commodities']['commodity_objects'][c]

            # exchange current_node columns with corresponding branch names
            node_list = c_branches['current_node'].tolist()  # list is unique as there can be just one branch at node or port with the same commodity
            branch_list = c_branches.index.tolist()
            new_column_list = []
            columns_to_keep = []
            for n in distances.columns:
                if n in node_list:
                    new_column_list.append(branch_list[node_list.index(n)])
                    columns_to_keep.append(n)

            c_distances = c_distances[columns_to_keep]  # keep only the columns where current branches are
            c_distances.columns = new_column_list  # rename columns to respective branche index

            # some locations are within tolerance. These are processed separately as we don't need transportation
            c_distances_stacked = c_distances.copy().transpose().stack().dropna()
            in_tolerance_distances = c_distances_stacked[c_distances_stacked <= configuration['tolerance_distance']]
            if not in_tolerance_distances.empty:
                in_tolerance_distances.loc[:] = 0  # in tolerance means 0 distance
                all_road_distances.append(in_tolerance_distances)

            if commodity_object.get_transportation_options()['Road']:

                road_distances = c_distances.copy().transpose()
                columns = road_distances.columns  # get initial columns before new ones are added

                # remove all branches where road is not applicable (remove rows)
                road_distances['road_applicable'] = c_branches['Road_applicable'].tolist()
                road_distances = road_distances[road_distances['road_applicable']]

                # remove all options where max length exceeds distance (remove columns)
                max_length_road_costs \
                    = list((benchmark - c_branches['current_total_costs']) * 1000
                           / c_branches['road_transportation_costs'] / no_road_multiplier)
                max_length_road_list = [min(n, max_length_road / no_road_multiplier) for n in max_length_road_costs]
                road_distances['max_length'] = max_length_road_list

                mask = road_distances[columns].values > road_distances['max_length'].values[:, None]
                road_distances.loc[:, columns] = np.where(mask, np.nan, road_distances[columns].values)
                road_distances.drop(columns=['max_length', 'road_applicable'], inplace=True)

                # remove options based on previous used infrastructure
                # todo: das scheint weder bei Häfen noch bei Pipelines richtig zu funktionieren --> new pipelines zwischen gleichen grafen
                for visited_infrastructure in branches_to_remove_based_on_visited_infrastructure.keys():
                    affected_branches \
                        = list(set(road_distances.index.tolist()).intersection(branches_to_remove_based_on_visited_infrastructure[visited_infrastructure]['branches']))
                    road_distances.loc[affected_branches, branches_to_remove_based_on_visited_infrastructure[visited_infrastructure]['nodes']] = np.nan

                # drop all nodes which cannot be visited
                road_distances = road_distances.stack().dropna()
                road_distances = road_distances.apply(pd.to_numeric, errors='coerce').dropna()

                # todo: some values are b'' --> why?

                all_road_distances.append(road_distances)

            # create and process new infrastructure distances
            if (commodity_object.get_transportation_options()['Pipeline_Gas']
                    | commodity_object.get_transportation_options()['Pipeline_Liquid']) & build_new_infrastructure:

                new_distances = c_distances.copy().transpose()
                columns = new_distances.columns

                # add information before any change to distances is made
                new_distances['pipeline_gas_applicable'] = c_branches['Pipeline_Gas_applicable'].tolist()
                new_distances['pipeline_liquid_applicable'] = c_branches['Pipeline_Liquid_applicable'].tolist()
                new_distances['max_length'] = max_length_new_segment / no_road_multiplier
                new_distances['minimal_distance'] = minimal_distances.loc[columns_to_keep, 'minimal_distance'].tolist()

                # remove branches where all minimal distances are already higher than minimal distance to next node
                new_distances = \
                    new_distances[new_distances['minimal_distance'] <= max_length_new_segment / no_road_multiplier]

                # only choose branches which are applicable for new infrastructure
                new_distances = new_distances[(new_distances['pipeline_gas_applicable']) | (new_distances['pipeline_liquid_applicable'])]

                # remove all options where distance is larger than max length
                mask = new_distances[columns].values > new_distances['max_length'].values[:, None]
                new_distances.loc[:, columns] = np.where(mask, np.nan, new_distances[columns].values)

                # remove unnecessary columns
                new_distances.drop(
                    columns=['minimal_distance', 'max_length', 'pipeline_gas_applicable', 'pipeline_liquid_applicable'],
                    inplace=True)

                # remove used infrastructure
                for visited_infrastructure in branches_to_remove_based_on_visited_infrastructure.keys():
                    affected_branches \
                        = list(set(new_distances.index.tolist()).intersection(branches_to_remove_based_on_visited_infrastructure[visited_infrastructure]['branches']))
                    new_distances.loc[affected_branches, branches_to_remove_based_on_visited_infrastructure[visited_infrastructure]['nodes']] = np.nan

                # restructure
                new_distances = new_distances.stack().dropna()
                new_distances = new_distances.apply(pd.to_numeric, errors='coerce').dropna()
                all_new_distances.append(new_distances)

        if all_road_distances:
            road_options = pd.concat(all_road_distances)
            road_options = road_options.reset_index()
            road_options.columns = ['previous_branch', 'current_node', 'current_distance']
        else:
            road_options = pd.DataFrame()

        if all_new_distances:
            new_infrastructure_options = pd.concat(all_new_distances).transpose()
            new_infrastructure_options = new_infrastructure_options.reset_index()
            new_infrastructure_options.columns = ['previous_branch', 'current_node', 'current_distance']
        else:
            new_infrastructure_options = pd.DataFrame()

    # Create new branches for road options
    if not road_options.empty:  # todo: Man könnte noch alle optionen entfernen, die gleiche start und end node haben (--> kein transport)

        # all distance below tolerance are 0
        below_tolerance = road_options[road_options['current_distance'] <= tolerance_distance].index
        road_options.loc[below_tolerance, 'current_distance'] = 0

        # add further information
        road_options['current_distance'] = road_options['current_distance'] * no_road_multiplier

        branch_list = road_options['previous_branch'].tolist()
        options_list = road_options['current_node'].tolist()

        road_options['current_commodity'] = branches.loc[branch_list, 'current_commodity'].tolist()
        road_options['current_commodity_object'] = branches.loc[branch_list, 'current_commodity_object'].tolist()
        road_options['specific_transportation_costs'] = road_transportation_costs.loc[branch_list].tolist()
        road_options['previous_total_costs'] = branches.loc[branch_list, 'current_total_costs'].tolist()
        road_options['current_transport_mean'] = 'Road'
        road_options['latitude'] = complete_infrastructure.loc[options_list, 'latitude'].tolist()
        road_options['longitude'] = complete_infrastructure.loc[options_list, 'longitude'].tolist()
        road_options['current_continent'] = branches.loc[branch_list, 'current_continent'].tolist()

        taken_route = [(branches.at[road_options.at[i, 'previous_branch'], 'current_node'], 'Road',
                        road_options.at[i, 'current_distance'], road_options.at[i, 'current_node'])
                       for i in road_options.index]
        road_options['taken_route'] = taken_route

        # calculate costs and remove all above benchmark
        road_options['current_transportation_costs'] \
            = road_options['current_distance'] * road_options['specific_transportation_costs'] / 1000
        road_options['current_total_costs'] \
            = road_options['previous_total_costs'] + road_options['current_transportation_costs']

        road_options = road_options[road_options['current_total_costs'] <= benchmark]

        road_options['distance_type'] = 'road'

        # remove duplicates based on node/port, commodity and costs
        road_options['comparison_index'] = [road_options.at[ind, 'current_node'] + '-'
                                            + road_options.at[ind, 'current_commodity']
                                            for ind in road_options.index]
        road_options.sort_values(['current_total_costs'], inplace=True)
        road_options = road_options.drop_duplicates(subset=['comparison_index'], keep='first')

    else:
        road_options = pd.DataFrame()

    # Create new branches for infrastructure options
    if not new_infrastructure_options.empty:

        # all distance below tolerance are 0
        below_tolerance = new_infrastructure_options[new_infrastructure_options['current_distance'] <= tolerance_distance].index
        new_infrastructure_options.loc[below_tolerance, 'current_distance'] = 0

        new_infrastructure_options['current_distance'] = new_infrastructure_options['current_distance'] * no_road_multiplier

        branch_list = new_infrastructure_options['previous_branch'].tolist()
        options_list = new_infrastructure_options['current_node'].tolist()

        # Add additional information
        new_infrastructure_options['current_commodity'] = branches.loc[branch_list, 'current_commodity'].tolist()
        new_infrastructure_options['current_commodity_object'] = branches.loc[branch_list, 'current_commodity_object'].tolist()
        new_infrastructure_options['previous_total_costs'] = branches.loc[branch_list, 'current_total_costs'].tolist()
        new_infrastructure_options['current_continent'] = branches.loc[branch_list, 'current_commodity'].tolist()

        new_infrastructure_options['specific_transportation_costs'] = new_transportation_costs.loc[branch_list].tolist()

        new_infrastructure_options['latitude'] = complete_infrastructure.loc[options_list, 'latitude'].tolist()
        new_infrastructure_options['longitude'] = complete_infrastructure.loc[options_list, 'longitude'].tolist()

        new_infrastructure_options['current_transportation_costs'] \
            = new_infrastructure_options['current_distance'] * new_infrastructure_options['specific_transportation_costs'] / 1000

        new_infrastructure_options['current_total_costs'] \
            = new_infrastructure_options['previous_total_costs'] + new_infrastructure_options['current_transportation_costs']

        pipeline_gas_branches = branches[branches['Pipeline_Gas_applicable']].index
        pg_options \
            = new_infrastructure_options[new_infrastructure_options['previous_branch'].isin(pipeline_gas_branches)].index
        new_infrastructure_options.loc[pg_options, 'current_transport_mean'] = 'New_Pipeline_Gas'

        pipeline_liquid_branches = branches[branches['Pipeline_Liquid_applicable']].index
        pl_options \
            = new_infrastructure_options[new_infrastructure_options['previous_branch'].isin(pipeline_liquid_branches)].index
        new_infrastructure_options.loc[pl_options, 'current_transport_mean'] = 'New_Pipeline_Liquid'

        taken_route = [(branches.at[new_infrastructure_options.at[i, 'previous_branch'], 'current_node'],
                        new_infrastructure_options.at[i, 'current_transport_mean'],
                        new_infrastructure_options.at[i, 'current_distance'],
                        new_infrastructure_options.at[i, 'current_node']) for i in new_infrastructure_options.index]
        new_infrastructure_options['taken_route'] = taken_route

        # remove all options higher than benchmark
        new_infrastructure_options \
            = new_infrastructure_options[new_infrastructure_options['current_total_costs'] <= benchmark]

        new_infrastructure_options['distance_type'] = 'new'

        # remove duplicates based on node/port, commodity and costs
        new_infrastructure_options['comparison_index'] = [new_infrastructure_options.at[ind, 'current_node'] + '-'
                                                          + new_infrastructure_options.at[ind, 'current_commodity']
                                                          for ind in new_infrastructure_options.index]
        new_infrastructure_options.sort_values(['current_total_costs'], inplace=True)
        new_infrastructure_options = new_infrastructure_options.drop_duplicates(subset=['comparison_index'], keep='first')
    else:
        new_infrastructure_options = pd.DataFrame()

    # Concatenate all options
    outside_options = pd.concat([road_options, new_infrastructure_options], ignore_index=True)

    if not outside_options.empty:

        final_destination = data['destination']['location']

        outside_options['distance_to_final_destination'] = calc_distance_list_to_single(outside_options['latitude'],
                                                                                        outside_options['longitude'],
                                                                                        final_destination.y,
                                                                                        final_destination.x)

        in_destination_tolerance \
            = outside_options[outside_options['distance_to_final_destination']
                              <= configuration['to_final_destination_tolerance']].index
        outside_options.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

        # get costs for all options outside tolerance
        outside_options['minimal_total_costs'] \
            = calculate_cheapest_option_to_final_destination(data, outside_options, benchmark, 'current_total_costs')

        # throw out options to expensive
        outside_options \
            = outside_options[outside_options['minimal_total_costs'] <= benchmark]

        # add further information
        outside_options['current_infrastructure'] = None

    return outside_options


def process_in_tolerance_branches_high_memory(data, branches, complete_infrastructure, benchmark, configuration,
                                              with_assessment=True):

    """
    This method iterates over all branches, gets the distance to all connected nodes or ports, and then processes
    all branches together. Processing all branches together results in high memory usage but is faster

    @param dict data: dictionary with common data
    @param pandas.DataFrame branches: DataFrame with current branches
    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param float benchmark: current benchmark
    @param dict configuration: dictionary with configuration
    @param bool with_assessment: boolean to start assessment of resulting dataframe

    @return: pandas.DataFrame with new branches
    """

    destination_continent = data['destination']['continent']

    processed_infrastructure = {}
    considered_branches = []
    starting_points = []
    branch_list = []
    transport_mean = []
    current_infrastructure = []
    transportation_costs_list = []
    all_costs = []
    comparison_index = []
    taken_route = []

    for mot in branches['current_transport_mean'].unique():
        options_m = branches.loc[branches['current_transport_mean'] == mot].copy()

        if options_m.empty:
            continue

        if mot == 'Shipping':

            shipping_infrastructure = data['Shipping']['ports'].copy()
            shipping_infrastructure['current_transport_mean'] = 'Shipping'

            # Only use ports which are on the same continent as the final destination
            if destination_continent in ['Europe', 'Asia', 'Africa']:
                shipping_infrastructure = shipping_infrastructure[shipping_infrastructure['continent'].isin(['Europe',
                                                                                                             'Asia',
                                                                                                             'Africa'])]
            else:
                shipping_infrastructure = shipping_infrastructure[
                    shipping_infrastructure['continent'].isin([destination_continent])]

            shipping_distances = pd.read_csv(configuration['path_processed_data']
                                             + 'inner_infrastructure_distances/port_distances.csv',
                                             index_col=0)

            # create one big target_infrastructure dataframe for all shipping options
            for s in options_m.index:

                current_commodity_object = options_m.at[s, 'current_commodity_object']
                if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                    continue

                previous_transport_means = options_m.at[s, 'all_previous_transport_means']
                if 'Shipping' in previous_transport_means:
                    # pass branch because cannot ship twice
                    continue

                transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
                current_total_costs = options_m.at[s, 'current_total_costs']

                used_infrastructure = options_m.at[s, 'all_previous_infrastructure']

                start_infrastructure = options_m.loc[s, 'current_node']
                if start_infrastructure in used_infrastructure:
                    # pass branch because infrastructure was already used
                    continue

                distances = shipping_distances.loc[start_infrastructure, shipping_infrastructure.index].copy()
                current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs
                current_total_costs_distances = current_total_costs_distances[current_total_costs_distances <= benchmark].dropna()

                if current_total_costs_distances.empty:
                    continue

                length_index = len(current_total_costs_distances.index)

                all_costs += current_total_costs_distances.values.tolist()

                shipping_infrastructure.loc[current_total_costs_distances.index, s] \
                    = distances.loc[current_total_costs_distances.index]

                considered_branches.append(s)
                starting_points += [start_infrastructure] * length_index
                branch_list += [s] * length_index

                current_infrastructure += current_total_costs_distances.index.tolist()
                transport_mean += ['Shipping'] * length_index

                transportation_costs_list += [transportation_costs] * length_index

                for i in current_total_costs_distances.index:
                    comparison_index.append(i + '-' + current_commodity_object.get_name())

                taken_route += [(start_infrastructure, mot, distances.at[i], i) for i in current_total_costs_distances.index]

            processed_infrastructure['Shipping'] = shipping_infrastructure

        else:

            graph_distances = {}
            if configuration['use_low_storage']:  # todo: check if it works as it's planned
                # if not precalculated, calculate network distances for respective nodes

                for g in options_m['graph'].unique():
                    graph = data[mot][g]['Graph']

                    g_options_m = options_m[options_m['graph'] == g]
                    nodes = g_options_m['current_node'].unique()

                    for n in nodes:
                        distances = nx.single_source_dijkstra_path_length(graph, n)
                        distances = pd.DataFrame(distances.values(), index=[*distances.keys()], columns=[n])
                        distances = distances.loc[n]

                        graph_distances[n] = distances

            for s in options_m.index:

                current_commodity_object = options_m.at[s, 'current_commodity_object']
                if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                    continue

                if (current_commodity_object.get_name() == 'Hydrogen_Gas') & (not configuration['H2_ready_infrastructure']):
                    # if pipelines are not H2 ready, we cannot use pipelines if current commodity is H2
                    continue

                transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
                current_total_costs = options_m.at[s, 'current_total_costs']

                # for removing already used infrastructure
                used_infrastructure = options_m.at[s, 'all_previous_infrastructure']

                graph_id = options_m.at[s, 'graph']

                if graph_id in used_infrastructure:
                    continue

                if graph_id in processed_infrastructure.keys():
                    pipeline_infrastructure = processed_infrastructure[graph_id]
                else:
                    pipeline_infrastructure = data[mot][graph_id]['NodeLocations'].copy()

                start_infrastructure = options_m.at[s, 'current_node']

                if not configuration['use_low_storage']:
                    path_processed_data = configuration['path_processed_data']
                    distances \
                        = pd.read_hdf(path_processed_data + '/inner_infrastructure_distances/' + start_infrastructure + '.h5',
                                      mode='r', title=graph_id, dtype=np.float16)

                    distances = pd.Series(distances.transpose().values[0], index=distances.index)
                    distances = distances.loc[pipeline_infrastructure.index]
                else:
                    distances = graph_distances[start_infrastructure]

                current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs
                current_total_costs_distances = current_total_costs_distances[current_total_costs_distances <= benchmark].dropna()

                if current_total_costs_distances.empty:
                    continue

                length_index = len(current_total_costs_distances.index)

                all_costs += current_total_costs_distances.values.tolist()

                pipeline_infrastructure.loc[current_total_costs_distances.index, s] \
                    = distances.loc[current_total_costs_distances.index]

                considered_branches.append(s)
                starting_points += [start_infrastructure] * length_index
                branch_list += [s] * length_index

                current_infrastructure += [graph_id] * length_index
                transport_mean += [mot] * length_index

                transportation_costs_list += [transportation_costs] * length_index

                processed_infrastructure[graph_id] = pipeline_infrastructure

                comparison_index += list(map(lambda x: x + '-' + current_commodity_object.get_name(),
                                             current_total_costs_distances.index.tolist()))

                taken_route += [(start_infrastructure, mot, distances.at[i], i) for i in current_total_costs_distances.index]

    if list(processed_infrastructure.values()):

        all_infrastructures = pd.concat(list(processed_infrastructure.values()),
                                        ignore_index=False, verify_integrity=True).transpose()
        all_infrastructures = all_infrastructures.loc[considered_branches, :].stack().dropna().reset_index()
        all_infrastructures.columns = ['previous_branch', 'current_node', 'current_distance']

        del processed_infrastructure
        gc.collect()

        if all_infrastructures.empty:
            return pd.DataFrame()

        nodes_list = all_infrastructures['current_node'].tolist()

        all_infrastructures['starting_point'] = starting_points
        all_infrastructures['previous_branch'] = branch_list
        all_infrastructures['current_transport_mean'] = transport_mean
        all_infrastructures['current_infrastructure'] = current_infrastructure
        all_infrastructures['current_total_costs'] = all_costs
        all_infrastructures['specific_transportation_costs'] = transportation_costs_list
        all_infrastructures['comparison_index'] = comparison_index
        all_infrastructures['taken_route'] = taken_route

        all_infrastructures['current_commodity'] = branches.loc[branch_list, 'current_commodity'].tolist()
        all_infrastructures['current_commodity_object'] \
            = branches.loc[branch_list, 'current_commodity_object'].tolist()
        all_infrastructures['latitude'] = complete_infrastructure.loc[nodes_list, 'latitude'].tolist()
        all_infrastructures['longitude'] = complete_infrastructure.loc[nodes_list, 'longitude'].tolist()

        # remove duplicates
        all_infrastructures.sort_values(['current_total_costs'], inplace=True)
        all_infrastructures = all_infrastructures.drop_duplicates(subset=['comparison_index'], keep='first')

        # costs assessment for benchmark comparing and anticipation of costs to the closest infrastructure
        if with_assessment:

            # add costs to options
            all_infrastructures['current_transportation_costs'] \
                = all_infrastructures['current_distance'] / 1000 * all_infrastructures['specific_transportation_costs']

            # calculate minimal potential costs to final destination
            final_destination = data['destination']['location']
            all_infrastructures['distance_to_final_destination'] \
                = calc_distance_list_to_single(all_infrastructures['latitude'],
                                               all_infrastructures['longitude'],
                                               final_destination.y, final_destination.x)

            # asses costs to final destination based on distance to final destination
            # get options in tolerance to final destination and set distance to 0
            in_destination_tolerance \
                = all_infrastructures[all_infrastructures['distance_to_final_destination']
                                      <= configuration['to_final_destination_tolerance']].index
            all_infrastructures.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

            # get costs for all options outside tolerance
            all_infrastructures['minimal_total_costs'] \
                = calculate_cheapest_option_to_closest_infrastructure(data, all_infrastructures, configuration,
                                                                      benchmark, 'current_total_costs')

            # throw out options to expensive
            all_infrastructures \
                = all_infrastructures[all_infrastructures['minimal_total_costs'] <= benchmark]

            # next iteration either uses road or new pipeline. Remove options where the closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            all_infrastructures['minimal_distances'] \
                = minimal_distances.loc[all_infrastructures['current_node'].tolist(), 'minimal_distance'].tolist()

            all_infrastructures = all_infrastructures[all_infrastructures['minimal_distances'] <= max_length]
            all_infrastructures.drop(['minimal_distances'], axis=1, inplace=True)

        return all_infrastructures
    else:
        return pd.DataFrame()


def process_in_tolerance_branches_low_memory(data, branches, complete_infrastructure, benchmark, configuration,
                                             with_assessment=True):

    """
    This method iterates over all branches, gets the distance to all connected nodes or ports, and then processes
    each branch alone. Processing all branches alone saves memory but takes more time

    @param dict data: dictionary with common data
    @param pandas.DataFrame branches: DataFrame with current branches
    @param pandas.DataFrame complete_infrastructure: DataFrame with all nodes, ports and destination
    @param float benchmark: current benchmark
    @param dict configuration: dictionary with configuration
    @param bool with_assessment: boolean to start assessment of resulting dataframe

    @return: pandas.DataFrame with new branches
    """

    destination_continent = data['destination']['continent']
    all_infrastructure = pd.DataFrame()

    for o in branches.index:

        mot = branches.at[o, 'current_transport_mean']
        if mot == 'Shipping':

            shipping_infrastructure = data['Shipping']['ports'].copy()
            shipping_infrastructure['current_transport_mean'] = 'Shipping'

            shipping_distances = pd.read_csv(configuration['path_processed_data']
                                             + 'inner_infrastructure_distances/port_distances.csv',
                                             index_col=0)

            # Only use ports which are on the same continent as the final destination
            if destination_continent in ['Europe', 'Asia', 'Africa']:
                shipping_infrastructure = shipping_infrastructure[shipping_infrastructure['continent'].isin(['Europe',
                                                                                                             'Asia',
                                                                                                             'Africa'])]
            else:
                shipping_infrastructure = shipping_infrastructure[
                    shipping_infrastructure['continent'].isin([destination_continent])]

            shipping_distances_columns = [c for c in shipping_distances.columns if c in shipping_infrastructure.index]
            shipping_distances = shipping_distances[shipping_distances_columns]

            # create one big target_infrastructure dataframe for all shipping options
            current_commodity_object = branches.at[o, 'current_commodity_object']
            if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                # pass branch because commodity cannot be transported via ship
                continue

            previous_transport_means = branches.at[o, 'all_previous_transport_means']
            if 'Shipping' in previous_transport_means:
                # pass branch because cannot ship twice
                continue

            transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
            current_total_costs = branches.at[o, 'current_total_costs']

            used_infrastructure = branches.at[o, 'all_previous_infrastructure']

            # pass branch if port has already been used
            start_infrastructure = branches.loc[o, 'current_node']
            if start_infrastructure in used_infrastructure:
                continue

            distances = shipping_distances.loc[start_infrastructure, :].copy()
            current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs

            current_infrastructure = 'Shipping'

            comparison_index = []
            taken_route = []
            for i in distances.index:
                comparison_index.append(i + '-' + current_commodity_object.get_name())
                taken_route.append((start_infrastructure, mot, distances.at[i], i))

        else:

            graph_id = branches.at[o, 'graph']

            current_commodity_object = branches.at[o, 'current_commodity_object']
            if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                # pass branch if commodity cannot be transported via pipeline
                continue

            if (current_commodity_object.get_name() == 'Hydrogen_Gas') & (not configuration['H2_ready_infrastructure']):
                # pass branch if commodity is H2 but pipeline is set to not H2 ready pipelines
                continue

            transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
            current_total_costs = branches.at[o, 'current_total_costs']

            used_infrastructure = branches.at[o, 'all_previous_infrastructure']
            if graph_id in used_infrastructure:
                # pass branch if node has already been used
                continue

            start_infrastructure = branches.at[o, 'current_node']

            if not configuration['use_low_storage']:
                # uses precalculated distances
                path_processed_data = configuration['path_processed_data']
                distances \
                    = pd.read_hdf(path_processed_data + '/inner_infrastructure_distances/' + start_infrastructure + '.h5',
                                  mode='r', title=graph_id, dtype=np.float16)
            else:
                # calculates distances from current node
                graph = data[mot][graph_id]['Graph']
                distances = nx.single_source_dijkstra_path_length(graph, start_infrastructure)
                distances = pd.DataFrame(distances.values(), index=[*distances.keys()], columns=[start_infrastructure])
                distances = distances.loc[start_infrastructure]  # todo: check how it looks like

            current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs

            current_infrastructure = graph_id

            comparison_index = []
            taken_route = []
            for i in distances.index:
                comparison_index.append(i + '-' + current_commodity_object.get_name())
                taken_route.append((start_infrastructure, mot, distances.at[i, start_infrastructure], i))

        infrastructure = pd.DataFrame(distances.values, index=distances.index, columns=['current_distance'])
        infrastructure['previous_branch'] = o
        infrastructure['current_node'] = distances.index

        if infrastructure.empty:
            return pd.DataFrame()

        infrastructure['current_total_costs'] = current_total_costs_distances
        infrastructure['comparison_index'] = comparison_index
        infrastructure['taken_route'] = taken_route

        infrastructure = infrastructure[infrastructure['current_total_costs'] <= benchmark].dropna()
        nodes_list = infrastructure['current_node'].tolist()

        infrastructure['starting_point'] = start_infrastructure
        infrastructure['previous_branch'] = o
        infrastructure['current_transport_mean'] = mot
        infrastructure['current_infrastructure'] = current_infrastructure
        infrastructure['specific_transportation_costs'] = transportation_costs

        infrastructure['current_commodity'] = current_commodity_object.get_name()
        infrastructure['current_commodity_object'] = current_commodity_object
        infrastructure['latitude'] = complete_infrastructure.loc[nodes_list, 'latitude'].tolist()
        infrastructure['longitude'] = complete_infrastructure.loc[nodes_list, 'longitude'].tolist()

        # remove duplicates
        infrastructure.sort_values(['current_total_costs'], inplace=True)
        infrastructure = infrastructure.drop_duplicates(subset=['comparison_index'], keep='first')

        # costs assessment for benchmark comparing and anticipation of costs to the closest infrastructure
        if with_assessment:

            # add costs to options
            infrastructure['current_transportation_costs'] \
                = infrastructure['current_distance'] / 1000 * infrastructure['specific_transportation_costs']

            # calculate minimal potential costs to final destination
            final_destination = data['destination']['location']
            infrastructure['distance_to_final_destination'] \
                = calc_distance_list_to_single(infrastructure['latitude'],
                                               infrastructure['longitude'],
                                               final_destination.y, final_destination.x)

            # asses costs to final destination based on distance to final destination
            # get options in tolerance to final destination and set distance to 0
            in_destination_tolerance \
                = infrastructure[infrastructure['distance_to_final_destination']
                                 <= configuration['to_final_destination_tolerance']].index
            infrastructure.loc[in_destination_tolerance, 'distance_to_final_destination'] = 0

            # get costs for all options outside tolerance
            infrastructure['minimal_total_costs'] \
                = calculate_cheapest_option_to_closest_infrastructure(data, infrastructure, configuration, benchmark,
                                                                      'current_total_costs')

            # throws out options to expensive
            infrastructure \
                = infrastructure[infrastructure['minimal_total_costs'] <= benchmark]

            # next iteration either uses road or new pipeline. Remove options where the closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            infrastructure['minimal_distances'] \
                = minimal_distances.loc[infrastructure['current_node'].tolist(), 'minimal_distance'].tolist()

            infrastructure = infrastructure[infrastructure['minimal_distances'] <= max_length]
            infrastructure.drop(['minimal_distances'], axis=1, inplace=True)

        all_infrastructure = pd.concat([all_infrastructure, infrastructure])

    return all_infrastructure
