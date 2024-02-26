import math

import pandas as pd
import numpy as np

import time

from _helpers import calc_distance_list_to_single,\
    calculate_cheapest_option_to_final_destination, calc_distance_list_to_list

import gc

# Ignore runtime warnings as they
# import os
# os.environ['PYTHONWARNINGS'] = 'ignore::[RuntimeWarning]'
# os.environ['PYTHONWARNINGS'] = 'ignore::[FutureWarning]'

import warnings
warnings.filterwarnings("ignore")


def get_complete_infrastructure(data):

    options = pd.DataFrame()
    options_to_concat = []
    final_destination = data['destination']['location']
    for m in ['Road', 'Shipping', 'Pipeline_Gas', 'Pipeline_Liquid']: #'data['Means_of_Transport']:

        if m == 'Road':
            # Check final destination and add to option outside tolerance if applicable
            options.loc['Destination', 'latitude'] = final_destination.y
            options.loc['Destination', 'longitude'] = final_destination.x
            options.loc['Destination', 'current_transport_mean'] = m
            options.loc['Destination', 'graph'] = None
            options.loc['Destination', 'continent'] = data['destination']['continent']

            continue

        # get all options of current mean of transport
        if m == 'Shipping':

            # get all options of current mean of transport
            options_shipping = data[m]['ports']
            options_shipping['current_transport_mean'] = m
            options_shipping['graph'] = None

            options_to_concat.append(options_shipping)  # todo: use used_infrastructure to remove ports already

        else:
            networks = data[m].keys()
            for n in networks:
                options_network = data[m][n]['GeoData'].copy()
                options_network['current_transport_mean'] = m
                options_to_concat.append(options_network)

    options = pd.concat([options] + options_to_concat)

    # create common infrastructure column
    options['infrastructure'] = options.index
    graph_df = options[options['graph'].apply(lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) if isinstance(x, (list, float)) else False)]
    options.loc[graph_df.index, 'infrastructure'] = options.loc[graph_df.index, 'infrastructure']

    return options


def process_out_tolerance_solutions(options, solutions, configuration, iteration, data, benchmark, local_benchmarks,
                                    use_minimal_distance=False, limitation=None):

    now = time.time()

    tolerance_distance = configuration['tolerance_distance']
    max_length_new_segment = configuration['max_length_new_segment']
    max_length_road = configuration['max_length_road']
    no_road_multiplier = configuration['no_road_multiplier']
    build_new_infrastructure = configuration['build_new_infrastructure']

    if iteration == 0:

        # only use options which are actually reachable from start
        options = options[options['reachable_from_start']]

        if limitation == 'no_pipeline_gas':
            reduced_options_index = [i for i in options.index if 'PG' not in i]
            distances = calc_distance_list_to_list(options.loc[reduced_options_index, 'latitude'],
                                                   options.loc[reduced_options_index, 'longitude'],
                                                   solutions['latitude'], solutions['longitude'])

        elif limitation == 'no_pipeline_liquid':
            reduced_options_index = [i for i in options.index if 'PL' not in i]
            distances = calc_distance_list_to_list(options.loc[reduced_options_index, 'latitude'],
                                                   options.loc[reduced_options_index, 'longitude'],
                                                   solutions['latitude'], solutions['longitude'])

        elif limitation == 'no_pipelines':
            reduced_options_index = [i for i in options.index if 'H' in i]
            distances = calc_distance_list_to_list(options.loc[reduced_options_index, 'latitude'],
                                                   options.loc[reduced_options_index, 'longitude'],
                                                   solutions['latitude'], solutions['longitude'])

        else:
            reduced_options_index = options.index
            distances = calc_distance_list_to_list(options['latitude'], options['longitude'],
                                                   solutions['latitude'], solutions['longitude'])

        road_transportation_costs = {}

        new_transportation_costs = {}

        solutions_to_keep_road = []
        solutions_to_keep_new = []

        road_distances = pd.DataFrame(distances.transpose(), index=reduced_options_index, columns=solutions.index)
        for i in solutions.index:
            current_commodity_object = solutions.at[i, 'current_commodity_object']

            pipeline_applicable \
                = current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas') \
                | current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Liquid')

            was_not_road = solutions.at[i, 'current_transport_mean'] != 'Road'
            road_applicable = current_commodity_object.get_transportation_options_specific_mean_of_transport('Road')
            road_applicable = road_applicable & was_not_road

            was_not_new = solutions.at[i, 'current_transport_mean'] not in ['New_Pipeline_Gas', 'New_Pipeline_Liquid']

            if road_applicable:
                road_transportation_costs[i]\
                    = current_commodity_object.get_transportation_costs_specific_mean_of_transport('Road')
                solutions_to_keep_road.append(i)
            else:
                in_tolerance_options = road_distances[i]
                in_tolerance_options = in_tolerance_options[in_tolerance_options <= configuration['tolerance_distance']].index
                road_distances[i] = math.nan
                road_distances.loc[in_tolerance_options, i] = 0

                road_transportation_costs[i] = 0
                solutions_to_keep_road.append(i)

            if was_not_new & pipeline_applicable & build_new_infrastructure:
                if current_commodity_object.get_transportation_options_specific_mean_of_transport('Pipeline_Gas'):
                    new_transportation_costs[i]\
                        = current_commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Gas')
                else:
                    new_transportation_costs[i]\
                        = current_commodity_object.get_transportation_costs_specific_mean_of_transport('New_Pipeline_Liquid')
                solutions_to_keep_new.append(i)

        road_distances = road_distances[solutions_to_keep_road]
        road_transportation_costs = pd.Series(road_transportation_costs.values(), index=road_transportation_costs.keys())

        road_options = road_distances[road_distances <= max_length_road / no_road_multiplier]
        road_options = road_options.transpose().stack().dropna().reset_index()
        road_options.columns = ['previous_solution', 'current_node', 'current_distance']

        if build_new_infrastructure:
            new_infrastructure_distances = pd.DataFrame(distances.transpose(), index=reduced_options_index,
                                                        columns=solutions.index)

            new_infrastructure_distances = new_infrastructure_distances[solutions_to_keep_new]
            new_transportation_costs = pd.Series(new_transportation_costs.values(), index=new_transportation_costs.keys())

            new_infrastructure_options \
                = new_infrastructure_distances[new_infrastructure_distances <= max_length_new_segment / no_road_multiplier]
            new_infrastructure_options = new_infrastructure_options.transpose().stack().dropna().reset_index()
            new_infrastructure_options.columns = ['previous_solution', 'current_node', 'current_distance']
        else:
            new_infrastructure_options = pd.DataFrame()

    else:

        road_transportation_costs = solutions['road_transportation_costs']

        new_transportation_costs = solutions['new_transportation_costs']

        minimal_distances = data['minimal_distances']

        if True:
            time_new_approach = time.time()
            time_calculate_distances = time.time()

            all_road_distances = []
            all_new_distances = []

            if not use_minimal_distance:

                solutions_no_duplicates = solutions.drop_duplicates(subset=['current_node'],
                                                                    keep='first')

                if limitation == 'no_pipeline_gas':
                    reduced_options_index = [i for i in options.index if 'PG' not in i]
                    distances = calc_distance_list_to_list(options.loc[reduced_options_index, 'latitude'],
                                                           options.loc[reduced_options_index, 'longitude'],
                                                           solutions_no_duplicates['latitude'],
                                                           solutions_no_duplicates['longitude'])

                elif limitation == 'no_pipeline_liquid':
                    reduced_options_index = [i for i in options.index if 'PL' not in i]
                    distances = calc_distance_list_to_list(options.loc[reduced_options_index, 'latitude'],
                                                           options.loc[reduced_options_index, 'longitude'],
                                                           solutions_no_duplicates['latitude'],
                                                           solutions_no_duplicates['longitude'])

                elif limitation == 'no_pipelines':
                    reduced_options_index = [i for i in options.index if 'H' in i]
                    distances = calc_distance_list_to_list(options.loc[reduced_options_index, 'latitude'],
                                                           options.loc[reduced_options_index, 'longitude'],
                                                           solutions_no_duplicates['latitude'],
                                                           solutions_no_duplicates['longitude'])

                else:
                    reduced_options_index = options.index
                    distances = calc_distance_list_to_list(options['latitude'], options['longitude'],
                                                           solutions_no_duplicates['latitude'],
                                                           solutions_no_duplicates['longitude'])

                distances = pd.DataFrame(distances.transpose(),
                                         index=reduced_options_index,
                                         columns=solutions_no_duplicates['current_node'])

            else:
                # Check distance not to all infrastructure but just closest one to assess, if the closest
                # infrastructure is already too expensive

                solutions_no_duplicates = solutions.drop_duplicates(subset=['current_node'],
                                                                    keep='first')

                solutions_no_duplicates['minimal_distance'] \
                    = minimal_distances.loc[solutions_no_duplicates['current_node'].tolist(), 'minimal_distance'].tolist()
                solutions_no_duplicates['closest_node'] \
                    = minimal_distances.loc[solutions_no_duplicates['current_node'].tolist(), 'closest_node'].tolist()

                distances \
                    = solutions_no_duplicates.set_index([solutions_no_duplicates['current_node'],
                                                         solutions_no_duplicates['closest_node']])['minimal_distance'].unstack().transpose()

            # get options to remove
            if True:

                def flatten_list(nested_list):
                    flattened_list = []
                    for element in nested_list:
                        if isinstance(element, list):
                            flattened_list.extend(flatten_list(element))
                        else:
                            flattened_list.append(element)
                    return flattened_list

                all_previous_infrastructure = list(set(flatten_list(solutions['all_previous_infrastructure'].tolist())))

                indexes_to_remove = {}
                for i in all_previous_infrastructure:
                    if i is not None:
                        if 'PG' in i:
                            affected_solutions = solutions[solutions['all_previous_infrastructure'].apply(lambda x: i in x)].index
                            indexes_to_remove[i] = {'nodes': data['Pipeline_Gas'][i]['GeoData'].index.tolist(),
                                                    'solutions': affected_solutions}

                        elif 'PL' in i:
                            affected_solutions = solutions[solutions['all_previous_infrastructure'].apply(lambda x: i in x)].index
                            indexes_to_remove[i] = {'nodes': data['Pipeline_Liquid'][i]['GeoData'].index.tolist(),
                                                    'solutions': affected_solutions}


            options_to_remove = {}
            if False: # iteration > 0:

                unstacked_benchmark \
                    = local_benchmarks.set_index(['current_commodity', 'current_node'])['current_total_costs'].unstack()

                for i in unstacked_benchmark.index:
                    if i in solutions['current_commodity'].tolist():
                        c_benchmarks = unstacked_benchmark.loc[i, :].dropna()
                        c_solutions = solutions[solutions['current_commodity'] == i]['anticipated_costs']

                        result = c_solutions.apply(lambda x: c_benchmarks[c_benchmarks < x].index.tolist())

                        options_to_remove[i] = {}
                        for n in result.index:
                            options_to_remove[i][n] = result.at[n]

            for c in solutions['current_commodity'].unique():
                c_solutions = solutions[solutions['current_commodity'] == c]
                if c_solutions.empty:
                    continue

                c_distances = distances.copy()
                commodity_object = data['commodities']['commodity_objects'][c]

                time_change_columns = time.time()
                # exchange current_node columns with corresponding solution names
                node_list = c_solutions['current_node'].tolist()  # list is unique
                solution_list = c_solutions.index.tolist()
                new_column_list = []
                columns_to_keep = []
                # todo: why is list values (c_solutions['Pipeline_Gas_applicable']) larger than new distances index?
                #  only possible if not all elements in node list are used
                for n in distances.columns:
                    if n in node_list:
                        new_column_list.append(solution_list[node_list.index(n)])
                        columns_to_keep.append(n)

                c_distances = c_distances[columns_to_keep]
                c_distances.columns = new_column_list  # rename columns to solutions

                time_road = time.time()

                if commodity_object.get_transportation_options()['Road']:

                    road_distances = c_distances.copy().transpose()
                    columns = road_distances.columns  # get initial columns before new ones are added

                    # remove all solutions where road is not applicable (remove rows)
                    road_distances['road_applicable'] = c_solutions['Road_applicable'].tolist()
                    road_distances = road_distances[road_distances['road_applicable']]

                    # remove all options where max length exceeds distance (remove columns)
                    max_length_road_costs \
                        = list((benchmark - c_solutions['current_total_costs']) * 1000
                               / c_solutions['road_transportation_costs'] / no_road_multiplier)
                    max_length_road_list = [min(n, max_length_road / no_road_multiplier) for n in max_length_road_costs]
                    road_distances['max_length'] = max_length_road_list

                    mask = road_distances[columns].values > road_distances['max_length'].values[:, None]
                    road_distances.loc[:, columns] = np.where(mask, np.nan, road_distances[columns].values)
                    road_distances.drop(columns=['max_length', 'road_applicable'], inplace=True)

                    # remove options based on previous used infrastructure
                    for k in indexes_to_remove.keys():
                        affected_solutions = list(set(road_distances.index.tolist()).intersection(indexes_to_remove[k]['solutions']))
                        road_distances.loc[affected_solutions, indexes_to_remove[k]['nodes']] = np.nan

                    road_distances = road_distances.stack().dropna()

                    all_road_distances.append(road_distances)

                # create and process new infrastructure distances
                time_new = time.time()

                if (commodity_object.get_transportation_options()['Pipeline_Gas']
                        | commodity_object.get_transportation_options()['Pipeline_Liquid']) & build_new_infrastructure:

                    new_distances = c_distances.copy().transpose()
                    columns = new_distances.columns

                    # add information before any change to distances is made
                    # todo: more values than new_distance index
                    new_distances['pipeline_gas_applicable'] = c_solutions['Pipeline_Gas_applicable'].tolist()
                    new_distances['pipeline_liquid_applicable'] = c_solutions['Pipeline_Liquid_applicable'].tolist()
                    new_distances['max_length'] = max_length_new_segment / no_road_multiplier
                    new_distances['minimal_distance'] = minimal_distances.loc[columns_to_keep, 'minimal_distance'].tolist()

                    # remove solutions where all minimal distances are already higher than minimal distance to next node
                    new_distances = \
                        new_distances[new_distances['minimal_distance'] <= max_length_new_segment / no_road_multiplier]

                    # only choose solutions which are applicable for new infrastructure
                    new_distances = new_distances[(new_distances['pipeline_gas_applicable']) | (new_distances['pipeline_liquid_applicable'])]

                    # remove all options where distance is larger than max length
                    mask = new_distances[columns].values > new_distances['max_length'].values[:, None]
                    new_distances.loc[:, columns] = np.where(mask, np.nan, new_distances[columns].values)

                    new_distances.drop(columns=['minimal_distance', 'max_length', 'pipeline_gas_applicable', 'pipeline_liquid_applicable'],
                                       inplace=True)
                    new_distances = new_distances.stack().dropna()
                    all_new_distances.append(new_distances)

            time_postprocessing = time.time()
            if all_road_distances:
                road_options = pd.concat(all_road_distances)
                road_options = road_options.reset_index()
                road_options.columns = ['previous_solution', 'current_node', 'current_distance']
            else:
                road_options = pd.DataFrame()

            if all_new_distances:
                new_infrastructure_options = pd.concat(all_new_distances).transpose()
                new_infrastructure_options = new_infrastructure_options.reset_index()
                new_infrastructure_options.columns = ['previous_solution', 'current_node', 'current_distance']
            else:
                new_infrastructure_options = pd.DataFrame()

    time_distance = time.time() - now

    # Create for road options
    now = time.time()

    if not road_options.empty:

        if not road_options.empty:

            # all distance below tolerance are 0
            below_tolerance = road_options[road_options['current_distance'] <= tolerance_distance].index
            road_options.loc[below_tolerance, 'current_distance'] = 0

            road_options['current_distance'] = road_options['current_distance'] * no_road_multiplier

            solution_list = road_options['previous_solution'].tolist()
            options_list = road_options['current_node'].tolist()

            road_options['current_commodity'] = solutions.loc[solution_list, 'current_commodity'].tolist()
            road_options['current_commodity_object'] = solutions.loc[solution_list, 'current_commodity_object'].tolist()
            road_options['specific_transportation_costs'] = road_transportation_costs.loc[solution_list].tolist()
            road_options['previous_total_costs'] = solutions.loc[solution_list, 'current_total_costs'].tolist()
            road_options['current_transport_mean'] = 'Road'
            road_options['latitude'] = options.loc[options_list, 'latitude'].tolist()
            road_options['longitude'] = options.loc[options_list, 'longitude'].tolist()
            road_options['current_continent'] = solutions.loc[solution_list, 'current_continent'].tolist()

            taken_route = [(solutions.at[road_options.at[i, 'previous_solution'], 'current_node'], 'Road',
                            road_options.at[i, 'current_distance'], road_options.at[i, 'current_node'])
                           for i in road_options.index]
            road_options['taken_route'] = taken_route

            road_options['current_transportation_costs'] \
                = road_options['current_distance'] * road_options['specific_transportation_costs'] / 1000
            road_options['current_total_costs'] \
                = road_options['previous_total_costs'] + road_options['current_transportation_costs']

            road_options = road_options[road_options['current_total_costs'] <= benchmark]

            road_options['distance_type'] = 'road'

            road_options['comparison_index'] = [road_options.at[ind, 'current_node'] + '-'
                                                + road_options.at[ind, 'current_commodity']
                                                for ind in road_options.index]
            road_options.sort_values(['current_total_costs'], inplace=True)
            road_options = road_options.drop_duplicates(subset=['comparison_index'], keep='first')

        else:
            road_options = pd.DataFrame()

    else:
        road_options = pd.DataFrame()

    time_road = time.time() - now

    # Create new infrastructure options
    now = time.time()
    if not new_infrastructure_options.empty:

        # all distance below tolerance are 0
        below_tolerance = new_infrastructure_options[new_infrastructure_options['current_distance'] <= tolerance_distance].index
        new_infrastructure_options.loc[below_tolerance, 'current_distance'] = 0

        new_infrastructure_options['current_distance'] = new_infrastructure_options['current_distance'] * no_road_multiplier

        solution_list = new_infrastructure_options['previous_solution'].tolist()
        options_list = new_infrastructure_options['current_node'].tolist()

        # Add additional columns
        new_infrastructure_options['current_commodity'] = solutions.loc[solution_list, 'current_commodity'].tolist()
        new_infrastructure_options['current_commodity_object'] = solutions.loc[solution_list, 'current_commodity_object'].tolist()
        new_infrastructure_options['previous_total_costs'] = solutions.loc[solution_list, 'current_total_costs'].tolist()
        new_infrastructure_options['current_continent'] = solutions.loc[solution_list, 'current_commodity'].tolist()

        new_infrastructure_options['specific_transportation_costs'] = new_transportation_costs.loc[solution_list].tolist()

        new_infrastructure_options['latitude'] = options.loc[options_list, 'latitude'].tolist()
        new_infrastructure_options['longitude'] = options.loc[options_list, 'longitude'].tolist()

        new_infrastructure_options['current_transportation_costs'] \
            = new_infrastructure_options['current_distance'] * new_infrastructure_options['specific_transportation_costs'] / 1000

        new_infrastructure_options['current_total_costs'] \
            = new_infrastructure_options['previous_total_costs'] + new_infrastructure_options['current_transportation_costs']

        pipeline_gas_solutions = solutions[solutions['Pipeline_Gas_applicable']].index
        pg_options \
            = new_infrastructure_options[new_infrastructure_options['previous_solution'].isin(pipeline_gas_solutions)].index
        new_infrastructure_options.loc[pg_options, 'current_transport_mean'] = 'New_Pipeline_Gas'

        pipeline_liquid_solutions = solutions[solutions['Pipeline_Liquid_applicable']].index
        pl_options \
            = new_infrastructure_options[new_infrastructure_options['previous_solution'].isin(pipeline_liquid_solutions)].index
        new_infrastructure_options.loc[pl_options, 'current_transport_mean'] = 'New_Pipeline_Liquid'

        taken_route = [(solutions.at[new_infrastructure_options.at[i, 'previous_solution'], 'current_node'],
                        new_infrastructure_options.at[i, 'current_transport_mean'],
                        new_infrastructure_options.at[i, 'current_distance'],
                        new_infrastructure_options.at[i, 'current_node']) for i in new_infrastructure_options.index]
        new_infrastructure_options['taken_route'] = taken_route

        # remove all options higher than benchmark
        new_infrastructure_options \
            = new_infrastructure_options[new_infrastructure_options['current_total_costs'] <= benchmark]

        new_infrastructure_options['distance_type'] = 'new'

        new_infrastructure_options['comparison_index'] = [new_infrastructure_options.at[ind, 'current_node'] + '-'
                                                          + new_infrastructure_options.at[ind, 'current_commodity']
                                                          for ind in new_infrastructure_options.index]
        new_infrastructure_options.sort_values(['current_total_costs'], inplace=True)
        new_infrastructure_options = new_infrastructure_options.drop_duplicates(subset=['comparison_index'], keep='first')
    else:
        new_infrastructure_options = pd.DataFrame()

    time_new = time.time() - now

    # Concatenate all options
    now = time.time()
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
        outside_options['minimal_total_costs_to_final_destination'] \
            = calculate_cheapest_option_to_final_destination(data, outside_options,
                                                             configuration, benchmark,
                                                             'current_total_costs')

        # throws out options to expensive
        outside_options \
            = outside_options[outside_options['minimal_total_costs_to_final_destination'] <= benchmark]

        # add further information
        outside_options['current_infrastructure'] = None

    time_concat = time.time() - now

    """print('number solutions: ' + str(len(solutions.index)) + ' | prepare distances: ' + str(time_distance)
          + ' | prepare road: ' + str(time_road) + ' | prepare new: ' + str(time_new) + ' | postprocessing: ' + str(time_concat))"""

    return outside_options


def _remove_ports_based_on_continent(options, continent_column):

    def consider_continent(continent):
        if continent in ['Europe', 'Asia', 'Africa']:
            return ['Europe', 'Asia', 'Africa']
        else:
            return [continent]

    options['considered_continent'] = options[continent_column].apply(consider_continent)

    options_shipping = options[options.current_transport_mean == 'Shipping'].copy()

    options_shipping = options_shipping[
        options_shipping.apply(lambda row: row['continent'] in row['considered_continent'], axis=1)].copy()

    options_not_shipping = options.loc[options.current_transport_mean != 'Shipping', :].copy()

    options = pd.concat([options_shipping, options_not_shipping])

    return options


def process_in_tolerance_solutions(data, options, all_options, local_benchmarks, benchmark, configuration, k,
                                   with_assessment=True):

    destination_continent = data['destination']['continent']

    processed_infrastructure = {}
    considered_solutions = []
    starting_points = []
    solution_list = []
    transport_mean = []
    current_infrastructure = []
    transportation_costs_list = []
    all_costs = []
    comparison_index = []
    taken_route = []

    now = time.time()
    for mot in options['current_transport_mean'].unique():
        options_m = options.loc[options['current_transport_mean'] == mot].copy()

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

            shipping_distances = pd.DataFrame(data['Shipping']['Distances']['value'],
                                              index=data['Shipping']['Distances']['index'],
                                              columns=data['Shipping']['Distances']['columns'])

            # create one big target_infrastructure dataframe for all shipping options
            for s in options_m.index:

                current_commodity_object = options_m.at[s, 'current_commodity_object']
                if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                    continue

                previous_transport_means = options_m.at[s, 'all_previous_transport_means']
                if 'Shipping' in previous_transport_means:
                    # cannot ship twice
                    continue

                transportation_costs = current_commodity_object.get_transportation_costs_specific_mean_of_transport(mot)
                current_total_costs = options_m.at[s, 'current_total_costs']

                used_infrastructure = options_m.at[s, 'all_previous_infrastructure']

                # remove also starting locations if they were already used
                start_location = options_m.loc[s, 'current_node']

                if start_location in used_infrastructure:
                    continue

                distances = shipping_distances.loc[start_location, shipping_infrastructure.index].copy()
                current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs
                current_total_costs_distances = current_total_costs_distances[current_total_costs_distances <= benchmark].dropna()

                if current_total_costs_distances.empty:
                    continue

                length_index = len(current_total_costs_distances.index)

                all_costs += current_total_costs_distances.values.tolist()

                shipping_infrastructure.loc[current_total_costs_distances.index, s] \
                    = distances.loc[current_total_costs_distances.index]

                considered_solutions.append(s)
                starting_points += [start_location] * length_index
                solution_list += [s] * length_index

                current_infrastructure += ['Shipping'] * length_index
                transport_mean += ['Shipping'] * length_index

                transportation_costs_list += [transportation_costs] * length_index

                for i in current_total_costs_distances.index:
                    comparison_index.append(i + '-' + current_commodity_object.get_name())

                taken_route += [(start_location, mot, distances.at[i], i) for i in current_total_costs_distances.index]

            processed_infrastructure['Shipping'] = shipping_infrastructure

        else:
            for s in options_m.index:

                current_commodity_object = options_m.at[s, 'current_commodity_object']
                if not current_commodity_object.get_transportation_options_specific_mean_of_transport(mot):
                    continue

                if (current_commodity_object == 'Hydrogen_Gas') & (not configuration['H2_ready_infrastructure']):
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
                    pipeline_infrastructure = data[mot][graph_id]['GeoData'].copy()

                start_location = options_m.at[s, 'current_node']

                path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'
                distances \
                    = pd.read_hdf(path_data + '/inner_infrastructure_distances/' + start_location + '.h5',
                                  mode='r', title=graph_id, dtype=np.float16)

                distances = pd.Series(distances.transpose().values[0], index=distances.index)
                distances = distances.loc[pipeline_infrastructure.index]

                current_total_costs_distances = distances / 1000 * transportation_costs + current_total_costs
                current_total_costs_distances = current_total_costs_distances[current_total_costs_distances <= benchmark].dropna()

                if current_total_costs_distances.empty:
                    continue

                length_index = len(current_total_costs_distances.index)

                all_costs += current_total_costs_distances.values.tolist()

                pipeline_infrastructure.loc[current_total_costs_distances.index, s] \
                    = distances.loc[current_total_costs_distances.index]

                considered_solutions.append(s)
                starting_points += [start_location] * length_index
                solution_list += [s] * length_index

                current_infrastructure += [graph_id] * length_index
                transport_mean += [mot] * length_index

                transportation_costs_list += [transportation_costs] * length_index

                processed_infrastructure[graph_id] = pipeline_infrastructure

                comparison_index += list(map(lambda x: x + '-' + current_commodity_object.get_name(),
                                             current_total_costs_distances.index.tolist()))

                taken_route += [(start_location, mot, distances.at[i], i) for i in current_total_costs_distances.index]

    time_get_options = time.time() - now
    now = time.time()

    if list(processed_infrastructure.values()):

        time_create = time.time()

        all_infrastructures = pd.concat(list(processed_infrastructure.values()),
                                        ignore_index=False, verify_integrity=True).transpose()
        all_infrastructures = all_infrastructures.loc[considered_solutions, :].stack().dropna().reset_index()
        all_infrastructures.columns = ['previous_solution', 'current_node', 'current_distance']

        del processed_infrastructure
        gc.collect()

        if all_infrastructures.empty:
            return pd.DataFrame()

        time_create = round(time.time() - time_create, 2)
        time_add_data = time.time()

        nodes_list = all_infrastructures['current_node'].tolist()

        all_infrastructures['starting_point'] = starting_points
        all_infrastructures['previous_solution'] = solution_list
        all_infrastructures['current_transport_mean'] = transport_mean
        all_infrastructures['current_infrastructure'] = current_infrastructure
        all_infrastructures['current_total_costs'] = all_costs
        all_infrastructures['specific_transportation_costs'] = transportation_costs_list
        all_infrastructures['comparison_index'] = comparison_index
        all_infrastructures['taken_route'] = taken_route

        all_infrastructures['current_commodity'] = options.loc[solution_list, 'current_commodity'].tolist()
        all_infrastructures['current_commodity_object'] \
            = options.loc[solution_list, 'current_commodity_object'].tolist()
        all_infrastructures['latitude'] = all_options.loc[nodes_list, 'latitude'].tolist()
        all_infrastructures['longitude'] = all_options.loc[nodes_list, 'longitude'].tolist()

        # remove duplicates
        time_assessment = time.time()
        all_infrastructures.sort_values(['current_total_costs'], inplace=True)
        all_infrastructures = all_infrastructures.drop_duplicates(subset=['comparison_index'], keep='first')

        time_assessment = round(time.time() - time_assessment, 2)

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
            all_infrastructures['minimal_total_costs_to_final_destination'] \
                = calculate_cheapest_option_to_final_destination(data, all_infrastructures,
                                                                 configuration, benchmark,
                                                                 'current_total_costs', check_minimal_distance=True)

            # throws out options to expensive
            all_infrastructures \
                = all_infrastructures[all_infrastructures['minimal_total_costs_to_final_destination'] <= benchmark]

            # next iteration either uses road or new pipeline. Remove options where closest infrastructure is already
            # further away than this distance
            max_length = max(configuration['max_length_new_segment'],
                             configuration['max_length_road']) / configuration['no_road_multiplier']
            minimal_distances = data['minimal_distances']
            all_infrastructures['minimal_distances'] \
                = minimal_distances.loc[all_infrastructures['current_node'].tolist(), 'minimal_distance'].tolist()

            all_infrastructures = all_infrastructures[all_infrastructures['minimal_distances'] <= max_length]
            all_infrastructures.drop(['minimal_distances'], axis=1, inplace=True)

        time_assess_options = time.time() - now
        # print('get: ' + str(time_get_options) + ' | assess: ' + str(time_assess_options) + ' (' + str(time_create) + ', ' + str(time_add_data) + ', ' + str(time_assessment) + ')')

        return all_infrastructures
    else:
        return pd.DataFrame()


def compare_to_local_benchmarks(solution, potential_destinations,
                                name_infrastructure_column, name_costs_column,
                                local_benchmarks):
    commodity = solution.get_current_commodity_object()

    available_benchmarks = local_benchmarks['index'].values.tolist()

    new_benchmarks = [(inf, commodity.get_name()) for inf in potential_destinations[name_infrastructure_column]]

    # index to check --> these are only unique index
    common_index = list(set(new_benchmarks).intersection(set(available_benchmarks)))
    common_index.sort(key=lambda x: x[0])

    affected_infrastructure = [ind[0] for ind in common_index]

    local_benchmarks_to_check = local_benchmarks[local_benchmarks['index'].isin(common_index)].copy()
    local_benchmarks_to_check.sort_values(['index'], inplace=True)

    affected_destinations\
        = potential_destinations[potential_destinations[name_infrastructure_column].isin(affected_infrastructure)].copy()
    affected_destinations.sort_index(inplace=True)

    not_affected_destinations\
        = potential_destinations[~potential_destinations[name_infrastructure_column].isin(affected_infrastructure)].copy()

    affected_destinations['costs_local_benchmark'] = local_benchmarks_to_check.loc[:, 'total_costs']

    # todo: check again if sorted correctly

    affected_destinations\
        = affected_destinations[affected_destinations[name_costs_column]
                                <= affected_destinations['costs_local_benchmark']].copy()

    potential_destinations = pd.concat([affected_destinations, not_affected_destinations])

    return potential_destinations
