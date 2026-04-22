import math

import pandas as pd
import numpy as np

from collections import defaultdict


def _get_global_conversion_lower_bound(
        commodity_object,
        source_commodity,
        target_commodity,
        convertible_locations,
        conversion_lower_bound_cache,
):
    if source_commodity == target_commodity:
        return 0.0, 1.0

    cache_key = (source_commodity, target_commodity)
    if cache_key not in conversion_lower_bound_cache:
        conversion_costs = commodity_object.get_conversion_costs_specific_commodity(
            convertible_locations,
            target_commodity,
        ).to_numpy(dtype=float, copy=False)
        conversion_efficiencies = commodity_object.get_conversion_efficiency_specific_commodity(
            convertible_locations,
            target_commodity,
        ).to_numpy(dtype=float, copy=False)

        finite_costs = conversion_costs[np.isfinite(conversion_costs)]
        finite_efficiencies = conversion_efficiencies[np.isfinite(conversion_efficiencies) & (conversion_efficiencies > 0)]

        if finite_costs.size == 0 or finite_efficiencies.size == 0:
            conversion_lower_bound_cache[cache_key] = (math.inf, 0.0)
        else:
            conversion_lower_bound_cache[cache_key] = (float(finite_costs.min()), float(finite_efficiencies.max()))

    return conversion_lower_bound_cache[cache_key]


def _calculate_shipping_required_candidates(
        positions,
        current_costs,
        distances,
        used_shipping_mask,
        c_start,
        c_start_object,
        commodity_names,
        commodity_objects,
        final_commodities,
        destination_conversion_cache,
        locations_in_destination,
        convertible_locations,
        conversion_lower_bound_cache,
):
    if positions.size == 0:
        return np.array([], dtype=float), np.array([], dtype=object)

    best_values = np.full(positions.size, math.inf, dtype=float)
    best_commodities = np.full(positions.size, 'basic_costs', dtype=object)
    c_start_conversion_options = c_start_object.get_conversion_options()

    for c_ship in commodity_names:
        c_ship_object = commodity_objects[c_ship]
        if not c_ship_object.get_transportation_options().get('Shipping', False):
            continue

        if c_start != c_ship:
            if not c_start_conversion_options[c_ship]:
                continue

            global_conversion_costs, global_conversion_efficiency = _get_global_conversion_lower_bound(
                c_start_object,
                c_start,
                c_ship,
                convertible_locations,
                conversion_lower_bound_cache,
            )
            if (not np.isfinite(global_conversion_costs)) or global_conversion_efficiency <= 0:
                continue

            shipping_base_costs = (current_costs + global_conversion_costs) / global_conversion_efficiency
        else:
            shipping_base_costs = current_costs.copy()

        distance_series = pd.Series(distances, copy=False)
        shipping_base_series = pd.Series(shipping_base_costs, copy=False)
        duration_series = distance_series / 1000 / c_ship_object.get_shipping_speed()
        _, shipping_total_costs = c_ship_object.get_distance_and_duration_based_costs_and_efficiency_shipping(
            distance_series,
            duration_series,
            shipping_base_series,
        )
        with np.errstate(invalid='ignore'):
            shipping_transport_costs = shipping_total_costs.to_numpy(dtype=float, copy=False) - shipping_base_costs

        if used_shipping_mask.any():
            shipping_transport_costs = np.asarray(shipping_transport_costs, dtype=float).copy()
            shipping_transport_costs[used_shipping_mask] = math.inf

        c_ship_conversion_options = c_ship_object.get_conversion_options()
        for c_end in commodity_names:
            if c_end not in final_commodities:
                continue

            if c_ship != c_end:
                if not c_ship_conversion_options[c_end]:
                    continue

                cache_key = (c_ship, c_end)
                if cache_key not in destination_conversion_cache:
                    conversion_costs_at_destination = c_ship_object.get_conversion_costs_specific_commodity(
                        locations_in_destination,
                        c_end,
                    ).fillna(math.inf)
                    conversion_efficiency_at_destination = c_ship_object.get_conversion_efficiency_specific_commodity(
                        locations_in_destination,
                        c_end,
                    ).fillna(math.inf)
                    destination_conversion_cache[cache_key] = (
                        conversion_costs_at_destination.to_numpy(dtype=float, copy=False),
                        conversion_efficiency_at_destination.to_numpy(dtype=float, copy=False),
                    )

                reconversion_costs, reconversion_efficiency = destination_conversion_cache[cache_key]
                candidate_values = np.min(
                    (shipping_base_costs[:, None] + shipping_transport_costs[:, None] + reconversion_costs[None, :])
                    / reconversion_efficiency[None, :],
                    axis=1,
                )
            else:
                candidate_values = shipping_base_costs + shipping_transport_costs

            update_mask = candidate_values < best_values
            if np.any(update_mask):
                best_values[update_mask] = candidate_values[update_mask]
                best_commodities[update_mask] = c_end

    return best_values, best_commodities


def calculate_cheapest_option_to_final_destination(data, branches, benchmarks, cost_column_name):

    """
    The method iterates over all possible combinations of
    conversion - transportation based on direct distance to final destination - conversion
    to calculate the lowest possible cost to the final destination

    @param dict data: dictionary with general data
    @param pandas.DataFrame branches: current branches which includes current commodity and location
    @param dict benchmarks: current benchmarks
    @param str cost_column_name: name of the total cost column in options
    @return: options with 'costs to final destination' column representing minimal total costs possible
    """

    number_branches = len(branches.index)
    if number_branches == 0:
        return [], []

    means_of_transport = data['transport_means']
    commodity_objects = data['commodities']['commodity_objects']
    commodity_names = list(commodity_objects.keys())
    final_commodities = set(data['commodities']['final_commodities'])

    all_locations = data['conversion_costs_and_efficiencies']
    conversion_possible = all_locations['conversion_possible']

    branches['current_node_conversion'] = True
    if not conversion_possible.all():
        no_conversion_possible_locations = set(conversion_possible[~conversion_possible].index)
        no_conversion_mask = branches['current_node'].isin(no_conversion_possible_locations).to_numpy()
        branches.loc[no_conversion_mask, 'current_node_conversion'] = False

    current_commodities = branches['current_commodity'].to_numpy()
    current_costs_all = branches[cost_column_name].to_numpy(dtype=float, copy=False)
    distances_all = branches['distance_to_final_destination'].to_numpy(dtype=float, copy=False)
    current_transport_means = branches['current_transport_mean'].to_numpy()
    current_nodes_all = branches['current_node'].to_numpy()
    current_node_conversion_all = branches['current_node_conversion'].to_numpy(dtype=bool, copy=False)
    current_continents_all = branches['current_continent'].to_numpy()

    best_values = np.full(number_branches, math.inf, dtype=float)
    best_commodities = np.full(number_branches, 'basic_costs', dtype=object)

    locations_in_destination = data['destination']['infrastructure'].index.tolist()
    destination_conversion_cache = {}
    continent_connections = data.get('continent_connections', {})
    reachable_continents = continent_connections.get('reachable_continents', {})
    destination_continent = data['destination'].get('continent')
    convertible_locations = all_locations.index[conversion_possible].tolist()
    conversion_lower_bound_cache = {}

    for c_start in pd.unique(current_commodities):
        positions = np.flatnonzero(current_commodities == c_start)
        if positions.size == 0:
            continue

        c_start_object = commodity_objects[c_start]
        c_start_conversion_options = c_start_object.get_conversion_options()

        current_costs = current_costs_all[positions]
        distances = distances_all[positions]
        current_nodes = current_nodes_all[positions]
        can_convert_at_start = current_node_conversion_all[positions]
        used_shipping_mask = current_transport_means[positions] == 'Shipping'
        current_continents = current_continents_all[positions]
        shipping_required_mask = np.array([
            destination_continent not in reachable_continents.get(current_continent, [current_continent])
            for current_continent in current_continents
        ], dtype=bool)

        if shipping_required_mask.any():
            shipping_required_values, shipping_required_commodities = _calculate_shipping_required_candidates(
                positions[shipping_required_mask],
                current_costs[shipping_required_mask],
                distances[shipping_required_mask],
                used_shipping_mask[shipping_required_mask],
                c_start,
                c_start_object,
                commodity_names,
                commodity_objects,
                final_commodities,
                destination_conversion_cache,
                locations_in_destination,
                convertible_locations,
                conversion_lower_bound_cache,
            )

            update_mask = shipping_required_values < best_values[positions[shipping_required_mask]]
            if np.any(update_mask):
                positions_to_update = positions[shipping_required_mask][update_mask]
                best_values[positions_to_update] = shipping_required_values[update_mask]
                best_commodities[positions_to_update] = shipping_required_commodities[update_mask]

        non_shipping_required_positions = positions[~shipping_required_mask]
        if non_shipping_required_positions.size == 0:
            continue

        for c_transported in commodity_names:
            if c_start != c_transported:
                if not c_start_conversion_options[c_transported]:
                    continue

                conversion_costs = np.full(non_shipping_required_positions.size, math.inf, dtype=float)
                non_shipping_can_convert_at_start = can_convert_at_start[~shipping_required_mask]
                non_shipping_current_nodes = current_nodes[~shipping_required_mask]
                non_shipping_current_costs = current_costs[~shipping_required_mask]
                if non_shipping_can_convert_at_start.any():
                    convertible_nodes = non_shipping_current_nodes[non_shipping_can_convert_at_start].tolist()
                    conversion_costs_first = c_start_object.get_conversion_costs_specific_commodity(convertible_nodes, c_transported)
                    conversion_efficiency_first = c_start_object.get_conversion_efficiency_specific_commodity(convertible_nodes, c_transported)
                    conversion_costs[non_shipping_can_convert_at_start] = (
                        non_shipping_current_costs[non_shipping_can_convert_at_start]
                        + conversion_costs_first.to_numpy(dtype=float, copy=False)
                    ) / conversion_efficiency_first.to_numpy(dtype=float, copy=False)
            else:
                conversion_costs = current_costs[~shipping_required_mask].copy()

            if conversion_costs.min() > benchmarks[c_transported]:
                continue

            c_transported_object = commodity_objects[c_transported]
            transportation_options = c_transported_object.get_transportation_options()
            c_transported_conversion_options = c_transported_object.get_conversion_options()
            non_shipping_distances = distances[~shipping_required_mask]
            non_shipping_used_shipping_mask = used_shipping_mask[~shipping_required_mask]

            for mean_of_transport in means_of_transport:
                if not transportation_options[mean_of_transport]:
                    continue

                if mean_of_transport != 'Shipping':
                    transport_costs = (
                        c_transported_object.get_transportation_costs_specific_mean_of_transport(mean_of_transport) / 1000
                    ) * non_shipping_distances
                else:
                    distance_series = pd.Series(non_shipping_distances, copy=False)
                    conversion_series = pd.Series(conversion_costs, copy=False)
                    duration_series = distance_series / 1000 / c_transported_object.get_shipping_speed()
                    _, new_costs = c_transported_object.get_distance_and_duration_based_costs_and_efficiency_shipping(
                        distance_series,
                        duration_series,
                        conversion_series,
                    )
                    with np.errstate(invalid='ignore'):
                        transport_costs = new_costs.to_numpy(dtype=float, copy=False) - conversion_costs
                    if non_shipping_used_shipping_mask.any():
                        transport_costs = np.asarray(transport_costs, dtype=float).copy()
                        transport_costs[non_shipping_used_shipping_mask] = math.inf

                for c_end in commodity_names:
                    if c_end not in final_commodities:
                        continue

                    if c_transported != c_end:
                        if not c_transported_conversion_options[c_end]:
                            continue

                        cache_key = (c_transported, c_end)
                        if cache_key not in destination_conversion_cache:
                            conversion_costs_at_destination = c_transported_object.get_conversion_costs_specific_commodity(
                                locations_in_destination,
                                c_end,
                            ).fillna(math.inf)
                            conversion_efficiency_at_destination = c_transported_object.get_conversion_efficiency_specific_commodity(
                                locations_in_destination,
                                c_end,
                            ).fillna(math.inf)
                            destination_conversion_cache[cache_key] = (
                                conversion_costs_at_destination.to_numpy(dtype=float, copy=False),
                                conversion_efficiency_at_destination.to_numpy(dtype=float, copy=False),
                            )

                        reconversion_costs, reconversion_efficiency = destination_conversion_cache[cache_key]
                        candidate_values = np.min(
                            (conversion_costs[:, None] + transport_costs[:, None] + reconversion_costs[None, :])
                            / reconversion_efficiency[None, :],
                            axis=1,
                        )
                    else:
                        candidate_values = conversion_costs + transport_costs

                    update_mask = candidate_values < best_values[non_shipping_required_positions]
                    if np.any(update_mask):
                        positions_to_update = non_shipping_required_positions[update_mask]
                        best_values[positions_to_update] = candidate_values[update_mask]
                        best_commodities[positions_to_update] = c_end

    return best_values.tolist(), best_commodities.tolist()


def calculate_cheapest_option_to_closest_infrastructure(data, branches, configuration, benchmarks, cost_column_name):

    """
    This method calculates the lowest transportation costs from different locations to their closest infrastructure
    conversion - transportation to the closest infrastructure via road or new infrastructure - conversion

    This method is applied after the in-tolerance-transportation took place. That means that next transportation
    will either be road or new pipeline. We can calculate the minimal costs to the next infrastructure by using
    the distance to the closest infrastructure of each location. As road and new pipelines are quite expensive
    transport means, we might exceed the benchmark quite fast and are able to remove several branches because if the
    transport to the closest infrastructure is already more expensive than the benchmark, so will be the transport
    to all other infrastructure nodes

    @param dict data: dictionary with general data
    @param pandas.DataFrame branches: current branches which include current commodity and location
    @param dict configuration: dictionary with assumptions and settings
    @param dict benchmarks: current benchmarks for assessment
    @param str cost_column_name: column of options where information on current costs is saved
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

    if False in all_locations['conversion_possible']:
        no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    else:
        no_conversion_possible_locations = []

    no_conversion_possible_branches = branches[branches['current_node'].isin(no_conversion_possible_locations)].index
    branches.loc[no_conversion_possible_branches, 'current_node_conversion'] = False

    branches['closest_node_conversion'] = True

    if False in all_locations['conversion_possible']:
        no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    else:
        no_conversion_possible_locations = []

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
                    c_start_df.loc[first_location_conversion_possible, c_transported + '-conversion_costs'] = \
                        (c_start_df.loc[first_location_conversion_possible, cost_column_name] + conversion_costs_first) / conversion_efficiency_first

                    # calculate conversion costs for locations where conversion is impossible
                    c_start_df.loc[first_location_conversion_not_possible, c_transported + '-conversion_costs'] = math.inf
                else:
                    continue
            else:
                # also no conversion is possible and c_start = c_transported is transported
                c_start_df[c_transported + '-conversion_costs'] = c_start_df[cost_column_name]

            fuel_price = data['commodities']['strike_prices'][c_transported]
            if c_start_df[c_transported + '-conversion_costs'].min() > benchmarks[c_transported]:
                # if all conversion costs are already higher than benchmark no further calculations will be made
                # as benchmark is already violated
                continue

            c_transported_object = data['commodities']['commodity_objects'][c_transported]
            transportation_options = c_transported_object.get_transportation_options()
            c_transported_conversion_options = c_transported_object.get_conversion_options()
            c_transported_transportation_costs = c_transported_object.get_transportation_costs()

            for m in ['New_Pipeline_Gas', 'New_Pipeline_Liquid', 'Road']:

                c_start_df[c_transported + '-transportation_costs-' + m] = math.inf

                if m != 'Road':
                    if transportation_options[m]:
                        # new possible
                        c_start_df['new_distance'] = c_start_df['minimal_distance'].apply(lambda x: min(max_length_new_segment / no_road_multiplier, x))

                        if transportation_options['Road']:
                            # road and new possible --> cover as much with new, rest with road
                            c_start_df.loc[possible, 'road_distance'] \
                                = (c_start_df.loc[possible, 'minimal_distance'] - c_start_df.loc[possible, 'new_distance']).apply(lambda x: min(max_length_road / no_road_multiplier, x))

                            c_start_df.loc[possible, c_transported + '-transportation_costs-' + m] \
                                = c_transported_transportation_costs['Road'] / 1000 * c_start_df.loc[possible, 'road_distance'] \
                                + c_transported_transportation_costs[m] / 1000 * c_start_df.loc[possible, 'new_distance'] * no_road_multiplier

                        else:
                            # no road possible, only new
                            c_start_df.loc[possible_only_new, c_transported + '-transportation_costs-' + m] \
                                = c_transported_transportation_costs[m] / 1000 * c_start_df.loc[possible_only_new, 'new_distance'] * no_road_multiplier

                    else:
                        # new not possible
                        if transportation_options['Road']:
                            # no new possible, only road --> use only road

                            c_start_df.loc[possible_only_road, 'road_distance'] = c_start_df.loc[possible_only_road, 'minimal_distance']
                            c_start_df.loc[possible_only_road, c_transported + '-transportation_costs-' + m] \
                                = c_start_df.loc[possible_only_road, 'road_distance'] / 1000 * c_transported_transportation_costs['Road'] * no_road_multiplier
                        else:
                            # not possible to transport at all
                            continue

                else:
                    if transportation_options['Road']:
                        c_start_df.loc[possible_only_road, 'road_distance'] = c_start_df.loc[possible_only_road, 'minimal_distance']
                        c_start_df.loc[possible_only_road, c_transported + '-transportation_costs-' + m] \
                            = c_start_df.loc[possible_only_road, 'road_distance'] / 1000 * c_transported_transportation_costs['Road'] * no_road_multiplier
                    else:
                        # not possible to transport at all
                        continue

                # on top to the transportation costs, we can calculate the minimal conversion costs to one of the
                # final commodity at the closest node
                for c_end in [*data['commodities']['commodity_objects'].keys()]:

                    name_column = 'costs-' + c_start + '-' + c_transported + '-' + m + '-' + c_end

                    if c_end in final_commodities:
                        if c_transported != c_end:
                            if c_transported_conversion_options[c_end]:

                                # # get conversion costs and efficiency of locations where conversion is possible
                                # conversion_costs_second \
                                #     = c_transported_object.get_conversion_costs_specific_commodity(c_start_df.loc[second_location_conversion_possible, 'closest_node'],
                                #                                                                    c_end)
                                # conversion_costs_second.index = second_location_conversion_possible
                                #
                                # conversion_efficiency_second \
                                #     = c_transported_object.get_conversion_efficiency_specific_commodity(c_start_df.loc[second_location_conversion_possible, 'closest_node'],
                                #                                                                         c_end)
                                # conversion_efficiency_second.index = second_location_conversion_possible

                                # get conversion costs and efficiency of locations where conversion is possible
                                conversion_costs_second = c_transported_object.get_conversion_costs()[c_end]
                                conversion_efficiency_second = c_transported_object.get_conversion_efficiencies()[c_end]

                                total_costs = []
                                for efficiency in conversion_efficiency_second.unique():
                                    min_eff_index = conversion_efficiency_second[conversion_efficiency_second == efficiency].index
                                    min_costs = conversion_costs_second.loc[min_eff_index].min()

                                    costs = (c_start_df.loc[:, c_transported + '-conversion_costs']
                                     + c_start_df.loc[:, c_transported + '-transportation_costs-' + m]
                                     + min_costs) / efficiency

                                    total_costs.append(costs)

                                total_costs = pd.concat(total_costs, axis=1)
                                cheapest_options.loc[:, name_column] = total_costs.min(axis=1)

                                # # calculate conversion costs for locations where conversion is possible
                                # cheapest_options.loc[second_location_conversion_possible, name_column] = \
                                #     (c_start_df.loc[second_location_conversion_possible, c_transported + '-conversion_costs']
                                #      + c_start_df.loc[second_location_conversion_possible, c_transported + '-transportation_costs-' + m]
                                #      + conversion_costs_second) / conversion_efficiency_second

                                # # if no conversion at location possible, we use minimal conversion costs and efficiency
                                # min_conversion_costs = c_transported_object.get_minimal_conversion_costs(c_end)
                                # min_conversion_efficiency = c_transported_object.get_minimal_conversion_efficiency(c_end)
                                #
                                # cheapest_options.loc[second_location_conversion_not_possible, name_column] = \
                                #     (c_start_df.loc[second_location_conversion_not_possible, c_transported + '-conversion_costs']
                                #      + c_start_df.loc[second_location_conversion_not_possible, c_transported + '-transportation_costs-' + m]
                                #      + min_conversion_costs) / min_conversion_efficiency

                                created_columns.append(name_column)
                            else:
                                continue
                        else:
                            cheapest_options.loc[c_start_df.index, name_column] \
                                = c_start_df[c_transported + '-conversion_costs'] \
                                + c_start_df[c_transported + '-transportation_costs-' + m]
                            created_columns.append(name_column)
                    else:
                        continue

    # make sure that number columns are number columns
    cheapest_options = cheapest_options.apply(pd.to_numeric, errors="ignore")

    # get min values and respective commodity
    min_values = cheapest_options[created_columns].min(axis=1).tolist()
    if min_values:
        columns = cheapest_options[created_columns].select_dtypes(include="number").idxmin(axis=1).tolist()
        min_commodities = [c.split('-')[-1] for c in columns]
    else:
        min_commodities = []

    return min_values, min_commodities


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

    if False in all_locations['conversion_possible'].tolist():
        no_conversion_possible_locations = all_locations[~all_locations['conversion_possible']].index.tolist()
    else:
        no_conversion_possible_locations = []

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

    end_commodity_columns = defaultdict(list)  # to consider fuel price

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
                created_columns.append(c_start + '_' + m)
                end_commodity_columns[c_start].append(c_start + '_' + m)

        for c_conversion in [*data['commodities']['commodity_objects'].keys()]:
            # todo: actually you require two conversions if the conversion from e.g. ammonia to FTF is not defined
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
                            cheapest_options.loc[location_conversion_possible, c_start + '_' + c_conversion + '_' + m] = \
                                (c_start_df.loc[location_conversion_possible, cost_column_name] + conversion_costs) / conversion_efficiency

                            # set not possible conversion branches to infinity
                            cheapest_options.loc[location_conversion_not_possible, c_start + '_' + c_conversion + '_' + m] = math.inf

                            created_columns.append(c_start + '_' + c_conversion + '_' + m)
                            end_commodity_columns[c_conversion].append(c_start + '_' + c_conversion + '_' + m)

    # reduce cheapest costs by fuel price
    for commodity in end_commodity_columns.keys():
        affected_columns = end_commodity_columns[commodity]

        fuel_price = data['commodities']['strike_prices'][commodity]

        cheapest_options[affected_columns] = cheapest_options[affected_columns] - fuel_price

    pipeline_gas_columns = [c for c in created_columns if 'Pipeline_Gas' in c]
    pipeline_gas_cheapest_options = cheapest_options[pipeline_gas_columns].min(axis=1).tolist()

    pipeline_liquid_columns = [c for c in created_columns if 'Pipeline_Liquid' in c]
    pipeline_liquid_cheapest_options = cheapest_options[pipeline_liquid_columns].min(axis=1).tolist()

    return pipeline_gas_cheapest_options, pipeline_liquid_cheapest_options
