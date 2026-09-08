import os
import tempfile
from typing import NamedTuple

import networkx as nx
import numpy as np
import pandas as pd

from algorithm.methods_conversion import calculate_conversion_costs, calculate_conversion_costs_increase
from algorithm.methods_geographic import calc_distance_list_to_list
from algorithm.object_commodity import create_commodity_objects
from data_processing.configuration import load_technology_data


class ExportRouteStep(NamedTuple):
    branch_index: str
    parent: object
    commodity: str
    total_costs: float
    transportation_costs: float
    conversion_costs: float
    transport_mean: object
    infrastructure: object
    node: str
    distance: float
    taken_route: object


def _make_export_route_step(branch, parent=None):
    return ExportRouteStep(
        branch_index=branch['branch_index'], parent=parent,
        commodity=branch['current_commodity'], total_costs=branch['current_total_costs'],
        transportation_costs=branch['current_transportation_costs'],
        conversion_costs=branch['current_conversion_costs'],
        transport_mean=branch['current_transport_mean'],
        infrastructure=branch['current_infrastructure'], node=branch['current_node'],
        distance=branch['current_distance'], taken_route=branch.get('taken_route'))


def materialize_export_branches(branches, parent_branches):
    """Attach shared parent-chain state without copying complete route histories."""
    if branches.empty:
        return branches.copy()
    result = branches.copy()
    route_steps = []
    visited_nodes = []
    visited_infrastructure = []
    starting_latitudes = []
    starting_longitudes = []
    previous_commodities = []
    for _, branch in result.iterrows():
        parent = parent_branches.loc[branch['previous_branch']]
        route_steps.append(_make_export_route_step(branch, parent['_route_step']))
        visited_nodes.append(parent['_visited_nodes'] | frozenset([branch['current_node']]))
        infrastructure = branch['current_infrastructure']
        addition = (frozenset([infrastructure]) if isinstance(infrastructure, str)
                    else frozenset())
        visited_infrastructure.append(parent['_visited_infrastructure'] | addition)
        starting_latitudes.append(parent['starting_latitude'])
        starting_longitudes.append(parent['starting_longitude'])
        previous_commodities.append(parent['current_commodity'])
    result['_route_step'] = route_steps
    result['_visited_nodes'] = visited_nodes
    result['_visited_infrastructure'] = visited_infrastructure
    result['starting_latitude'] = starting_latitudes
    result['starting_longitude'] = starting_longitudes
    result['previous_commodity'] = previous_commodities
    return result


def _collect_k_best_candidate(candidates, candidate, number_k_best_routes):
    """Keep only k cheapest not-yet-materialized candidates per node/commodity."""
    state = (str(candidate['current_node']), str(candidate['current_commodity']))
    state_candidates = candidates.setdefault(state, [])
    if len(state_candidates) < number_k_best_routes:
        state_candidates.append(candidate)
        return
    worst_position = max(
        range(len(state_candidates)),
        key=lambda position: state_candidates[position]['current_total_costs'])
    if candidate['current_total_costs'] < state_candidates[worst_position]['current_total_costs']:
        state_candidates[worst_position] = candidate


def _candidate_frame(candidates):
    return pd.DataFrame([
        candidate for state_candidates in candidates.values()
        for candidate in state_candidates
    ])


def _route_contains_invalid(route_step, invalid_branches):
    while route_step is not None:
        if route_step.branch_index in invalid_branches:
            return True
        route_step = route_step.parent
    return False


def apply_export_k_best(branches, k_best_routes, invalid_branches,
                        number_k_best_routes):
    """Keep k cheapest branches per node/commodity and invalidate descendants."""
    if branches.empty:
        return branches.copy(), 0, k_best_routes, invalid_branches
    if number_k_best_routes < 1:
        raise ValueError('number_k_best_routes must be at least 1.')

    invalid = set(invalid_branches)
    ordered = branches.assign(
        _k_best_branch_order=branches['branch_index'].astype(str)).sort_values(
            ['current_total_costs', '_k_best_branch_order'], kind='stable')
    ordered.drop(columns=['_k_best_branch_order'], inplace=True)
    for branch_index, branch in ordered.iterrows():
        if _route_contains_invalid(branch['_route_step'].parent, invalid):
            invalid.add(branch_index)
            continue
        state = (str(branch['current_node']), str(branch['current_commodity']))
        ranking = [entry for entry in k_best_routes.get(state, [])
                   if not _route_contains_invalid(entry['route_step'], invalid)]
        existing = next((entry for entry in ranking
                         if entry['branch_index'] == branch_index), None)
        if existing is not None:
            k_best_routes[state] = ranking
            continue
        entry = {'branch_index': branch_index,
                 'current_total_costs': branch['current_total_costs'],
                 'route_step': branch['_route_step']}
        ranking.append(entry)
        ranking.sort(key=lambda item: (item['current_total_costs'], item['branch_index']))
        if len(ranking) > number_k_best_routes:
            removed = ranking.pop()
            invalid.add(removed['branch_index'])
        k_best_routes[state] = ranking

    changed = True
    while changed:
        changed = False
        for state, ranking in list(k_best_routes.items()):
            valid_ranking = []
            for entry in ranking:
                if _route_contains_invalid(entry['route_step'], invalid):
                    if entry['branch_index'] not in invalid:
                        invalid.add(entry['branch_index'])
                        changed = True
                else:
                    valid_ranking.append(entry)
            k_best_routes[state] = valid_ranking

    surviving_mask = branches['_route_step'].map(
        lambda route_step: not _route_contains_invalid(route_step, invalid))
    surviving = branches.loc[surviving_mask].copy()
    return (surviving, int((~surviving_mask).sum()),
            k_best_routes, invalid)


def prepare_export_commodities(config_file, location_data, data):
    """Create target commodities plus intermediates required for conversion."""
    conversion_data, transportation_data = load_technology_data(config_file)
    config_file = config_file.copy()
    targets = list(dict.fromkeys(config_file['target_commodity']))
    unknown_targets = [commodity for commodity in targets
                       if commodity not in config_file['available_commodity']]
    if unknown_targets:
        raise ValueError('Target commodities are not available commodities: '
                         + ', '.join(unknown_targets))
    config_file['available_commodity'] = [
        commodity for commodity in config_file['available_commodity']
        if commodity in targets
        or conversion_data[commodity]['potential_conversions']
    ]
    return create_commodity_objects(
        location_data, data['conversion_costs_and_efficiencies'], conversion_data,
        transportation_data, config_file)


def get_complete_export_infrastructure(data):
    """Collect ports and pipeline nodes used by export routing."""
    frames = []
    ports = data.get('Shipping', {}).get('ports')
    if ports is not None and not ports.empty:
        ports = ports.copy()
        ports['current_transport_mean'] = 'Shipping'
        ports['graph'] = None
        frames.append(ports)
    for transport_mean in ('Pipeline_Gas', 'Pipeline_Liquid'):
        for graph_name, network in data.get(transport_mean, {}).items():
            nodes = network['NodeLocations'].copy()
            nodes['current_transport_mean'] = transport_mean
            if 'graph' not in nodes.columns:
                nodes['graph'] = graph_name
            frames.append(nodes)
    if not frames:
        return pd.DataFrame(columns=['latitude', 'longitude', 'country',
                                     'current_transport_mean', 'graph', 'infrastructure'])
    infrastructure = pd.concat(frames)
    infrastructure['infrastructure'] = infrastructure.index
    return infrastructure


def create_export_branches_at_start(data):
    """Create initial commodity branches with production state only."""
    location = data['start']['location']
    rows = []
    for number, commodity in enumerate(data['commodities']['commodity_objects'].values()):
        name = commodity.get_name()
        branch = 'S' + str(number)
        production_costs = commodity.get_production_costs()
        efficiency = commodity.get_starting_efficiency()
        row = {
            'branch_index': branch,
            'starting_latitude': location.y,
            'starting_longitude': location.x,
            'latitude': location.y,
            'longitude': location.x,
            'previous_branch': None,
            'current_commodity': name,
            'current_commodity_object': commodity,
            'current_total_costs': production_costs,
            'current_transportation_costs': 0,
            'current_conversion_costs': 0,
            'current_transport_mean': None,
            'current_infrastructure': None,
            'current_node': 'Start',
            'current_distance': 0,
            'taken_route': (name, efficiency),
            'total_efficiency': efficiency,
        }
        row['_route_step'] = _make_export_route_step(pd.Series(row))
        row['_visited_nodes'] = frozenset(['Start'])
        row['_visited_infrastructure'] = frozenset()
        rows.append(row)
    branches = pd.DataFrame(rows).set_index('branch_index', drop=False)
    branches.index.name = None
    return branches, len(branches)


def normalize_country(country):
    """Return a canonical country label without repairing aliases at runtime."""
    if country is None or pd.isna(country):
        return None
    return str(country).strip()


def get_start_country(location_data):
    """Read the country of a single start location."""
    for column in ('country_start', 'country'):
        if column in location_data.columns:
            country = location_data.iloc[0][column]
            if normalize_country(country) is not None:
                return country
    raise ValueError('The start location has no country_start value.')


def attach_infrastructure_countries(complete_infrastructure, world, target_country=None):
    """Assign a target country after a cheap bounding-box prefilter."""
    infrastructure = complete_infrastructure.copy()
    if 'country' not in infrastructure.columns:
        infrastructure['country'] = None

    if target_country is None:
        missing = infrastructure['country'].isna()
        if not missing.any() or world is None or world.empty:
            return infrastructure
        country_column = next((c for c in ('NAME_EN', 'name', 'country') if c in world.columns), None)
        if country_column is None:
            return infrastructure
        for node, row in infrastructure.loc[missing].iterrows():
            from shapely.geometry import Point
            point = Point(row['longitude'], row['latitude'])
            matches = world[world.geometry.apply(lambda geometry: geometry.covers(point))]
            countries = matches[country_column].dropna().unique().tolist()
            if len(countries) == 1:
                infrastructure.at[node, 'country'] = countries[0]
        return infrastructure

    home = normalize_country(target_country)
    explicit_home = infrastructure['country'].map(normalize_country) == home
    explicit = infrastructure.loc[explicit_home].copy()
    missing = infrastructure['country'].isna()
    if not missing.any():
        return explicit
    if world is None or world.empty:
        return explicit

    country_column = next((c for c in ('NAME_EN', 'name', 'country') if c in world.columns), None)
    if country_column is None:
        return explicit
    country_rows = world[world[country_column].map(normalize_country) == home]
    if country_rows.empty:
        raise ValueError('Start country not found in world polygons: ' + str(target_country))

    country_geometry = country_rows.geometry.unary_union
    min_longitude, min_latitude, max_longitude, max_latitude = country_geometry.bounds
    candidates = infrastructure.loc[missing]
    candidates = candidates[
        candidates['longitude'].between(min_longitude, max_longitude)
        & candidates['latitude'].between(min_latitude, max_latitude)
    ].copy()

    # Only the usually much smaller bounding-box subset reaches the exact test.
    from shapely.geometry import Point
    inside = []
    for node, row in candidates.iterrows():
        point = Point(row['longitude'], row['latitude'])
        if country_geometry.covers(point):
            inside.append(node)
    candidates = candidates.loc[inside].copy()
    candidates['country'] = target_country
    return pd.concat([explicit, candidates], axis=0)


def process_export_out_tolerance_branches(domestic_infrastructure, branches, configuration,
                                          number_k_best_routes):
    """Create every technically valid road/new-pipeline branch."""
    if domestic_infrastructure.empty or branches.empty:
        return pd.DataFrame()

    distances = calc_distance_list_to_list(
        domestic_infrastructure['latitude'], domestic_infrastructure['longitude'],
        branches['latitude'], branches['longitude'])
    values = np.asarray(distances).transpose()
    results = {}
    for column, branch_index in enumerate(branches.index):
        branch = branches.loc[branch_index]
        commodity = branch['current_commodity_object']
        visited = branch['_visited_nodes']
        visited_infrastructure = branch['_visited_infrastructure']
        for row, node in enumerate(domestic_infrastructure.index):
            if node == branch['current_node'] or node in visited:
                continue
            node_infrastructure = domestic_infrastructure.at[node, 'graph']
            if isinstance(node_infrastructure, str):
                node_infrastructure = {node_infrastructure}
            elif isinstance(node_infrastructure, (list, tuple, set)):
                node_infrastructure = set(node_infrastructure)
            else:
                node_infrastructure = set()
            if visited_infrastructure.intersection(node_infrastructure):
                continue
            direct_distance = float(values[row, column])
            options = []
            if (commodity.get_transportation_options_specific_mean_of_transport('Road')
                    and branch['current_transport_mean'] not in
                    ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid']
                    and direct_distance <= (configuration['max_length_road']
                                            / configuration['no_road_multiplier'])):
                options.append(('Road', commodity.get_transportation_costs_specific_mean_of_transport('Road')))
            for transport_mean in ('New_Pipeline_Gas', 'New_Pipeline_Liquid'):
                if (commodity.get_transportation_options_specific_mean_of_transport(transport_mean)
                        and branch['current_transport_mean'] not in
                        ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid']
                        and direct_distance <= (configuration['max_length_new_segment']
                                                / configuration['no_road_multiplier'])):
                    options.append((transport_mean,
                                    commodity.get_transportation_costs_specific_mean_of_transport(transport_mean)))
            for transport_mean, specific_costs in options:
                routed_distance = (0 if direct_distance <= configuration['tolerance_distance']
                                   else direct_distance * configuration['no_road_multiplier'])
                transport_costs = routed_distance * specific_costs / 1000
                candidate = {
                    'previous_branch': branch_index,
                    'current_node': node,
                    'current_distance': routed_distance,
                    'current_transport_mean': transport_mean,
                    'current_infrastructure': None,
                    'current_commodity': branch['current_commodity'],
                    'current_commodity_object': commodity,
                    'current_transportation_costs': transport_costs,
                    'current_total_costs': branch['current_total_costs'] + transport_costs,
                    'latitude': domestic_infrastructure.at[node, 'latitude'],
                    'longitude': domestic_infrastructure.at[node, 'longitude'],
                    'taken_route': (branch['current_node'], transport_mean, routed_distance, node, 1),
                    'total_efficiency': branch['total_efficiency'],
                }
                _collect_k_best_candidate(results, candidate, number_k_best_routes)

    return _candidate_frame(results)


def prepare_export_infrastructure_branches(branches, complete_infrastructure):
    """Attach the existing transport network used at the current node."""
    prepared = branches.copy()
    if prepared.empty:
        return prepared
    prepared['graph'] = None
    for index in prepared.index:
        node = prepared.at[index, 'current_node']
        if node not in complete_infrastructure.index:
            continue
        transport_mean = complete_infrastructure.at[node, 'current_transport_mean']
        if transport_mean in ('Pipeline_Gas', 'Pipeline_Liquid', 'Shipping'):
            prepared.at[index, 'current_transport_mean'] = transport_mean
            prepared.at[index, 'graph'] = complete_infrastructure.at[node, 'graph']
    return prepared


def process_export_zero_distance_branches(data, branches, complete_infrastructure,
                                          number_k_best_routes):
    """Create co-located infrastructure transfers without costs or target assessment."""
    results = {}
    tolerance_locations = data.get('in_tolerance_locations', {})
    for branch_index, branch in branches.iterrows():
        visited_nodes = branch['_visited_nodes']
        visited_infrastructure = branch['_visited_infrastructure']
        for node in tolerance_locations.get(branch['current_node'], []):
            if node == branch['current_node'] or node in visited_nodes:
                continue
            if node not in complete_infrastructure.index:
                continue
            graph = complete_infrastructure.at[node, 'graph']
            if (isinstance(graph, str) and graph in visited_infrastructure):
                continue
            candidate = {
                'previous_branch': branch_index,
                'current_node': node,
                'current_distance': 0,
                'current_transport_mean': 'Road',
                'current_infrastructure': None,
                'current_commodity': branch['current_commodity'],
                'current_commodity_object': branch['current_commodity_object'],
                'current_transportation_costs': 0,
                'current_total_costs': branch['current_total_costs'],
                'latitude': complete_infrastructure.at[node, 'latitude'],
                'longitude': complete_infrastructure.at[node, 'longitude'],
                'taken_route': (branch['current_node'], 'Road', 0, node, 1),
                'total_efficiency': branch['total_efficiency'],
            }
            _collect_k_best_candidate(results, candidate, number_k_best_routes)
    return _candidate_frame(results)


def process_export_infrastructure_branches(data, branches, complete_infrastructure, configuration,
                                           number_k_best_routes):
    """Create existing-pipeline continuations; ports are terminal nodes."""
    results = {}
    for branch_index, branch in branches.iterrows():
        transport_mean = branch['current_transport_mean']
        if transport_mean not in ('Pipeline_Gas', 'Pipeline_Liquid'):
            continue
        if not np.isfinite(branch['current_total_costs']):
            continue
        commodity = branch['current_commodity_object']
        if not commodity.get_transportation_options_specific_mean_of_transport(transport_mean):
            continue
        if (transport_mean.startswith('Pipeline') and commodity.get_name() == 'Hydrogen_Gas'
                and not configuration['H2_ready_infrastructure']):
            continue
        graph_id = branch.get('graph')
        used = branch['_visited_infrastructure']
        if graph_id in used or graph_id not in data[transport_mean]:
            continue
        if configuration['use_low_storage']:
            distances = pd.Series(nx.single_source_dijkstra_path_length(
                data[transport_mean][graph_id]['Graph'], branch['current_node']))
        else:
            path = os.path.join(configuration['path_processed_data'],
                                'inner_infrastructure_distances', branch['current_node'] + '.h5')
            stored = pd.read_hdf(path, mode='r', title=graph_id)
            distances = pd.Series(np.ceil(stored.iloc[:, 0].to_numpy()), index=stored.index)
        distances = distances.loc[distances.index.intersection(complete_infrastructure.index)]
        infrastructure_id = graph_id
        distances = distances.drop(index=branch['current_node'], errors='ignore').dropna()
        specific_costs = commodity.get_transportation_costs_specific_mean_of_transport(transport_mean)
        for node, distance in distances.items():
            if node in branch['_visited_nodes']:
                continue
            route_efficiency = 1
            transport_costs = distance / 1000 * specific_costs
            total_costs = branch['current_total_costs'] + transport_costs
            total_efficiency = branch['total_efficiency']
            candidate = {
                'previous_branch': branch_index,
                'current_node': node,
                'current_distance': distance,
                'current_transport_mean': transport_mean,
                'current_infrastructure': infrastructure_id,
                'current_commodity': branch['current_commodity'],
                'current_commodity_object': commodity,
                'current_transportation_costs': transport_costs,
                'current_total_costs': total_costs,
                'latitude': complete_infrastructure.at[node, 'latitude'],
                'longitude': complete_infrastructure.at[node, 'longitude'],
                'taken_route': (branch['current_node'], transport_mean, distance, node, route_efficiency),
                'total_efficiency': total_efficiency,
            }
            _collect_k_best_candidate(results, candidate, number_k_best_routes)
    return _candidate_frame(results)


def export_branch_snapshot(branches, path_results, location_index, iteration, stage):
    """Atomically stream a complete snapshot reconstructed from parent chains."""
    folder = os.path.join(path_results, 'export_infrastructure_branches', str(location_index))
    os.makedirs(folder, exist_ok=True)
    filename = f'{iteration:05d}_{stage}.csv'
    destination = os.path.join(folder, filename)
    handle, temporary = tempfile.mkstemp(prefix=filename + '.', suffix='.tmp', dir=folder)
    os.close(handle)
    try:
        if '_route_step' not in branches.columns:
            branches.to_csv(temporary, index=False)
        else:
            first_chunk = True
            chunk_size = 10000
            for start in range(0, len(branches), chunk_size):
                chunk = branches.iloc[start:start + chunk_size].copy()
                histories = [_reconstruct_export_history(step)
                             for step in chunk['_route_step']]
                for column in histories[0]:
                    chunk[column] = [history[column] for history in histories]
                chunk.drop(columns=['_route_step', '_visited_nodes',
                                    '_visited_infrastructure',
                                    'current_commodity_object'],
                           errors='ignore', inplace=True)
                chunk.to_csv(temporary, mode='w' if first_chunk else 'a',
                             header=first_chunk, index=False)
                first_chunk = False
            if first_chunk:
                branches.drop(columns=['_route_step', '_visited_nodes',
                                       '_visited_infrastructure',
                                       'current_commodity_object'],
                              errors='ignore').to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination


def _reconstruct_export_history(route_step):
    steps = []
    while route_step is not None:
        steps.append(route_step)
        route_step = route_step.parent
    steps.reverse()
    return {
        'all_previous_transport_means': [step.transport_mean for step in steps],
        'all_previous_infrastructure': [step.infrastructure for step in steps[1:]],
        'all_previous_nodes': [step.node for step in steps],
        'all_previous_branches': [step.branch_index for step in steps],
        'all_previous_distances': [step.distance for step in steps],
        'all_previous_transportation_costs': [
            step.transportation_costs for step in steps[1:]],
        'all_previous_conversion_costs': [step.conversion_costs for step in steps[1:]],
        'all_previous_total_costs': [step.total_costs for step in steps],
        'all_previous_commodities': [step.commodity for step in steps],
        'taken_routes': [step.taken_route for step in steps],
    }


def update_export_node_results(node_results, branches, target_commodities,
                               infrastructure_nodes):
    """Record minima for output without using them to prune active branches."""
    if branches.empty:
        return node_results
    targets = set(target_commodities)
    nodes = set(str(node) for node in infrastructure_nodes)
    relevant = branches[
        branches['current_commodity'].isin(targets)
        & branches['current_node'].astype(str).isin(nodes)
    ]
    for _, branch in relevant.iterrows():
        key = (str(branch['current_node']), str(branch['current_commodity']))
        costs = branch['current_total_costs']
        previous = node_results.get(key)
        if previous is None or costs < previous['current_total_costs']:
            node_results[key] = {
                'current_total_costs': costs,
                'total_efficiency': branch['total_efficiency'],
            }
    return node_results


def export_node_results_snapshot(node_results, path_results, location_index, iteration,
                                 stage='node_results'):
    """Write passive minimum-cost results without branch histories."""
    rows = [{
        'current_node': node,
        'current_commodity': commodity,
        'current_total_costs': result['current_total_costs'],
        'total_efficiency': result['total_efficiency'],
    } for (node, commodity), result in node_results.items()]
    snapshot = pd.DataFrame(rows, columns=[
        'current_node', 'current_commodity', 'current_total_costs',
        'total_efficiency'])
    if not snapshot.empty:
        snapshot.sort_values(
            ['current_total_costs'], inplace=True, kind='stable')
        snapshot.drop_duplicates(
            subset=['current_node', 'current_commodity'], keep='first', inplace=True)
        snapshot.sort_values(
            ['current_node', 'current_commodity'], inplace=True, kind='stable')
    return export_branch_snapshot(
        snapshot, path_results, location_index, iteration, stage)


def apply_export_conversion(branches, data, branch_number):
    """Create every technically feasible conversion branch without cost pruning."""
    if branches.empty:
        return branches.copy(), branch_number
    rows = []
    commodities = data['commodities']['commodity_objects']
    for previous_branch, branch in branches.iterrows():
        start_name = branch['current_commodity']
        start = commodities[start_name]
        conversion_options = start.get_conversion_options()
        for end_name, end in commodities.items():
            if end_name == start_name:
                continue
            if not conversion_options[end_name]:
                continue
            nodes = pd.Series([branch['current_node']], index=[previous_branch])
            conversion_costs = start.get_conversion_costs_specific_commodity(nodes, end_name).iloc[0]
            efficiency = start.get_conversion_efficiency_specific_commodity(nodes, end_name).iloc[0]
            if (not np.isfinite(branch['current_total_costs'])
                    or not np.isfinite(conversion_costs)
                    or not np.isfinite(efficiency)
                    or efficiency <= 0):
                continue
            total_costs = calculate_conversion_costs(
                branch['current_total_costs'], conversion_costs, efficiency)
            if not np.isfinite(total_costs):
                continue
            row = branch.copy()
            row['previous_branch'] = previous_branch
            row['current_commodity'] = end_name
            row['current_commodity_object'] = end
            row['current_total_costs'] = total_costs
            row['current_conversion_costs'] = calculate_conversion_costs_increase(
                branch['current_total_costs'], conversion_costs, efficiency)
            row['current_transportation_costs'] = 0
            row['current_distance'] = 0
            row['taken_route'] = (start_name, end_name, efficiency)
            row['total_efficiency'] = branch['total_efficiency'] * efficiency
            rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return branches.copy(), branch_number
    candidates.drop(columns=['_route_step', '_visited_nodes', '_visited_infrastructure'],
                    errors='ignore', inplace=True)
    candidates['branch_index'] = ['S' + str(branch_number + i) for i in range(len(candidates))]
    converted = candidates
    converted.index = converted['branch_index']
    converted.index.name = None
    branch_number += len(converted)
    converted = materialize_export_branches(converted, branches)
    return pd.concat([converted, branches], ignore_index=False), branch_number
