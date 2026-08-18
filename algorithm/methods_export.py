import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd

from algorithm.methods_algorithm import (drop_branch_comparison_columns,
                                          postprocessing_branches,
                                          update_branch_comparison_index)
from algorithm.methods_geographic import calc_distance_list_to_list
from algorithm.object_commodity import create_commodity_objects
from data_processing.configuration import load_technology_data


def prepare_export_commodities(config_file, location_data, data):
    """Create every configured commodity relevant to export routing."""
    conversion_data, transportation_data = load_technology_data(config_file)
    return create_commodity_objects(
        location_data, data['conversion_costs_and_efficiencies'], conversion_data,
        transportation_data, config_file.copy())


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
    continent = data['start']['continent']
    rows = []
    for number, commodity in enumerate(data['commodities']['commodity_objects'].values()):
        name = commodity.get_name()
        branch = 'S' + str(number)
        production_costs = commodity.get_production_costs()
        efficiency = commodity.get_starting_efficiency()
        rows.append({
            'branch_index': branch,
            'starting_latitude': location.y,
            'starting_longitude': location.x,
            'latitude': location.y,
            'longitude': location.x,
            'previous_branch': None,
            'current_commodity': name,
            'current_commodity_object': commodity,
            'current_continent': continent,
            'current_total_costs': production_costs,
            'current_transportation_costs': 0,
            'current_conversion_costs': 0,
            'current_transport_mean': None,
            'current_infrastructure': None,
            'current_node': 'Start',
            'current_distance': 0,
            'all_previous_commodities': [name],
            'all_previous_total_costs': [production_costs],
            'all_previous_transportation_costs': [],
            'all_previous_conversion_costs': [],
            'all_previous_transport_means': [None],
            'all_previous_infrastructure': [],
            'all_previous_nodes': ['Start'],
            'all_previous_branches': [branch],
            'all_previous_distances': [0],
            'taken_routes': [(name, efficiency)],
            'total_efficiency': efficiency,
        })
    branches = pd.DataFrame(rows).set_index('branch_index', drop=False)
    branches.index.name = None
    return branches, len(branches)


def normalize_country(country):
    """Return a stable country label for comparisons."""
    if country is None or pd.isna(country):
        return None
    return str(country).strip().casefold()


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


def get_export_infrastructure_scope(complete_infrastructure, start_country):
    """Keep infrastructure located in the start country only."""
    countries = complete_infrastructure['country'].map(normalize_country)
    home = normalize_country(start_country)
    return complete_infrastructure.loc[countries == home].copy()


def apply_export_local_benchmark(branches, local_benchmarks):
    """Keep the cheapest branch for each local node/commodity/connector state."""
    if branches.empty:
        return branches.copy(), branches.copy(), local_benchmarks
    assessed = update_branch_comparison_index(branches.copy())
    assessed.sort_values('current_total_costs', inplace=True, kind='stable')
    keep = []
    remove = []
    seen_this_batch = set()
    for branch_index, branch in assessed.iterrows():
        state = branch['comparison_index']
        costs = branch['current_total_costs']
        previous = local_benchmarks.get(state)
        if state in seen_this_batch:
            remove.append(branch_index)
            continue
        if previous is None or costs <= previous['current_total_costs']:
            keep.append(branch_index)
            seen_this_batch.add(state)
            local_benchmarks[state] = {
                'current_total_costs': costs,
                'branch_index': branch['branch_index'],
            }
        else:
            remove.append(branch_index)
    surviving = drop_branch_comparison_columns(assessed.loc[keep].copy())
    terminated = drop_branch_comparison_columns(assessed.loc[remove].copy())
    if not terminated.empty:
        terminated['status'] = 'local_benchmark_pruned'
    return surviving, terminated, local_benchmarks


def process_export_out_tolerance_branches(domestic_infrastructure, branches,
                                          configuration):
    """Create every reachable domestic road/new-pipeline branch without cost pruning."""
    if domestic_infrastructure.empty or branches.empty:
        return pd.DataFrame()

    distances = calc_distance_list_to_list(
        domestic_infrastructure['latitude'], domestic_infrastructure['longitude'],
        branches['latitude'], branches['longitude'])
    values = np.asarray(distances).transpose()
    results = []
    for column, branch_index in enumerate(branches.index):
        branch = branches.loc[branch_index]
        commodity = branch['current_commodity_object']
        visited = set(branch['all_previous_nodes'])
        for row, node in enumerate(domestic_infrastructure.index):
            if node == branch['current_node'] or node in visited:
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
                results.append({
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
                })
    return pd.DataFrame(results)


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


def process_export_zero_distance_branches(data, branches, complete_infrastructure):
    """Create co-located infrastructure transfers without costs or target assessment."""
    results = []
    tolerance_locations = data.get('in_tolerance_locations', {})
    for branch_index, branch in branches.iterrows():
        visited_nodes = set(branch['all_previous_nodes'])
        visited_infrastructure = set(branch['all_previous_infrastructure'])
        for node in tolerance_locations.get(branch['current_node'], []):
            if node == branch['current_node'] or node in visited_nodes:
                continue
            if node not in complete_infrastructure.index:
                continue
            graph = complete_infrastructure.at[node, 'graph']
            if (isinstance(graph, str) and graph in visited_infrastructure):
                continue
            results.append({
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
            })
    return pd.DataFrame(results)


def _shipping_distances(configuration):
    path = os.path.join(configuration['path_processed_data'],
                        'inner_infrastructure_distances', 'port_distances.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    distances = pd.read_csv(path, index_col=0, dtype=str, sep=None, engine='python',
                            keep_default_na=False)
    return np.ceil(distances.apply(pd.to_numeric, errors='raise'))


def process_export_infrastructure_branches(data, branches, complete_infrastructure, configuration):
    """Create every valid shipping and existing-pipeline continuation."""
    results = []
    shipping_distances = None
    for branch_index, branch in branches.iterrows():
        transport_mean = branch['current_transport_mean']
        if transport_mean not in ('Shipping', 'Pipeline_Gas', 'Pipeline_Liquid'):
            continue
        commodity = branch['current_commodity_object']
        if not commodity.get_transportation_options_specific_mean_of_transport(transport_mean):
            continue
        if (transport_mean.startswith('Pipeline') and commodity.get_name() == 'Hydrogen_Gas'
                and not configuration['H2_ready_infrastructure']):
            continue
        graph_id = branch.get('graph')
        used = set(branch['all_previous_infrastructure'])
        if transport_mean == 'Shipping':
            if 'Shipping' in branch['all_previous_transport_means']:
                continue
            if shipping_distances is None:
                shipping_distances = _shipping_distances(configuration)
            if shipping_distances.empty or branch['current_node'] not in shipping_distances.index:
                continue
            targets = complete_infrastructure.index.intersection(shipping_distances.columns)
            distances = shipping_distances.loc[branch['current_node'], targets]
            infrastructure_id = 'Shipping'
        else:
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
            if node in set(branch['all_previous_nodes']):
                continue
            if transport_mean == 'Shipping':
                duration = distance / 1000 / commodity.get_shipping_speed()
                efficiency, totals = commodity.get_distance_and_duration_based_costs_and_efficiency_shipping(
                    pd.Series([distance], index=[node]), pd.Series([duration], index=[node]),
                    branch['current_total_costs'])
                total_costs = totals.at[node]
                route_efficiency = efficiency.at[node]
                total_efficiency = branch['total_efficiency'] * route_efficiency
                transport_costs = total_costs - branch['current_total_costs']
            else:
                route_efficiency = 1
                transport_costs = distance / 1000 * specific_costs
                total_costs = branch['current_total_costs'] + transport_costs
                total_efficiency = branch['total_efficiency']
            results.append({
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
            })
    return pd.DataFrame(results)


def split_export_branches(branches, complete_infrastructure, start_country):
    """Separate branches that first reached infrastructure in another country."""
    if branches.empty:
        return branches.copy(), branches.copy()
    countries = complete_infrastructure['country'].map(normalize_country)
    branch_countries = branches['current_node'].map(countries)
    exported = branch_countries.notna() & (branch_countries != normalize_country(start_country))
    active = branches.loc[~exported].copy()
    completed = branches.loc[exported].copy()
    if not completed.empty:
        completed['export_country'] = branch_countries.loc[completed.index]
        completed['status'] = 'export_infrastructure_reached'
    return active, completed


def export_branch_snapshot(branches, path_results, location_index, iteration, stage):
    """Atomically write a regular, complete branch snapshot."""
    folder = os.path.join(path_results, 'export_infrastructure_branches', str(location_index))
    os.makedirs(folder, exist_ok=True)
    filename = f'{iteration:05d}_{stage}.csv'
    destination = os.path.join(folder, filename)
    handle, temporary = tempfile.mkstemp(prefix=filename + '.', suffix='.tmp', dir=folder)
    os.close(handle)
    try:
        branches.to_csv(temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination


def apply_export_conversion(branches, data, branch_number):
    """Create every technically feasible conversion branch."""
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
                conversion_costs = 0
                efficiency = 1
            elif not conversion_options[end_name]:
                continue
            else:
                nodes = pd.Series([branch['current_node']], index=[previous_branch])
                conversion_costs = start.get_conversion_costs_specific_commodity(nodes, end_name).iloc[0]
                efficiency = start.get_conversion_efficiency_specific_commodity(nodes, end_name).iloc[0]
            total_costs = ((branch['current_total_costs'] + conversion_costs) / efficiency
                           if end_name != start_name else branch['current_total_costs'])
            row = branch.copy()
            row['previous_branch'] = previous_branch
            row['current_commodity'] = end_name
            row['current_commodity_object'] = end
            row['current_total_costs'] = total_costs
            row['current_conversion_costs'] = total_costs - branch['current_total_costs']
            row['current_transportation_costs'] = 0
            row['current_distance'] = 0
            row['taken_route'] = (start_name, end_name, efficiency)
            row['total_efficiency'] = branch['total_efficiency'] * efficiency
            row['branch_index'] = 'S' + str(branch_number + len(rows))
            rows.append(row)
    converted = pd.DataFrame(rows)
    if converted.empty:
        return converted, branch_number
    inherited_columns = [column for column in converted.columns
                         if column.startswith('all_previous_')]
    inherited_columns += [column for column in
                          ('taken_routes', 'starting_latitude', 'starting_longitude')
                          if column in converted.columns]
    converted.drop(columns=inherited_columns, inplace=True)
    converted.index = converted['branch_index']
    converted.index.name = None
    branch_number += len(converted)
    converted = postprocessing_branches(converted, branches)
    return converted, branch_number
