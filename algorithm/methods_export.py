import os
import tempfile

import pandas as pd

from algorithm.methods_algorithm import (create_new_branches_based_on_conversion,
                                          drop_branch_comparison_columns,
                                          postprocessing_branches)


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


def attach_infrastructure_countries(complete_infrastructure, world):
    """Fill missing node countries; explicit port countries always take precedence."""
    infrastructure = complete_infrastructure.copy()
    if 'country' not in infrastructure.columns:
        infrastructure['country'] = None

    missing = infrastructure['country'].isna()
    if not missing.any() or world is None or world.empty:
        return infrastructure

    country_column = next((c for c in ('NAME_EN', 'name', 'country') if c in world.columns), None)
    if country_column is None:
        return infrastructure

    # Covers is intentional: a node exactly on a border may match more than one
    # polygon. Such ambiguity is left unresolved instead of assigning it silently.
    for node, row in infrastructure.loc[missing].iterrows():
        from shapely.geometry import Point
        point = Point(row['longitude'], row['latitude'])
        matches = world[world.geometry.apply(lambda geometry: geometry.covers(point))]
        countries = matches[country_column].dropna().unique().tolist()
        if len(countries) == 1:
            infrastructure.at[node, 'country'] = countries[0]
    return infrastructure


def _foreign_pipeline_neighbours(data, domestic_nodes):
    neighbours = set()
    for transport_mean in ('Pipeline_Gas', 'Pipeline_Liquid'):
        for network in data.get(transport_mean, {}).values():
            graph = network.get('Graph')
            if graph is None:
                continue
            graph_nodes = set(graph.nodes)
            for node in graph_nodes.intersection(domestic_nodes):
                neighbours.update(graph.neighbors(node))
    return neighbours


def get_export_infrastructure_scope(data, complete_infrastructure, start_country):
    """Limit expansion to domestic infrastructure and genuine border terminals."""
    countries = complete_infrastructure['country'].map(normalize_country)
    home = normalize_country(start_country)
    domestic = set(complete_infrastructure.index[countries == home])

    # Every foreign port is a possible landing point for shipping, but it becomes
    # terminal immediately. Explicit port metadata avoids coastal border errors.
    foreign_ports = set(complete_infrastructure.index[
        (complete_infrastructure['current_transport_mean'] == 'Shipping')
        & countries.notna()
        & (countries != home)
    ])
    foreign_pipeline_nodes = _foreign_pipeline_neighbours(data, domestic)
    foreign_pipeline_nodes = {
        node for node in foreign_pipeline_nodes
        if node in complete_infrastructure.index
        and countries.at[node] is not None
        and countries.at[node] != home
    }

    allowed = domestic | foreign_ports | foreign_pipeline_nodes
    return complete_infrastructure.loc[complete_infrastructure.index.intersection(allowed)].copy()


def process_export_out_tolerance_branches(complete_infrastructure, branches,
                                          configuration, iteration, data, benchmarks,
                                          limitation=None, use_minimal_distance=False):
    """Create road/new-pipeline approaches only towards domestic infrastructure."""
    from algorithm.methods_routing import process_out_tolerance_branches

    home = normalize_country(data['export_start_country'])
    domestic = complete_infrastructure[
        complete_infrastructure['country'].map(normalize_country) == home].copy()
    if domestic.empty or branches.empty:
        return pd.DataFrame()
    if use_minimal_distance:
        # The main runner's global closest-node cache may point abroad. The
        # export runner therefore skips that preselection and lets every input
        # branch reach the country-scoped calculation below.
        return pd.DataFrame({'previous_branch': branches.index}, index=branches.index)
    return process_out_tolerance_branches(
        domestic, branches, configuration, iteration, data, benchmarks,
        limitation=limitation, use_minimal_distance=False)


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
    """Create every feasible conversion branch without benchmark comparison."""
    if branches.empty:
        return branches.copy(), branch_number
    unlimited = {commodity: float('inf')
                 for commodity in data['commodities']['commodity_objects']}
    converted, branch_number = create_new_branches_based_on_conversion(
        branches, data, branch_number, unlimited)
    converted = drop_branch_comparison_columns(converted)
    converted = postprocessing_branches(converted, branches)
    return converted, branch_number
