"""Derive reusable k-best cross-country routes between pipeline border nodes."""
from __future__ import annotations

import json
from itertools import combinations

import networkx as nx
import pandas as pd
from tqdm import tqdm

BORDER_NODE_COLUMNS = [
    'pipeline_type', 'graph', 'country', 'node', 'neighbor_country',
    'neighbor_node', 'cross_border_edge_id',
]
ROUTE_COLUMNS = [
    'pipeline_type', 'graph', 'origin_country', 'destination_country',
    'origin_node', 'destination_node', 'rank', 'distance_m', 'nodes_json',
    'edge_ids_json', 'countries_json', 'bidirectional',
]


def _country(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _edge_distance(value):
    distance = float(value)
    if distance < 0:
        raise ValueError('Pipeline edge distances must be non-negative.')
    return distance


def _append_route(rows, pipeline_type, graph_id, origin, destination,
                  origin_country, destination_country, rank, path, graph,
                  node_countries):
    edge_ids = [graph[left][right]['edge_id']
                for left, right in zip(path, path[1:])]
    distance = sum(graph[left][right]['distance']
                   for left, right in zip(path, path[1:]))
    countries = []
    for node in path:
        country = node_countries.get(node)
        if country and (not countries or countries[-1] != country):
            countries.append(country)
    rows.append(dict(
        pipeline_type=pipeline_type, graph=graph_id,
        origin_country=origin_country, destination_country=destination_country,
        origin_node=origin, destination_node=destination, rank=rank,
        distance_m=distance, nodes_json=json.dumps(path),
        edge_ids_json=json.dumps(edge_ids), countries_json=json.dumps(countries),
        bidirectional=True,
    ))


def calculate_pipeline_border_routes(network_graph_data, nodes, pipeline_type,
                                      number_k_best_routes, show_progress=False):
    """Return border nodes and k best full-network cross-country routes.

    A border node is an endpoint of an edge whose other endpoint belongs to a
    different country.  The k best paths are calculated only between endpoint
    pairs in *different* countries.  The search graph is the full pipeline
    network, therefore a route can traverse one or more transit countries and
    retains all physical nodes and edge IDs.  Raw infrastructure is undirected,
    so each physical route is emitted in both directions for later directional
    flow constraints.
    """
    if number_k_best_routes < 1:
        raise ValueError('number_k_best_routes must be at least 1.')
    required_edges = {'graph', 'node_start', 'node_end', 'distance'}
    if (network_graph_data is None or network_graph_data.empty or nodes is None
            or nodes.empty or not required_edges.issubset(network_graph_data.columns)
            or 'country' not in nodes.columns):
        return (pd.DataFrame(columns=BORDER_NODE_COLUMNS),
                pd.DataFrame(columns=ROUTE_COLUMNS))

    node_countries = {str(node): _country(country)
                      for node, country in nodes['country'].items()}
    border_rows, route_rows = [], []

    for graph_id, graph_edges in network_graph_data.groupby('graph', sort=True):
        graph_edges = graph_edges.copy()
        graph_edges['_start'] = graph_edges['node_start'].astype(str)
        graph_edges['_end'] = graph_edges['node_end'].astype(str)
        border_nodes = {}
        graph = nx.Graph()

        for edge_id, edge in graph_edges.iterrows():
            start, end = edge['_start'], edge['_end']
            distance = _edge_distance(edge['distance'])
            existing = graph.get_edge_data(start, end)
            # NetworkX shortest_simple_paths requires a simple graph.  Keep the
            # shortest parallel physical segment and retain its original ID.
            if existing is None or distance < existing['distance']:
                graph.add_edge(start, end, distance=distance, edge_id=str(edge_id))

            start_country, end_country = node_countries.get(start), node_countries.get(end)
            if not start_country or not end_country or start_country == end_country:
                continue
            border_nodes[start] = start_country
            border_nodes[end] = end_country
            border_rows.extend((
                dict(pipeline_type=pipeline_type, graph=graph_id,
                     country=start_country, node=start, neighbor_country=end_country,
                     neighbor_node=end, cross_border_edge_id=str(edge_id)),
                dict(pipeline_type=pipeline_type, graph=graph_id,
                     country=end_country, node=end, neighbor_country=start_country,
                     neighbor_node=start, cross_border_edge_id=str(edge_id)),
            ))

        cross_country_pairs = [
            (origin, destination) for origin, destination in combinations(sorted(border_nodes), 2)
            if border_nodes[origin] != border_nodes[destination]
        ]
        if show_progress:
            print(
                f'[{pipeline_type}] Network {graph_id}: {len(border_nodes)} border nodes · '
                f'{len(cross_country_pairs)} cross-country combinations', flush=True)
        iterator = tqdm(
            cross_country_pairs,
            desc=f'{pipeline_type} network {graph_id}',
            unit='combination',
            disable=not show_progress,
        )
        for origin, destination in iterator:
            origin_country = border_nodes[origin]
            destination_country = border_nodes[destination]
            try:
                paths = nx.shortest_simple_paths(
                    graph, origin, destination, weight='distance')
                for rank, path in enumerate(paths, start=1):
                    _append_route(route_rows, pipeline_type, graph_id, origin,
                                  destination, origin_country, destination_country,
                                  rank, path, graph, node_countries)
                    _append_route(route_rows, pipeline_type, graph_id, destination,
                                  origin, destination_country, origin_country,
                                  rank, list(reversed(path)), graph, node_countries)
                    if rank >= number_k_best_routes:
                        break
            except nx.NetworkXNoPath:
                continue
        if show_progress:
            print(f'[{pipeline_type}] Network {graph_id}: completed', flush=True)

    border_nodes = pd.DataFrame(border_rows, columns=BORDER_NODE_COLUMNS)
    routes = pd.DataFrame(route_rows, columns=ROUTE_COLUMNS)
    if not border_nodes.empty:
        border_nodes.sort_values(
            ['pipeline_type', 'graph', 'country', 'node', 'neighbor_node'],
            kind='stable', inplace=True, ignore_index=True)
    if not routes.empty:
        routes.sort_values(
            ['pipeline_type', 'graph', 'origin_country', 'destination_country',
             'origin_node', 'destination_node', 'rank'],
            kind='stable', inplace=True, ignore_index=True)
    return border_nodes, routes


def export_pipeline_border_routes(gas_graph, gas_nodes, oil_graph, oil_nodes,
                                  output_folder, number_k_best_routes, show_progress=False):
    """Calculate and persist gas and liquid-pipeline border-route datasets."""
    border_frames, route_frames = [], []
    for pipeline_type, graph, nodes in (
            ('gas', gas_graph, gas_nodes), ('oil', oil_graph, oil_nodes)):
        if show_progress:
            print(f'[{pipeline_type}] Start pipeline border-route calculation', flush=True)
        border_nodes, routes = calculate_pipeline_border_routes(
            graph, nodes, pipeline_type, number_k_best_routes,
            show_progress=show_progress)
        border_frames.append(border_nodes)
        route_frames.append(routes)
    nonempty_borders = [frame for frame in border_frames if not frame.empty]
    nonempty_routes = [frame for frame in route_frames if not frame.empty]
    all_border_nodes = (pd.concat(nonempty_borders, ignore_index=True)
                        if nonempty_borders else pd.DataFrame(columns=BORDER_NODE_COLUMNS))
    all_routes = (pd.concat(nonempty_routes, ignore_index=True)
                  if nonempty_routes else pd.DataFrame(columns=ROUTE_COLUMNS))
    all_border_nodes.to_csv(
        str(output_folder) + '/pipeline_border_nodes.csv', index=False,
        encoding='utf-8')
    all_routes.to_csv(
        str(output_folder) + '/pipeline_border_routes.csv', index=False,
        encoding='utf-8')
    return all_border_nodes, all_routes
