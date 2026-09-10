"""Derive reusable k-best cross-country routes between pipeline border nodes."""
from __future__ import annotations

import json
from itertools import combinations
from multiprocessing import get_context

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
_WORKER_GRAPHS = {}
_WORKER_COUNTRIES = {}
_WORKER_PIPELINE_TYPE = ''
_WORKER_K = 1


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


def _build_network(graph_edges, node_countries, pipeline_type, graph_id):
    """Build one graph and identify its cross-country border nodes once."""
    graph = nx.Graph()
    border_nodes, border_rows = {}, []
    for edge_id, edge in graph_edges.iterrows():
        start, end = str(edge['node_start']), str(edge['node_end'])
        distance = _edge_distance(edge['distance'])
        existing = graph.get_edge_data(start, end)
        # shortest_simple_paths requires a simple graph.  Retain the shortest
        # parallel raw segment and its original identifier.
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
    pairs = [
        (graph_id, origin, destination, border_nodes[origin], border_nodes[destination])
        for origin, destination in combinations(sorted(border_nodes), 2)
        if border_nodes[origin] != border_nodes[destination]
    ]
    return graph, border_rows, pairs


def _calculate_pair(graph, node_countries, pipeline_type, graph_id, origin,
                    destination, origin_country, destination_country,
                    number_k_best_routes):
    """Calculate all k paths for one independent border-node combination."""
    route_rows = []
    try:
        paths = nx.shortest_simple_paths(graph, origin, destination,
                                         weight='distance')
        for rank, path in enumerate(paths, start=1):
            _append_route(route_rows, pipeline_type, graph_id, origin, destination,
                          origin_country, destination_country, rank, path, graph,
                          node_countries)
            _append_route(route_rows, pipeline_type, graph_id, destination, origin,
                          destination_country, origin_country, rank,
                          list(reversed(path)), graph, node_countries)
            if rank >= number_k_best_routes:
                break
    except nx.NetworkXNoPath:
        pass
    return graph_id, route_rows


def _initialize_worker(graphs, node_countries, pipeline_type, number_k_best_routes):
    global _WORKER_GRAPHS, _WORKER_COUNTRIES, _WORKER_PIPELINE_TYPE, _WORKER_K
    _WORKER_GRAPHS = graphs
    _WORKER_COUNTRIES = node_countries
    _WORKER_PIPELINE_TYPE = pipeline_type
    _WORKER_K = number_k_best_routes


def _calculate_pair_worker(task):
    graph_id, origin, destination, origin_country, destination_country = task
    return _calculate_pair(
        _WORKER_GRAPHS[graph_id], _WORKER_COUNTRIES, _WORKER_PIPELINE_TYPE,
        graph_id, origin, destination, origin_country, destination_country,
        _WORKER_K)


def calculate_pipeline_border_routes(network_graph_data, nodes, pipeline_type,
                                      number_k_best_routes, number_workers=1,
                                      show_progress=False):
    """Return k-best full-network routes between border nodes in different countries.

    Transit countries remain in each path's node, edge and country sequences.
    Each cross-country border-node combination is one parallel task, while each
    worker receives the built pipeline graphs only once.
    """
    if number_k_best_routes < 1:
        raise ValueError('number_k_best_routes must be at least 1.')
    if number_workers < 1:
        raise ValueError('number_workers must be at least 1.')
    required_edges = {'graph', 'node_start', 'node_end', 'distance'}
    if (network_graph_data is None or network_graph_data.empty or nodes is None
            or nodes.empty or not required_edges.issubset(network_graph_data.columns)
            or 'country' not in nodes.columns):
        return (pd.DataFrame(columns=BORDER_NODE_COLUMNS),
                pd.DataFrame(columns=ROUTE_COLUMNS))

    node_countries = {str(node): _country(country)
                      for node, country in nodes['country'].items()}
    graphs, border_rows, tasks, total_by_graph = {}, [], [], {}
    for graph_id, graph_edges in network_graph_data.groupby('graph', sort=True):
        graph, network_borders, network_tasks = _build_network(
            graph_edges, node_countries, pipeline_type, graph_id)
        graphs[graph_id] = graph
        border_rows.extend(network_borders)
        tasks.extend(network_tasks)
        total_by_graph[graph_id] = len(network_tasks)
        if show_progress:
            border_count = len({row['node'] for row in network_borders})
            print(f'[{pipeline_type}] Network {graph_id}: {border_count} border nodes · '
                  f'{len(network_tasks)} cross-country combinations', flush=True)

    route_rows, complete_by_graph = [], {graph_id: 0 for graph_id in graphs}
    workers = min(int(number_workers), len(tasks)) if tasks else 1
    if workers > 1:
        # The spawn context is safe on Windows.  Graphs are supplied once per
        # worker through the initializer rather than once per combination.
        context = get_context('spawn')
        with context.Pool(
                processes=workers, initializer=_initialize_worker,
                initargs=(graphs, node_countries, pipeline_type, number_k_best_routes)) as pool:
            iterator = tqdm(
                pool.imap_unordered(_calculate_pair_worker, tasks, chunksize=1),
                total=len(tasks), desc=f'{pipeline_type} border combinations',
                unit='combination', disable=not show_progress)
            for graph_id, rows in iterator:
                route_rows.extend(rows)
                complete_by_graph[graph_id] += 1
                if show_progress and complete_by_graph[graph_id] == total_by_graph[graph_id]:
                    print(f'[{pipeline_type}] Network {graph_id}: completed', flush=True)
    else:
        iterator = tqdm(tasks, total=len(tasks), desc=f'{pipeline_type} border combinations',
                        unit='combination', disable=not show_progress)
        for graph_id, origin, destination, origin_country, destination_country in iterator:
            _, rows = _calculate_pair(
                graphs[graph_id], node_countries, pipeline_type, graph_id, origin,
                destination, origin_country, destination_country,
                number_k_best_routes)
            route_rows.extend(rows)
            complete_by_graph[graph_id] += 1
            if show_progress and complete_by_graph[graph_id] == total_by_graph[graph_id]:
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
                                  output_folder, number_k_best_routes,
                                  number_workers=1, show_progress=False):
    """Calculate and persist gas and liquid-pipeline border-route datasets."""
    border_frames, route_frames = [], []
    for pipeline_type, graph, nodes in (
            ('gas', gas_graph, gas_nodes), ('oil', oil_graph, oil_nodes)):
        if show_progress:
            print(f'[{pipeline_type}] Start pipeline border-route calculation', flush=True)
        border_nodes, routes = calculate_pipeline_border_routes(
            graph, nodes, pipeline_type, number_k_best_routes,
            number_workers=number_workers, show_progress=show_progress)
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
