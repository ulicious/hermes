import logging
import math
import os

import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

from algorithm.methods_geographic import calc_distance_list_to_list, calc_distance_single_to_single


logger = logging.getLogger(__name__)


def _as_float_tolerance(value):
    if value is None:
        return 0
    if isinstance(value, str):
        if value == 'math.inf':
            return math.inf
        return float(value)
    return value


def _graph_name_from_node(node):
    if not isinstance(node, str) or '_Node_' not in node:
        return None
    return node.rsplit('_Node_', 1)[0]


def create_transport_edges(distance_options, commodities, techno_economic_data_transport,
                           show_progress=False):
    """Attach permitted commodities and techno-economic values to directed segments."""
    edges = {}
    max_costs = 0

    for transport_mean, distances in distance_options.items():
        if distances.empty:
            continue
        if not {'pointA', 'pointB', 'distance'}.issubset(distances.columns):
            logger.warning('Skip %s transport edges because distance columns are incomplete', transport_mean)
            continue
        logger.debug('Create %s transport edges from %s directed distances',
                     transport_mean, len(distances))
        row_iterator = tqdm(distances.itertuples(), total=len(distances),
                            desc='Create ' + transport_mean + ' edges',
                            disable=not show_progress)
        for row in row_iterator:
            if row.pointA == row.pointB:
                continue
            if transport_mean in {'Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'}:
                start_graph = _graph_name_from_node(row.pointA)
                end_graph = _graph_name_from_node(row.pointB)
                if start_graph is not None and start_graph == end_graph:
                    continue
            for commodity in commodities:
                if commodity not in techno_economic_data_transport:
                    logger.warning('Skip commodity %s for %s because transportation techno-economic data is missing',
                                   commodity, transport_mean)
                    continue
                if transport_mean not in techno_economic_data_transport[commodity]['potential_transportation']:
                    continue

                transport_costs = row.distance / 1000 * \
                    techno_economic_data_transport[commodity][transport_mean] / 1000
                transport_losses = 0
                max_costs = max(max_costs, transport_costs)

                if transport_mean == 'Shipping':
                    technology = techno_economic_data_transport[commodity]
                    boil_off = 0
                    if technology['Boil_Off'] > 0:
                        duration = row.distance / 1000 / technology['Shipping_Speed']
                        boil_off = duration / 24 * technology['Boil_Off']

                    self_consumption = 0
                    if technology['Uses_Commodity_as_Shipping_Fuel']:
                        self_consumption = row.distance / 1000 * technology['Self_Consumption']
                        transport_costs = 0

                    transport_losses = max(boil_off, self_consumption)

                key = row.pointA + '+' + commodity + '-' + row.pointB + '+' + commodity + '-' + transport_mean
                edges[key] = ('transport', row.pointA + '+' + commodity, row.pointB + '+' + commodity,
                              transport_costs, transport_losses, commodity, transport_mean)

    return edges, max_costs


def load_static_mip_graph_legacy(path_mip_data):
    """Legacy CSV loader retained for comparison and backwards-compatible diagnostics."""
    logger.debug('Load static MIP graph artifacts from CSV using legacy reconstruction')
    nodes = pd.read_csv(path_mip_data + 'static_nodes.csv', index_col=0).index.tolist()
    conversion_data = pd.read_csv(path_mip_data + 'static_conversion_edges.csv', index_col=0)
    transport_data = pd.read_csv(path_mip_data + 'static_transport_edges.csv', index_col=0)
    conversion_edges = {
        key: (row.start, row.end, row.costs, row.efficiency, row.end_commodity)
        for key, row in tqdm(conversion_data.iterrows(), total=len(conversion_data),
                             desc='Load static conversion edges')
    }
    transport_edges = {
        key: (row.start, row.end, row.costs, row.efficiency, row.commodity, row['mean'])
        for key, row in tqdm(transport_data.iterrows(), total=len(transport_data),
                             desc='Load static transport edges')
    }
    logger.debug('Reconstructed %s conversion and %s transport edges',
                 len(conversion_edges), len(transport_edges))
    edges = {key: ('conversion',) + value for key, value in conversion_edges.items()}
    edges.update({key: ('transport',) + value for key, value in transport_edges.items()})
    max_costs = transport_data['costs'].max() if not transport_data.empty else 0
    return {
        'nodes': nodes,
        'edges': edges,
        'conversion_edges': conversion_data,
        'transport_edges': transport_data,
        'max_costs': max_costs,
    }


def load_static_mip_graph(path_mip_data):
    """
    Load static MIP graph data efficiently.

    Newly processed input contains one binary file holding all global nodes
    and edges. For existing CSV-only preprocessing output, reconstruction
    uses column arrays instead of `iterrows()` and migrates it to that file.
    """
    graph_file = path_mip_data + 'static_graph.pkl'
    if os.path.exists(graph_file):
        logger.debug('Load static MIP graph from %s', graph_file)
        static_graph = pd.read_pickle(graph_file)
        logger.debug('Loaded %s nodes and %s static edges from cache',
                     len(static_graph['nodes']), len(static_graph['edges']))
        return static_graph

    logger.debug('Binary graph file absent; migrate existing CSV artifacts to %s', graph_file)
    nodes = pd.read_csv(path_mip_data + 'static_nodes.csv', index_col=0).index.tolist()
    conversion_data = pd.read_csv(path_mip_data + 'static_conversion_edges.csv', index_col=0)
    transport_data = pd.read_csv(path_mip_data + 'static_transport_edges.csv', index_col=0)

    conversion_values = zip(
        conversion_data['start'], conversion_data['end'], conversion_data['costs'],
        conversion_data['efficiency'], conversion_data['end_commodity'])
    conversion_edges = dict(zip(conversion_data.index, conversion_values))
    transport_values = zip(
        transport_data['start'], transport_data['end'], transport_data['costs'],
        transport_data['efficiency'], transport_data['commodity'], transport_data['mean'])
    transport_edges = dict(zip(transport_data.index, transport_values))
    edges = {key: ('conversion',) + value for key, value in conversion_edges.items()}
    edges.update({key: ('transport',) + value for key, value in transport_edges.items()})
    static_graph = {
        'nodes': nodes,
        'edges': edges,
        'conversion_edges': conversion_data,
        'transport_edges': transport_data,
        'max_costs': transport_data['costs'].max() if not transport_data.empty else 0,
    }
    pd.to_pickle(static_graph, graph_file)
    logger.debug('Reconstructed %s conversion and %s transport edges; saved binary graph file',
                 len(conversion_edges), len(transport_edges))
    return static_graph


def prepare_destination_mip_data(options, destination, path_processed_data=None, destination_tolerance=0,
                                 include_tolerance_for_polygons=False):
    """Determine the infrastructure nodes accepted as sinks for one destination.

    Country/polygon destinations mirror the heuristic: infrastructure inside
    the destination polygon is accepted directly. Infrastructure outside the
    polygon is accepted if it lies within the destination tolerance to any
    infrastructure point inside the destination polygon. The tolerance is not
    measured to the polygon boundary.
    """
    destination_tolerance = _as_float_tolerance(destination_tolerance)
    destination_infrastructure = []
    if options.empty or not {'longitude', 'latitude'}.issubset(options.columns):
        logger.warning('No destination infrastructure can be selected because options are empty '
                       'or longitude/latitude columns are missing')
        result = pd.DataFrame(destination_infrastructure, columns=['destination_infrastructure'])
        if path_processed_data is not None:
            result.to_csv(path_processed_data + 'mip_data/destination_infrastructure.csv',
                          encoding='utf-8', index=True)
        return result

    if hasattr(destination, 'covers'):
        option_points = {
            option: Point([options.loc[option, 'longitude'], options.loc[option, 'latitude']])
            for option in options.index
        }
        destination_infrastructure = [
            option for option, option_point in option_points.items()
            if destination.covers(option_point)
        ]

        if destination_tolerance > 0 and destination_infrastructure:
            inside_options = options.loc[destination_infrastructure]
            outside_options = options.loc[
                [option for option in options.index if option not in destination_infrastructure]]
            if not outside_options.empty:
                distances = calc_distance_list_to_list(
                    outside_options['latitude'],
                    outside_options['longitude'],
                    inside_options['latitude'],
                    inside_options['longitude'])
                minimal_distances_to_destination_infrastructure = pd.DataFrame(
                    distances, index=inside_options.index,
                    columns=outside_options.index).min(axis=0)
                tolerated_options = minimal_distances_to_destination_infrastructure[
                    minimal_distances_to_destination_infrastructure <= destination_tolerance].index.tolist()
                destination_infrastructure.extend(tolerated_options)
    elif isinstance(destination, Point):
        for option in options.index:
            option_point = Point([options.loc[option, 'longitude'], options.loc[option, 'latitude']])
            distance_to_destination = calc_distance_single_to_single(
                option_point.y, option_point.x, destination.y, destination.x)
            if distance_to_destination <= destination_tolerance:
                destination_infrastructure.append(option)

    result = pd.DataFrame(destination_infrastructure, columns=['destination_infrastructure'])
    if path_processed_data is not None:
        result.to_csv(path_processed_data + 'mip_data/destination_infrastructure.csv',
                      encoding='utf-8', index=True)
    return result


def load_minimal_mip_case(path_processed_data):
    """Load the preprocessed hardcoded MIP test case."""
    case_file = path_processed_data + 'mip_data/minimal_case.pkl'
    logger.info('Load preprocessed minimal MIP case from %s', case_file)
    return pd.read_pickle(case_file)
