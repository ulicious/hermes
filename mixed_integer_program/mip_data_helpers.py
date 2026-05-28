import logging
import os

import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


logger = logging.getLogger(__name__)


def create_transport_edges(distance_options, commodities, techno_economic_data_transport,
                           show_progress=False):
    """Attach permitted commodities and techno-economic values to directed segments."""
    edges = {}
    max_costs = 0

    for transport_mean, distances in distance_options.items():
        if distances.empty:
            continue
        logger.info('Create %s transport edges from %s directed distances',
                    transport_mean, len(distances))
        row_iterator = tqdm(distances.itertuples(), total=len(distances),
                            desc='Create ' + transport_mean + ' edges',
                            disable=not show_progress)
        for row in row_iterator:
            if row.pointA == row.pointB:
                continue
            for commodity in commodities:
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
    logger.info('Load static MIP graph artifacts from CSV using legacy reconstruction')
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
    logger.info('Reconstructed %s conversion and %s transport edges',
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
        logger.info('Load static MIP graph from %s', graph_file)
        static_graph = pd.read_pickle(graph_file)
        logger.info('Loaded %s nodes and %s static edges from cache',
                    len(static_graph['nodes']), len(static_graph['edges']))
        return static_graph

    logger.info('Binary graph file absent; migrate existing CSV artifacts to %s', graph_file)
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
    logger.info('Reconstructed %s conversion and %s transport edges; saved binary graph file',
                len(conversion_edges), len(transport_edges))
    return static_graph


def prepare_destination_mip_data(options, destination, path_processed_data=None):
    """Determine the infrastructure nodes accepted as sinks for one destination."""
    destination_infrastructure = []
    if hasattr(destination, 'contains'):
        for option in options.index:
            option_point = Point([options.loc[option, 'longitude'], options.loc[option, 'latitude']])
            if destination.contains(option_point):
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
