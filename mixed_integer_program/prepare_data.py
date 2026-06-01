import ast
import logging
import math

import pandas as pd
import networkx as nx

from mixed_integer_program.mip_data_helpers import create_transport_edges


logger = logging.getLogger(__name__)


def prepare_data(start_location_data, static_graph, start_road_distances, start_new_pipeline_distances,
                 end_node, config_file, techno_economic_data_transport,
                 warm_start_route=None, filter_edges_above_warm_start=False,
                 filter_start_options_above_warm_start=False):
    """
    Complete the commodity-expanded graph for one optimization location.

    Static graph data is created before optimization by
    `data_processing.process_mip_data.build_static_mip_graph`:

    1. Infrastructure/commodity nodes:
       For every physical node N in the processed infrastructure and every
       commodity C, one model node `N_C` is created. The node can exist even
       if no edge for that commodity reaches it; edge generation determines
       which of these nodes are usable in a route.
    2. Conversion edges:
       At one physical, conversion-capable node N, a permitted conversion
       C1 -> C2 produces `N_C1 -> N_C2`.
    3. Infrastructure transport edges:
       A directed physical segment A -> B for transport mean M and a
       commodity C permitted for M produces `A_C -> B_C`.

    This function adds only run-specific graph data:

    1. Start nodes and start transport edges:
       For every producible commodity C, `start_C` is created. Transport
       edges from the current start location connect it to the static graph.
       These nodes carry the production cost for C later in the model.
    2. Sink node and destination edges:
       A single technical node `end` represents arrival with any permitted
       target commodity. Zero-cost sink edges connect destination
       infrastructure nodes to it. The selected destination nodes are supplied
       by the caller and can be reused for every origin with the same target.

    Static completeness is stored once in the generated `static_graph.pkl`;
    repeated calls here do not rebuild or reload those nodes and edges.

    Warm starts are optional and independent from graph construction:
    - `warm_start_route` supplies an already known list of generated edge
      keys directly, as used by the minimal infrastructure example;
    - `create_results=True` retains the legacy option of rebuilding a route
      from an existing result file for a real start location.

    Configuration-dependent feasibility is applied after the broad graph is
    assembled for this location. The global preprocessed MIP artifacts stay
    broad, but one concrete optimization run does not carry impossible edges.
    """

    all_commodities = config_file['available_commodity']
    start_commodities = [commodity for commodity in all_commodities if commodity in start_location_data.index]
    target_commodities = config_file['target_commodity']
    target_nodes = ['end']
    transport_means = config_file['available_transport_means']
    production_costs = {commodity: start_location_data.loc[commodity] for commodity in start_commodities}

    all_nodes_adjusted = list(static_graph['nodes'])
    all_nodes_adjusted += ['start+' + commodity for commodity in start_commodities] + target_nodes
    edges = dict(static_graph['edges'])
    conversion_edges = static_graph['conversion_edges'].copy()
    transport_edges = static_graph['transport_edges'].copy()
    max_costs = static_graph['max_costs']

    # START-SPECIFIC TRANSPORT EDGES
    # Only these physical segments depend on the selected origin.
    start_options = {
        'Road': start_road_distances,
        'New_Pipeline_Gas': start_new_pipeline_distances,
        'New_Pipeline_Liquid': start_new_pipeline_distances,
    }
    start_edges, start_max_costs = create_transport_edges(
        start_options, start_commodities, techno_economic_data_transport)
    edges.update(start_edges)
    max_costs = max(max_costs, start_max_costs)
    columns = ['start', 'end', 'costs', 'efficiency', 'commodity', 'mean']
    start_edges_df = pd.DataFrame.from_dict(
        {key: value[1:] for key, value in start_edges.items()}, orient='index', columns=columns)
    transport_edges = pd.concat([transport_edges, start_edges_df], axis=0)

    # DESTINATION-SPECIFIC SINK EDGES
    # The selected destination is represented only by terminal connections;
    # no static infrastructure edge is rebuilt for a new optimization run.
    sink_edges = {}
    for destination in end_node:
        for commodity in target_commodities:
            key = destination + '+' + commodity + '-end'
            sink_edges[key] = ('transport', destination + '+' + commodity, 'end', 0, 0,
                               commodity, 'Destination')
    if warm_start_route is not None:
        for edge_key in warm_start_route:
            if not edge_key.startswith('start+') or not edge_key.endswith('-end'):
                continue
            commodity = edge_key.split('+', 1)[1].rsplit('-', 1)[0]
            if commodity not in target_commodities:
                continue
            sink_edges[edge_key] = ('transport', 'start+' + commodity, 'end', 0, 0,
                                    commodity, 'Destination')
    edges.update(sink_edges)
    sink_edges_df = pd.DataFrame.from_dict(
        {key: value[1:] for key, value in sink_edges.items()}, orient='index', columns=columns)
    transport_edges = pd.concat([transport_edges, sink_edges_df], axis=0)

    edges, transport_edges = filter_transport_edges_by_configuration(
        edges, transport_edges, config_file, techno_economic_data_transport)

    if filter_edges_above_warm_start:
        edges, conversion_edges, transport_edges = filter_edges_by_warm_start_costs(
            edges, conversion_edges, transport_edges, production_costs, warm_start_route)

    if filter_start_options_above_warm_start:
        all_nodes_adjusted, edges, transport_edges, production_costs = \
            filter_start_options_by_warm_start_costs(
                all_nodes_adjusted, edges, transport_edges, production_costs, warm_start_route)

    return all_nodes_adjusted, target_nodes, edges, production_costs, transport_means, max_costs, conversion_edges, transport_edges


def _as_float_config(value):
    """Parse numeric configuration values, including the local `math.inf` convention."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == 'math.inf':
            return math.inf
        return float(value)
    return float(value)


def _physical_node_name(node):
    if '+' not in node:
        return node
    return node.split('+', 1)[0]


def _estimate_transport_distance(edge, techno_economic_data_transport):
    commodity = edge[5]
    transport_mean = edge[6]
    if transport_mean == 'Destination':
        return 0.0
    cost_rate = techno_economic_data_transport.get(commodity, {}).get(transport_mean)
    if cost_rate in (None, 0):
        return None
    return edge[3] * 1000000 / cost_rate


def transport_edge_forbidden_reason(edge, config_file, techno_economic_data_transport):
    """Return why a transport edge is not allowed in this run, or None."""
    if edge[0] != 'transport':
        return None

    start = edge[1]
    end = edge[2]
    commodity = edge[5]
    transport_mean = edge[6]

    if transport_mean == 'Destination':
        return None

    if _physical_node_name(start) == _physical_node_name(end):
        return 'self_loop'

    if transport_mean not in config_file['available_transport_means']:
        return 'transport_mean_not_available'

    transport_options = techno_economic_data_transport.get(commodity, {}).get('potential_transportation', [])
    if transport_mean not in transport_options:
        return 'commodity_transport_not_allowed'

    if 'New' in transport_mean and not config_file['build_new_infrastructure']:
        return 'new_infrastructure_disabled'

    if (commodity == 'Hydrogen_Gas'
            and transport_mean == 'Pipeline_Gas'
            and not config_file['H2_ready_infrastructure']):
        return 'hydrogen_in_existing_gas_pipeline_not_ready'

    distance = _estimate_transport_distance(edge, techno_economic_data_transport)
    if distance is not None and math.isfinite(distance):
        if transport_mean == 'Road':
            max_length_road = _as_float_config(config_file['max_length_road'])
            if max_length_road is not None and distance > max_length_road + 1e-9:
                return 'road_distance_above_max_length'
        elif transport_mean in {'New_Pipeline_Gas', 'New_Pipeline_Liquid'}:
            max_length_new_segment = _as_float_config(config_file['max_length_new_segment'])
            if max_length_new_segment is not None and distance > max_length_new_segment + 1e-9:
                return 'new_pipeline_distance_above_max_length'

    return None


def filter_transport_edges_by_configuration(edges, transport_edges, config_file, techno_economic_data_transport):
    """Remove run-specific infeasible transport edges before building the MIP."""
    removed_edges = []
    removed_by_reason = {}
    for key, edge in list(edges.items()):
        reason = transport_edge_forbidden_reason(edge, config_file, techno_economic_data_transport)
        if reason is None:
            continue
        removed_edges.append(key)
        removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1
        edges.pop(key, None)

    if not removed_edges:
        logger.info('Configuration transport filter removed no edges')
        return edges, transport_edges

    transport_edges = transport_edges.drop(
        index=transport_edges.index.intersection(removed_edges))
    logger.info('Configuration transport filter removed %s edges before model build: %s',
                len(removed_edges), removed_by_reason)
    return edges, transport_edges


def calculate_route_objective(edges, production_costs, route):
    """Calculate route costs with the same cost propagation used by the MIP."""
    if not route:
        raise ValueError('Cannot calculate route objective for an empty route')

    missing_edges = [edge for edge in route if edge not in edges]
    if missing_edges:
        raise ValueError('Cannot calculate route objective because '
                         + str(len(missing_edges))
                         + ' route edges are absent from the current MIP graph. First missing edge: '
                         + missing_edges[0])

    first_edge = edges[route[0]]
    start_commodity = first_edge[1].split('+', 1)[1]
    total_costs = production_costs[start_commodity]

    for edge_key in route:
        edge = edges[edge_key]
        edge_costs = edge[3]
        edge_loss = edge[4]
        total_costs = (total_costs + edge_costs) / (1 - edge_loss)

    return total_costs


def filter_edges_by_warm_start_costs(edges, conversion_edges, transport_edges,
                                     production_costs, warm_start_route):
    """
    Remove edges that cannot be part of a route cheaper than the warm start.

    Edge position 4 stores loss, not efficiency. If all losses are
    non-negative, costs cannot decrease along a path. In that case every edge
    must at least carry the cheapest available start production cost before its
    own costs and losses are applied. Edges whose resulting lower bound already
    exceeds the warm-start objective cannot improve the incumbent.
    """
    if warm_start_route is None:
        logger.warning('Warm-start edge filter requested, but no warm-start route was provided')
        return edges, conversion_edges, transport_edges
    missing_edges = [edge for edge in warm_start_route if edge not in edges]
    if missing_edges:
        logger.warning('Warm-start edge filter skipped because %s route edges are absent from the current MIP graph. '
                       'First missing edge: %s', len(missing_edges), missing_edges[0])
        return edges, conversion_edges, transport_edges

    warm_start_costs = calculate_route_objective(edges, production_costs, warm_start_route)
    if not production_costs:
        logger.warning('Warm-start edge filter requested, but no production costs are available')
        return edges, conversion_edges, transport_edges

    has_negative_losses = any(edge[4] < -1e-12 for edge in edges.values())
    if has_negative_losses:
        logger.warning('Warm-start edge filter skipped because at least one edge has negative loss')
        return edges, conversion_edges, transport_edges

    minimal_production_costs = min(production_costs.values())
    warm_start_edges = set(warm_start_route)
    removed_edges = set()
    for key, edge in edges.items():
        if key in warm_start_edges:
            continue

        edge_costs = edge[3]
        edge_loss = edge[4]
        if edge_loss >= 1:
            removed_edges.add(key)
            continue

        edge_lower_bound = (minimal_production_costs + edge_costs) / (1 - edge_loss)
        if edge_lower_bound > warm_start_costs:
            removed_edges.add(key)

    if not removed_edges:
        logger.info('Warm-start edge filter removed no edges; route costs %.6f, '
                    'minimal production costs %.6f',
                    warm_start_costs, minimal_production_costs)
        return edges, conversion_edges, transport_edges

    for key in removed_edges:
        edges.pop(key, None)
    conversion_edges = conversion_edges.drop(
        index=conversion_edges.index.intersection(removed_edges))
    transport_edges = transport_edges.drop(
        index=transport_edges.index.intersection(removed_edges))
    logger.info('Warm-start edge filter removed %s edges with edge lower bound above %.6f; '
                'minimal production costs %.6f',
                len(removed_edges), warm_start_costs, minimal_production_costs)
    return edges, conversion_edges, transport_edges


def filter_start_options_by_warm_start_costs(all_nodes_adjusted, edges, transport_edges,
                                             production_costs, warm_start_route):
    """
    Remove start options whose production cost is already above the warm-start objective.

    This is valid only when all losses are non-negative. Then every route can only
    add costs or lose material, so a start option that is already more expensive
    than a known complete route cannot improve the incumbent.

    This does not remove the commodity from the infrastructure graph. It only
    removes `start+Commodity`, its outgoing start edges and the corresponding
    production-cost fixation.
    """
    if warm_start_route is None:
        logger.warning('Start-option warm-start filter requested, but no warm-start route was provided')
        return all_nodes_adjusted, edges, transport_edges, production_costs

    missing_edges = [edge for edge in warm_start_route if edge not in edges]
    if missing_edges:
        logger.warning('Start-option warm-start filter skipped because %s route edges are absent from '
                       'the current MIP graph. First missing edge: %s',
                       len(missing_edges), missing_edges[0])
        return all_nodes_adjusted, edges, transport_edges, production_costs

    has_negative_losses = any(edge[4] < -1e-12 for edge in edges.values())
    if has_negative_losses:
        logger.warning('Start-option warm-start filter skipped because at least one edge has negative loss')
        return all_nodes_adjusted, edges, transport_edges, production_costs

    warm_start_costs = calculate_route_objective(edges, production_costs, warm_start_route)
    removed_start_options = {
        commodity for commodity, production_cost in production_costs.items()
        if production_cost > warm_start_costs
    }
    if not removed_start_options:
        logger.info('Start-option warm-start filter removed no start options; route costs %.6f',
                    warm_start_costs)
        return all_nodes_adjusted, edges, transport_edges, production_costs

    removed_start_nodes = {'start+' + commodity for commodity in removed_start_options}
    removed_edges = {
        key for key, edge in edges.items()
        if edge[1] in removed_start_nodes or edge[2] in removed_start_nodes
    }
    for edge in removed_edges:
        edges.pop(edge, None)

    transport_edges = transport_edges.drop(
        index=transport_edges.index.intersection(removed_edges))
    all_nodes_adjusted = [
        node for node in all_nodes_adjusted
        if node not in removed_start_nodes
    ]
    production_costs = {
        commodity: production_cost for commodity, production_cost in production_costs.items()
        if commodity not in removed_start_options
    }
    logger.info('Start-option warm-start filter removed %s start options and %s start edges above %.6f: %s',
                len(removed_start_options), len(removed_edges), warm_start_costs,
                sorted(removed_start_options))
    return all_nodes_adjusted, edges, transport_edges, production_costs


def create_edges_from_distance_only(df_list, transport_means, techno_economic_data_transport,
                                    all_commodities, start_commodities):

    edges = {}

    distances = pd.concat(df_list, axis=0)
    distances.reset_index(inplace=True)

    for transport_mean in transport_means:

        for i in distances.index:

            start = distances.loc[i, 'pointA']
            end = distances.loc[i, 'pointB']
            distance = distances.loc[i, 'distance']

            for com in all_commodities:

                if ('start' == start) & (com not in start_commodities):
                    continue

                if transport_mean in techno_economic_data_transport[com]['potential_transportation']:

                    transport_costs = distance * techno_economic_data_transport[com][transport_mean] / 1000
                    transport_losses = 0

                    edges[start + '-' + com + '-' + end + '-' + com + '-' + transport_mean] = \
                        ('transport', start + '-' + com, end + '-' + com, transport_costs, transport_losses, com, transport_mean)

    return edges


def create_graph(edges, nodes):

    graph = nx.Graph()

    for edge in edges.keys():

        data = edges[edge]

        if data[0] == 'conversion':  # conversion edge
            start = data[1]
            end = data[2]

            name = start + '-' + end + '_conversion'

        else:  # transport edge
            start = data[1]
            end = data[2]

            name = start + '-' + end + '_transport'

        graph.add_edge(start, end, name=name)

    max_length = 0
    for n_1 in nodes:
        for n_2 in nodes:

            if n_1 == n_2:
                continue

            paths = nx.all_simple_paths(graph, source=n_1, target=n_2)
            print(len(list(paths)))
            for p in paths:
                if len(p) > max_length:
                    max_length = len(p)

    print(max_length)
