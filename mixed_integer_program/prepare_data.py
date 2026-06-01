import ast
import logging

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
    edges.update(sink_edges)
    sink_edges_df = pd.DataFrame.from_dict(
        {key: value[1:] for key, value in sink_edges.items()}, orient='index', columns=columns)
    transport_edges = pd.concat([transport_edges, sink_edges_df], axis=0)

    if filter_edges_above_warm_start:
        edges, conversion_edges, transport_edges = filter_edges_by_warm_start_costs(
            edges, conversion_edges, transport_edges, production_costs, warm_start_route)

    if filter_start_options_above_warm_start:
        all_nodes_adjusted, edges, transport_edges, production_costs = \
            filter_start_options_by_warm_start_costs(
                all_nodes_adjusted, edges, transport_edges, production_costs, warm_start_route)

    return all_nodes_adjusted, target_nodes, edges, production_costs, transport_means, max_costs, conversion_edges, transport_edges


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
    Remove loss-free edges whose additive costs plus start hydrogen production
    costs already exceed the warm-start route.

    Edge position 4 stores loss, not efficiency. Therefore an edge with
    technology efficiency 1 has loss 0 and contributes only absolute costs.
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
    hydrogen_production_costs = production_costs.get('Hydrogen_Gas')
    if hydrogen_production_costs is None:
        logger.warning('Warm-start edge filter requested, but Hydrogen_Gas production costs are unavailable')
        return edges, conversion_edges, transport_edges

    warm_start_edges = set(warm_start_route)
    removed_edges = {
        key for key, edge in edges.items()
        if key not in warm_start_edges
        and abs(edge[4]) <= 1e-12
        and edge[3] + hydrogen_production_costs > warm_start_costs
    }
    if not removed_edges:
        logger.info('Warm-start edge filter removed no edges; route costs %.6f, hydrogen production costs %.6f',
                    warm_start_costs, hydrogen_production_costs)
        return edges, conversion_edges, transport_edges

    for key in removed_edges:
        edges.pop(key, None)
    conversion_edges = conversion_edges.drop(
        index=conversion_edges.index.intersection(removed_edges))
    transport_edges = transport_edges.drop(
        index=transport_edges.index.intersection(removed_edges))
    logger.info('Warm-start edge filter removed %s loss-free edges with edge costs plus hydrogen production costs '
                'above %.6f; hydrogen production costs %.6f',
                len(removed_edges), warm_start_costs, hydrogen_production_costs)
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
