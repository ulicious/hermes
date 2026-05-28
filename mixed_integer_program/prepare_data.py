import ast

import pandas as pd
import networkx as nx

from data_processing.process_mip_data import create_transport_edges


def prepare_data(start_location_data, static_graph, start_road_distances, start_new_pipeline_distances,
                 end_node, config_file, techno_economic_data_transport):
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

    return all_nodes_adjusted, target_nodes, edges, production_costs, transport_means, max_costs, conversion_edges, transport_edges


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
