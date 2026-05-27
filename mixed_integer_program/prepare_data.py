import yaml
import os
import ast

from algorithm.methods_geographic import calc_distance_list_to_list_no_matrix

import pandas as pd
from shapely.ops import unary_union
import networkx as nx

import geopandas as gpd
import cartopy.io.shapereader as shpreader

from itertools import combinations


def create_bidirectional_distances(distances):
    """
    Convert an undirected physical connection list into directed MIP arcs.

    `calculate_road_distances()` writes every physical pair only once, for
    example A--B. The MIP selects directed transport edges, so both A->B and
    B->A have to be present before commodities are attached to the segment.
    Existing pipeline and shipping matrices are already directed after they
    are stacked and therefore do not use this helper.
    """
    reverse_distances = distances.copy()
    reverse_distances[['pointA', 'pointB']] = distances[['pointB', 'pointA']].to_numpy()

    return pd.concat([distances, reverse_distances], ignore_index=True)


def _calculate_conversion_data_from_technologies(all_nodes, all_commodities, techno_economic_data_conversion):
    """
    Calculate conversion inputs for an in-memory infrastructure example.

    The full-data workflow reads location-specific preprocessed conversion
    values. The minimal infrastructure has no geographic cost preprocessing, so it
    uses `uniform_costs` from `techno_economic_data_conversion.yaml` for all
    dummy nodes and calculates the same conversion columns consumed below.
    """
    uniform_costs = techno_economic_data_conversion['uniform_costs']
    interest_rate = uniform_costs['interest_rate']
    electricity_costs = uniform_costs['Electricity']
    co2_costs = uniform_costs['CO2']
    nitrogen_costs = uniform_costs['Nitrogen']

    conversion_data = pd.DataFrame(index=all_nodes)
    conversion_data['conversion_possible'] = True

    for commodity_start in all_commodities:
        for commodity_end in techno_economic_data_conversion[commodity_start]['potential_conversions']:
            technology = techno_economic_data_conversion[commodity_start][commodity_end]
            annuity_factor = (interest_rate * (1 + interest_rate) ** technology['lifetime']) / \
                ((1 + interest_rate) ** technology['lifetime'] - 1)
            conversion_costs = technology['specific_investment'] * \
                (annuity_factor + technology['fixed_maintenance']) / technology['operating_hours'] + \
                electricity_costs * technology['electricity_demand'] + \
                co2_costs * technology['co2_demand'] + \
                nitrogen_costs * technology['nitrogen_demand']

            # Minimal infrastructure does not provide external process heat,
            # matching the default `*_heat_available_at_*: False` settings.
            conversion_efficiency = technology['efficiency_autothermal']
            conversion_data[commodity_start + '-' + commodity_end + '-conversion_costs'] = conversion_costs
            conversion_data[commodity_start + '-' + commodity_end + '-conversion_efficiency'] = conversion_efficiency

    return conversion_data


def prepare_data(start_location_data, end_node=None, create_results=False, infrastructure_data=None,
                 warm_start_route=None):
    """
    Build the commodity-expanded graph that is passed to the MIP.

    There are three kinds of model nodes:

    1. Infrastructure/commodity nodes:
       For every physical node N in `options.csv` and every configured
       commodity C, one model node `N_C` is created. The node can exist even
       if no edge for that commodity reaches it; edge generation determines
       which of these nodes are usable in a route.
    2. Start nodes:
       For every producible commodity C, `start_C` is created. Transport
       edges leaving these nodes carry the production cost for C later in the
       optimization model.
    3. Sink node:
       A single technical node `end` represents arrival with any permitted
       target commodity. Zero-cost sink edges connect destination
       infrastructure nodes to it.

    There are three kinds of model edges:

    1. Conversion edge:
       At one physical, conversion-capable node N, a permitted conversion
       C1 -> C2 produces `N_C1 -> N_C2`.
    2. Transport edge:
       A directed physical segment A -> B for transport mean M and a
       commodity C permitted for M produces `A_C -> B_C`. Transportation
       never changes the commodity.
    3. Sink edge:
       If a transport edge arrives at a destination infrastructure node in a
       target commodity, `destination_C -> end` is added.

    Completeness can therefore be checked independently:
    - physical directed segments per transport mean are collected in
      `available_options`;
    - commodities per segment are filtered only through
      `potential_transportation`;
    - conversions are filtered only through `conversion_possible` and
      `potential_conversions`.

    Warm starts are optional and independent from graph construction:
    - `warm_start_route` supplies an already known list of generated edge
      keys directly, as used by the minimal infrastructure example;
    - `create_results=True` retains the legacy option of rebuilding a route
      from an existing result file for a real start location.
    """

    path_config = os.getcwd()
    path_config = os.path.dirname(path_config) + '/algorithm_configuration.yaml'

    yaml_file = open(path_config)
    config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    path_overall_data = config_file['project_folder_path']
    path_raw_data = path_overall_data + 'raw_data/'

    # load techno economic data
    yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
    techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

    yaml_file = open(path_raw_data + 'techno_economic_data_transportation.yaml')
    techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Physical infrastructure nodes only: ports and existing gas/oil pipeline
    # nodes written by preprocessing or supplied by a minimal example.
    nodes = infrastructure_data['options']

    all_nodes = nodes.index.tolist()
    all_commodities = config_file['available_commodity']
    filtered_commodities = []
    for c in all_commodities:
        if c in start_location_data.index:
            filtered_commodities.append(c)

    all_commodities = filtered_commodities

    # NODE CREATION: expand each physical infrastructure location into one
    # node per commodity: N -> N_Hydrogen_Gas, N_Ammonia, ..., N_FTF.
    # Not every created node necessarily has incident edges; that depends on
    # the transportation and conversion permissions below.
    all_nodes_adjusted = [n + '+' + commodity for n in all_nodes for commodity in all_commodities]

    # The start is not contained in options.csv, because it changes for each
    # optimization run. It receives the same commodity expansion separately.
    start_commodities = config_file['available_commodity']
    start_nodes = ['start+' + c for c in all_commodities]

    # The destination is represented by one shared sink rather than by one
    # sink per target commodity. The commodity is retained until the final
    # zero-cost edge into `end`.
    target_commodities = config_file['target_commodity']
    filtered_commodities = []
    for c in target_commodities:
        if c in start_location_data.index:
            filtered_commodities.append(c)
    target_commodities = filtered_commodities
    target_nodes = ['end']

    all_nodes_adjusted += start_nodes + target_nodes

    # Only configured transport means are allowed to contribute transport
    # edges, even if physical distance data for another category exists.
    transport_means = config_file['available_transport_means']

    # Costs at the chosen origin are indexed by commodity and are later used
    # for transport edges leaving `start_<commodity>`.

    production_costs = {}
    for com in all_commodities:
        production_costs[com] = start_location_data.loc[com]

    edges = {}
    conversion_edges = {}
    transport_edges = {}

    # CONVERSION EDGE CREATION
    #
    # For every physical node N with `conversion_possible == True`, create a
    # directed edge N_C1 -> N_C2 for every conversion C1 -> C2 listed in
    # techno_economic_data_conversion.yaml. No movement occurs here: node_1
    # and node_2 are required to be the identical physical location.
    if True:
        if infrastructure_data is None:
            conversion_costs_and_efficiencies = pd.read_csv(path_overall_data + 'processed_data/conversion_costs_and_efficiency.csv', index_col=0)
        else:
            conversion_costs_and_efficiencies = _calculate_conversion_data_from_technologies(
                all_nodes, all_commodities, techno_economic_data_conversion)
        for node_1 in all_nodes:

            # Origins are handled through `start_*` and do not receive local
            # conversion edges in this graph-construction step.
            if 'origin' in node_1:
                continue

            # This is the location-level conversion filter. Pipeline and port
            # nodes remain eligible if preprocessing marked them True.
            if not conversion_costs_and_efficiencies.loc[node_1, 'conversion_possible']:
                continue

            for node_2 in all_nodes:
                if 'origin' in node_2:
                    continue

                if node_1 != node_2:
                    continue

                # This is the commodity-level conversion filter. For each
                # permissible ordered pair, exactly one directed edge is
                # created at the current physical node.
                for com_1 in all_commodities:
                    for com_2 in all_commodities:
                        if com_2 in techno_economic_data_conversion[com_1]['potential_conversions']:
                            conversion_costs = conversion_costs_and_efficiencies.loc[node_1, com_1 + '-' + com_2 + '-conversion_costs']
                            conversion_efficiency = 1 - conversion_costs_and_efficiencies.loc[node_1, com_1 + '-' + com_2 + '-conversion_efficiency']

                            edges[node_1 + '+' + com_1 + '-' + node_2 + '+' + com_2] = \
                                ('conversion', node_1 + '+' + com_1, node_2 + '+' + com_2, conversion_costs,
                                 conversion_efficiency, com_2)

                            conversion_edges[node_1 + '+' + com_1 + '-' + node_2 + '+' + com_2] = \
                                (node_1 + '+' + com_1, node_2 + '+' + com_2, conversion_costs,
                                 conversion_efficiency, com_2)

                            # if node_2 == end_node:
                            #     edges[node_1 + '_' + com_1 + '-end'] =\
                            #         ('conversion', node_1 + '_' + com_1, 'end', 0, 0, com_2)

    columns = ['start', 'end', 'costs', 'efficiency', 'end_commodity']
    conversion_edges = pd.DataFrame.from_dict(conversion_edges, orient="index", columns=columns)

    # TRANSPORT SEGMENT COLLECTION
    #
    # The following data frames describe physical links first. Commodity
    # combinations and MIP transport edges are attached only afterwards.
    #
    # Road and new pipeline files contain unordered location pairs produced
    # via `combinations`, so they need an explicit reverse direction.
    road_distances = infrastructure_data['road_distances']
    start_road_distances = infrastructure_data['start_road_distances']
    new_pipeline_distances = infrastructure_data['new_pipeline_distances']
    start_new_pipeline_distances = infrastructure_data['start_new_pipeline_distances']

    # Example: physical row A--B becomes directed segments A->B and B->A.
    # Start distance files are not mirrored: the start is the route origin,
    # so a transport edge back into `start` is not required.
    road_distances = create_bidirectional_distances(road_distances)
    new_pipeline_distances = create_bidirectional_distances(new_pipeline_distances)

    # Port distances are stored as a square matrix. `stack()` directly emits
    # all directed entries A->B and B->A that are present in that matrix.
    port_distances = infrastructure_data['port_distances']
    port_distances = port_distances.stack().reset_index()
    port_distances.columns = ['pointA', 'pointB', 'distance']

    if False:  # removes road transport to pipelines --> remove in final version

        road_distances = road_distances[~road_distances['pointA'].str.contains('PG', na=False)]
        road_distances = road_distances[~road_distances['pointB'].str.contains('PG', na=False)]

        road_distances = road_distances[~road_distances['pointA'].str.contains('PL', na=False)]
        road_distances = road_distances[~road_distances['pointB'].str.contains('PL', na=False)]

        start_road_distances = start_road_distances[~start_road_distances['pointA'].str.contains('PG', na=False)]
        start_road_distances = start_road_distances[~start_road_distances['pointB'].str.contains('PG', na=False)]

        start_road_distances = start_road_distances[~start_road_distances['pointA'].str.contains('PL', na=False)]
        start_road_distances = start_road_distances[~start_road_distances['pointB'].str.contains('PL', na=False)]

    # Existing gas pipeline networks are stored as one square distance matrix
    # per PG graph. These matrices are created from an undirected networkx
    # graph, so stacking them already yields both A->B and B->A segments.
    gas_pipelines_distances = []
    gas_pipeline_matrices = infrastructure_data['gas_pipeline_matrices']

    for file_data in gas_pipeline_matrices:
        file_data = file_data.stack().reset_index()
        file_data.columns = ['pointA', 'pointB', 'distance']
        file_data = file_data[file_data['pointA'] != file_data['pointB']]  # remove same start and end
        gas_pipelines_distances.append(file_data)

    gas_pipelines_distances = pd.concat(gas_pipelines_distances, ignore_index=True)

    # Existing oil pipeline networks use PL file names. In the technology and
    # optimization nomenclature the matching transport mean is
    # `Pipeline_Liquid`; PL therefore means physical oil pipeline here.
    # As for gas, the stacked matrix already contains both directions.
    oil_pipelines_distances = []
    oil_pipeline_matrices = infrastructure_data['oil_pipeline_matrices']

    for file_data in oil_pipeline_matrices:
        file_data = file_data.stack().reset_index()
        file_data.columns = ['pointA', 'pointB', 'distance']
        file_data = file_data[file_data['pointA'] != file_data['pointB']]
        oil_pipelines_distances.append(file_data)

    oil_pipelines_distances = pd.concat(oil_pipelines_distances, ignore_index=True)

    # Map each configured transport mean to all of its directed physical
    # segments. The one new-pipeline segment set is intentionally offered to
    # both gas and liquid/oil construction alternatives; the commodity filter
    # below decides which commodity can use each alternative.
    available_options = {'Road': [road_distances, start_road_distances],
                         'New_Pipeline_Gas': [new_pipeline_distances, start_new_pipeline_distances],
                         'New_Pipeline_Liquid': [new_pipeline_distances, start_new_pipeline_distances],
                         'Shipping': [port_distances],
                         'Pipeline_Gas': [gas_pipelines_distances],
                         'Pipeline_Liquid': [oil_pipelines_distances]}
    options = {transport_mean: available_options[transport_mean]
               for transport_mean in transport_means
               if transport_mean in available_options}

    # TRANSPORT EDGE CREATION
    #
    # The nested iteration implements:
    #   directed physical segment x allowed commodity -> one directed edge.
    # Edge names contain the direction, commodity and transport mean:
    # `A_C-B_C-M`. Consequently, two transport means using the same physical
    # endpoints result in separate selectable MIP edges.
    max_costs = 0
    if True:
        for transport_mean in options.keys():

            distances = pd.concat(options[transport_mean], axis=0)
            distances.reset_index(inplace=True)

            for i in distances.index:

                start = distances.loc[i, 'pointA']
                end = distances.loc[i, 'pointB']
                distance = distances.loc[i, 'distance']

                if start == end:
                    continue

                for com in all_commodities:

                    # Relevant only for source-segment data: do not create an
                    # outgoing edge for a commodity unavailable at the start.
                    if ('start' == start) & (com not in start_commodities):
                        continue

                    # Central completeness criterion for transport edges:
                    # every permitted commodity receives one edge on this
                    # directed segment; every non-permitted commodity receives
                    # no edge.
                    if transport_mean in techno_economic_data_transport[com]['potential_transportation']:

                        transport_costs = distance / 1000 * techno_economic_data_transport[com][transport_mean] / 1000
                        transport_losses = 0

                        if transport_costs > max_costs:
                            max_costs = transport_costs

                        if transport_mean == 'Shipping':

                            if techno_economic_data_transport[com]['Boil_Off'] > 0:
                                duration = distance / 1000 / techno_economic_data_transport[com]['Shipping_Speed']
                                boil_off = duration / 24 * techno_economic_data_transport[com]['Boil_Off']
                            else:
                                boil_off = 0

                            if techno_economic_data_transport[com]['Uses_Commodity_as_Shipping_Fuel']:
                                self_consumption = distance / 1000 * techno_economic_data_transport[com]['Self_Consumption']
                                transport_costs = 0
                            else:
                                self_consumption = 0

                            transport_losses = max(boil_off, self_consumption)

                        edges[start + '+' + com + '-' + end + '+' + com + '-' + transport_mean] = \
                            ('transport', start + '+' + com, end + '+' + com, transport_costs, transport_losses, com,
                             transport_mean)

                        transport_edges[start + '+' + com + '-' + end + '+' + com + '-' + transport_mean] = \
                            (start + '+' + com, end + '+' + com, transport_costs, transport_losses, com,
                             transport_mean)

                        # SINK EDGE CREATION
                        #
                        # A zero-cost terminal edge is created once arrival at
                        # an infrastructure located in the destination is
                        # possible in an accepted target commodity. The key
                        # intentionally omits `transport_mean`: all arrivals
                        # in the same commodity use the same final sink edge.
                        if (end in end_node) & (com in target_commodities):
                            edges[end + '+' + com + '-' + 'end'] = \
                                ('transport', end + '+' + com, 'end', 0, 0, com, transport_mean)

                            transport_edges[end + '+' + com + '-' + 'end'] = \
                                (end + '+' + com, 'end', 0, 0, com, transport_mean)

    columns = ['start', 'end', 'costs', 'efficiency', 'commodity', 'mean']
    transport_edges = pd.DataFrame.from_dict(transport_edges, orient="index", columns=columns)

    # A warm-start route, when requested, merely selects edge keys created
    # above. It does not create additional graph nodes or edges. A directly
    # supplied route avoids any dependency on an external results file.
    if (warm_start_route is not None) and create_results:
        raise ValueError('Pass either warm_start_route or create_results=True, not both.')

    if warm_start_route is not None:
        missing_edges = [edge for edge in warm_start_route if edge not in edges]
        if missing_edges:
            raise ValueError('Warm-start route contains edges absent from generated graph: ' +
                             ', '.join(missing_edges))
        solution_route = list(warm_start_route)
        cost_route = None
    elif create_results:
        result = pd.read_csv(path_overall_data + '/results/location_results/' +
                             str(start_location_data.name) + '_final_solution.csv', index_col=0)
        result = result[result.columns[0]]
        route = ast.literal_eval(result.loc['taken_routes'])
        total_costs = result.loc['current_total_costs']

        cost_route = ast.literal_eval(result.loc['all_previous_total_costs'])
        cost_route = list(set(cost_route))

        commodity = None
        transport_mean = None
        start = None
        end = None
        solution_route = []  # same commodity conversion required?
        for n, segment in enumerate(route):
            if n > 0:
                if len(segment) == 5:  # transport

                    start = segment[0]
                    end = segment[3]

                    if start == 'Start':
                        start = 'start'

                    transport_mean = segment[1]

                    solution_route.append(start + '+' + commodity + '-' + end + '+' + commodity + '-' + transport_mean)
                    start = end
                elif len(segment) == 3:  # conversion

                    if commodity == segment[1]:
                        continue

                    solution_route.append(start + '+' + commodity + '-' + end + '+' + segment[1])
                    commodity = segment[1]
            else:
                commodity = segment[0]
                start = 'start'

        solution_route += [end + '+' + commodity + '-end']

        print(total_costs)
        print(solution_route)
    else:
        solution_route = None
        cost_route = None

    return all_nodes_adjusted, target_nodes, edges, production_costs, transport_means, solution_route, cost_route, max_costs, conversion_edges, transport_edges


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
