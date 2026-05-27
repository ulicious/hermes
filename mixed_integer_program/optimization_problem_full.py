import itertools
import yaml
import os
import logging
import time

import pandas as pd
import gurobipy as gp
import geopandas as gpd

from gurobipy import GRB

from shapely.geometry import Point

from prepare_data import prepare_data, create_edges_from_distance_only, create_graph
from data_processing.process_mip_data import (
    calculate_road_distances,
    build_static_mip_graph,
    load_static_mip_graph,
    prepare_destination_mip_data,
)
from data_processing.helpers_geometry import get_destination


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# noinspection PyTypeChecker
class OptimizationGurobiModel:

    def attach_edges(self):
        logger.info('Create Gurobi variables for %s nodes and %s edges',
                    len(self.all_nodes_adjusted), len(self.edges))

        self.costs = self.model.addVars(self.all_nodes_adjusted, name=self.all_nodes_adjusted)

        self.capacity_at_node = self.model.addVars(self.all_nodes_adjusted, vtype='I', name=self.all_nodes_adjusted)

        self.edge_binaries = self.model.addVars([*self.edges], vtype='B', name=[*self.edges.keys()])

        now = time.time()
        # for row in self.conversion_edges.itertuples():
        #     name = 'conversion_' + row.start + '_' + row.end
        #
        #     self.model.addGenConstrIndicator(
        #         self.edge_binaries[row.Index],
        #         1,
        #         (self.costs[row.start] + row.costs) / (1 - row.efficiency) - self.costs[row.end],
        #         GRB.LESS_EQUAL,
        #         0.0,
        #         name=f"ind_{name}"
        #     )
        #
        # for row in self.transport_edges.itertuples():
        #     name = 'transport_' + row.start + '_' + row.end + '_' + row.commodity + '_' + row.mean
        #
        #     self.model.addGenConstrIndicator(
        #         self.edge_binaries[row.Index],
        #         1,
        #         (self.costs[row.start] + row.costs) / (1 - row.efficiency) - self.costs[row.end],
        #         GRB.LESS_EQUAL,
        #         0.0,
        #         name=f"ind_{name}"
        #     )

        for edge in self.edges.keys():

            data = self.edges[edge]

            if data[0] == 'conversion':  # conversion edge
                start = data[1]
                end = data[2]
                costs = data[3]
                efficiency = data[4]
                commodity = data[5]

                name = 'conversion-' + start + '-' + end
            else:  # transport edge
                start = data[1]
                end = data[2]
                costs = data[3]
                efficiency = data[4]
                commodity = data[5]
                transport_mean = data[6]

                name = 'transport-' + start + '-' + end + '-' + commodity + '-' + transport_mean

            if 'start' in start:
                self.model.addConstr(self.costs[start] == self.production_costs[commodity],
                                     name=name)

            if True:
                # self.model.addGenConstrIndicator(self.edge_binaries[edge], 1, (self.costs[start] + costs) / (1 - efficiency) - self.costs[end] <= 0)
                self.model.addGenConstrIndicator(self.edge_binaries[edge], 1, (self.costs[start] + costs) / (1 - efficiency) - self.costs[end], GRB.LESS_EQUAL, 0.0)
            else:
                self.model.addConstr((self.costs[start] + costs) / (1 - efficiency) - self.costs[end]
                                     <= (1 - self.edge_binaries[edge]) * self.BigM,
                                     name=name)

            # if 'start' not in start:
            #     self.model.addConstr((self.costs[start] + costs) / (1 - efficiency) - self.costs[end]
            #                          <= (1 - self.edge_binaries[edge]) * self.BigM,
            #                          name=name)
            #
            #     # self.model.addConstr(self.capacity_at_node[start] - self.capacity_at_node[end] - 1 >= - (1 - self.edge_binaries[edge]) * 100,
            #     #                      name=name + '_capacity')
            #
            # else:
            #     self.model.addConstr((self.production_costs[commodity] + costs) / (1 - efficiency) - self.costs[end]
            #                          <= (1 - self.edge_binaries[edge]) * (self.production_costs[commodity] + costs) * 1.1 / (1 - efficiency),
            #                          name=name)

        logger.info('Attached edge cost propagation constraints in %.2f s', time.time() - now)

        # for node in self.all_nodes_adjusted:
        #     name = 'max_costs_' + node
        #     self.model.addConstr(self.costs[node] <= self.BigM, name=name)

        self.model.addConstr(sum(self.costs[node] for node in self.target_nodes) >= min(self.production_costs.values()), name='min_costs')

        # self.model.addConstr(sum(self.capacity_at_node[node] for node in self.target_nodes) <= 1, name='min_costs')  # seq 1 because maybe one conversion at final node

        # ensure transport out of start node
        self.model.addConstr(sum(self.edge_binaries[key]
                                 for key in self.edges.keys()
                                 if ((self.edges[key][0] == 'transport') & ('start' in self.edges[key][1])))
                             == 1,
                             name='out_of_origin')

        logger.info('Attached origin flow constraint in %.2f s', time.time() - now)

        # ensure that the final node is reached by setting that transport to destination or conversion at destination is correct
        # self.model.addConstr(sum(self.edge_binaries[key]
        #                          for key in self.edges.keys()
        #                          if ((self.edges[key][0] == 'transport') & (self.edges[key][2] in self.target_nodes)))
        #                      + sum(self.edge_binaries[key]
        #                          for key in self.edges.keys()
        #                          if ((self.edges[key][0] == 'conversion') & (self.edges[key][2] in self.target_nodes))) == 1,
        #                      name='into_destination')

        # for e in self.edges.keys():
        #     if (self.edges[e][0] == 'transport') & (self.edges[e][2] == 'end'):
        #         print(e)
        #         print(self.edges[e])

        self.model.addConstr(sum(self.edge_binaries[key]
                                 for key in self.edges.keys()
                                 if ((self.edges[key][0] == 'transport') & (self.edges[key][2] == 'end'))) == 1,
                             name='into_destination')

        self.model.addConstr(sum(self.edge_binaries[key]
                                 for key in self.edges.keys()
                                 if ((self.edges[key][0] == 'transport') & (self.edges[key][1] == 'end'))) == 0,
                             name='no_out_destination')

        logger.info('Attached destination flow constraints in %.2f s', time.time() - now)

        from collections import defaultdict

        incoming_edges = defaultdict(list)  # edges with node in position 2
        outgoing_edges = defaultdict(list)  # edges with node in position 1

        for key, edge in self.edges.items():
            # Assuming edge is something like (…, tail_node, head_node)
            outgoing_edges[edge[1]].append(key)
            incoming_edges[edge[2]].append(key)

        for node in self.all_nodes_adjusted:
            # sum of binaries where node is in position 2
            sum_edge_key_2 = gp.quicksum(self.edge_binaries[key]
                                         for key in incoming_edges[node])

            self.model.addConstr(sum_edge_key_2 <= 1,
                                 name=f'only_visit_node_once[{node}]')

            if ('start' not in node) and ('end' not in node):
                # sum of binaries where node is in position 1
                sum_edge_key_1 = gp.quicksum(self.edge_binaries[key]
                                             for key in outgoing_edges[node])

                self.model.addConstr(sum_edge_key_1 == sum_edge_key_2,
                                     name=f'balance[{node}]')

        # # ensure that each node is visited max. once or a specific conversion takes place max. once
        # for node in self.all_nodes_adjusted:
        #
        #     sum_edge_key_2 = gp.quicksum(self.edge_binaries[key] for key in self.edges.keys() if self.edges[key][2] == node)
        #
        #     self.model.addConstr(sum_edge_key_2 <= 1, name='only_visit_node_once')
        #
        #     if ('start' not in node) & ('end' not in node):
        #         sum_edge_key_1 = gp.quicksum(self.edge_binaries[key] for key in self.edges.keys() if self.edges[key][1] == node)
        #
        #         self.model.addConstr(sum_edge_key_1 == sum_edge_key_2, name='balance')

            # self.model.addConstr(sum(self.edge_binaries[key]
            #                          for key in self.edges.keys()
            #                          if self.edges[key][2] == node) <= 1,
            #                      name='only_visit_node_once')
            #
            # if ('start' not in node) & ('end' not in node):
            #     self.model.addConstr(sum(self.edge_binaries[key]
            #                              for key in self.edges.keys()
            #                              if self.edges[key][1] == node)
            #                          == sum(self.edge_binaries[key]
            #                              for key in self.edges.keys()
            #                              if self.edges[key][2] == node),
            #                          name='balance')

        # # balance activities of all nodes but start and end
        # for node in self.all_nodes_adjusted:
        #     if ('start' not in node) & ('end' not in node):
        #         self.model.addConstr(sum(self.edge_binaries[key]
        #                                  for key in self.edges.keys()
        #                                  if self.edges[key][1] == node)
        #                              == sum(self.edge_binaries[key]
        #                                  for key in self.edges.keys()
        #                                  if self.edges[key][2] == node),
        #                              name='balance')

        logger.info('Finished graph balance constraints in %.2f s', time.time() - now)

        self.model.setObjective(sum(self.costs[node] for node in self.target_nodes), gp.GRB.MINIMIZE)

    def attach_edges_test(self):

        self.costs = self.model.addVars(self.edges, lb=0)
        self.quantity = self.model.addVars(self.edges, lb=0)

        for edge in self.edges.keys():

            data = self.edges[edge]

            if data[0] == 'conversion':  # conversion edge
                start = data[1]
                end = data[2]
                costs = data[3]
                efficiency = data[4]
                commodity = data[5]

                name = 'conversion_' + start + '_' + end
            else:  # transport edge
                start = data[1]
                end = data[2]
                costs = data[3]
                efficiency = data[4]
                commodity = data[5]
                transport_mean = data[6]

                name = 'transport_' + start + '_' + end + '_' + commodity + '_' + transport_mean

            if 'start' not in start:
                self.model.addConstr(self.costs[edge] == self.quantity[edge] * costs, name=name)

            else:
                self.model.addConstr(self.costs[edge] == self.quantity[edge] * (self.production_costs[commodity] + costs),
                                     name=name)

        # balance quantities
        for node in self.all_nodes_adjusted:
            if ('start' not in node) & ('end' not in node):

                self.model.addConstr(sum(self.quantity[edge]
                                         for edge in self.edges.keys()
                                         if self.edges[edge][1] == node)
                                     == sum(self.quantity[edge] * (1 - self.edges[edge][4])
                                            for edge in self.edges.keys()
                                            if self.edges[edge][2] == node),
                                     name=node + '_balance')

        for edge in self.edges.keys():
            if 'end' in self.edges[edge][2]:
                print(edge)
                print(self.edges[edge])

        self.model.addConstr(sum(self.quantity[edge] * (1 - self.edges[edge][4])
                                 for edge in self.edges.keys()
                                 if 'end' in self.edges[edge][2]) == 1,
                             name='end_balance')

        self.model.setObjective(sum(self.costs[edge] for edge in self.edges), gp.GRB.MINIMIZE)

    def optimize(self):
        logger.info('Build optimization constraints')
        self.attach_edges()

        logger.info('All constraints attached')

        # Optional MIP start: selected route edges start at 1; every other
        # generated edge starts at 0. Gurobi receives these values below when
        # `self.model.optimize(...)` is called.
        if self.solution_route is not None:
            logger.info('Apply warm-start route with %s active edges', len(self.solution_route))
            logger.debug('Warm-start route: %s', self.solution_route)
            self.model.update()

            for edge in self.edge_binaries.keys():
                self.edge_binaries[edge].Start = int(edge in self.solution_route)

        self.model.update()

        # self.model.Params.BestObjStop = 149.91814

        # self.model.Params.LogToConsole = 0
        # self.model.setParam("MIPGap", 0.01)

        self.model.Params.IntFeasTol = 1e-9
        self.model.Params.FeasibilityTol  = 1e-9
        self.model.Params.OptimalityTol   = 1e-9

        self.model.Params.Method  = 2
        self.model.Params.Crossover  = 0
        self.model.Params.BarHomogeneous = 1

        self.model.Params.MIPFocus = 3  # 2 = focus on proving optimality (tightening bounds)
        self.model.Params.Heuristics = 0  # turn off heuristics for finding new incumbents
        self.model.Params.Cuts = 3  # aggressive cuts to improve bounds
        self.model.Params.Presolve = 2  # full presolve, to tighten constraints

        self.model.Params.Heuristics = 0
        self.model.Params.PoolSearchMode = 0  # don’t look for additional solutions

        self.model.Params.Threads = 1
        logger.info('Solver parameters set; start optimization')

        binaries = [v for v in self.model.getVars() if v.VType == gp.GRB.BINARY]
        self.model._binaries = binaries

        continuous = [v for v in self.model.getVars() if v.VType == gp.GRB.CONTINUOUS]
        self.model._continuous = continuous

        self.model._best_incumbent = float('inf')  # keep track of improvement

        def incumbent_callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                # Get new incumbent objective
                obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)

                # Check if it's strictly better than the previous one
                if obj + 1e-12 < model._best_incumbent:
                    model._best_incumbent = obj

                    # Extract binary solution values
                    sol = model.cbGetSolution(model._binaries)

                    # Which binaries are 1?
                    active = [v.VarName for v, val in zip(model._binaries, sol) if val > 0.5]

                    print(f"\n=== New incumbent found at {model.cbGet(gp.GRB.Callback.RUNTIME):.2f}s ===")
                    print(f"Objective: {obj:.6f}")
                    print(f"{len(active)} binaries = 1:")
                    print(", ".join(active[:50]))  # print only first 50 to keep log readable
                    if len(active) > 50:
                        print(f"... ({len(active) - 50} more omitted)\n")

                    # total_costs = [self.production_costs[active[0].split('-')[0].split('start_')[1]]]

                    mapping = {}
                    start_node = None
                    for edge in active:
                        parts = edge.split("-")

                        # Start- und Zielknoten zusammensetzen
                        start = parts[0]
                        end = parts[1]

                        if 'start' in start:
                            start_node = start

                        mapping[start] = edge

                    # Reihenfolge erzeugen
                    ordered = []

                    current = start_node

                    while current in mapping:
                        edge = mapping[current]
                        ordered.append(edge)

                        parts = edge.split("-")

                        # nächsten Knoten bestimmen
                        current = parts[1]

                    print("\n".join(ordered))

                    start_node = [a for a in active if 'start' in a][0].split('-')[0].split('+')[1]
                    total_costs = [self.production_costs[start_node]]

                    new_costs = sum(total_costs)
                    for binary in active:

                        data = self.edges[binary]

                        if data[0] == 'conversion':  # conversion edge
                            costs = data[3]
                            efficiency = data[4]
                        else:  # transport edge
                            costs = data[3]
                            efficiency = data[4]

                        new_costs = (sum(total_costs) + costs) / (1 - efficiency)
                        total_costs += [new_costs - sum(total_costs)]

                    # print(total_costs)
                    print(new_costs)
                    #
                    # sol = model.cbGetSolution(model._continuous)
                    # for v, val in zip(model._continuous, sol):
                    #     if val > 0.1:
                    #         print(v)
                    #         print(val)

        # run optimization with callback
        self.model.optimize(incumbent_callback)
        self.instance = self

        self.status = self.model.status
        logger.info('Optimization finished with Gurobi status %s', self.status)

        if self.status == 2:

            self.objective_function_value = self.model.objVal
            logger.info('Optimal objective value: %.6f', self.objective_function_value)

    def __init__(self, static_graph, start_location_data, start_road_distances,
                 start_new_pipeline_distances, end_location, config_file,
                 techno_economic_data_conversion, techno_economic_data_transport,
                 warm_start_route=None, create_results=False):

        # ----------------------------------
        # Set up problem
        self.solver = 'gurobi'
        self.instance = None
        self.status = None
        self.objective = None

        self.model_type = 'gurobi'

        path_overall_data = config_file['project_folder_path']
        self.techno_economic_data_conversion = techno_economic_data_conversion
        self.techno_economic_data_transport = techno_economic_data_transport

        self.BigM = 200
        self.eps = 0.001

        logger.info('Add origin- and destination-specific graph data')
        self.all_nodes_adjusted, self.target_nodes, self.edges, self.production_costs, self.transport_means, \
            self.solution_route, self.cost_route, self.max_costs, self.conversion_edges, self.transport_edges = \
            prepare_data(start_location_data, static_graph, start_road_distances,
                         start_new_pipeline_distances, end_location, config_file,
                         self.techno_economic_data_transport,
                         create_results=create_results, warm_start_route=warm_start_route)

        logger.info('Optimization graph contains %s nodes and %s edges (%s conversion, %s transport)',
                    len(self.all_nodes_adjusted), len(self.edges),
                    len(self.conversion_edges), len(self.transport_edges))

        six = []
        seven = []

        for e in self.edges:
            if len(self.edges[e]) == 6:
                six.append(e)
            else:
                seven.append(e)

        subset_six = {k: self.edges[k] for k in six}
        subset_seven = {k: self.edges[k] for k in seven}

        pd.DataFrame(subset_six).transpose().to_csv(path_overall_data + 'conversion_edges.csv')
        pd.DataFrame(subset_seven).transpose().to_csv(path_overall_data + 'transport_edges.csv')
        logger.info('Exported current conversion and transport edge tables')

        if False:
            start_distance = calculate_road_distances(config_file['tolerance_distance'], options, start_point, start_name)
            self.start_edges = create_edges_from_distance_only([start_distance],
                                                               ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'],
                                                               techno_economic_data_transport, all_commodities, start_commodities)

        self.model = gp.Model()
        logger.info('Created Gurobi model instance')
        self.optimize()

        if False:
            for edge in self.edges:
                if self.quantity[edge].X > 0.1:
                    print(edge)
                    print(self.quantity[edge].X)
        else:

            # read and organize solution
            chosen_binaries = []
            for k in self.edge_binaries.keys():
                # print(str(k) + ': ' + str(self.edge_binaries[k].X))

                if self.edge_binaries[k].X:
                    print(k)
                    print(self.edge_binaries[k].X)
                    chosen_binaries.append(k)

            for k in self.all_nodes_adjusted:
                # print(str(k) + ': ' + str(self.edge_binaries[k].X))

                if self.costs[k].X:
                    print(k)
                    print(self.costs[k].X)

            path = []
            last_node_binary = [n for n in chosen_binaries if 'start' in n][0]

            path.append(last_node_binary)
            chosen_binaries.remove(last_node_binary)
            if len(chosen_binaries) > 0:
                while chosen_binaries:
                    last_node = last_node_binary.split('-')[1]

                    for n in chosen_binaries:
                        if last_node in n:
                            chosen_binaries.remove(n)
                            last_node_binary = n
                            path.append(n)

                            break

            print(path)
            total_costs = [self.production_costs[path[0].split('-')[0].split('start_')[1]]]
            for p in path:

                data = self.edges[p]

                if data[0] == 'conversion':  # conversion edge
                    costs = data[3]
                    efficiency = data[4]
                else:  # transport edge
                    costs = data[3]
                    efficiency = data[4]

                new_costs = (sum(total_costs) + costs) / (1 - efficiency)
                total_costs += [new_costs - sum(total_costs)]

            print(total_costs)
            print(new_costs)

path_config = os.getcwd()
path_config = os.path.dirname(path_config)

yaml_file = open(path_config + '/algorithm_configuration.yaml')
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

yaml_file = open(path_config + '/data/techno_economic_data_transportation.yaml')
techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

yaml_file = open(path_config + '/data/techno_economic_data_conversion.yaml')
techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

all_commodities = config_file['available_commodity']
start_commodities = config_file['available_commodity']
target_commodities = config_file['target_commodity']

path_overall_data = config_file['project_folder_path']
path_raw_data = path_overall_data + 'raw_data/'
path_processed_data = path_overall_data + 'processed_data/'
techno_economic_path = path_config + '/data/'

# This switch changes only the physical infrastructure input. Model nodes and
# edges are always constructed by `prepare_data`, using the same technology
# data files configured above.
USE_MINIMAL_INFRASTRUCTURE = True

# Hardcoded minimal infrastructure in the same forms read by `prepare_data`:
# options.csv, road/new-pipeline edge tables, and square shipping/pipeline
# distance matrices. No expanded commodity nodes or MIP edges are defined here.
# Intended cheapest path:
# start -(New_Pipeline_Gas/Hydrogen_Gas)-> PG_0
#       -(Pipeline_Gas/Hydrogen_Gas)-> PG_1
#       -(Road/Hydrogen_Gas)-> s_0
#       -(conversion: Hydrogen_Gas -> FTF)-> s_0
#       -(Shipping/FTF)-> s_1
#       -(New_Pipeline_Liquid/FTF)-> PL_0 -(sink)-> end
# Existing liquid/oil pipeline edges are still generated below, but they are
# intentionally not forced into the cheap route because only FTF can use them.
# The route therefore exercises every transport mean except Pipeline_Liquid.
# Longer branches give the optimizer feasible alternatives without making a
# route that skips the intended sequence competitive.
minimal_nodes = ['s_0', 's_1', 's_2', 's_3',
                 'PG_0', 'PG_1', 'PG_2',
                 'PL_0', 'PL_1']
minimal_infrastructure = {
    'options': pd.DataFrame(index=minimal_nodes),
    'road_distances': pd.DataFrame([
        # Cheap required connector before changing Hydrogen_Gas into FTF.
        ('PG_1', 's_0', 1000),
        # Alternative connections from the longer gas branch to ports.
        ('PG_2', 's_0', 30_000_000),
        ('PG_2', 's_2', 25_000_000),
        # Alternative final approach for road-capable liquid commodities.
        ('s_3', 'PL_0', 30_000_000),
    ], columns=['pointA', 'pointB', 'distance']),
    'start_road_distances': pd.DataFrame([
        # Complete but expensive shortcut that avoids both gas pipeline means.
        ('start', 's_0', 100_000_000_000),
        # Less extreme port entry for testing an alternate road/shipping path.
        ('start', 's_2', 50_000_000),
    ], columns=['pointA', 'pointB', 'distance']),
    'new_pipeline_distances': pd.DataFrame([
        # Cheap required liquid/oil connection after shipping FTF.
        ('s_1', 'PL_0', 1000),
        # Liquid detour that may continue through the existing oil pipeline.
        ('s_1', 'PL_1', 1_000_000_000),
        # Additional terminal approach from a secondary port.
        ('s_3', 'PL_0', 1_000_000_000),
    ], columns=['pointA', 'pointB', 'distance']),
    'start_new_pipeline_distances': pd.DataFrame([
        # Only short entry into the cheap corridor; Hydrogen_Gas uses New_Pipeline_Gas.
        ('start', 'PG_0', 1000),
        # Complete but expensive shortcuts that skip required corridor steps.
        ('start', 'PG_1', 100_000_000_000),
        ('start', 'PL_0', 100_000_000_000),
        # Alternative gas branch: feasible but more expensive than PG_0.
        ('start', 'PG_2', 50_000_000),
    ], columns=['pointA', 'pointB', 'distance']),
    'port_distances': pd.DataFrame([
        # s_0 -> s_1 is the cheap shipping step; secondary ports add detours.
        [0, 1000, 2_000_000, 3_000_000],
        [1000, 0, 2_000_000, 2_500_000],
        [2_000_000, 2_000_000, 0, 1_500_000],
        [3_000_000, 2_500_000, 1_500_000, 0],
    ], index=['s_0', 's_1', 's_2', 's_3'],
       columns=['s_0', 's_1', 's_2', 's_3']),
    'gas_pipeline_matrices': [pd.DataFrame([
        # PG_0 -> PG_1 is cheap; PG_2 supplies a longer pipeline alternative.
        [0, 1000, 40_000_000],
        [1000, 0, 20_000_000],
        [40_000_000, 20_000_000, 0],
    ], index=['PG_0', 'PG_1', 'PG_2'],
       columns=['PG_0', 'PG_1', 'PG_2'])],
    'oil_pipeline_matrices': [pd.DataFrame([
        # Oil/liquid pipelines remain available for edge-generation checks,
        # but PL_0 is already the minimal-example destination.
        [0, 1000],
        [1000, 0],
    ], index=['PL_0', 'PL_1'], columns=['PL_0', 'PL_1'])],
}

minimal_destination_infrastructure = ['PL_0']

# The minimal network has no preprocessing CSVs. Run the same global builder
# used by `_1_script_process_raw_data` once before adding start/end edges.
minimal_static_graph = build_static_mip_graph(
    minimal_infrastructure, config_file, techno_economic_data_conversion,
    techno_economic_data_transport)

# Optional MIP start for the minimal infrastructure. These are edge keys
# generated by `prepare_data`, not separately created optimization edges.
minimal_warm_start_route = [
    'start+Hydrogen_Gas-PG_0+Hydrogen_Gas-New_Pipeline_Gas',
    'PG_0+Hydrogen_Gas-PG_1+Hydrogen_Gas-Pipeline_Gas',
    'PG_1+Hydrogen_Gas-s_0+Hydrogen_Gas-Road',
    's_0+Hydrogen_Gas-s_0+FTF',
    's_0+FTF-s_1+FTF-Shipping',
    's_1+FTF-PL_0+FTF-New_Pipeline_Liquid',
    'PL_0+FTF-end',
]

if USE_MINIMAL_INFRASTRUCTURE:
    logger.info('Run optimization with hardcoded minimal infrastructure')
    # Fixed origin costs keep the minimal example independent from prepared
    # location files; graph expansion still uses the normal technology YAMLs.
    minimal_start_location = {
        'Hydrogen_Gas': 0, 'Ammonia': 10, 'Methane_Gas': 5,
        'Methane_Liquid': 10, 'Methanol': 12, 'FTF': 15
    }
    minimal_start_location = pd.Series(minimal_start_location)

    problem = OptimizationGurobiModel(
        static_graph=minimal_static_graph,
        start_location_data=minimal_start_location,
        start_road_distances=minimal_infrastructure['start_road_distances'],
        start_new_pipeline_distances=minimal_infrastructure['start_new_pipeline_distances'],
        end_location=minimal_destination_infrastructure,
        config_file=config_file,
        techno_economic_data_conversion=techno_economic_data_conversion,
        techno_economic_data_transport=techno_economic_data_transport,
        warm_start_route=minimal_warm_start_route)
else:
    logger.info('Load origin-independent graph for real-data optimization runs')
    static_graph = load_static_mip_graph(path_processed_data + 'mip_data/')
    options = pd.read_csv(path_processed_data + 'mip_data/options.csv', index_col=0)
    start_locations = pd.read_csv(path_overall_data + '/' + 'start_destination_combinations.csv', index_col=0)
    destination = get_destination(config_file)
    end_location = prepare_destination_mip_data(options, destination)['destination_infrastructure'].tolist()
    logger.info('Loaded %s global nodes, %s global edges and %s destination infrastructure nodes',
                len(static_graph['nodes']), len(static_graph['edges']), len(end_location))

    for i in start_locations.index:
        logger.info('Prepare optimization run for origin %s', i)
        start_point = Point([start_locations.loc[i, 'longitude'], start_locations.loc[i, 'latitude']])
        start_name = 'start'
        start_road_distances = pd.read_csv(
            path_processed_data + 'mip_data/' + str(i) + '_start_road_distances.csv', index_col=0)
        start_new_pipeline_distances = pd.read_csv(
            path_processed_data + 'mip_data/' + str(i) + '_start_new_pipeline_distances.csv', index_col=0)

        problem = OptimizationGurobiModel(
            static_graph=static_graph,
            start_location_data=start_locations.loc[i, :],
            start_road_distances=start_road_distances,
            start_new_pipeline_distances=start_new_pipeline_distances,
            end_location=end_location,
            config_file=config_file,
            techno_economic_data_conversion=techno_economic_data_conversion,
            techno_economic_data_transport=techno_economic_data_transport)

if False:
    result = pd.read_csv(path_overall_data + 'results/location_results/' + str(n) + '_final_solution.csv', index_col=0)
    cols = result.columns[0]

    end_location = result.at['current_infrastructure', result.columns[0]]

    problem = OptimizationGurobiModel()
