import itertools
import yaml
import os

import pandas as pd
import gurobipy as gp
import geopandas as gpd

from prepare_data import prepare_data, create_edges_from_distance_only, prepare_dummy_data, create_graph
from data_processing.process_mip_data import calculate_road_distances


# noinspection PyTypeChecker
class OptimizationGurobiModel:

    def attach_edges(self):

        self.costs = self.model.addVars(self.all_nodes_adjusted, name=self.all_nodes_adjusted)

        self.capacity_at_node = self.model.addVars(self.all_nodes_adjusted, vtype='I', name=self.all_nodes_adjusted)

        self.edge_binaries = self.model.addVars([*self.edges], vtype='B', name=[*self.edges.keys()])

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

            if 'start' in start:
                self.model.addConstr(self.costs[start] == self.production_costs[commodity],
                                     name=name)

            if True:
                self.model.addGenConstrIndicator(self.edge_binaries[edge], 1, (self.costs[start] + costs) / (1 - efficiency) - self.costs[end] <= 0)
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

        # for node in self.all_nodes_adjusted:
        #     name = 'max_costs_' + node
        #     self.model.addConstr(self.costs[node] <= self.BigM, name=name)

        self.model.addConstr(sum(self.costs[node] for node in self.target_nodes) >= min(self.production_costs.values()), name='min_costs')

        # self.model.addConstr(sum(self.capacity_at_node[node] for node in self.target_nodes) <= 1, name='min_costs')  # seq 1 because maybe one conversion at final node

        # todo: unterscheidung zwischen normalen nodes und end nodes. Aktuell nur ein end node definiert --> egal, oder? Dann ist end node halt der, der im ergebnis drin ist
        # ensure transport out of start node
        self.model.addConstr(sum(self.edge_binaries[key]
                                 for key in self.edges.keys()
                                 if ((self.edges[key][0] == 'transport') & ('start' in self.edges[key][1])))
                             == 1,
                             name='out_of_origin')

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

        # ensure that each node is visited max. once or a specific conversion takes place max. once
        for node in self.all_nodes_adjusted:
            self.model.addConstr(sum(self.edge_binaries[key]
                                     for key in self.edges.keys()
                                     if self.edges[key][2] == node) <= 1,
                                 name='only_visit_node_once')

        # balance activities of all nodes but start and end
        for node in self.all_nodes_adjusted:
            if ('start' not in node) & ('end' not in node):
                self.model.addConstr(sum(self.edge_binaries[key]
                                         for key in self.edges.keys()
                                         if self.edges[key][1] == node)
                                     == sum(self.edge_binaries[key]
                                         for key in self.edges.keys()
                                         if self.edges[key][2] == node),
                                     name='balance')

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

        self.attach_edges_test()

        if False:
            if self.solution_route is not None:
                print(self.solution_route)
                self.model.update()

                for edge in self.edge_binaries.keys():
                    if edge in self.solution_route:
                        self.edge_binaries[edge].Start = 1
                    else:
                        self.edge_binaries[edge].Start = 0

                # for i, n in enumerate(self.solution_route):
                #
                #     print(n)
                #     print(self.cost_route[i])
                #
                #     self.costs[n.split('-')[0]].Start = self.cost_route[i]
                #
                # self.costs['end'].Start = self.cost_route[-1]
                #
                # for n in self.all_nodes_adjusted:
                #     if 'start' in n:
                #         commodity = n.split('start_')[1]
                #         self.costs[n] = self.production_costs[commodity]
                #     else:
                #         self.costs[n].Start = 0

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

                    start_node = [a for a in active if 'start' in a][0].split('-')[0].split('_')[1]
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

                    print(total_costs)
                    print(new_costs)

                    sol = model.cbGetSolution(model._continuous)
                    for v, val in zip(model._continuous, sol):
                        if val > 0.1:
                            print(v)
                            print(val)

        # run optimization with callback
        self.model.optimize(incumbent_callback)
        self.instance = self

        self.status = self.model.status

        if self.status == 2:

            self.objective_function_value = self.model.objVal

    def __init__(self):

        # ----------------------------------
        # Set up problem
        self.solver = 'gurobi'
        self.instance = None
        self.status = None
        self.objective = None

        self.model_type = 'gurobi'

        # load techno economic data
        yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
        self.techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

        yaml_file = open(path_raw_data + 'techno_economic_data_transportation.yaml')
        self.techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self.BigM = 200
        self.eps = 0.001

        if True:

            self.all_nodes_adjusted, self.target_nodes, self.edges, self.production_costs, self.transport_means,\
                self.solution_route, self.cost_route, self.max_costs = prepare_data(0, end_location)

            # create_graph(self.edges, self.all_nodes_adjusted)

            six = []
            seven = []

            for e in self.edges:
                if len(self.edges[e]) == 6:
                    six.append(e)
                else:
                    seven.append(e)

            subset_six = {k: self.edges[k] for k in six}
            subset_seven = {k: self.edges[k] for k in seven}

            df = pd.DataFrame(subset_six).transpose()
            df.to_excel(path_overall_data + 'conversion_full_edges.xlsx')

            df = pd.DataFrame(subset_seven).transpose()
            df.to_excel(path_overall_data + 'transport_full_edges.xlsx')

        else:
            self.all_nodes_adjusted, self.target_nodes, self.edges, self.production_costs, self.transport_means = \
                prepare_dummy_data()

            self.solution_route = None

            six = []
            seven = []

            for e in self.edges:
                if len(self.edges[e]) == 6:
                    six.append(e)
                else:
                    seven.append(e)

            subset_six = {k: self.edges[k] for k in six}
            subset_seven = {k: self.edges[k] for k in seven}

            df = pd.DataFrame(subset_six).transpose()
            df.to_excel(path_overall_data + 'conversion_dummy_edges.xlsx')

            df = pd.DataFrame(subset_seven).transpose()
            df.to_excel(path_overall_data + 'transport_dummy_edges.xlsx')

        if False:
            start_distance = calculate_road_distances(config_file['tolerance_distance'], options, start_point, start_name)
            self.start_edges = create_edges_from_distance_only([start_distance],
                                                               ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'],
                                                               techno_economic_data_transport, all_commodities, start_commodities)

        self.model = gp.Model()
        self.optimize()

        if True:
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

all_commodities = config_file['available_commodity']
start_commodities = config_file['available_commodity']
target_commodities = config_file['target_commodity']

path_overall_data = config_file['project_folder_path']
path_raw_data = path_overall_data + 'raw_data/'
path_processed_data = path_overall_data + 'processed_data/'

options = pd.read_csv(path_processed_data + 'mip_data/' + 'options.csv', index_col=0)

n = 0
result = pd.read_csv(path_overall_data + 'results/location_results/' + str(n) + '_final_solution.csv', index_col=0)
cols = result.columns[0]

start_location = result.loc[['starting_latitude', 'starting_longitude'], :]
start_point = gpd.points_from_xy(start_location.loc['starting_longitude', :], start_location.loc['starting_latitude', :])
start_name = 'start'

end_location = result.at['current_infrastructure', result.columns[0]]

problem = OptimizationGurobiModel()