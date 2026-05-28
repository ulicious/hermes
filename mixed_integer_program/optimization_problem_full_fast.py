import logging
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy import sparse

try:
    from .prepare_data import prepare_data
except ImportError:
    from prepare_data import prepare_data


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Gurobi writes its optimization progress directly to the console.
logging.getLogger('gurobipy').setLevel(logging.WARNING)


class FastOptimizationGurobiModel:
    """
    Matrix-API implementation of the full MIP graph model.

    The original model creates one Gurobi indicator constraint per edge in a
    Python loop. This variant keeps the same active mathematical constraints,
    but sends edge propagation and graph-flow constraints to Gurobi in bulk.
    """

    def __init__(self, static_graph, start_location_data, start_road_distances,
                 start_new_pipeline_distances, end_location, config_file,
                 techno_economic_data_conversion, techno_economic_data_transport,
                 warm_start_route=None, export_edges=False,
                 use_warm_start_as_lower_bound=False,
                 solve=False):

        self.config_file = config_file
        self.techno_economic_data_conversion = techno_economic_data_conversion
        self.techno_economic_data_transport = techno_economic_data_transport
        self.solution_route = warm_start_route
        self.use_warm_start_as_lower_bound = use_warm_start_as_lower_bound
        self.warm_start_objective_lower_bound = None

        logger.info('Add origin- and destination-specific graph data')
        self.all_nodes_adjusted, self.target_nodes, self.edges, self.production_costs, \
            self.transport_means, self.max_costs, \
            self.conversion_edges, self.transport_edges = prepare_data(
                start_location_data, static_graph, start_road_distances,
                start_new_pipeline_distances, end_location, config_file,
                techno_economic_data_transport)

        logger.info('Optimization graph contains %s nodes and %s edges (%s conversion, %s transport)',
                    len(self.all_nodes_adjusted), len(self.edges),
                    len(self.conversion_edges), len(self.transport_edges))

        if export_edges:
            self._export_edges(config_file['project_folder_path'])

        self.model = gp.Model()
        self.status = None
        self.objective_function_value = None
        self.chosen_edges = []
        self._prepare_index_arrays()
        self._prepare_warm_start_lower_bound()
        self.build_model()
        if solve:
            self.optimize()

    def _export_edges(self, path_overall_data):
        """Optional diagnostic export; disabled by default for large graphs."""
        self.conversion_edges.to_csv(os.path.join(path_overall_data, 'conversion_edges.csv'))
        self.transport_edges.to_csv(os.path.join(path_overall_data, 'transport_edges.csv'))
        logger.info('Exported current conversion and transport edge tables')

    def _prepare_index_arrays(self):
        self.node_names = list(self.all_nodes_adjusted)
        self.node_index = {node: index for index, node in enumerate(self.node_names)}
        self.edge_names = list(self.edges)
        self.edge_index = {edge: index for index, edge in enumerate(self.edge_names)}
        edge_values = [self.edges[edge] for edge in self.edge_names]

        self.start_indices = np.fromiter(
            (self.node_index[edge[1]] for edge in edge_values), dtype=np.int64)
        self.end_indices = np.fromiter(
            (self.node_index[edge[2]] for edge in edge_values), dtype=np.int64)
        self.edge_costs = np.fromiter((edge[3] for edge in edge_values), dtype=float)
        self.edge_losses = np.fromiter((edge[4] for edge in edge_values), dtype=float)
        self.edge_scales = 1.0 / (1.0 - self.edge_losses)

        self.origin_edge_indices = np.fromiter(
            (index for index, edge in enumerate(edge_values)
             if edge[0] == 'transport' and 'start' in edge[1]), dtype=np.int64)
        self.destination_in_indices = np.fromiter(
            (index for index, edge in enumerate(edge_values)
             if edge[0] == 'transport' and edge[2] == 'end'), dtype=np.int64)
        self.destination_out_indices = np.fromiter(
            (index for index, edge in enumerate(edge_values)
             if edge[0] == 'transport' and edge[1] == 'end'), dtype=np.int64)

    def _prepare_warm_start_lower_bound(self):
        """Optionally calculate the lower bound from the actual warm-start route."""
        if not self.use_warm_start_as_lower_bound:
            return
        if self.solution_route is None:
            logger.warning('Warm-start lower bound requested, but no warm-start route was provided')
            return

        self.warm_start_objective_lower_bound = self._calculate_route_objective(
            self.solution_route)
        logger.info('Calculated warm-start lower bound from route: %.6f',
                    self.warm_start_objective_lower_bound)

    def _calculate_route_objective(self, route):
        """Calculate total route costs using the same edge propagation logic as the MIP."""
        if not route:
            raise ValueError('Cannot calculate warm-start lower bound for an empty route')

        first_edge = self.edges[route[0]]
        start_commodity = first_edge[1].split('+', 1)[1]
        total_costs = self.production_costs[start_commodity]

        for edge_key in route:
            edge = self.edges[edge_key]
            edge_costs = edge[3]
            edge_loss = edge[4]
            total_costs = (total_costs + edge_costs) / (1 - edge_loss)

        return total_costs

    def build_model(self):
        """Create variables and constraints using the Gurobi matrix API."""
        now = time.time()
        node_count = len(self.node_names)
        edge_count = len(self.edge_names)
        logger.info('Create matrix model for %s nodes and %s edges', node_count, edge_count)

        self.costs = self.model.addMVar(node_count, lb=0.0, name='cost')
        self.edge_binaries = self.model.addMVar(edge_count, vtype=GRB.BINARY, name='use')

        propagation_lhs = (
            self.edge_scales * self.costs[self.start_indices] -
            self.costs[self.end_indices])
        propagation_rhs = -self.edge_scales * self.edge_costs
        self.model.addGenConstrIndicator(
            self.edge_binaries, True, propagation_lhs, GRB.LESS_EQUAL,
            propagation_rhs, name='cost_propagation')

        start_node_values = {
            self.node_index['start+' + commodity]: value
            for commodity, value in self.production_costs.items()
            if 'start+' + commodity in self.node_index
        }
        if start_node_values:
            production_indices = np.fromiter(start_node_values, dtype=np.int64)
            production_values = np.fromiter(start_node_values.values(), dtype=float)
            self.model.addConstr(
                self.costs[production_indices] == production_values,
                name='production_cost')

        target_indices = np.array(
            [self.node_index[node] for node in self.target_nodes], dtype=np.int64)
        self.model.addConstr(
            self.costs[target_indices].sum() >= min(self.production_costs.values()),
            name='min_costs')

        self.model.addConstr(
            self.edge_binaries[self.origin_edge_indices].sum() == 1,
            name='out_of_origin')
        self.model.addConstr(
            self.edge_binaries[self.destination_in_indices].sum() == 1,
            name='into_destination')
        if self.destination_out_indices.size:
            self.model.addConstr(
                self.edge_binaries[self.destination_out_indices].sum() == 0,
                name='no_out_destination')

        columns = np.arange(edge_count, dtype=np.int64)
        incidence_values = np.ones(edge_count)
        incoming = sparse.csr_matrix(
            (incidence_values, (self.end_indices, columns)),
            shape=(node_count, edge_count))
        outgoing = sparse.csr_matrix(
            (incidence_values, (self.start_indices, columns)),
            shape=(node_count, edge_count))
        self.model.addMConstr(
            incoming, self.edge_binaries, GRB.LESS_EQUAL,
            np.ones(node_count), name='only_visit_node_once')

        internal_indices = np.array(
            [index for index, node in enumerate(self.node_names)
             if 'start' not in node and 'end' not in node],
            dtype=np.int64)
        balance = (outgoing - incoming)[internal_indices, :]
        self.model.addMConstr(
            balance, self.edge_binaries, GRB.EQUAL,
            np.zeros(len(internal_indices)), name='balance')

        self.objective_expr = self.costs[target_indices].sum()
        if self.warm_start_objective_lower_bound is not None:
            self.model.addConstr(
                self.objective_expr >= self.warm_start_objective_lower_bound,
                name='warm_start_objective_lower_bound')
            logger.info('Added objective lower bound from warm-start value: %.6f',
                        self.warm_start_objective_lower_bound)
        self.model.setObjective(self.objective_expr, GRB.MINIMIZE)
        self.model.update()
        logger.info('Finished matrix model construction in %.2f s', time.time() - now)

    def optimize(self):
        if self.solution_route is not None:
            start_values = np.zeros(len(self.edge_names))
            for edge in self.solution_route:
                start_values[self.edge_index[edge]] = 1
            self.edge_binaries.Start = start_values
            logger.info('Applied warm-start route with %s active edges', len(self.solution_route))

        self.model.Params.IntFeasTol = 1e-9
        self.model.Params.FeasibilityTol = 1e-9
        self.model.Params.OptimalityTol = 1e-9
        self.model.Params.Method = 2
        self.model.Params.Crossover = 0
        self.model.Params.BarHomogeneous = 1
        self.model.Params.MIPFocus = 3
        self.model.Params.Heuristics = 0
        self.model.Params.Cuts = 3
        self.model.Params.Presolve = 2
        self.model.Params.PoolSearchMode = 0
        self.model.Params.Threads = 1

        logger.info('Start optimization')
        self.model.optimize()
        self.status = self.model.Status
        if self.status == GRB.OPTIMAL:
            self.objective_function_value = self.model.ObjVal
            selected = np.flatnonzero(self.edge_binaries.X > 0.5)
            self.chosen_edges = [self.edge_names[index] for index in selected]
            logger.info('Optimal objective value: %.6f', self.objective_function_value)
