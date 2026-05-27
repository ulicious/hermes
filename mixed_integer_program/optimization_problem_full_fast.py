import logging
import os
import sys
import time

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB
from scipy import sparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from .prepare_data import prepare_data
except ImportError:
    from prepare_data import prepare_data
from data_processing.helpers_geometry import get_destination
from data_processing.process_mip_data import (
    load_minimal_mip_case,
    load_static_mip_graph,
    prepare_destination_mip_data,
)


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
                 warm_start_route=None, create_results=False, export_edges=False,
                 solve=False):
        self.config_file = config_file
        self.techno_economic_data_conversion = techno_economic_data_conversion
        self.techno_economic_data_transport = techno_economic_data_transport

        logger.info('Add origin- and destination-specific graph data')
        self.all_nodes_adjusted, self.target_nodes, self.edges, self.production_costs, \
            self.transport_means, self.solution_route, self.cost_route, self.max_costs, \
            self.conversion_edges, self.transport_edges = prepare_data(
                start_location_data, static_graph, start_road_distances,
                start_new_pipeline_distances, end_location, config_file,
                techno_economic_data_transport, create_results=create_results,
                warm_start_route=warm_start_route)

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

        self.model.setObjective(self.costs[target_indices].sum(), GRB.MINIMIZE)
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


def load_inputs():
    with open(os.path.join(PROJECT_ROOT, 'algorithm_configuration.yaml')) as yaml_file:
        config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    with open(os.path.join(PROJECT_ROOT, 'data', 'techno_economic_data_transportation.yaml')) as yaml_file:
        transport_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    with open(os.path.join(PROJECT_ROOT, 'data', 'techno_economic_data_conversion.yaml')) as yaml_file:
        conversion_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config_file, conversion_data, transport_data


def run_optimization(use_minimal_infrastructure=None, solve=False):
    config_file, conversion_data, transport_data = load_inputs()
    if use_minimal_infrastructure is None:
        use_minimal_infrastructure = config_file.get('use_minimal_example', False)
    processed_path = os.path.join(config_file['project_folder_path'], 'processed_data') + os.sep

    if use_minimal_infrastructure:
        logger.info('Load preprocessed minimal infrastructure case')
        case = load_minimal_mip_case(processed_path)
        return FastOptimizationGurobiModel(
            static_graph=case['static_graph'],
            start_location_data=case['start_location_data'],
            start_road_distances=case['start_road_distances'],
            start_new_pipeline_distances=case['start_new_pipeline_distances'],
            end_location=case['end_location'],
            config_file=config_file,
            techno_economic_data_conversion=conversion_data,
            techno_economic_data_transport=transport_data,
            warm_start_route=case['warm_start_route'],
            solve=solve)

    logger.info('Load origin-independent graph for real-data optimization runs')
    static_graph = load_static_mip_graph(os.path.join(processed_path, 'mip_data') + os.sep)
    options = pd.read_csv(os.path.join(processed_path, 'mip_data', 'options.csv'), index_col=0)
    start_locations = pd.read_csv(
        os.path.join(config_file['project_folder_path'], 'start_destination_combinations.csv'),
        index_col=0)
    destination = get_destination(config_file)
    end_location = prepare_destination_mip_data(options, destination)['destination_infrastructure'].tolist()
    problems = []
    for location in start_locations.index:
        logger.info('Prepare optimization run for origin %s', location)
        problems.append(FastOptimizationGurobiModel(
            static_graph=static_graph,
            start_location_data=start_locations.loc[location, :],
            start_road_distances=pd.read_csv(
                os.path.join(processed_path, 'mip_data', str(location) + '_start_road_distances.csv'),
                index_col=0),
            start_new_pipeline_distances=pd.read_csv(
                os.path.join(processed_path, 'mip_data', str(location) + '_start_new_pipeline_distances.csv'),
                index_col=0),
            end_location=end_location,
            config_file=config_file,
            techno_economic_data_conversion=conversion_data,
            techno_economic_data_transport=transport_data,
            solve=solve))
    return problems


if __name__ == '__main__':
    run_optimization()
