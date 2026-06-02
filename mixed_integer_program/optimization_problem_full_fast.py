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
incumbent_logger = logging.getLogger(__name__ + '.incumbent')
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

    CONNECTOR_TRANSPORT_MEANS = {'Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'}

    def __init__(self, static_graph, start_location_data, start_road_distances,
                 start_new_pipeline_distances, end_location, config_file,
                 techno_economic_data_conversion, techno_economic_data_transport,
                 warm_start_route=None, warm_start_bound_route=None, export_edges=False,
                 use_warm_start_as_lower_bound=False,
                 use_warm_start_as_upper_bound=False,
                 mip_gap=None, time_limit=None,
                 filter_edges_above_warm_start=False,
                 filter_start_options_above_warm_start=False,
                 filter_unreachable_edges=False,
                 solve=False):

        self.config_file = config_file
        self.techno_economic_data_conversion = techno_economic_data_conversion
        self.techno_economic_data_transport = techno_economic_data_transport
        self.solution_route = warm_start_route
        self.warm_start_bound_route = warm_start_bound_route if warm_start_bound_route is not None else warm_start_route
        self.use_warm_start_as_lower_bound = use_warm_start_as_lower_bound
        self.use_warm_start_as_upper_bound = use_warm_start_as_upper_bound
        self.warm_start_objective = None
        self.warm_start_objective_lower_bound = None
        self.warm_start_objective_upper_bound = None
        self.big_m_cost_propagation = None
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.filter_edges_above_warm_start = filter_edges_above_warm_start
        self.filter_start_options_above_warm_start = filter_start_options_above_warm_start
        self.filter_unreachable_edges = filter_unreachable_edges

        logger.debug('Add origin- and destination-specific graph data')
        self.all_nodes_adjusted, self.target_nodes, self.edges, self.production_costs, \
            self.transport_means, self.max_costs, \
            self.conversion_edges, self.transport_edges = prepare_data(
                start_location_data, static_graph, start_road_distances,
                start_new_pipeline_distances, end_location, config_file,
                techno_economic_data_transport, techno_economic_data_conversion,
                self.warm_start_bound_route,
                filter_edges_above_warm_start,
                filter_start_options_above_warm_start,
                filter_unreachable_edges)

        logger.debug('Optimization graph contains %s nodes and %s edges (%s conversion, %s transport)',
                     len(self.all_nodes_adjusted), len(self.edges),
                     len(self.conversion_edges), len(self.transport_edges))
        self.solution_route = self._validate_route_against_graph(self.solution_route, 'warm-start route')
        self.warm_start_bound_route = self._validate_route_against_graph(
            self.warm_start_bound_route, 'warm-start bound route')

        if export_edges:
            self._export_edges(config_file['project_folder_path'])

        self.model = gp.Model()
        self.status = None
        self.objective_function_value = None
        self.chosen_edges = []
        self.incumbent_history = []
        self._prepare_index_arrays()
        self._prepare_warm_start_objective_bounds()
        self.use_big_m_constraints = self.warm_start_objective is not None
        self.build_model()
        self._log_setup_summary()
        if solve:
            self.optimize()

    def _export_edges(self, path_overall_data):
        """Optional diagnostic export; disabled by default for large graphs."""
        self.conversion_edges.to_csv(os.path.join(path_overall_data, 'conversion_edges.csv'))
        self.transport_edges.to_csv(os.path.join(path_overall_data, 'transport_edges.csv'))
        logger.info('Exported current conversion and transport edge tables')

    def _validate_route_against_graph(self, route, route_name):
        """Disable optional route-based features if the route does not fit the current graph."""
        if route is None:
            return None
        missing_edges = [edge for edge in route if edge not in self.edges]
        if missing_edges:
            logger.warning('%s ignored because %s route edges are absent from the current MIP graph. '
                           'First missing edge: %s',
                           route_name, len(missing_edges), missing_edges[0])
            return None
        return route

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

    @staticmethod
    def _physical_node_name(node):
        """Return the infrastructure node without the commodity suffix."""
        if '+' not in node:
            return node
        return node.split('+', 1)[0]

    def _connector_edge_indices_by_physical_node(self):
        """Collect connector edges that enter or leave each physical infrastructure node."""
        incoming = {}
        outgoing = {}

        for edge_index, edge_key in enumerate(self.edge_names):
            edge = self.edges[edge_key]
            if edge[0] != 'transport':
                continue
            if edge[6] not in self.CONNECTOR_TRANSPORT_MEANS:
                continue
            if edge[2] != 'end':
                incoming.setdefault(self._physical_node_name(edge[2]), []).append(edge_index)
            if not edge[1].startswith('start+'):
                outgoing.setdefault(self._physical_node_name(edge[1]), []).append(edge_index)

        return incoming, outgoing

    @staticmethod
    def _pipeline_graph_name_from_node(node):
        """Return the physical pipeline graph id, e.g. PG_Graph_5, from a commodity-expanded node."""
        physical_node = FastOptimizationGurobiModel._physical_node_name(node)
        if '_Node_' not in physical_node:
            return None
        return physical_node.rsplit('_Node_', 1)[0]

    def _existing_pipeline_edge_indices_by_graph(self):
        """Collect existing pipeline edges by physical graph to prevent repeated graph use."""
        edges_by_graph = {}
        for edge_index, edge_key in enumerate(self.edge_names):
            edge = self.edges[edge_key]
            if edge[0] != 'transport':
                continue
            if edge[6] not in {'Pipeline_Gas', 'Pipeline_Liquid'}:
                continue

            start_graph = self._pipeline_graph_name_from_node(edge[1])
            end_graph = self._pipeline_graph_name_from_node(edge[2])
            if start_graph is None or start_graph != end_graph:
                continue
            edges_by_graph.setdefault(start_graph, []).append(edge_index)
        return edges_by_graph

    def _prepare_warm_start_objective_bounds(self):
        """Calculate warm-start objective value and derive optional model bounds."""
        if self.warm_start_bound_route is None:
            if self.use_warm_start_as_lower_bound or self.use_warm_start_as_upper_bound:
                logger.warning('Warm-start objective bound requested, but no warm-start route was provided')
            return

        warm_start_objective = self._calculate_route_objective(self.warm_start_bound_route)
        self.warm_start_objective = warm_start_objective
        max_production_costs = max(self.production_costs.values()) if self.production_costs else 0
        finite_edge_mask = self.edge_losses < 1
        if finite_edge_mask.any():
            max_edge_requirement = np.max(
                (max_production_costs + self.edge_costs[finite_edge_mask]) * self.edge_scales[finite_edge_mask])
        else:
            max_edge_requirement = 0
        self.big_m_cost_propagation = max(warm_start_objective, max_production_costs, max_edge_requirement)
        if self.big_m_cost_propagation > warm_start_objective:
            logger.debug('Increase effective Big-M from warm-start objective %.6f to %.6f '
                         'to cover fixed start production costs and edge propagation',
                         warm_start_objective, self.big_m_cost_propagation)
        if self.use_warm_start_as_lower_bound:
            self.warm_start_objective_lower_bound = warm_start_objective
            logger.debug('Calculated warm-start lower bound from route: %.6f',
                         self.warm_start_objective_lower_bound)
        self.warm_start_objective_upper_bound = warm_start_objective
        logger.debug('Calculated automatic warm-start upper bound from route: %.6f',
                     self.warm_start_objective_upper_bound)

    def _estimate_transport_distance_km(self, edge):
        """Estimate route-section distance from the MIP transport cost definition."""
        edge_costs = edge[3]
        commodity = edge[5]
        transport_mean = edge[6]
        if transport_mean == 'Destination':
            return 0.0

        cost_rate = self.techno_economic_data_transport[commodity].get(transport_mean)
        if cost_rate in (None, 0):
            return None

        return edge_costs * 1000000 / cost_rate

    @staticmethod
    def _as_float(value):
        """Parse numeric configuration values, including the local `math.inf` convention."""
        if value is None:
            return None
        if isinstance(value, str):
            if value == 'math.inf':
                return np.inf
            return float(value)
        return float(value)

    def _configuration_forbidden_reason(self, edge):
        """Return why an edge is forbidden by run-specific heuristic rules, or None."""
        edge_type = edge[0]
        start = edge[1]
        end = edge[2]
        edge_costs = edge[3]
        edge_loss = edge[4]

        if edge_type == 'conversion':
            start_commodity = start.split('+', 1)[1]
            end_commodity = end.split('+', 1)[1]
            conversion_options = self.techno_economic_data_conversion.get(
                start_commodity, {}).get('potential_conversions', [])
            if end_commodity not in conversion_options:
                return 'conversion_not_allowed'
            if not np.isfinite(edge_costs) or not np.isfinite(edge_loss):
                return 'conversion_cost_or_loss_not_finite'
            return None

        if edge_type != 'transport':
            return None

        commodity = edge[5]
        transport_mean = edge[6]
        if transport_mean == 'Destination':
            return None

        if self._physical_node_name(start) == self._physical_node_name(end):
            return 'self_loop'

        if transport_mean not in self.config_file['available_transport_means']:
            return 'transport_mean_not_available'

        transport_options = self.techno_economic_data_transport.get(
            commodity, {}).get('potential_transportation', [])
        if transport_mean not in transport_options:
            return 'commodity_transport_not_allowed'

        if 'New' in transport_mean and not self.config_file['build_new_infrastructure']:
            return 'new_infrastructure_disabled'

        if (commodity == 'Hydrogen_Gas'
                and transport_mean == 'Pipeline_Gas'
                and not self.config_file['H2_ready_infrastructure']):
            return 'hydrogen_in_existing_gas_pipeline_not_ready'

        distance = self._estimate_transport_distance_km(edge)
        if distance is not None and np.isfinite(distance):
            if transport_mean == 'Road':
                max_length_road = self._as_float(self.config_file['max_length_road'])
                if max_length_road is not None and distance > max_length_road + 1e-9:
                    return 'road_distance_above_max_length'
            elif transport_mean in {'New_Pipeline_Gas', 'New_Pipeline_Liquid'}:
                max_length_new_segment = self._as_float(self.config_file['max_length_new_segment'])
                if max_length_new_segment is not None and distance > max_length_new_segment + 1e-9:
                    return 'new_pipeline_distance_above_max_length'

        return None

    def _configuration_forbidden_edge_indices(self):
        """Collect edges disabled by run-specific heuristic feasibility assumptions."""
        forbidden_indices = []
        forbidden_by_reason = {}
        for edge_index, edge_key in enumerate(self.edge_names):
            reason = self._configuration_forbidden_reason(self.edges[edge_key])
            if reason is None:
                continue
            forbidden_indices.append(edge_index)
            forbidden_by_reason[reason] = forbidden_by_reason.get(reason, 0) + 1
        return forbidden_indices, forbidden_by_reason

    def log_warm_start_route_details(self, route, route_name='warm-start route'):
        """Log a compact cost summary for a known route."""
        if not route:
            logger.warning('Cannot log %s because it is empty', route_name)
            return
        missing_edges = [edge for edge in route if edge not in self.edges]
        if missing_edges:
            logger.warning('Cannot log %s because %s route edges are absent from the current MIP graph. '
                           'First missing edge: %s',
                           route_name, len(missing_edges), missing_edges[0])
            return

        first_edge = self.edges[route[0]]
        start_commodity = first_edge[1].split('+', 1)[1]
        total_costs = self.production_costs[start_commodity]
        route_parts = [f'production {start_commodity}={total_costs:.6f}']

        for edge_key in route:
            edge = self.edges[edge_key]
            edge_costs = edge[3]
            edge_loss = edge[4]
            total_costs = (total_costs + edge_costs) / (1 - edge_loss)
            if edge[0] == 'transport':
                route_parts.append(
                    f'{self._physical_node_name(edge[1])}->{self._physical_node_name(edge[2])} '
                    f'{edge[6]} +{edge_costs:.6f}')
            else:
                route_parts.append(
                    f'{edge[1]}->{edge[2]} conversion +{edge_costs:.6f}')

        logger.info('%s | objective %.6f | %s',
                    route_name, total_costs, ' | '.join(route_parts))

    def _calculate_route_objective(self, route):
        """Calculate total route costs using the same edge propagation logic as the MIP."""
        if not route:
            raise ValueError('Cannot calculate warm-start lower bound for an empty route')
        missing_edges = [edge for edge in route if edge not in self.edges]
        if missing_edges:
            raise ValueError('Cannot calculate warm-start objective because '
                             + str(len(missing_edges))
                             + ' route edges are absent from the current MIP graph. First missing edge: '
                             + missing_edges[0])

        first_edge = self.edges[route[0]]
        start_commodity = first_edge[1].split('+', 1)[1]
        total_costs = self.production_costs[start_commodity]

        for edge_key in route:
            edge = self.edges[edge_key]
            edge_costs = edge[3]
            edge_loss = edge[4]
            total_costs = (total_costs + edge_costs) / (1 - edge_loss)

        return total_costs

    def _route_summary(self, route):
        """Return objective and compact route text for a route represented by edge keys."""
        if not route:
            return None, 'route unavailable'

        first_edge = self.edges[route[0]]
        start_commodity = first_edge[1].split('+', 1)[1]
        total_costs = self.production_costs[start_commodity]
        route_parts = [f'production {start_commodity}={total_costs:.6f}']

        for edge_key in route:
            edge = self.edges[edge_key]
            edge_costs = edge[3]
            edge_loss = edge[4]
            total_costs = (total_costs + edge_costs) / (1 - edge_loss)
            if edge[0] == 'transport':
                route_parts.append(
                    f'{self._physical_node_name(edge[1])}->{self._physical_node_name(edge[2])} '
                    f'{edge[6]} +{edge_costs:.6f}')
            else:
                route_parts.append(f'{edge[1]}->{edge[2]} conversion +{edge_costs:.6f}')

        return total_costs, ' | '.join(route_parts)

    def _append_route_history(self, kind, objective, route_text, runtime=None):
        """Store warm-start and incumbent routes for a readable diagnostic file."""
        self.incumbent_history.append({
            'kind': kind,
            'runtime': runtime,
            'objective': objective,
            'route': route_text,
        })

    def _route_from_active_indices(self, active_indices):
        """Reconstruct the selected origin-to-destination path from active edge indices."""
        active_edges = [self.edge_names[index] for index in active_indices]
        edge_by_start = {}
        for edge_key in active_edges:
            start = self.edges[edge_key][1]
            edge_by_start[start] = edge_key

        start_edges = [
            edge_key for edge_key in active_edges
            if self.edges[edge_key][1].startswith('start+')
        ]
        if not start_edges:
            return [], active_edges, 'no active start edge'

        route = []
        visited_nodes = set()
        current_node = self.edges[start_edges[0]][1]
        issue = None
        while current_node in edge_by_start:
            if current_node in visited_nodes:
                issue = 'cycle detected while reconstructing incumbent route'
                break
            visited_nodes.add(current_node)
            edge_key = edge_by_start[current_node]
            route.append(edge_key)
            current_node = self.edges[edge_key][2]
            if current_node == 'end':
                break

        if current_node != 'end' and issue is None:
            issue = 'active path does not reach end'

        route_set = set(route)
        extra_edges = [edge_key for edge_key in active_edges if edge_key not in route_set]
        return route, extra_edges, issue

    def _log_incumbent_route_details(self, objective_value, runtime, binary_values, cost_values):
        """Log every new incumbent with compact route and validation diagnostics."""
        active_indices = np.flatnonzero(binary_values > 0.5)
        route, extra_edges, route_issue = self._route_from_active_indices(active_indices)

        fractionality = np.minimum(np.abs(binary_values), np.abs(1 - binary_values))
        max_fractionality = float(fractionality.max()) if fractionality.size else 0.0
        if not route:
            incumbent_logger.info(
                'New incumbent after %.2f s | objective %.9f | active edges %s | route unavailable',
                runtime, objective_value, len(active_indices))
            self._append_route_history('incumbent', objective_value, 'route unavailable', runtime)
            if route_issue is not None:
                incumbent_logger.warning('Incumbent route reconstruction issue: %s', route_issue)
            return

        first_edge = self.edges[route[0]]
        start_commodity = first_edge[1].split('+', 1)[1]
        recomputed_costs = self.production_costs[start_commodity]
        largest_active_propagation_violation = 0.0
        for edge_key in route:
            edge = self.edges[edge_key]
            start = edge[1]
            end = edge[2]
            edge_costs = edge[3]
            edge_loss = edge[4]
            recomputed_costs = (recomputed_costs + edge_costs) / (1 - edge_loss)

            solver_start_costs = cost_values[self.node_index[start]]
            solver_end_costs = cost_values[self.node_index[end]]
            required_end_costs = (solver_start_costs + edge_costs) / (1 - edge_loss)
            propagation_violation = max(0.0, required_end_costs - solver_end_costs)
            largest_active_propagation_violation = max(
                largest_active_propagation_violation, propagation_violation)

        objective_difference = abs(recomputed_costs - objective_value)
        _, route_text = self._route_summary(route)
        self._append_route_history('incumbent', objective_value, route_text, runtime)
        incumbent_logger.info('New incumbent after %.2f s | objective %.6f | path %.6f | edges %s | '
                              'diff %.1e | frac %.1e | route: %s',
                              runtime, objective_value, recomputed_costs, len(route),
                              objective_difference, max_fractionality, route_text)
        if route_issue is not None:
            incumbent_logger.warning('Incumbent route reconstruction issue: %s', route_issue)
        if extra_edges:
            incumbent_logger.warning('Incumbent contains %s active edges outside the reconstructed path: %s',
                                     len(extra_edges), extra_edges[:10])
        if objective_difference > 1e-6 or largest_active_propagation_violation > 1e-6:
            incumbent_logger.warning('Incumbent validation issue | objective/path diff %.3e | '
                                     'largest active propagation violation %.3e',
                                     objective_difference, largest_active_propagation_violation)

    def _log_solution_comparison(self):
        """Compare warm-start route with the final incumbent after optimization."""
        if self.model.SolCount == 0:
            return

        selected = np.flatnonzero(self.edge_binaries.X > 0.5)
        optimal_route, extra_edges, route_issue = self._route_from_active_indices(selected)
        optimal_costs, optimal_text = self._route_summary(optimal_route)

        if self.solution_route is not None:
            if list(self.solution_route) == list(optimal_route):
                return
            warm_start_costs, warm_start_text = self._route_summary(self.solution_route)
            improvement = warm_start_costs - optimal_costs
            improvement_percent = improvement / warm_start_costs * 100 if warm_start_costs else 0
            incumbent_logger.info('Solution comparison | warm start %.6f -> final %.6f | '
                                  'improvement %.6f (%.2f%%)',
                                  warm_start_costs, optimal_costs, improvement, improvement_percent)
            incumbent_logger.info('Warm start route: %s', warm_start_text)
        else:
            incumbent_logger.info('Solution comparison | no warm start route | final %.6f',
                                  optimal_costs)

        incumbent_logger.info('Final route: %s', optimal_text)
        if route_issue is not None:
            incumbent_logger.warning('Final route reconstruction issue: %s', route_issue)
        if extra_edges:
            incumbent_logger.warning('Final solution contains %s active edges outside the reconstructed path: %s',
                                     len(extra_edges), extra_edges[:10])

    def _write_incumbent_history_file(self):
        """Append warm-start and incumbent routes to the shared run diagnostic file."""
        if not self.incumbent_history:
            return

        path_results = os.path.join(self.config_file['project_folder_path'], 'results')
        os.makedirs(path_results, exist_ok=True)
        location = self.config_file.get('current_mip_location', 'unknown')
        warm_start_objectives = [
            entry['objective'] for entry in self.incumbent_history
            if entry['kind'] == 'warm_start'
        ]
        incumbent_objectives = [
            entry['objective'] for entry in self.incumbent_history
            if entry['kind'] == 'incumbent'
        ]
        has_improvement = (
            bool(warm_start_objectives)
            and bool(incumbent_objectives)
            and min(incumbent_objectives) < warm_start_objectives[0] - 1e-9
        )
        file_path = self.config_file.get(
            'mip_incumbent_history_file',
            os.path.join(path_results, 'mip_incumbents_all_locations_running.txt')
        )

        lines = ['', '=' * 100,
                 f'MIP incumbent history for location {location}',
                 f'Improved objective: {has_improvement}', '']
        for number, entry in enumerate(self.incumbent_history, start=1):
            runtime = ''
            if entry['runtime'] is not None:
                runtime = f" after {entry['runtime']:.2f} s"
            lines.append(
                f"{number}. {entry['kind']}{runtime} | objective {entry['objective']:.6f}")
            lines.append(f"   route: {entry['route']}")
            lines.append('')

        with open(file_path, 'a', encoding='utf-8') as file:
            file.write('\n'.join(lines))

    def _incumbent_callback(self, model, where):
        """Gurobi callback that logs every strict improvement over the current incumbent."""
        if where != GRB.Callback.MIPSOL:
            return

        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if objective_value >= model._best_incumbent - 1e-9:
            return

        model._best_incumbent = objective_value
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        binary_values = np.array(model.cbGetSolution(model._edge_binary_vars), dtype=float)
        cost_values = np.array(model.cbGetSolution(model._cost_vars), dtype=float)
        self._log_incumbent_route_details(
            objective_value, runtime, binary_values, cost_values)

    def _create_cost_start_values(self, route):
        """Create continuous MIP-start values consistent with production and route propagation."""
        cost_start_values = np.full(len(self.node_names), GRB.UNDEFINED, dtype=float)
        for commodity, production_cost in self.production_costs.items():
            node = 'start+' + commodity
            if node in self.node_index:
                cost_start_values[self.node_index[node]] = production_cost

        if not route:
            return cost_start_values

        first_edge = self.edges[route[0]]
        start_commodity = first_edge[1].split('+', 1)[1]
        total_costs = self.production_costs[start_commodity]
        cost_start_values[self.node_index[first_edge[1]]] = total_costs

        for edge_key in route:
            edge = self.edges[edge_key]
            total_costs = (total_costs + edge[3]) / (1 - edge[4])
            cost_start_values[self.node_index[edge[2]]] = total_costs

        return cost_start_values

    def _log_setup_summary(self):
        """Log compact grouped information for one MIP location."""
        location = self.config_file.get('current_mip_location', 'unknown')
        config_summary = {
            'available_transport_means': self.config_file.get('available_transport_means'),
            'build_new_infrastructure': self.config_file.get('build_new_infrastructure'),
            'H2_ready_infrastructure': self.config_file.get('H2_ready_infrastructure'),
            'max_length_road': self.config_file.get('max_length_road'),
            'max_length_new_segment': self.config_file.get('max_length_new_segment'),
            'target_commodity': self.config_file.get('target_commodity'),
            'filter_edges_above_warm_start': self.filter_edges_above_warm_start,
            'filter_start_options_above_warm_start': self.filter_start_options_above_warm_start,
            'filter_unreachable_edges': self.filter_unreachable_edges,
        }
        logger.info('MIP config | location %s | %s', location, config_summary)

        edge_summary = self.config_file.get('current_mip_edge_summary', {})
        filters = edge_summary.get('filters', {})
        logger.info(
            'MIP edges | location %s | static %s -> assembled %s -> final %s | '
            'start added %s | destination added %s | final by type %s',
            location,
            edge_summary.get('static_edges'),
            edge_summary.get('assembled_edges_before_filters'),
            edge_summary.get('final_edges'),
            edge_summary.get('start_edges_added'),
            edge_summary.get('sink_edges_added'),
            edge_summary.get('final_edge_counts'),
        )
        logger.info(
            'MIP edge filters | location %s | configuration %s | warm-start edge-cost %s | '
            'warm-start start-option %s | start-end reachability %s | '
            'backup forbidden constraints %s',
            location,
            filters.get('configuration'),
            filters.get('warm_start_edge_cost'),
            filters.get('warm_start_start_option'),
            filters.get('start_end_reachability'),
            getattr(self, 'forbidden_edge_summary', {'removed': 0, 'reasons': {}}),
        )

        cost_propagation = (
            f'Big-M M={self.big_m_cost_propagation:.6f}'
            if self.use_big_m_constraints
            else 'indicator constraints'
        )
        logger.info(
            'MIP optimization setup | location %s | warm start %s | warm-start objective %s | '
            'cost propagation %s | lower bound %s | upper bound %s | connector constraints %s | '
            'pipeline graph constraints %s | MIPGap %s | TimeLimit %s',
            location,
            self.solution_route is not None,
            self.warm_start_objective,
            cost_propagation,
            self.warm_start_objective_lower_bound,
            self.warm_start_objective_upper_bound,
            getattr(self, 'connector_constraints', 0),
            getattr(self, 'pipeline_graph_constraints', 0),
            self.mip_gap,
            self.time_limit,
        )

    def build_model(self):
        """Create variables and constraints using the Gurobi matrix API."""
        now = time.time()
        node_count = len(self.node_names)
        edge_count = len(self.edge_names)
        logger.debug('Create matrix model for %s nodes and %s edges', node_count, edge_count)

        self.costs = self.model.addMVar(node_count, lb=0.0, name='cost')
        self.edge_binaries = self.model.addMVar(edge_count, vtype=GRB.BINARY, name='use')

        forbidden_edge_indices, forbidden_by_reason = self._configuration_forbidden_edge_indices()
        if forbidden_edge_indices:
            logger.warning(
                '%s configuration-forbidden edges reached the OR model although they should '
                'have been removed before model construction: %s',
                len(forbidden_edge_indices), forbidden_by_reason)
        else:
            logger.debug('No edges forbidden by configuration feasibility constraints')
        self.forbidden_edge_summary = {
            'removed': len(forbidden_edge_indices),
            'reasons': forbidden_by_reason,
        }

        propagation_lhs = (
            self.edge_scales * self.costs[self.start_indices] -
            self.costs[self.end_indices])
        propagation_rhs = -self.edge_scales * self.edge_costs
        if self.use_big_m_constraints:
            big_m = self.big_m_cost_propagation
            self.model.addConstr(
                propagation_lhs <= propagation_rhs + big_m * (1 - self.edge_binaries),
                name='cost_propagation_big_m')
            logger.debug('Use Big-M cost propagation constraints with M=%.6f', big_m)
        else:
            self.model.addGenConstrIndicator(
                self.edge_binaries, True, propagation_lhs, GRB.LESS_EQUAL,
                propagation_rhs, name='cost_propagation')
            logger.debug('Use indicator cost propagation constraints because no warm-start objective is available')

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

        connector_incoming, connector_outgoing = self._connector_edge_indices_by_physical_node()
        connector_constraints = 0
        connector_nodes = set(connector_incoming).intersection(connector_outgoing)
        for node in connector_nodes:
            incoming_connector_edges = connector_incoming[node]
            outgoing_connector_edges = connector_outgoing[node]
            if not incoming_connector_edges or not outgoing_connector_edges:
                continue

            connector_indices = np.array(
                incoming_connector_edges + outgoing_connector_edges, dtype=np.int64)
            self.model.addConstr(
                self.edge_binaries[connector_indices].sum() <= 1,
                name='no_consecutive_connector_edges[' + node + ']')
            connector_constraints += 1
        self.connector_constraints = connector_constraints
        logger.debug('Attached %s physical-node connector sequencing constraints for Road/New Pipeline edges',
                     connector_constraints)

        pipeline_graph_constraints = 0
        for graph_name, graph_edge_indices in self._existing_pipeline_edge_indices_by_graph().items():
            if len(graph_edge_indices) <= 1:
                continue
            self.model.addConstr(
                self.edge_binaries[np.array(graph_edge_indices, dtype=np.int64)].sum() <= 1,
                name='use_existing_pipeline_graph_once[' + graph_name + ']')
            pipeline_graph_constraints += 1
        self.pipeline_graph_constraints = pipeline_graph_constraints
        logger.debug('Attached %s existing-pipeline graph usage constraints',
                     pipeline_graph_constraints)

        self.objective_expr = self.costs[target_indices].sum()
        if self.warm_start_objective_lower_bound is not None:
            self.model.addConstr(
                self.objective_expr >= self.warm_start_objective_lower_bound,
                name='warm_start_objective_lower_bound')
            logger.debug('Added objective lower bound from warm-start value: %.6f',
                         self.warm_start_objective_lower_bound)
        if self.warm_start_objective_upper_bound is not None:
            self.model.addConstr(
                self.objective_expr <= self.warm_start_objective_upper_bound,
                name='warm_start_objective_upper_bound')
            logger.debug('Added objective upper bound from warm-start value: %.6f',
                         self.warm_start_objective_upper_bound)
        self.model.setObjective(self.objective_expr, GRB.MINIMIZE)
        self.model.update()
        self.model_build_time = time.time() - now
        logger.debug('Finished matrix model construction in %.2f s', self.model_build_time)

    def optimize(self):
        if self.solution_route is not None:
            warm_start_objective, warm_start_text = self._route_summary(self.solution_route)
            self._append_route_history('warm_start', warm_start_objective, warm_start_text)
            start_values = np.zeros(len(self.edge_names))
            for edge in self.solution_route:
                start_values[self.edge_index[edge]] = 1
            self.edge_binaries.Start = start_values
            self.costs.Start = self._create_cost_start_values(self.solution_route)
            logger.debug('Applied warm-start route with %s active edges', len(self.solution_route))

        # self.model.Params.IntFeasTol = 1e-9
        # self.model.Params.FeasibilityTol = 1e-9
        # self.model.Params.OptimalityTol = 1e-9
        # self.model.Params.Method = 2
        # self.model.Params.Crossover = 0
        # self.model.Params.BarHomogeneous = 1
        # self.model.Params.MIPFocus = 3
        # self.model.Params.Heuristics = 0
        # self.model.Params.Cuts = 3
        # self.model.Params.Presolve = 1
        # self.model.Params.PoolSearchMode = 0
        # self.model.Params.Threads = 1

        if self.mip_gap is not None:
            self.model.Params.MIPGap = self.mip_gap
            logger.debug('Set Gurobi MIPGap to %s', self.mip_gap)
        if self.time_limit is not None:
            self.model.Params.TimeLimit = self.time_limit
            logger.debug('Set Gurobi TimeLimit to %s seconds', self.time_limit)

        logger.debug('Start optimization')
        self.model._best_incumbent = (
            self.warm_start_objective if self.solution_route is not None
            and self.warm_start_objective is not None
            else float('inf')
        )
        self.model._edge_binary_vars = self.edge_binaries.tolist()
        self.model._cost_vars = self.costs.tolist()
        solve_start = time.time()
        self.model.optimize(self._incumbent_callback)
        self.solve_time = time.time() - solve_start
        self.status = self.model.Status
        if self.model.SolCount > 0:
            self.objective_function_value = self.model.ObjVal
            selected = np.flatnonzero(self.edge_binaries.X > 0.5)
            self.chosen_edges = [self.edge_names[index] for index in selected]
            self._log_solution_comparison()
            self._write_incumbent_history_file()
            if self.status == GRB.OPTIMAL:
                logger.info('Optimal objective value: %.6f', self.objective_function_value)
            else:
                logger.info('Best incumbent objective value: %.6f with status %s',
                            self.objective_function_value, self.status)
