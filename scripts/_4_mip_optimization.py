import logging
import os
import time
import ast
import math

import pandas as pd

from data_processing.helpers_geometry import get_destination
from data_processing.configuration import load_algorithm_configuration, load_technology_data
from mixed_integer_program.mip_data_helpers import (
    load_minimal_mip_case,
    load_static_mip_graph,
    prepare_destination_mip_data,
)
from mixed_integer_program.optimization_problem_full_fast import FastOptimizationGurobiModel


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('gurobipy').setLevel(logging.WARNING)

def initialize_mip_incumbent_history_file(config_file):
    """Create one shared incumbent history file for the current optimization run."""
    path_results = os.path.join(config_file['project_folder_path'], 'results')
    os.makedirs(path_results, exist_ok=True)
    file_path = os.path.join(path_results, 'mip_incumbents_all_locations_running.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('MIP incumbent history for all locations\n')
    config_file['mip_incumbent_history_file'] = file_path
    config_file['mip_optimization_run_statistics'] = []
    return file_path


def collect_mip_run_statistics(config_file, location, model):
    """Store compact per-location statistics for the shared optimization info file."""
    edge_summary = config_file.get('current_mip_edge_summary', {})
    edges_before = edge_summary.get('assembled_edges_before_filters')
    edges_after = edge_summary.get('final_edges')
    edge_difference = None
    if edges_before is not None and edges_after is not None:
        edge_difference = edges_before - edges_after

    config_file.setdefault('mip_optimization_run_statistics', []).append({
        'location': location,
        'solve_time': getattr(model, 'solve_time', None),
        'edges_before_filter': edges_before,
        'edges_after_filter': edges_after,
        'edge_filter_difference': edge_difference,
    })


def _format_filter_summary(filter_summary):
    if not filter_summary:
        return 'none'

    formatted_filters = []
    for name, details in filter_summary.items():
        if not isinstance(details, dict):
            formatted_filters.append(f'{name}: {details}')
            continue

        removed = details.get('removed', details.get('removed_edges', 0))
        enabled = details.get('enabled')
        reasons = details.get('reasons', details.get('removed_by_reason', {}))
        parts = [f'removed {removed}']
        if enabled is not None:
            parts.append(f'enabled {enabled}')
        if reasons:
            parts.append(f'reasons {reasons}')
        if 'removed_nodes' in details:
            parts.append(f"removed_nodes {details['removed_nodes']}")
        if 'removed_start_options' in details:
            parts.append(f"removed_start_options {details['removed_start_options']}")
        formatted_filters.append(f"{name}: " + ', '.join(parts))

    return ' | '.join(formatted_filters)


def append_mip_location_build_summary(config_file, location, model, solve):
    """Append model-size diagnostics for every location, also for build-only runs."""
    file_path = config_file.get('mip_incumbent_history_file')
    if not file_path:
        return

    edge_summary = config_file.get('current_mip_edge_summary', {})
    status = getattr(model, 'status', None) if solve else 'model_built'
    lines = [
        '',
        '-' * 100,
        f'MIP model summary for location {location}',
        f'Status: {status}',
        'Edges | static {static_edges} | before pruning {before_edges} | after pruning {after_edges}'.format(
            static_edges=edge_summary.get('static_edges'),
            before_edges=edge_summary.get('assembled_edges_before_filters'),
            after_edges=edge_summary.get('final_edges')),
        'Nodes | static {static_nodes} | final {final_nodes}'.format(
            static_nodes=edge_summary.get('static_nodes'),
            final_nodes=edge_summary.get('final_nodes')),
        'Added edges | start {start_edges} | sink {sink_edges}'.format(
            start_edges=edge_summary.get('start_edges_added'),
            sink_edges=edge_summary.get('sink_edges_added')),
        'Edge counts before pruning: {counts}'.format(
            counts=edge_summary.get('assembled_edge_counts_before_filters')),
        'Edge counts after pruning: {counts}'.format(
            counts=edge_summary.get('final_edge_counts')),
        'Filters: ' + _format_filter_summary(edge_summary.get('filters', {})),
        ''
    ]

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('\n'.join(lines))


def finalize_mip_incumbent_history_file(config_file, solve=True):
    """Rename the shared incumbent file after the run based on whether any location improved."""
    file_path = config_file.get('mip_incumbent_history_file')
    if not file_path or not os.path.exists(file_path):
        return None

    run_statistics = config_file.get('mip_optimization_run_statistics', [])
    solve_times = [
        entry['solve_time'] for entry in run_statistics
        if entry.get('solve_time') is not None
    ]
    edge_differences = [
        entry['edge_filter_difference'] for entry in run_statistics
        if entry.get('edge_filter_difference') is not None
    ]
    edges_before_filters = [
        entry['edges_before_filter'] for entry in run_statistics
        if entry.get('edges_before_filter') is not None
    ]
    if solve_times or edge_differences or edges_before_filters:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write('\n' + '=' * 100 + '\n')
            file.write('Run summary\n')
            if solve_times:
                file.write(
                    'Solving time | shortest %.2f s | longest %.2f s | average %.2f s\n'
                    % (min(solve_times), max(solve_times), sum(solve_times) / len(solve_times)))
            if edge_differences:
                min_entry = min(
                    (entry for entry in run_statistics if entry.get('edge_filter_difference') is not None),
                    key=lambda entry: entry['edge_filter_difference'])
                max_entry = max(
                    (entry for entry in run_statistics if entry.get('edge_filter_difference') is not None),
                    key=lambda entry: entry['edge_filter_difference'])
                file.write(
                    'Edge filter difference | lowest %s at location %s | highest %s at location %s\n'
                    % (min_entry['edge_filter_difference'], min_entry['location'],
                       max_entry['edge_filter_difference'], max_entry['location']))
            if edges_before_filters:
                min_entry = min(
                    (entry for entry in run_statistics if entry.get('edges_before_filter') is not None),
                    key=lambda entry: entry['edges_before_filter'])
                max_entry = max(
                    (entry for entry in run_statistics if entry.get('edges_before_filter') is not None),
                    key=lambda entry: entry['edges_before_filter'])
                file.write(
                    'Model size before pruning | smallest %s edges at location %s | largest %s edges at location %s\n'
                    % (min_entry['edges_before_filter'], min_entry['location'],
                       max_entry['edges_before_filter'], max_entry['location']))

    with open(file_path, 'r', encoding='utf-8') as file:
        has_improvement = 'Improved objective: True' in file.read()

    path_results = os.path.dirname(file_path)
    if solve:
        improvement_label = 'improved' if has_improvement else 'no_improvement'
    else:
        improvement_label = 'built_only'
    final_path = os.path.join(
        path_results, f'mip_incumbents_all_locations_{improvement_label}.txt')
    os.replace(file_path, final_path)
    config_file['mip_incumbent_history_file'] = final_path
    return final_path


def configure_mip_logging(show_mip_logs=True):
    """Keep optional MIP progress logs quiet while preserving incumbent updates."""
    incumbent_logger = logging.getLogger(
        'mixed_integer_program.optimization_problem_full_fast.incumbent')
    regular_loggers = [
        logging.getLogger('mixed_integer_program'),
        logging.getLogger('data_processing'),
        logging.getLogger('algorithm'),
        logger,
    ]
    if show_mip_logs:
        for regular_logger in regular_loggers:
            regular_logger.setLevel(logging.INFO)
    else:
        for regular_logger in regular_loggers:
            regular_logger.setLevel(logging.CRITICAL + 1)

    incumbent_logger.setLevel(logging.INFO)
    incumbent_logger.propagate = True


def load_configuration_and_technology_data():
    """Load settings and techno-economic assumptions used by every MIP run."""
    config_file = load_algorithm_configuration()
    conversion_data, transport_data = load_technology_data(config_file)
    return config_file, conversion_data, transport_data


def create_model(static_graph, start_location_data, start_road_distances,
                 start_new_pipeline_distances, end_location, config_file,
                 conversion_data, transport_data, solve, warm_start_route=None,
                 use_warm_start_as_lower_bound=False,
                 warm_start_bound_route=None,
                 mip_gap=None, time_limit=None,
                 filter_edges_above_warm_start=False,
                 filter_start_options_above_warm_start=False,
                 filter_unreachable_edges=False,
                 destination_tolerance_nodes=None):
    """Build one Gurobi MIP instance and optionally start its optimization."""
    return FastOptimizationGurobiModel(
        static_graph=static_graph,
        start_location_data=start_location_data,
        start_road_distances=start_road_distances,
        start_new_pipeline_distances=start_new_pipeline_distances,
        end_location=end_location,
        config_file=config_file,
        techno_economic_data_conversion=conversion_data,
        techno_economic_data_transport=transport_data,
        warm_start_route=warm_start_route,
        warm_start_bound_route=warm_start_bound_route,
        use_warm_start_as_lower_bound=use_warm_start_as_lower_bound,
        mip_gap=mip_gap,
        time_limit=time_limit,
        filter_edges_above_warm_start=filter_edges_above_warm_start,
        filter_start_options_above_warm_start=filter_start_options_above_warm_start,
        filter_unreachable_edges=filter_unreachable_edges,
        destination_tolerance_nodes=destination_tolerance_nodes,
        solve=solve)


def create_warm_start_solution(result):
    """Convert a stored heuristic result row into ordered MIP edge keys."""
    result = result[result.columns[0]]
    if 'taken_routes' not in result.index or pd.isna(result.loc['taken_routes']):
        logger.warning('Warm-start result has no taken_routes entry')
        return None
    taken_routes_text = str(result.loc['taken_routes'])
    try:
        route = ast.literal_eval(taken_routes_text)
    except (SyntaxError, ValueError):
        route = ast.literal_eval(
            taken_routes_text.replace('nan', 'None').replace('inf', '1e309'))
    if not route:
        logger.warning('Warm-start result contains an empty route')
        return None

    commodity = None
    start = None
    end = None
    solution_route = []
    for n, segment in enumerate(route):
        if n == 0:
            commodity = segment[0]
            start = 'start'
            continue

        if len(segment) == 5:
            start = segment[0]
            end = segment[3]
            if start == 'Start':
                start = 'start'
            transport_mean = segment[1]
            if transport_mean is None or (isinstance(transport_mean, float) and math.isnan(transport_mean)):
                distance_type = result.get('distance_type')
                if distance_type == 'new':
                    if commodity in {'Hydrogen_Gas', 'Methane_Gas'}:
                        transport_mean = 'New_Pipeline_Gas'
                    else:
                        transport_mean = 'New_Pipeline_Liquid'
                else:
                    transport_mean = result.get('current_transport_mean')
                    if pd.isna(transport_mean):
                        transport_mean = 'Road'
                logger.debug('Warm-start route segment has no transport mean; inferred %s for %s -> %s',
                             transport_mean, start, end)
            if start == end:
                logger.debug('Skip warm-start transport self-loop %s via %s', start, transport_mean)
                continue
            solution_route.append(
                start + '+' + commodity + '-' + end + '+' + commodity + '-' + transport_mean)
            start = end
        elif len(segment) == 3:
            if start is None or end is None:
                logger.warning('Warm-start route contains a conversion before any transport segment: %s', segment)
                return None
            if commodity == segment[1]:
                continue
            solution_route.append(start + '+' + commodity + '-' + end + '+' + segment[1])
            commodity = segment[1]

    if commodity is not None and not solution_route:
        logger.debug('Warm-start route for commodity %s is already complete at the start location',
                     commodity)
        return ['start+' + commodity + '-end']

    if end is None or commodity is None:
        logger.warning('Warm-start route has no usable transport path')
        return None

    solution_route += [end + '+' + commodity + '-end']
    return solution_route


def load_warm_start_from_result_file(project_path, location):
    """Load one warm-start route from the heuristic result folder."""
    result_file = os.path.join(
        project_path, 'results', 'location_results', str(location) + '_final_solution.csv')
    if not os.path.exists(result_file):
        logger.warning('Warm-start result file missing for location %s: %s', location, result_file)
        return None
    result = pd.read_csv(result_file, index_col=0)
    return create_warm_start_solution(result)


def load_warm_start_total_costs(project_path, location):
    """Read the heuristic total supply costs used to order MIP locations."""
    result_file = os.path.join(
        project_path, 'results', 'location_results', str(location) + '_final_solution.csv')
    if not os.path.exists(result_file):
        return math.inf

    try:
        result = pd.read_csv(result_file, index_col=0)
        value = result.iloc[:, 0].get('current_total_costs')
        return float(value) if value is not None and not pd.isna(value) else math.inf
    except (OSError, ValueError, IndexError, KeyError):
        logger.warning('Could not read warm-start total costs for location %s from %s',
                       location, result_file)
        return math.inf


def read_distance_file_or_empty(path):
    """Load optional origin-specific distance data without failing on missing layers."""
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    logger.warning('Distance file missing; continue with no edges from %s', path)
    return pd.DataFrame(columns=['pointA', 'pointB', 'distance'])


def run_minimal_case(config_file, conversion_data, transport_data, solve,
                     use_warm_start_as_lower_bound=False,
                     use_warm_start=False,
                     mip_gap=None, time_limit=None,
                     filter_edges_above_warm_start=False,
                     filter_start_options_above_warm_start=False,
                     filter_unreachable_edges=False):
    """Build or solve the preprocessed diagnostic example."""
    processed_path = os.path.join(config_file['project_folder_path'], 'processed_data') + os.sep
    logger.debug('Run preprocessed minimal MIP infrastructure case')
    config_file['current_mip_location'] = 'minimal'
    case = load_minimal_mip_case(processed_path)
    known_route = case['warm_start_route']
    warm_start_route = known_route if use_warm_start else None
    warm_start_bound_route = known_route if (
        use_warm_start
        or use_warm_start_as_lower_bound
        or filter_edges_above_warm_start
        or filter_start_options_above_warm_start
    ) else None
    model = create_model(
        static_graph=case['static_graph'],
        start_location_data=case['start_location_data'],
        start_road_distances=case['start_road_distances'],
        start_new_pipeline_distances=case['start_new_pipeline_distances'],
        end_location=case['end_location'],
        config_file=config_file,
        conversion_data=conversion_data,
        transport_data=transport_data,
        solve=solve,
        warm_start_route=warm_start_route,
        use_warm_start_as_lower_bound=use_warm_start_as_lower_bound,
        warm_start_bound_route=warm_start_bound_route,
        mip_gap=mip_gap,
        time_limit=time_limit,
        filter_edges_above_warm_start=filter_edges_above_warm_start,
        filter_start_options_above_warm_start=filter_start_options_above_warm_start,
        filter_unreachable_edges=filter_unreachable_edges,
        destination_tolerance_nodes=case.get('end_location'))
    collect_mip_run_statistics(config_file, 'minimal', model)
    append_mip_location_build_summary(config_file, 'minimal', model, solve)
    return model


def run_real_locations(config_file, conversion_data, transport_data, solve,
                       use_warm_start=False,
                       use_warm_start_as_lower_bound=False,
                       mip_gap=None, time_limit=None,
                       filter_edges_above_warm_start=False,
                       filter_start_options_above_warm_start=False,
                       filter_unreachable_edges=False):
    """
    Build or solve one MIP per real origin.

    The origin-independent graph and destination infrastructure are loaded
    once. Within the loop only origin-specific connections are added.
    """
    project_path = config_file['project_folder_path']
    needs_warm_start_route = (
        use_warm_start
        or use_warm_start_as_lower_bound
        or filter_edges_above_warm_start
        or filter_start_options_above_warm_start
    )

    processed_path = os.path.join(project_path, 'processed_data')
    mip_path = os.path.join(processed_path, 'mip_data')
    logger.debug('Load preprocessed static MIP graph for real locations')
    static_graph = load_static_mip_graph(mip_path + os.sep)
    options = pd.read_csv(os.path.join(mip_path, 'options.csv'), index_col=0)
    start_locations = pd.read_csv(
        os.path.join(project_path, 'start_destination_combinations.csv'), index_col=0)
    destination = get_destination(config_file)
    end_location = prepare_destination_mip_data(
        options, destination,
        destination_tolerance=config_file['to_final_destination_tolerance']
    )['destination_infrastructure'].tolist()
    logger.debug('Loaded %s static nodes, %s static edges, %s destinations and %s origins',
                 len(static_graph['nodes']), len(static_graph['edges']),
                 len(end_location), len(start_locations))

    if needs_warm_start_route:
        location_order = sorted(
            start_locations.index,
            key=lambda location: (load_warm_start_total_costs(project_path, location), location)
        )
        logger.info('Sort MIP locations by heuristic total supply costs; cheapest start solution first')
    else:
        location_order = start_locations.index

    results = []
    for location in location_order:

        logger.debug('Build MIP for origin %s', location)
        config_file['current_mip_location'] = location
        known_route = None
        if needs_warm_start_route:
            known_route = load_warm_start_from_result_file(project_path, location)
        warm_start_route = known_route if use_warm_start else None
        warm_start_bound_route = known_route if (
            use_warm_start
            or use_warm_start_as_lower_bound
            or filter_edges_above_warm_start
            or filter_start_options_above_warm_start
        ) else None

        model = create_model(
            static_graph=static_graph,
            start_location_data=start_locations.loc[location, :],
            start_road_distances=read_distance_file_or_empty(
                os.path.join(mip_path, str(location) + '_start_road_distances.csv')),
            start_new_pipeline_distances=read_distance_file_or_empty(
                os.path.join(mip_path, str(location) + '_start_new_pipeline_distances.csv')),
            end_location=end_location,
            config_file=config_file,
            conversion_data=conversion_data,
            transport_data=transport_data,
            solve=solve,
            warm_start_route=warm_start_route,
            use_warm_start_as_lower_bound=use_warm_start_as_lower_bound,
            warm_start_bound_route=warm_start_bound_route,
            mip_gap=mip_gap,
            time_limit=time_limit,
            filter_edges_above_warm_start=filter_edges_above_warm_start,
            filter_start_options_above_warm_start=filter_start_options_above_warm_start,
            filter_unreachable_edges=filter_unreachable_edges,
            destination_tolerance_nodes=end_location)
        collect_mip_run_statistics(config_file, location, model)
        append_mip_location_build_summary(config_file, location, model, solve)
        if solve:
            results.append({
                'location': location,
                'status': model.status,
                'objective': model.objective_function_value,
                'chosen_edges': model.chosen_edges,
            })
            model.model.dispose()
        else:
            results.append({
                'location': location,
                'status': 'model_built',
                'objective': None,
                'chosen_edges': None,
            })
            model.model.dispose()
    return results


def run_mip_optimization(use_minimal_example=False, solve=True,
                         use_warm_start=False,
                         use_warm_start_as_lower_bound=False,
                         mip_gap=None, time_limit=None,
                         filter_edges_above_warm_start=False,
                         filter_start_options_above_warm_start=False,
                         filter_unreachable_edges=False,
                         show_mip_logs=True):
    """Central entry point for both the diagnostic example and real MIP runs."""
    configure_mip_logging(show_mip_logs)
    config_file, conversion_data, transport_data = load_configuration_and_technology_data()
    initialize_mip_incumbent_history_file(config_file)
    start = time.time()
    if use_minimal_example:
        result = run_minimal_case(
            config_file, conversion_data, transport_data, solve,
            use_warm_start_as_lower_bound, use_warm_start,
            mip_gap, time_limit,
            filter_edges_above_warm_start,
            filter_start_options_above_warm_start,
            filter_unreachable_edges)
    else:
        result = run_real_locations(
            config_file, conversion_data, transport_data, solve,
            use_warm_start, use_warm_start_as_lower_bound,
            mip_gap, time_limit,
            filter_edges_above_warm_start,
            filter_start_options_above_warm_start,
            filter_unreachable_edges)
    logger.info('MIP run completed in %.2f s', time.time() - start)
    final_history_file = finalize_mip_incumbent_history_file(config_file, solve=solve)
    if final_history_file is not None:
        logger.info('Wrote shared MIP incumbent history to %s', final_history_file)
    return result


if __name__ == '__main__':
    USE_MINIMAL_EXAMPLE = False
    SOLVE = True
    USE_WARM_START = True
    USE_WARM_START_AS_LOWER_BOUND = False
    FILTER_EDGES_ABOVE_WARM_START = True
    FILTER_START_OPTIONS_ABOVE_WARM_START = True
    FILTER_UNREACHABLE_EDGES = True
    MIP_GAP = None
    TIME_LIMIT = 3600
    SHOW_MIP_LOGS = True

    run_mip_optimization(
        use_minimal_example=USE_MINIMAL_EXAMPLE,
        solve=SOLVE,
        use_warm_start=USE_WARM_START,
        use_warm_start_as_lower_bound=USE_WARM_START_AS_LOWER_BOUND,
        mip_gap=MIP_GAP,
        time_limit=TIME_LIMIT,
        filter_edges_above_warm_start=FILTER_EDGES_ABOVE_WARM_START,
        filter_start_options_above_warm_start=FILTER_START_OPTIONS_ABOVE_WARM_START,
        filter_unreachable_edges=FILTER_UNREACHABLE_EDGES,
        show_mip_logs=SHOW_MIP_LOGS)
