import logging
import os
import time
import ast

import pandas as pd
import yaml

from data_processing.helpers_geometry import get_destination
from mixed_integer_program.mip_data_helpers import (
    load_minimal_mip_case,
    load_static_mip_graph,
    prepare_destination_mip_data,
)
from mixed_integer_program.optimization_problem_full_fast import FastOptimizationGurobiModel


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('gurobipy').setLevel(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_configuration_and_technology_data():
    """Load settings and techno-economic assumptions used by every MIP run."""
    with open(os.path.join(PROJECT_ROOT, '_1_algorithm_configuration.yaml')) as yaml_file:
        config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    with open(os.path.join(PROJECT_ROOT, 'data', 'techno_economic_data_transportation.yaml')) as yaml_file:
        transport_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    with open(os.path.join(PROJECT_ROOT, 'data', 'techno_economic_data_conversion.yaml')) as yaml_file:
        conversion_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config_file, conversion_data, transport_data


def create_model(static_graph, start_location_data, start_road_distances,
                 start_new_pipeline_distances, end_location, config_file,
                 conversion_data, transport_data, solve, warm_start_route=None,
                 use_warm_start_as_lower_bound=False,
                 mip_gap=None, time_limit=None,
                 filter_edges_above_warm_start=False):
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
        use_warm_start_as_lower_bound=use_warm_start_as_lower_bound,
        mip_gap=mip_gap,
        time_limit=time_limit,
        filter_edges_above_warm_start=filter_edges_above_warm_start,
        solve=solve)


def create_warm_start_solution(result):
    """Convert a stored heuristic result row into ordered MIP edge keys."""
    result = result[result.columns[0]]
    route = ast.literal_eval(result.loc['taken_routes'])

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
            solution_route.append(
                start + '+' + commodity + '-' + end + '+' + commodity + '-' + transport_mean)
            start = end
        elif len(segment) == 3:
            if commodity == segment[1]:
                continue
            solution_route.append(start + '+' + commodity + '-' + end + '+' + segment[1])
            commodity = segment[1]

    solution_route += [end + '+' + commodity + '-end']
    return solution_route


def load_warm_start_from_result_file(project_path, location):
    """Load one warm-start route from the heuristic result folder."""
    result_file = os.path.join(
        project_path, 'results', 'location_results', str(location) + '_final_solution.csv')
    result = pd.read_csv(result_file, index_col=0)
    return create_warm_start_solution(result)


def run_minimal_case(config_file, conversion_data, transport_data, solve,
                     use_warm_start_as_lower_bound=False,
                     mip_gap=None, time_limit=None,
                     filter_edges_above_warm_start=False):
    """Build or solve the preprocessed diagnostic example."""
    processed_path = os.path.join(config_file['project_folder_path'], 'processed_data') + os.sep
    logger.info('Run preprocessed minimal MIP infrastructure case')
    case = load_minimal_mip_case(processed_path)
    return create_model(
        static_graph=case['static_graph'],
        start_location_data=case['start_location_data'],
        start_road_distances=case['start_road_distances'],
        start_new_pipeline_distances=case['start_new_pipeline_distances'],
        end_location=case['end_location'],
        config_file=config_file,
        conversion_data=conversion_data,
        transport_data=transport_data,
        solve=solve,
        warm_start_route=case['warm_start_route'],
        use_warm_start_as_lower_bound=use_warm_start_as_lower_bound,
        mip_gap=mip_gap,
        time_limit=time_limit,
        filter_edges_above_warm_start=filter_edges_above_warm_start)


def run_real_locations(config_file, conversion_data, transport_data, solve,
                       use_warm_start=False,
                       use_warm_start_as_lower_bound=False,
                       mip_gap=None, time_limit=None,
                       filter_edges_above_warm_start=False):
    """
    Build or solve one MIP per real origin.

    The origin-independent graph and destination infrastructure are loaded
    once. Within the loop only origin-specific connections are added.
    """
    project_path = config_file['project_folder_path']
    if use_warm_start_as_lower_bound and not use_warm_start:
        raise ValueError(
            'use_warm_start_as_lower_bound=True requires use_warm_start=True '
            'so the bound can be calculated from a route.')
    if filter_edges_above_warm_start and not use_warm_start:
        raise ValueError(
            'filter_edges_above_warm_start=True requires use_warm_start=True '
            'so the filter can calculate the route costs.')

    processed_path = os.path.join(project_path, 'processed_data')
    mip_path = os.path.join(processed_path, 'mip_data')
    logger.info('Load preprocessed static MIP graph for real locations')
    static_graph = load_static_mip_graph(mip_path + os.sep)
    options = pd.read_csv(os.path.join(mip_path, 'options.csv'), index_col=0)
    start_locations = pd.read_csv(
        os.path.join(project_path, 'start_destination_combinations.csv'), index_col=0)
    destination = get_destination(config_file)
    end_location = prepare_destination_mip_data(
        options, destination)['destination_infrastructure'].tolist()
    logger.info('Loaded %s static nodes, %s static edges, %s destinations and %s origins',
                len(static_graph['nodes']), len(static_graph['edges']),
                len(end_location), len(start_locations))

    results = []
    for location in start_locations.index:
        logger.info('Build MIP for origin %s', location)
        warm_start_route = None
        if use_warm_start:
            warm_start_route = load_warm_start_from_result_file(project_path, location)

        model = create_model(
            static_graph=static_graph,
            start_location_data=start_locations.loc[location, :],
            start_road_distances=pd.read_csv(
                os.path.join(mip_path, str(location) + '_start_road_distances.csv'),
                index_col=0),
            start_new_pipeline_distances=pd.read_csv(
                os.path.join(mip_path, str(location) + '_start_new_pipeline_distances.csv'),
                index_col=0),
            end_location=end_location,
            config_file=config_file,
            conversion_data=conversion_data,
            transport_data=transport_data,
            solve=solve,
            warm_start_route=warm_start_route,
            use_warm_start_as_lower_bound=use_warm_start_as_lower_bound,
            mip_gap=mip_gap,
            time_limit=time_limit,
            filter_edges_above_warm_start=filter_edges_above_warm_start)
        if solve:
            results.append({
                'location': location,
                'status': model.status,
                'objective': model.objective_function_value,
                'chosen_edges': model.chosen_edges,
            })
            model.model.dispose()
        else:
            results.append(model)
    return results


def run_mip_optimization(use_minimal_example=False, solve=True,
                         use_warm_start=False,
                         use_warm_start_as_lower_bound=False,
                         mip_gap=None, time_limit=None,
                         filter_edges_above_warm_start=False):
    """Central entry point for both the diagnostic example and real MIP runs."""
    config_file, conversion_data, transport_data = load_configuration_and_technology_data()
    start = time.time()
    if use_minimal_example:
        result = run_minimal_case(
            config_file, conversion_data, transport_data, solve,
            use_warm_start_as_lower_bound, mip_gap, time_limit,
            filter_edges_above_warm_start)
    else:
        result = run_real_locations(
            config_file, conversion_data, transport_data, solve,
            use_warm_start, use_warm_start_as_lower_bound, mip_gap, time_limit,
            filter_edges_above_warm_start)
    logger.info('MIP run completed in %.2f s', time.time() - start)
    return result


if __name__ == '__main__':
    USE_MINIMAL_EXAMPLE = False
    SOLVE = True
    USE_WARM_START = False
    USE_WARM_START_AS_LOWER_BOUND = False
    FILTER_EDGES_ABOVE_WARM_START = False
    MIP_GAP = None
    TIME_LIMIT = None

    run_mip_optimization(
        use_minimal_example=USE_MINIMAL_EXAMPLE,
        solve=SOLVE,
        use_warm_start=USE_WARM_START,
        use_warm_start_as_lower_bound=USE_WARM_START_AS_LOWER_BOUND,
        mip_gap=MIP_GAP,
        time_limit=TIME_LIMIT,
        filter_edges_above_warm_start=FILTER_EDGES_ABOVE_WARM_START)
