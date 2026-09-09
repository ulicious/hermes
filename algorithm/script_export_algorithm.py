import gc
import math
import os
import time
import warnings

import pandas as pd
from pandas.errors import PerformanceWarning
from shapely.geometry import Point

from algorithm.methods_export import (apply_export_conversion,
                                      apply_export_k_best,
                                      attach_infrastructure_countries,
                                      create_export_branches_at_start,
                                      export_branch_snapshot,
                                      export_k_best_routes_snapshot,
                                      export_node_results_snapshot,
                                      get_complete_export_infrastructure,
                                      get_start_country,
                                      materialize_export_branches,
                                      prepare_export_commodities,
                                      prepare_export_infrastructure_branches,
                                      process_export_infrastructure_branches,
                                      process_export_out_tolerance_branches,
                                      process_export_zero_distance_branches,
                                      update_export_node_results)
from algorithm.tracking import AlgorithmTracker, branch_count
from data_processing.configuration import load_technology_data
from data_processing.helpers_attach_costs import (
    attach_conversion_costs_and_efficiency_to_infrastructure,
    calculate_conversion_costs_and_efficiencies_for_all_combinations,
)


def _describe_export_infrastructure(complete_infrastructure):
    """Summarize infrastructure units and node types after the country filter."""
    transport_order = ('Pipeline_Gas', 'Pipeline_Liquid', 'Shipping')
    node_types = complete_infrastructure['current_transport_mean'].value_counts()
    network_parts = []
    network_total = 0
    for transport_mean in transport_order[:2]:
        matching = complete_infrastructure[
            complete_infrastructure['current_transport_mean'] == transport_mean]
        network_count = matching['graph'].dropna().astype(str).nunique()
        network_total += network_count
        network_parts.append(transport_mean + ': ' + str(network_count) + ' networks')

    port_count = int(node_types.get('Shipping', 0))
    infrastructure_total = network_total + port_count
    network_parts.append('Shipping: ' + str(port_count) + ' ports')
    type_parts = [transport_mean + ': ' + str(int(node_types.get(transport_mean, 0)))
                  for transport_mean in transport_order]
    other_types = [transport_mean for transport_mean in node_types.index
                   if transport_mean not in transport_order]
    type_parts.extend(transport_mean + ': ' + str(int(node_types[transport_mean]))
                      for transport_mean in other_types)
    return (infrastructure_total, ', '.join(network_parts), ', '.join(type_parts))


def _complete_generated_branches(branches, previous_branches, branch_number):
    if branches.empty:
        return branches, branch_number
    branches = branches.copy()
    branches['branch_index'] = ['S' + str(branch_number + i) for i in range(len(branches))]
    branches.index = branches['branch_index']
    branches.index.name = None
    branch_number += len(branches)
    branches['current_conversion_costs'] = 0
    branches = materialize_export_branches(branches, previous_branches)
    return branches, branch_number


def _write_complete_marker(path_results, location_index):
    marker = os.path.join(path_results, 'export_infrastructure_branches',
                          str(location_index), '_complete')
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, 'w', encoding='utf-8') as handle:
        handle.write('complete')


def _target_coverage(node_results, infrastructure_nodes, target_commodities):
    covered = {(str(node), str(commodity)) for node, commodity in node_results}
    missing = {
        commodity: [str(node) for node in infrastructure_nodes
                    if (str(node), str(commodity)) not in covered]
        for commodity in target_commodities
    }
    return {commodity: nodes for commodity, nodes in missing.items() if nodes}


def _prepare_location(location_index, location_data, data, config_file):
    location_data = location_data.copy().loc[[location_index]]
    location_data.index = ['Start']
    start_country = get_start_country(location_data)
    print(str(location_index) + ': Start country: ' + str(start_country))
    data = data.copy()
    data['start'] = {
        'location': Point(location_data.at['Start', 'longitude'],
                          location_data.at['Start', 'latitude']),
    }

    complete_infrastructure = get_complete_export_infrastructure(data)
    complete_infrastructure = attach_infrastructure_countries(
        complete_infrastructure, data.get('world'), target_country=start_country)

    technology_conversion, _ = load_technology_data(config_file)
    # These shared preparation methods emit a CRS warning for the one-row start
    # table and many pandas fragmentation warnings while adding conversion
    # columns. Neither warning affects the resulting values; keep this runner's
    # multiprocessing output readable without changing the shared methods.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='CRS mismatch.*', category=UserWarning)
        warnings.simplefilter('ignore', PerformanceWarning)
        start_conversions = attach_conversion_costs_and_efficiency_to_infrastructure(
            location_data, config_file, technology_conversion, with_tqdm=False)
        calculate_conversion_costs_and_efficiencies_for_all_combinations(
            config_file, start_conversions, technology_conversion)
    data['conversion_costs_and_efficiencies'] = pd.concat([
        data['conversion_costs_and_efficiencies'], start_conversions])

    commodities, commodity_names = prepare_export_commodities(config_file, location_data, data)
    data['commodities']['all_commodities'] = commodity_names
    data['commodities']['target_commodities'] = list(dict.fromkeys(config_file['target_commodity']))
    for commodity in commodities:
        data['commodities']['commodity_objects'][commodity.get_name()] = commodity

    branches, branch_number = create_export_branches_at_start(data)
    return data, complete_infrastructure, branches, branch_number


def run_export_algorithm(args):
    """Enumerate the k cheapest domestic routes per node and commodity."""
    location_index, location_data, common_data, config_file, configuration = args
    number_k_best_routes = int(config_file.get('number_k_best_routes', 10))
    export_intermediate_branches = bool(config_file.get(
        'export_intermediate_branches', False))
    if number_k_best_routes < 1:
        raise ValueError('number_k_best_routes must be at least 1.')
    print(str(location_index) + ': Start Processing export infrastructure')
    started = time.time()
    tracker = AlgorithmTracker(location_index, configuration['path_results'],
                               tracking_folder='export_infrastructure_tracking')
    preparation_started = time.time()
    data, complete_infrastructure, branches, branch_number = \
        _prepare_location(location_index, location_data, common_data, config_file)
    preparation_time = time.time() - preparation_started
    tracker.event(phase='initialization', method='_prepare_location', event='runtime',
                  after=branch_count(branches), runtime_s=preparation_time,
                  details={'country_infrastructure_nodes': len(complete_infrastructure)})
    infrastructure_total, infrastructure_types, node_types = \
        _describe_export_infrastructure(complete_infrastructure)
    print(str(location_index) + ': Preparation [s]: ' + str(round(preparation_time, 2))
          + ' | Country infrastructure nodes: ' + str(len(complete_infrastructure)))
    print(str(location_index) + ': Country infrastructures: ' + str(infrastructure_total)
          + ' | ' + infrastructure_types)
    print(str(location_index) + ': Infrastructure node types: ' + node_types)

    finite_start_costs = pd.to_numeric(
        branches['current_total_costs'], errors='coerce').map(math.isfinite)
    branches = branches.loc[finite_start_costs].copy()
    if branches.empty:
        export_node_results_snapshot(
            {}, configuration['path_results'], location_index, 0,
            stage='final_nodes', k_best_routes={})
        _write_complete_marker(configuration['path_results'], location_index)
        tracker.event(phase='location', method='run_export_algorithm', event='stop_no_potential',
                      after=0, runtime_s=time.time() - started)
        return None

    node_results = {}
    k_best_routes = {}
    invalid_branches = set()
    branches, _, k_best_routes, invalid_branches = apply_export_k_best(
        branches, k_best_routes, invalid_branches, number_k_best_routes)
    iteration = 0
    while not branches.empty:
        iteration_started = time.time()
        iteration_input_count = branch_count(branches)
        tracker.event(iteration=iteration, phase='iteration', method='run_export_algorithm',
                      event='start', before=iteration_input_count, runtime_s=0.0,
                      details={'branch_number': branch_number})
        if iteration > 0:
            conversion_locations = data['conversion_costs_and_efficiencies']
            possible_nodes = conversion_locations[conversion_locations['conversion_possible']].index
            convertible = branches[branches['current_node'].isin(possible_nodes)]
            unchanged = branches[~branches['current_node'].isin(possible_nodes)]
            with tracker.time_block(iteration=iteration, phase='conversion',
                                    method='apply_export_conversion', event='runtime'):
                converted, branch_number = apply_export_conversion(
                    convertible, data, branch_number)
                branches = pd.concat([converted, unchanged], ignore_index=False)
            with tracker.time_block(iteration=iteration, phase='pruning',
                                    method='apply_export_k_best_after_conversion',
                                    event='runtime'):
                branches, conversion_pruned, k_best_routes, invalid_branches = \
                    apply_export_k_best(
                        branches, k_best_routes, invalid_branches,
                        number_k_best_routes)
            update_export_node_results(
                node_results, branches, data['commodities']['target_commodities'],
                complete_infrastructure.index)
            if export_intermediate_branches:
                with tracker.time_block(iteration=iteration, phase='export',
                                        method='export_conversion_branches', event='runtime'):
                    export_branch_snapshot(
                        branches, configuration['path_results'], location_index, iteration,
                        'conversion_branches')
        if branches.empty:
            tracker.event(iteration=iteration, phase='iteration', method='run_export_algorithm',
                          event='stop_no_active_branches', before=iteration_input_count, after=0,
                          runtime_s=time.time() - iteration_started,
                          details={'branch_number': branch_number})
            break

        with tracker.time_block(iteration=iteration, phase='routing',
                                method='prepare_routing_inputs', event='runtime'):
            arrived_by_approach = branches['current_transport_mean'].isin(
                ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])
            infrastructure_inputs = prepare_export_infrastructure_branches(
                branches[arrived_by_approach], complete_infrastructure)
            approach_inputs = branches[~arrived_by_approach]
        with tracker.time_block(iteration=iteration, phase='routing',
                                method='process_export_infrastructure_branches', event='runtime'):
            infrastructure_options = process_export_infrastructure_branches(
                data, infrastructure_inputs, complete_infrastructure, configuration,
                number_k_best_routes)
        with tracker.time_block(iteration=iteration, phase='routing',
                                method='process_export_out_tolerance_branches', event='runtime'):
            approach_options = process_export_out_tolerance_branches(
                complete_infrastructure, approach_inputs, configuration,
                number_k_best_routes)
        with tracker.time_block(iteration=iteration, phase='routing',
                                method='process_export_zero_distance_branches', event='runtime'):
            zero_options = process_export_zero_distance_branches(
                data, branches, complete_infrastructure, number_k_best_routes)

        with tracker.time_block(iteration=iteration, phase='routing',
                                method='combine_generated_options', event='runtime'):
            candidate_frames = [frame for frame in
                                (infrastructure_options, approach_options, zero_options)
                                if not frame.empty]
            candidates = (pd.concat(candidate_frames, ignore_index=True)
                          if candidate_frames else pd.DataFrame())
        with tracker.time_block(iteration=iteration, phase='routing_finalize',
                                method='materialize_generated_branches', event='runtime'):
            branches, branch_number = _complete_generated_branches(
                candidates, branches, branch_number)
        with tracker.time_block(iteration=iteration, phase='pruning',
                                method='apply_export_k_best_after_transport',
                                event='runtime'):
            branches, transport_pruned, k_best_routes, invalid_branches = \
                apply_export_k_best(
                    branches, k_best_routes, invalid_branches,
                    number_k_best_routes)
        update_export_node_results(
            node_results, branches, data['commodities']['target_commodities'],
            complete_infrastructure.index)
        if export_intermediate_branches:
            with tracker.time_block(iteration=iteration, phase='export',
                                    method='export_transport_branches', event='runtime'):
                export_branch_snapshot(
                    branches, configuration['path_results'], location_index, iteration,
                    'transport_branches')
            with tracker.time_block(iteration=iteration, phase='export',
                                    method='export_node_results', event='runtime'):
                export_node_results_snapshot(
                    node_results, configuration['path_results'], location_index, iteration)
        iteration_runtime = time.time() - iteration_started
        tracker.event(iteration=iteration, phase='iteration', method='run_export_algorithm',
                      event='runtime', before=iteration_input_count,
                      after=branch_count(branches), runtime_s=iteration_runtime,
                      details={'branch_number': branch_number,
                               'generated_candidates': len(candidates),
                               'k_best_pruned_after_conversion': (
                                   conversion_pruned if iteration > 0 else 0),
                               'k_best_pruned_after_transport': transport_pruned,
                               'invalid_branches': len(invalid_branches),
                               'number_k_best_routes': number_k_best_routes})
        print(str(location_index) + '-' + str(iteration)
              + ': Active branches: ' + str(branch_count(branches))
              + ' | Created: ' + str(branch_number)
              + ' | Iteration [s]: ' + str(round(iteration_runtime, 2)))
        iteration += 1

    with tracker.time_block(iteration=iteration, phase='export',
                            method='export_final_node_results', event='runtime'):
        export_node_results_snapshot(
            node_results, configuration['path_results'], location_index, iteration,
            stage='final_nodes', k_best_routes=k_best_routes,
            invalid_branches=invalid_branches)
    with tracker.time_block(iteration=iteration, phase='export',
                            method='export_final_routes', event='runtime'):
        export_k_best_routes_snapshot(
            node_results, k_best_routes, invalid_branches,
            configuration['path_results'], location_index, iteration)
    missing_targets = _target_coverage(
        node_results, complete_infrastructure.index,
        data['commodities']['target_commodities'])
    expected_states = (len(complete_infrastructure)
                       * len(data['commodities']['target_commodities']))
    missing_states = sum(len(nodes) for nodes in missing_targets.values())
    print(str(location_index) + ': Target result coverage: '
          + str(expected_states - missing_states) + '/' + str(expected_states)
          + ' node-commodity combinations')
    if missing_targets:
        print(str(location_index) + ': Missing target combinations '
              + '(technically unreachable unless routing coverage is incomplete): '
              + ', '.join(commodity + '=' + str(len(nodes))
                          for commodity, nodes in missing_targets.items()))
    _write_complete_marker(configuration['path_results'], location_index)
    tracker.event(phase='location', method='run_export_algorithm', event='end',
                  runtime_s=time.time() - started,
                  details={'total_branches_created': branch_number,
                           'target_states_expected': expected_states,
                           'target_states_covered': expected_states - missing_states,
                           'target_states_missing': missing_states,
                           'missing_target_states_by_commodity': {
                               commodity: len(nodes)
                               for commodity, nodes in missing_targets.items()
                           }})
    print(str(location_index) + ': finished export enumeration in '
          + str(math.ceil((time.time() - started) / 60)) + ' minutes.')
    gc.collect()
    return None
