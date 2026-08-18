import gc
import math
import os
import time
import warnings

import pandas as pd
from pandas.errors import PerformanceWarning
from shapely.geometry import Point

from algorithm.methods_algorithm import postprocessing_branches
from algorithm.methods_export import (apply_export_conversion,
                                      apply_complete_commodity_benchmark,
                                      apply_export_local_benchmark,
                                      attach_infrastructure_countries,
                                      create_export_branches_at_start,
                                      export_branch_snapshot,
                                      export_final_local_benchmark_branches,
                                      export_local_benchmark_snapshot,
                                      get_complete_export_infrastructure,
                                      get_start_country,
                                      prefilter_export_branch_candidates,
                                      prepare_export_commodities,
                                      prepare_export_infrastructure_branches,
                                      preselect_export_infrastructure_branches,
                                      process_export_infrastructure_branches,
                                      process_export_out_tolerance_branches,
                                      process_export_zero_distance_branches)
from algorithm.methods_export import remove_superseded_branch_descendants
from algorithm.tracking import AlgorithmTracker, branch_count
from data_processing.configuration import load_technology_data
from data_processing.helpers_attach_costs import (
    attach_conversion_costs_and_efficiency_to_infrastructure,
    calculate_conversion_costs_and_efficiencies_for_all_combinations,
)


def _attach_transport_properties(branches, available_transport_means):
    branches = branches.copy()
    for transport_mean in available_transport_means:
        branches[transport_mean + '_applicable'] = branches['current_commodity_object'].apply(
            lambda commodity: commodity.get_transportation_options_specific_mean_of_transport(transport_mean))
    return branches


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
    branches = postprocessing_branches(branches, previous_branches)
    return branches, branch_number


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
    for commodity in commodities:
        data['commodities']['commodity_objects'][commodity.get_name()] = commodity

    branches, branch_number = create_export_branches_at_start(data)
    return data, complete_infrastructure, branches, branch_number


def run_export_algorithm(args):
    """Enumerate routes until they first reach infrastructure outside the start country."""
    location_index, location_data, common_data, config_file, configuration = args
    print(str(location_index) + ': Start Processing export infrastructure')
    started = time.time()
    tracker = AlgorithmTracker(location_index, configuration['path_results'])
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
        export_branch_snapshot(branches, configuration['path_results'], location_index, 0, 'no_potential')
        return None

    local_benchmarks = {}
    superseded_branches = set()
    export_branch_snapshot(branches, configuration['path_results'], location_index, 0, 'initial')
    iteration = 0
    while not branches.empty:
        iteration_started = time.time()
        iteration_input_count = branch_count(branches)
        filter_counts = {
            'local_benchmark_pre_routing': 0,
            'descendants_pre_routing': 0,
            'complete_commodity_pre_routing': 0,
            'global_descendants_pre_routing': 0,
            'dominated_infrastructure_entries': 0,
            'minimal_distance_parents': 0,
            'early_candidate_rejections': 0,
            'local_benchmark_post_routing': 0,
            'descendants_post_routing': 0,
            'complete_commodity_post_routing': 0,
            'global_descendants_post_routing': 0,
        }
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
                    convertible, data, branch_number, local_benchmarks,
                    complete_infrastructure.index)
                branches = pd.concat([converted, unchanged], ignore_index=False)
            with tracker.time_block(iteration=iteration, phase='benchmark',
                                    method='apply_export_local_benchmark_pre_routing', event='runtime'):
                branches, pruned_count, local_benchmarks, newly_superseded = apply_export_local_benchmark(
                    branches, local_benchmarks)
            filter_counts['local_benchmark_pre_routing'] = pruned_count
            superseded_branches.update(newly_superseded)
            with tracker.time_block(iteration=iteration, phase='benchmark',
                                    method='remove_superseded_descendants_pre_routing', event='runtime'):
                branches, descendant_count, local_benchmarks, superseded_branches = \
                    remove_superseded_branch_descendants(
                        branches, local_benchmarks, superseded_branches)
            filter_counts['descendants_pre_routing'] = descendant_count
            with tracker.time_block(iteration=iteration, phase='benchmark',
                                    method='apply_complete_commodity_benchmark_pre_routing', event='runtime'):
                branches, globally_pruned_ids = apply_complete_commodity_benchmark(
                    branches, local_benchmarks, complete_infrastructure.index)
            filter_counts['complete_commodity_pre_routing'] = len(globally_pruned_ids)
            if globally_pruned_ids:
                superseded_branches.update(globally_pruned_ids)
                with tracker.time_block(iteration=iteration, phase='benchmark',
                                        method='remove_global_descendants_pre_routing', event='runtime'):
                    branches, global_descendant_count, local_benchmarks, superseded_branches = \
                        remove_superseded_branch_descendants(
                            branches, local_benchmarks, superseded_branches)
                filter_counts['global_descendants_pre_routing'] = global_descendant_count
        if branches.empty:
            tracker.event(iteration=iteration, phase='iteration', method='run_export_algorithm',
                          event='stop_no_active_branches', before=iteration_input_count, after=0,
                          runtime_s=time.time() - iteration_started,
                          details={'branch_number': branch_number})
            break

        with tracker.time_block(iteration=iteration, phase='routing',
                                method='prepare_routing_inputs', event='runtime'):
            branches = _attach_transport_properties(branches, config_file['available_transport_means'])
            arrived_by_approach = branches['current_transport_mean'].isin(
                ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])
            infrastructure_inputs = prepare_export_infrastructure_branches(
                branches[arrived_by_approach], complete_infrastructure)
            approach_inputs = branches[~arrived_by_approach]
        with tracker.time_block(iteration=iteration, phase='routing',
                                method='preselect_export_infrastructure_branches', event='runtime'):
            infrastructure_inputs, dominated_entry_count = preselect_export_infrastructure_branches(
                data, infrastructure_inputs, complete_infrastructure, configuration)
        filter_counts['dominated_infrastructure_entries'] = dominated_entry_count

        with tracker.time_block(iteration=iteration, phase='routing',
                                method='process_export_infrastructure_branches', event='runtime'):
            infrastructure_options = process_export_infrastructure_branches(
                data, infrastructure_inputs, complete_infrastructure, configuration)
        with tracker.time_block(iteration=iteration, phase='routing',
                                method='process_export_out_tolerance_branches', event='runtime'):
            approach_options, minimal_distance_pruned_count = process_export_out_tolerance_branches(
                complete_infrastructure, approach_inputs, configuration, local_benchmarks)
        filter_counts['minimal_distance_parents'] = minimal_distance_pruned_count
        with tracker.time_block(iteration=iteration, phase='routing',
                                method='process_export_zero_distance_branches', event='runtime'):
            zero_options = process_export_zero_distance_branches(
                data, branches, complete_infrastructure)

        with tracker.time_block(iteration=iteration, phase='routing',
                                method='combine_generated_options', event='runtime'):
            candidate_frames = [frame for frame in
                                (infrastructure_options, approach_options, zero_options)
                                if not frame.empty]
            candidates = (pd.concat(candidate_frames, ignore_index=True)
                          if candidate_frames else pd.DataFrame())
        with tracker.time_block(iteration=iteration, phase='benchmark',
                                method='prefilter_generated_candidates', event='runtime'):
            candidates, early_rejected_count = prefilter_export_branch_candidates(
                candidates, local_benchmarks, complete_infrastructure.index)
        filter_counts['early_candidate_rejections'] = early_rejected_count
        with tracker.time_block(iteration=iteration, phase='routing_finalize',
                                method='materialize_generated_branches', event='runtime'):
            branches, branch_number = _complete_generated_branches(
                candidates, branches, branch_number)
        with tracker.time_block(iteration=iteration, phase='benchmark',
                                method='apply_export_local_benchmark_post_routing', event='runtime'):
            branches, pruned_count, local_benchmarks, newly_superseded = apply_export_local_benchmark(
                branches, local_benchmarks)
        filter_counts['local_benchmark_post_routing'] = pruned_count
        superseded_branches.update(newly_superseded)
        with tracker.time_block(iteration=iteration, phase='benchmark',
                                method='remove_superseded_descendants_post_routing', event='runtime'):
            branches, descendant_count, local_benchmarks, superseded_branches = \
                remove_superseded_branch_descendants(
                    branches, local_benchmarks, superseded_branches)
        filter_counts['descendants_post_routing'] = descendant_count
        with tracker.time_block(iteration=iteration, phase='benchmark',
                                method='apply_complete_commodity_benchmark_post_routing', event='runtime'):
            branches, globally_pruned_ids = apply_complete_commodity_benchmark(
                branches, local_benchmarks, complete_infrastructure.index)
        filter_counts['complete_commodity_post_routing'] = len(globally_pruned_ids)
        if globally_pruned_ids:
            superseded_branches.update(globally_pruned_ids)
            with tracker.time_block(iteration=iteration, phase='benchmark',
                                    method='remove_global_descendants_post_routing', event='runtime'):
                branches, global_descendant_count, local_benchmarks, superseded_branches = \
                    remove_superseded_branch_descendants(
                        branches, local_benchmarks, superseded_branches)
            filter_counts['global_descendants_post_routing'] = global_descendant_count
        with tracker.time_block(iteration=iteration, phase='export',
                                method='export_iteration_snapshots', event='runtime'):
            export_branch_snapshot(branches, configuration['path_results'], location_index,
                                   iteration, 'active')
            export_local_benchmark_snapshot(
                local_benchmarks, configuration['path_results'], location_index, iteration)
        iteration_runtime = time.time() - iteration_started
        tracker.event(iteration=iteration, phase='iteration', method='run_export_algorithm',
                      event='runtime', before=iteration_input_count,
                      after=branch_count(branches), runtime_s=iteration_runtime,
                      details={'branch_number': branch_number,
                               'filter_counts': filter_counts})
        print(str(location_index) + '-' + str(iteration)
              + ': Active branches: ' + str(branch_count(branches))
              + ' | Created: ' + str(branch_number)
              + ' | Iteration [s]: ' + str(round(iteration_runtime, 2)))
        iteration += 1

    with tracker.time_block(iteration=iteration, phase='export',
                            method='export_final_local_benchmark_branches', event='runtime'):
        export_final_local_benchmark_branches(
            local_benchmarks, configuration['path_results'], location_index, iteration)
    marker = os.path.join(configuration['path_results'], 'export_infrastructure_branches',
                          str(location_index), '_complete')
    with open(marker, 'w', encoding='utf-8') as handle:
        handle.write('complete')
    tracker.event(phase='location', method='run_export_algorithm', event='end',
                  runtime_s=time.time() - started,
                  details={'total_branches_created': branch_number})
    print(str(location_index) + ': finished export enumeration in '
          + str(math.ceil((time.time() - started) / 60)) + ' minutes.')
    gc.collect()
    return None
