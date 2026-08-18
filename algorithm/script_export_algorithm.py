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
                                      apply_export_local_benchmark,
                                      attach_infrastructure_countries,
                                      create_export_branches_at_start,
                                      export_branch_snapshot,
                                      get_complete_export_infrastructure,
                                      get_export_infrastructure_scope,
                                      get_start_country,
                                      prepare_export_commodities,
                                      prepare_export_infrastructure_branches,
                                      process_export_infrastructure_branches,
                                      process_export_out_tolerance_branches,
                                      process_export_zero_distance_branches,
                                      split_export_branches)
from algorithm.methods_geographic import update_branch_continents
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


def _complete_generated_branches(branches, previous_branches, branch_number,
                                 data, complete_infrastructure):
    if branches.empty:
        return branches, pd.DataFrame(), branch_number
    branches = branches.copy()
    branches['branch_index'] = ['S' + str(branch_number + i) for i in range(len(branches))]
    branches.index = branches['branch_index']
    branches.index.name = None
    branch_number += len(branches)
    branches['current_conversion_costs'] = 0
    branches = update_branch_continents(branches, complete_infrastructure, world=data['world'])
    branches = postprocessing_branches(branches, previous_branches)
    active, completed = split_export_branches(
        branches, complete_infrastructure, data['export_start_country'])
    return active, completed, branch_number


def _prepare_location(location_index, location_data, data, config_file, configuration):
    location_data = location_data.copy().loc[[location_index]]
    location_data.index = ['Start']
    start_country = get_start_country(location_data)
    print(str(location_index) + ': Start country: ' + str(start_country))
    data = data.copy()
    data['k'] = location_index
    data['location_index'] = location_index
    data['start_location_data'] = location_data.copy()
    data['export_start_country'] = start_country
    data['start'] = {
        'location': Point(location_data.at['Start', 'longitude'],
                          location_data.at['Start', 'latitude']),
        'continent': location_data.at['Start', 'continent_start'],
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
    complete_infrastructure = get_export_infrastructure_scope(
        complete_infrastructure, start_country)
    return location_data, data, complete_infrastructure, branches, branch_number


def run_export_algorithm(args):
    """Enumerate routes until they first reach infrastructure outside the start country."""
    location_index, location_data, common_data, config_file, configuration = args
    print(str(location_index) + ': Start Processing export infrastructure')
    started = time.time()
    tracker = AlgorithmTracker(location_index, configuration['path_results'])
    preparation_started = time.time()
    location_data, data, complete_infrastructure, branches, branch_number = \
        _prepare_location(location_index, location_data, common_data, config_file, configuration)
    data['tracker'] = tracker
    preparation_time = time.time() - preparation_started
    print(str(location_index) + ': Preparation [s]: ' + str(round(preparation_time, 2))
          + ' | Country infrastructure nodes: ' + str(len(complete_infrastructure)))

    if branches.empty or branches['current_total_costs'].map(math.isinf).all():
        export_branch_snapshot(branches, configuration['path_results'], location_index, 0, 'no_potential')
        return None

    completed_batches = []
    benchmark_pruned_batches = []
    local_benchmarks = {}
    export_branch_snapshot(branches, configuration['path_results'], location_index, 0, 'initial')
    iteration = 0
    while not branches.empty:
        data['current_iteration'] = iteration
        iteration_started = time.time()
        if iteration > 0:
            conversion_locations = data['conversion_costs_and_efficiencies']
            possible_nodes = conversion_locations[conversion_locations['conversion_possible']].index
            convertible = branches[branches['current_node'].isin(possible_nodes)]
            unchanged = branches[~branches['current_node'].isin(possible_nodes)]
            converted, branch_number = apply_export_conversion(convertible, data, branch_number)
            branches = pd.concat([converted, unchanged], ignore_index=False)
            branches, pruned, local_benchmarks = apply_export_local_benchmark(
                branches, local_benchmarks)
            if not pruned.empty:
                benchmark_pruned_batches.append(pruned)
        if branches.empty:
            break

        branches = _attach_transport_properties(branches, config_file['available_transport_means'])
        arrived_by_approach = branches['current_transport_mean'].isin(
            ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid'])
        infrastructure_inputs = prepare_export_infrastructure_branches(
            branches[arrived_by_approach], complete_infrastructure)
        approach_inputs = branches[~arrived_by_approach]

        infrastructure_options = process_export_infrastructure_branches(
            data, infrastructure_inputs, complete_infrastructure, configuration)
        infrastructure_options, completed_infrastructure, branch_number = _complete_generated_branches(
            infrastructure_options, infrastructure_inputs, branch_number, data, complete_infrastructure)
        approach_options = process_export_out_tolerance_branches(
            complete_infrastructure, approach_inputs, configuration)
        approach_options, completed_approaches, branch_number = _complete_generated_branches(
            approach_options, approach_inputs, branch_number, data, complete_infrastructure)
        zero_options = process_export_zero_distance_branches(
            data, branches, complete_infrastructure)
        zero_options, completed_zero, branch_number = _complete_generated_branches(
            zero_options, branches, branch_number, data, complete_infrastructure)

        completed_batches.extend([
            frame for frame in (completed_infrastructure, completed_approaches, completed_zero)
            if not frame.empty])
        active_frames = [frame for frame in (infrastructure_options, approach_options, zero_options)
                         if not frame.empty]
        branches = pd.concat(active_frames, ignore_index=False) if active_frames else pd.DataFrame()
        branches, pruned, local_benchmarks = apply_export_local_benchmark(
            branches, local_benchmarks)
        if not pruned.empty:
            benchmark_pruned_batches.append(pruned)
        export_branch_snapshot(branches, configuration['path_results'], location_index,
                               iteration, 'active')
        completed = (pd.concat(completed_batches, ignore_index=False)
                     if completed_batches else pd.DataFrame())
        export_branch_snapshot(completed, configuration['path_results'], location_index,
                               iteration, 'completed')
        pruned_branches = (pd.concat(benchmark_pruned_batches, ignore_index=False)
                           if benchmark_pruned_batches else pd.DataFrame())
        export_branch_snapshot(pruned_branches, configuration['path_results'], location_index,
                               iteration, 'local_benchmark_pruned')
        print(str(location_index) + '-' + str(iteration)
              + ': Active branches: ' + str(branch_count(branches))
              + ' | Export branches: ' + str(branch_count(completed))
              + ' | Created: ' + str(branch_number)
              + ' | Iteration [s]: ' + str(round(time.time() - iteration_started, 2)))
        iteration += 1

    completed = (pd.concat(completed_batches, ignore_index=False)
                 if completed_batches else pd.DataFrame())
    export_branch_snapshot(completed, configuration['path_results'], location_index,
                           iteration, 'all_completed')
    all_pruned = (pd.concat(benchmark_pruned_batches, ignore_index=False)
                  if benchmark_pruned_batches else pd.DataFrame())
    export_branch_snapshot(all_pruned, configuration['path_results'], location_index,
                           iteration, 'all_local_benchmark_pruned')
    marker = os.path.join(configuration['path_results'], 'export_infrastructure_branches',
                          str(location_index), '_complete')
    with open(marker, 'w', encoding='utf-8') as handle:
        handle.write(str(len(completed)))
    tracker.event(phase='location', method='run_export_algorithm', event='end',
                  runtime_s=time.time() - started,
                  details={'total_branches_created': branch_number,
                           'export_branches': len(completed)})
    print(str(location_index) + ': finished export enumeration in '
          + str(math.ceil((time.time() - started) / 60)) + ' minutes.')
    gc.collect()
    return None
