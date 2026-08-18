"""Runner for enumerating a country's export-infrastructure branches."""

import itertools
import multiprocessing
import os
import time

import numpy as np

from algorithm.methods_main import prepare_data_and_configuration_dictionary
from algorithm.script_export_algorithm import run_export_algorithm
from data_processing.configuration import load_algorithm_configuration
from algorithm.tracking import is_enabled


def _ensure_trailing_separator(path_folder):
    return path_folder if path_folder.endswith(('/', '\\')) else path_folder + os.sep


def _prepare_result_folders(config_file, configuration):
    path_results = os.path.join(config_file['project_folder_path'], 'results')
    if config_file.get('_configuration_path'):
        default = os.path.join(config_file['project_folder_path'], '1_algorithm_configuration.yaml')
        if os.path.abspath(config_file['_configuration_path']) != os.path.abspath(default):
            path_results = os.path.join(path_results, os.path.basename(config_file['_configuration_path']))
    configuration['path_results'] = _ensure_trailing_separator(path_results)
    os.makedirs(os.path.join(path_results, 'export_infrastructure_branches'), exist_ok=True)


def _processed_locations(configuration):
    folder = os.path.join(configuration['path_results'], 'export_infrastructure_branches')
    return [int(name) for name in os.listdir(folder) if name.isdigit()
            and os.path.exists(os.path.join(folder, name, '_complete'))]


def _num_cores(config_file):
    requested = config_file['number_cores']
    available = max(1, multiprocessing.cpu_count() - 1)
    return available if requested == 'max' else max(1, min(requested, available))


if __name__ == '__main__':
    config_file = load_algorithm_configuration()
    data, configuration, location_data = prepare_data_and_configuration_dictionary(config_file)
    _prepare_result_folders(config_file, configuration)
    location_data = location_data.drop(_processed_locations(configuration), errors='ignore')

    print('Start export-infrastructure branch enumeration')
    started = time.time()
    arguments = zip(np.random.default_rng(42).permutation(location_data.index),
                    itertools.repeat(location_data), itertools.repeat(data),
                    itertools.repeat(config_file), itertools.repeat(configuration))
    if is_enabled(configuration['use_low_memory']):
        for args in arguments:
            run_export_algorithm(args)
    else:
        with multiprocessing.Pool(_num_cores(config_file), maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(run_export_algorithm, arguments):
                pass
    print('Finished after [m]: ' + str(round((time.time() - started) / 60, 2)))
