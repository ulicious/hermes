import os
import time
import multiprocessing
import yaml
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from algorithm.script_algorithm import run_algorithm
from algorithm.methods_main import prepare_data_and_configuration_dictionary

# sys.path.append(os.path.dirname(os.getcwd()))

import warnings
warnings.filterwarnings('ignore')


_WORKER_LOCATION_DATA = None
_WORKER_DATA = None
_WORKER_CONFIG_FILE = None
_WORKER_CONFIGURATION = None


def get_project_folder_path(config_file):
    return config_file['project_folder_path']


def get_results_path(config_file):
    return get_project_folder_path(config_file) + 'results/'


def load_location_data(config_file):
    return pd.read_csv(get_project_folder_path(config_file) + 'start_destination_combinations.csv', index_col=0)


def choose_parallel_backend(config_file):
    configured_backend = config_file.get('parallel_backend', 'auto')
    if configured_backend != 'auto':
        return configured_backend

    if sys.platform.startswith('linux') and 'fork' in multiprocessing.get_all_start_methods():
        return 'processes_fork'

    return 'threads'


def get_number_location_workers(config_file, number_locations):
    configured_workers = config_file['number_cores']

    if configured_workers == 'max':
        configured_workers = multiprocessing.cpu_count() - 1

    configured_workers = int(configured_workers)
    configured_workers = max(1, configured_workers)
    configured_workers = min(configured_workers, multiprocessing.cpu_count() - 1)
    configured_workers = min(configured_workers, number_locations)

    return configured_workers


def initialize_worker(config_file):
    global _WORKER_CONFIG_FILE

    _WORKER_CONFIG_FILE = config_file


def initialize_shared_data(config_file, data, configuration, location_data):
    global _WORKER_LOCATION_DATA
    global _WORKER_DATA
    global _WORKER_CONFIG_FILE
    global _WORKER_CONFIGURATION

    _WORKER_CONFIG_FILE = config_file
    _WORKER_DATA = data
    _WORKER_CONFIGURATION = configuration
    _WORKER_LOCATION_DATA = location_data


def get_worker_data():
    global _WORKER_LOCATION_DATA
    global _WORKER_DATA
    global _WORKER_CONFIGURATION

    if _WORKER_DATA is None:
        _WORKER_DATA, _WORKER_CONFIGURATION, _WORKER_LOCATION_DATA \
            = prepare_data_and_configuration_dictionary(_WORKER_CONFIG_FILE)

    return _WORKER_DATA, _WORKER_CONFIGURATION, _WORKER_LOCATION_DATA


def run_algorithm_for_location(location_index):
    data, configuration, location_data = get_worker_data()

    return run_algorithm((location_index,
                          location_data,
                          data,
                          _WORKER_CONFIG_FILE,
                          configuration))


if __name__ == '__main__':

    # load configuration file
    path_config = os.getcwd() + '/algorithm_configuration.yaml'
    yaml_file = open(path_config)
    config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # used to remove processed results
    location_data = load_location_data(config_file)
    processed_locations = []
    files = os.listdir(get_results_path(config_file) + 'location_results/')
    for f in files:
        if 'global' in f:
            continue

        number = int(f.split('_')[0])
        processed_locations.append(number)

    location_data.drop(processed_locations, inplace=True)

    # location_data = location_data.loc[[7386], :]

    print_information = True

    if print_information:  # todo: all configuration should be shown here
        # paths
        path_project_folder = config_file['project_folder_path']
        path_raw_data = path_project_folder + 'raw_data/'
        path_processed_data = path_project_folder + 'processed_data/'

        yaml_file = open(path_raw_data + 'techno_economic_data_conversion.yaml')
        techno_economic_data_conversion = yaml.load(yaml_file, Loader=yaml.FullLoader)

        yaml_file = open(path_raw_data + 'techno_economic_data_transportation.yaml')
        techno_economic_data_transport = yaml.load(yaml_file, Loader=yaml.FullLoader)

        print('Main configuration:')

        target_commodity_text = ', '.join(config_file['target_commodity'])
        print('Target commodity: ' + target_commodity_text)

        print('Distance to destination requiring not transport: ' + str(config_file['to_final_destination_tolerance']))

        print('Distance between infrastructure requiring no transport: ' + str(config_file['tolerance_distance']))

        print('Maximal distance for road transport: ' + str(config_file['max_length_road']))

        if config_file['build_new_infrastructure']:
            print('Distance for new pipeline segments: ' + str(config_file['max_length_new_segment']))
        else:
            print('New pipeline segments not enabled')

        print('Multiplier to consider obstacles (road transport / new pipeline segments): ' + str(config_file['no_road_multiplier']))

        if config_file['H2_ready_infrastructure']:
            print('Retroffiting of gas pipelines enabled')
        else:
            print('Retrofitting if gas pipelines not enabled')

        if config_file['consider_commodity_prices']:
            print(techno_economic_data_conversion['strike_prices'])
        else:
            for k in techno_economic_data_conversion['strike_prices'].keys():
                techno_economic_data_conversion['strike_prices'][k] = 0
            print(techno_economic_data_conversion['strike_prices'])

        # print('Considered: ' + )

        # add more info like hydrogen retrofitting and distances

    print('start algorithm')
    time_start = time.time()
    if not config_file['use_low_memory']:

        rng = np.random.default_rng(seed=42)
        indexes = rng.permutation(location_data.index)
        number_location_workers = get_number_location_workers(config_file, len(indexes))

        parallel_backend = choose_parallel_backend(config_file)
        print('Parallel backend: ' + str(parallel_backend))
        print('Parallel location workers from number_cores: ' + str(number_location_workers))

        if parallel_backend == 'processes_fork':
            tasks_per_child = config_file.get('tasks_per_child', None)

            # Linux can share the large read-mostly data efficiently via fork/copy-on-write. Load once in the parent,
            # then fork workers so each location run still creates fresh mutable working data in run_algorithm.
            data, configuration, _ = prepare_data_and_configuration_dictionary(config_file)
            initialize_shared_data(config_file, data, configuration, location_data)

            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=number_location_workers, maxtasksperchild=tasks_per_child) as pool:
                for _ in pool.imap_unordered(run_algorithm_for_location, indexes, chunksize=1):
                    pass

        elif parallel_backend == 'processes':
            tasks_per_child = config_file.get('tasks_per_child', None)

            # Portable process mode. On Windows/macOS this uses spawn and each worker loads its own data, so prefer
            # threads unless enough memory is available.
            with multiprocessing.Pool(processes=number_location_workers,
                                      initializer=initialize_worker,
                                      initargs=(config_file,),
                                      maxtasksperchild=tasks_per_child) as pool:

                for _ in pool.imap_unordered(run_algorithm_for_location, indexes, chunksize=1):
                    pass

        else:
            # Threads share the large read-mostly pipeline data in one process. Each location run still receives fresh
            # local copies of mutable data inside run_algorithm.
            data, configuration, _ = prepare_data_and_configuration_dictionary(config_file)
            initialize_shared_data(config_file, data, configuration, location_data)

            with ThreadPoolExecutor(max_workers=number_location_workers) as executor:
                for _ in executor.map(run_algorithm_for_location, indexes):
                    pass

    else:
        data, configuration, _ = prepare_data_and_configuration_dictionary(config_file)
        for i in location_data.index:
            args = [i, location_data, data, config_file, configuration]
            run_algorithm(args)

    if time.time() - time_start < 60:
        print('total time [s]: ' + str(time.time() - time_start))
    elif time.time() - time_start < 3600:
        print('total time [m]: ' + str((time.time() - time_start) / 60))
    else:
        print('total time [h]: ' + str((time.time() - time_start) / 60 / 60))
