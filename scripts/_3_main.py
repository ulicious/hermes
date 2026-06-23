import os
import time
import multiprocessing
import itertools
import sys

import numpy as np

from algorithm.script_algorithm import run_algorithm
from algorithm.methods_main import prepare_data_and_configuration_dictionary
from data_processing.configuration import load_algorithm_configuration, load_technology_data
from algorithm.tracking import is_enabled

# sys.path.append(os.path.dirname(os.getcwd()))

import warnings
warnings.filterwarnings('ignore')

POOL_CHECK_INTERVAL_S = 30
DEAD_POOL_RESTART_AFTER_S = 300
MAX_POOL_RESTARTS = 10


def _ensure_trailing_separator(path_folder):
    if path_folder.endswith(('/', '\\')):
        return path_folder
    return path_folder + os.sep


def _prepare_result_folders(config_file, configuration):
    path_results = os.path.join(config_file['project_folder_path'], 'results')
    if config_file.get('_configuration_path'):
        default_config_path = os.path.join(config_file['project_folder_path'], '1_algorithm_configuration.yaml')
        if os.path.abspath(config_file['_configuration_path']) != os.path.abspath(default_config_path):
            path_results = os.path.join(path_results, os.path.basename(config_file['_configuration_path']))

    configuration['path_results'] = _ensure_trailing_separator(path_results)
    os.makedirs(os.path.join(configuration['path_results'], 'location_results'), exist_ok=True)
    os.makedirs(os.path.join(configuration['path_results'], 'algorithm_tracking'), exist_ok=True)


def _get_processed_locations(configuration):
    processed_locations = []
    path_location_results = os.path.join(configuration['path_results'], 'location_results')
    files = os.listdir(path_location_results)
    for f in files:
        if 'global' in f:
            continue

        try:
            number = int(f.split('_')[0])
        except ValueError:
            continue
        processed_locations.append(number)

    return processed_locations


def _get_unprocessed_location_data(location_data, configuration):
    processed_locations = _get_processed_locations(configuration)
    processed_locations = [i for i in processed_locations if i in location_data.index]
    return location_data.drop(processed_locations)


def _get_num_cores(config_file):
    num_cores = config_file['number_cores']
    if num_cores == 'max':
        num_cores = multiprocessing.cpu_count() - 1
    else:
        num_cores = min(num_cores, multiprocessing.cpu_count() - 1)
    return max(1, num_cores)


def _pool_has_live_workers(pool):
    return any(worker.is_alive() for worker in pool._pool)


def _run_parallel_algorithm(location_data, data, config_file, configuration):
    num_cores = _get_num_cores(config_file)
    restarts = 0

    while True:
        pending_location_data = _get_unprocessed_location_data(location_data, configuration)
        if pending_location_data.empty:
            return

        print('Open locations: ' + str(len(pending_location_data.index)))

        rng = np.random.default_rng(seed=42)
        indexes = rng.permutation(pending_location_data.index)

        task_args = zip(indexes,
                        itertools.repeat(pending_location_data),
                        itertools.repeat(data),
                        itertools.repeat(config_file),
                        itertools.repeat(configuration))

        pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=1)
        results = pool.imap_unordered(run_algorithm, task_args)
        dead_pool_since = None
        pool_finished = False

        try:
            while True:
                try:
                    results.next(timeout=POOL_CHECK_INTERVAL_S)
                    dead_pool_since = None
                except StopIteration:
                    pool.close()
                    pool.join()
                    pool_finished = True
                    break
                except multiprocessing.TimeoutError:
                    pending_location_data = _get_unprocessed_location_data(location_data, configuration)
                    if pending_location_data.empty:
                        pool.terminate()
                        pool.join()
                        pool_finished = True
                        return

                    if _pool_has_live_workers(pool):
                        dead_pool_since = None
                        continue

                    if dead_pool_since is None:
                        dead_pool_since = time.time()
                        print('No live worker processes detected. Waiting before restarting pool.')
                        continue

                    if time.time() - dead_pool_since < DEAD_POOL_RESTART_AFTER_S:
                        continue

                    restarts += 1
                    print('No live worker processes for '
                          + str(DEAD_POOL_RESTART_AFTER_S)
                          + ' seconds. Restart pool '
                          + str(restarts)
                          + '/'
                          + str(MAX_POOL_RESTARTS)
                          + '.')
                    pool.terminate()
                    pool.join()
                    pool_finished = True

                    if restarts > MAX_POOL_RESTARTS:
                        raise RuntimeError(
                            'Maximum number of pool restarts reached. '
                            'Remaining locations: ' + str(len(pending_location_data.index))
                        )
                    break
        except Exception:
            if not pool_finished:
                pool.terminate()
                pool.join()
            raise


if __name__ == '__main__':

    # load configuration file
    config_file = load_algorithm_configuration()

    data, configuration, location_data = prepare_data_and_configuration_dictionary(config_file)
    _prepare_result_folders(config_file, configuration)

    # used to remove processed results
    location_data = _get_unprocessed_location_data(location_data, configuration)

    # location_data = location_data.loc[[5869], :]

    print_information = True

    if print_information:  # todo: all configuration should be shown here
        # paths
        path_project_folder = config_file['project_folder_path']
        path_raw_data = path_project_folder + 'raw_data/'
        path_processed_data = path_project_folder + 'processed_data/'

        techno_economic_data_conversion, techno_economic_data_transport = load_technology_data(config_file)

        print('Main configuration:')
        print('Configuration file: ' + str(config_file.get('_configuration_path')))
        print('Project folder: ' + str(config_file['project_folder_path']))

        target_commodity_text = ', '.join(config_file['target_commodity'])
        print('Target commodity: ' + target_commodity_text)

        print('Distance to destination requiring not transport: ' + str(configuration['to_final_destination_tolerance']))

        print('Distance between infrastructure requiring no transport: ' + str(configuration['tolerance_distance']))

        print('Maximal distance for road transport: ' + str(configuration['max_length_road']))

        if is_enabled(configuration['build_new_infrastructure']):
            print('Distance for new pipeline segments: ' + str(configuration['max_length_new_segment']))
        else:
            print('New pipeline segments not enabled')

        print('Multiplier to consider obstacles (road transport / new pipeline segments): ' + str(configuration['no_road_multiplier']))

        if is_enabled(configuration['H2_ready_infrastructure']):
            print('Retrofitting of gas pipelines enabled')
        else:
            print('Retrofitting of gas pipelines not enabled')

        if is_enabled(config_file['consider_commodity_prices']):
            print(techno_economic_data_conversion['strike_prices'])
        else:
            for k in techno_economic_data_conversion['strike_prices'].keys():
                techno_economic_data_conversion['strike_prices'][k] = 0
            print(techno_economic_data_conversion['strike_prices'])

        # print('Considered: ' + )

        # add more info like hydrogen retrofitting and distances

    print('start algorithm')
    time_start = time.time()
    if not is_enabled(configuration['use_low_memory']):
        _run_parallel_algorithm(location_data, data, config_file, configuration)

    else:
        for i in location_data.index:
            args = [i, location_data, data, config_file, configuration]
            run_algorithm(args)

    if time.time() - time_start < 60:
        print('total time [s]: ' + str(time.time() - time_start))
    elif time.time() - time_start < 3600:
        print('total time [m]: ' + str((time.time() - time_start) / 60))
    else:
        print('total time [h]: ' + str((time.time() - time_start) / 60 / 60))
