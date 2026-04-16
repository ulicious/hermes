import os
import time
import multiprocessing
import yaml
import sys

import numpy as np

from algorithm.script_algorithm import run_algorithm
from algorithm.methods_main import prepare_data_and_configuration_dictionary

# sys.path.append(os.path.dirname(os.getcwd()))

import warnings
warnings.filterwarnings('ignore')


_WORKER_LOCATION_DATA = None
_WORKER_DATA = None
_WORKER_CONFIG_FILE = None
_WORKER_CONFIGURATION = None


def initialize_worker(location_data, data, config_file, configuration):
    global _WORKER_LOCATION_DATA
    global _WORKER_DATA
    global _WORKER_CONFIG_FILE
    global _WORKER_CONFIGURATION

    _WORKER_LOCATION_DATA = location_data
    _WORKER_DATA = data
    _WORKER_CONFIG_FILE = config_file
    _WORKER_CONFIGURATION = configuration


def run_algorithm_for_location(location_index):
    return run_algorithm((location_index,
                          _WORKER_LOCATION_DATA,
                          _WORKER_DATA,
                          _WORKER_CONFIG_FILE,
                          _WORKER_CONFIGURATION))


if __name__ == '__main__':

    # load configuration file
    path_config = os.getcwd() + '/algorithm_configuration.yaml'
    yaml_file = open(path_config)
    config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    data, configuration, location_data = prepare_data_and_configuration_dictionary(config_file)

    # used to remove processed results
    processed_locations = []
    files = os.listdir(configuration['path_results'] + 'location_results/')
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

        num_cores = config_file['number_cores']
        if num_cores == 'max':
            num_cores = multiprocessing.cpu_count() - 1
        else:
            num_cores = min(num_cores, multiprocessing.cpu_count() - 1)

        rng = np.random.default_rng(seed=42)
        indexes = rng.permutation(location_data.index)

        tasks_per_child = config_file.get('tasks_per_child', None)

        # Large read-mostly data is initialized once per worker. Each location run creates its own local working
        # copies inside run_algorithm, so repeated tasks in a worker do not inherit changed data from previous tasks.
        with multiprocessing.Pool(processes=num_cores,
                                  initializer=initialize_worker,
                                  initargs=(location_data, data, config_file, configuration),
                                  maxtasksperchild=tasks_per_child) as pool:

            for _ in pool.imap_unordered(run_algorithm_for_location, indexes, chunksize=1):
                pass

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
