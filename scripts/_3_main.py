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

if __name__ == '__main__':

    # load configuration file
    config_file = load_algorithm_configuration()

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

        num_cores = config_file['number_cores']
        if num_cores == 'max':
            num_cores = multiprocessing.cpu_count() - 1
        else:
            num_cores = min(num_cores, multiprocessing.cpu_count() - 1)

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=1)

        # Create an iterable of tuples, each containing the task ID and shared_dict
        rng = np.random.default_rng(seed=42)
        indexes = rng.permutation(location_data.index)

        task_args = zip(indexes,
                        itertools.repeat(location_data),
                        itertools.repeat(data),
                        itertools.repeat(config_file),
                        itertools.repeat(configuration))

        # Start processing tasks and ensure parallelism
        results = list(pool.imap(run_algorithm, task_args))

        # Close and join the worker pool
        pool.close()
        pool.join()

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
