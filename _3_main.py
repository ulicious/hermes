import os
import time
import multiprocessing
import itertools
import yaml
import sys

from algorithm.script_algorithm import run_algorithm
from algorithm.methods_main import prepare_data_and_configuration_dictionary

# sys.path.append(os.path.dirname(os.getcwd()))

import warnings
warnings.filterwarnings('ignore')

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

    print_information = True

    print('start algorithm')
    time_start = time.time()
    if not config_file['use_low_memory']:

        num_cores = config_file['number_cores']
        if num_cores == 'max':
            num_cores = multiprocessing.cpu_count() - 1
        else:
            num_cores = min(num_cores, multiprocessing.cpu_count() - 1)

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=1)

        # Create an iterable of tuples, each containing the task ID and shared_dict
        task_args = zip(location_data.index,
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
