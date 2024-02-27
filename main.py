import os
import time
import multiprocessing
import itertools
import yaml

from script_algorithm import run_algorithm
from methods_main import prepare_data_and_configuration_dictionary

import warnings
warnings.filterwarnings('ignore')

# todo: point has x,y --> lon, lat

if __name__ == '__main__':

    # todo: aufräumen --> methoden für unterschiedliche Schritte hier machen
    # todo: wenn liste oder dict nicht in einer Methodik verwendet wird, dann nicht übergeben --> außerhalb verarbeiten

    # path_config = os.getcwd() + '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/configuration.yaml'
    path_config = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/configuration.yaml'
    yaml_file = open(path_config)
    config_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    data, configuration, location_data = prepare_data_and_configuration_dictionary(config_yaml)

    processed_locations = []
    files = os.listdir(configuration['path_results'])
    for f in files:
        if 'global' in f:
            continue

        number = int(f.split('_')[0])
        processed_locations.append(number)

    location_data.drop(processed_locations, inplace=True)

    print_information = True

    num_cores = config_yaml['number_cores']
    if num_cores == 'max':
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = min(num_cores, multiprocessing.cpu_count())

    num_cores_left = 0

    print('start algorithm')
    time_start = time.time()

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=1)

    # Create an iterable of tuples, each containing the task ID and shared_dict
    task_args = zip(location_data.index,
                    itertools.repeat(location_data),
                    itertools.repeat(data),
                    itertools.repeat(configuration),
                    itertools.repeat(print_information))
    # Start processing tasks and ensure parallelism
    results = list(pool.imap(run_algorithm, task_args))

    # Close and join the worker pool
    pool.close()
    pool.join()

    if time.time() - time_start < 60:
        print('total time [s]: ' + str(time.time() - time_start))
    elif time.time() - time_start < 3600:
        print('total time [m]: ' + str((time.time() - time_start) / 60))
    else:
        print('total time [h]: ' + str((time.time() - time_start) / 60 / 60))
