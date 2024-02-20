import os
import time
import multiprocessing
import itertools
import yaml

from script_algorithm import run_algorithm
from methods_main import prepare_data_and_configuration_dictionary

from methods_routing import get_complete_infrastructure
from calculate_and_save_all_distances import calculate_and_save_shortest_distance

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

    # location_data = location_data.loc[[1251], :] # 6359 # todo: check 675 wegen nan als graph
    # location_data = location_data.iloc[54:60]
    # subset = [58, 54]
    # location_data = location_data.loc[subset, :]
    # location_data = location_data.loc[location_data.index[0], :]

    print_information = True

    num_cores = min(100, multiprocessing.cpu_count() - 1)
    num_cores_left = multiprocessing.cpu_count() - 1 - num_cores

    # all_options = get_complete_infrastructure(data)
    # calculate_and_save_all_distances(all_options)
    # calculate_and_save_shortest_distances(all_options, data)
    # calculate_and_save_shortest_distance(all_options, data)

    print('start algorithm')
    time_start = time.time()

    if True:
        # The number of worker processes
        num_workers = num_cores

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_workers, maxtasksperchild=1)

        # Create an iterable of tuples, each containing the task ID and shared_dict
        task_args = zip(location_data.index,
                        itertools.repeat(num_cores_left),
                        itertools.repeat(location_data),
                        itertools.repeat(data),
                        itertools.repeat(configuration),
                        itertools.repeat(print_information))
        # Start processing tasks and ensure parallelism
        results = list(pool.imap(run_algorithm, task_args))

        # Close and join the worker pool
        pool.close()
        pool.join()

    else:
        task_args = [location_data.index[0], num_cores_left, historic_most_cost_effective_routes, distances_dict,
                     graph_connector_data, location_data, data, configuration, coastlines, commodity_conversion_data,
                     commodity_conversion_loss_of_products_data, commodity_transportation_data, print_information,
                     path_csvs]
        run_algorithm(task_args)

    #routing_costs = pd.DataFrame(dict(shared_dict)).transpose()
    #routing_costs.to_csv(path_csvs + 'global_minimal_paths.csv')

    if time.time() - time_start < 60:
        print('total time [s]: ' + str(time.time() - time_start))
    elif time.time() - time_start < 3600:
        print('total time [m]: ' + str((time.time() - time_start) / 60))
    else:
        print('total time [h]: ' + str((time.time() - time_start) / 60 / 60))
