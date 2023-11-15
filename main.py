import time

import pandas as pd
from algorithm_test import run_algorithm
from _helpers import calc_distance_list_to_single

import geopandas as gpd
from shapely.wkt import loads

import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm
from shapely.geometry import Point

from process_input_data import process_network_data

from algorithm import start_algorithm

import itertools

import warnings
warnings.filterwarnings('ignore')

# todo: point has x,y --> lon, lat

if __name__ == '__main__':

    # todo: aufräumen --> methoden für unterschiedliche Schritte hier machen
    # todo: wenn liste oder dict nicht in einer Methodik verwendet wird, dann nicht übergeben --> außerhalb verarbeiten

    # Define path of data
    path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'

    # Define path of result

    # load configuration
    # Define assumptions
    configuration = {'use_OSRM': False,
                     'allow_first_iteration_conversion': True,
                     'tolerance_distance': 1000,
                     'to_final_destination_tolerance': 10000,
                     'no_road_multiplier': 1.5,
                     'Shipping': {'build_new_infrastructure': False,
                                  'find_only_closest_in_tolerance': False,
                                  'find_only_closest_outside_tolerance': False,
                                  'find_only_closest_to_destination': False},
                     'Pipeline_Gas': {'find_only_closest_in_tolerance': False,
                                      'find_only_closest_outside_tolerance': False,
                                      'find_only_closest_to_destination': False,
                                      'use_only_existing_nodes': True,
                                      'build_new_infrastructure': True,
                                      'build_consecutive_new_infrastructure': False,
                                      'follow_existing_roads': False,
                                      'use_direct_path': False,
                                      'max_length_new_segment': 50000},
                     'Pipeline_Liquid': {'find_only_closest_in_tolerance': True,
                                         'find_only_closest_outside_tolerance': True,
                                         'find_only_closest_to_destination': True,
                                         'use_only_existing_nodes': True,
                                         'build_new_infrastructure': True,
                                         'build_consecutive_new_infrastructure': False,
                                         'follow_existing_roads': False,
                                         'use_direct_path': False,
                                         'max_length_new_segment': 100000},
                     'Railroad': {'find_only_closest_in_tolerance': True,
                                  'find_only_closest_outside_tolerance': True,
                                  'find_only_closest_to_destination': True,
                                  'use_only_existing_nodes': True,
                                  'build_new_infrastructure': False,
                                  'build_consecutive_new_infrastructure': False,
                                  'follow_existing_roads': False,
                                  'use_direct_path': False,
                                  'max_length_new_segment': 10000}}

    # load input data
    location_data = pd.read_excel(path_data + 'start_destination_combinations.xlsx', index_col=0)
    commodity_conversion_data = pd.read_excel(path_data + 'commodities_conversions.xlsx',
                                              index_col=0)
    commodity_conversion_loss_of_products_data = pd.read_excel(path_data + 'commodities_conversion_efficiencies.xlsx',
                                                               index_col=0)
    commodity_transportation_data = pd.read_excel(path_data + 'commodities_transportation.xlsx',
                                                  index_col=0)
    pipeline_gas_geodata = pd.read_csv(path_data + 'gas_pipeline_geodata.csv', index_col=0)
    pipeline_gas_graphs = pd.read_csv(path_data + 'gas_pipeline_graphs.csv', index_col=0)
    pipeline_gas_graphs_objects = pd.read_csv(path_data + 'gas_pipeline_graphs_objects.csv', index_col=0)
    pipeline_liquid_geodata = pd.read_csv(path_data + 'oil_pipeline_geodata.csv', index_col=0)
    pipeline_liquid_graphs = pd.read_csv(path_data + 'oil_pipeline_graphs.csv', index_col=0)
    pipeline_liquid_graphs_objects = pd.read_csv(path_data + 'oil_pipeline_graphs_objects.csv', index_col=0)
    railroad_geodata = pd.read_csv(path_data + 'railroad_geodata.csv', index_col=0)
    railroad_graphs = pd.read_csv(path_data + 'railroad_graphs.csv', index_col=0)
    railroad_graphs_objects = pd.read_csv(path_data + 'railroad_graphs_objects.csv', index_col=0)
    ports = pd.read_excel(path_data + 'ports_processed.xlsx', index_col=0)
    ports_distances = pd.read_csv(path_data + '/inner_infrastructure_distances/ports.csv', index_col=0)
    # all_distances_inner_infrastructure = pd.read_csv(path_data + 'new_all_distances.csv', index_col=0)
    coastlines = pd.read_csv(path_data + 'coastlines.csv', index_col=0)

    if configuration['use_OSRM']:
        all_distances_road = pd.read_csv(path_data + 'test_distances.csv', index_col=0)
    else:
        all_distances_road = None

    coastlines = gpd.GeoDataFrame(coastlines)
    coastlines.geometry = coastlines.geometry.apply(loads)
    coastlines.set_geometry('geometry', inplace=True)

    colors = {'Road': 'red',
              'Shipping': 'darkblue',
              'Railroad': 'yellow',
              'Pipeline_Gas': 'purple',
              'New_Pipeline_Gas': 'indigo',
              'Pipeline_Liquid': 'green',
              'New_Pipeline_Liquid': 'turquoise'}

    # process input data
    # location_data = get_start_destination_combinations(location_data)

    # sort the starting locations by their distance to the final destination
    min_index = location_data.index[0]
    distances_from_start_to_destination = calc_distance_list_to_single(location_data['start_lat'],
                                                                       location_data['start_lon'],
                                                                       location_data.loc[
                                                                           min_index, 'destination_lat'],
                                                                       location_data.loc[
                                                                           min_index, 'destination_lon'])

    location_data['distance_to_final_destination'] = distances_from_start_to_destination
    location_data = location_data.sort_values(by=['distance_to_final_destination'])
    location_data.index = range(len(location_data.index))
    # location_data = location_data.loc[[117], :]
    # location_data = location_data.iloc[54:60]
    # subset = [54, 58]
    # location_data = location_data.loc[subset, :]

    # todo: return useful data for analysis

    path_plots = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/graphs/'
    path_csvs = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/csvs/'

    data = {'Shipping': {'ports': ports,
                         'Distances': {'value': ports_distances.to_numpy(),
                                       'index': ports_distances.index,
                                       'columns': ports_distances.columns}}}

    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_geodata,
                                pipeline_gas_graphs, path_data)
    data['Pipeline_Liquid'] = {}
    data['Railroad'] = {}
    data['all_distances_road'] = all_distances_road
    # data = process_network_data(data, 'Railroad', railroad_geodata, railroad_graphs)

    all_infrastructure = pd.concat(
        (ports[['latitude', 'longitude']], pipeline_gas_geodata[['latitude', 'longitude']],
         pipeline_liquid_geodata[['latitude', 'longitude']]))

    data['All_Infrastructure'] = all_infrastructure
    data['Coastline'] = coastlines

    print_information = True

    manager = Manager()
    historic_most_cost_effective_routes = manager.dict()
    graph_data = manager.dict()
    graph_connector_data = manager.dict()

    num_cores = min(100, multiprocessing.cpu_count() - 1)
    num_cores_left = multiprocessing.cpu_count() - 1 - num_cores

    print('start algorithm')

    if True:
        num_workers = num_cores  # The number of worker processes
        time_start = time.time()

        # Create a multiprocessing.Manager().dict() for shared data
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_workers)

        # Create an iterable of tuples, each containing the task ID and shared_dict
        task_args = zip(location_data.index,
                        itertools.repeat(num_cores_left),
                        itertools.repeat(historic_most_cost_effective_routes),
                        itertools.repeat(graph_data),
                        itertools.repeat(graph_connector_data),
                        itertools.repeat(location_data),
                        itertools.repeat(data),
                        itertools.repeat(configuration),
                        itertools.repeat(coastlines),
                        itertools.repeat(commodity_conversion_data),
                        itertools.repeat(commodity_conversion_loss_of_products_data),
                        itertools.repeat(commodity_transportation_data),
                        itertools.repeat(print_information),
                        itertools.repeat(path_csvs))

        # Start processing tasks and ensure parallelism
        results = list(pool.imap(run_algorithm, task_args))

        # Close and join the worker pool
        pool.close()
        pool.join()

    if False:
        routing_costs = pd.DataFrame(dict(shared_dict)).transpose()
        routing_costs.to_csv(path_csvs + 'global_minimal_paths.csv')

    print('total time: ' + str(time.time() - time_start))
