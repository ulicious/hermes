import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import math
import ast
import shapely
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from shapely.geometry import Point
from tqdm import tqdm

from plotting.helpers_plotting import load_data, get_complete_infrastructure
from plotting.get_figures import get_routes_figure, get_cost_figure, get_production_costs_figure, get_infrastructure_figure, \
    get_energy_carrier_figure, get_cost_and_quantity_figure, get_supply_curves

# script to process results

# load configuration file
path_config = os.getcwd() + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_config = os.getcwd() + '/plotting_configuration.yaml'
yaml_file = open(path_config)
config_file_plotting = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_production_costs = config_file['project_folder_path'] + 'start_destination_combinations.csv'
production_costs = pd.read_csv(path_production_costs, index_col=0)

# # convert polygon strings to shapely objects
if config_file['use_voronoi_cells']:
    production_costs['geometry'] \
        = production_costs['geometry'].apply(shapely.wkt.loads)

path_processed_data = config_file['project_folder_path'] + 'processed_data/'

infrastructure_data, destination = load_data(path_processed_data, config_file)
complete_infrastructure = get_complete_infrastructure(infrastructure_data, destination)

path_processed_results = config_file['project_folder_path'] + 'results/processed_results/'

if 'processed_results' not in os.listdir(config_file['project_folder_path'] + 'results/'):
    os.makedirs(path_processed_results)

min_costs = {'total_costs': np.inf, 'transportation_costs': np.inf, 'conversion_costs': np.inf}
max_costs = {'total_costs': 0, 'transportation_costs': 0, 'conversion_costs': 0}

result_folders = config_file_plotting['process_results']

for folder in result_folders:

    if path_processed_results + folder + '_processed_results.csv' in os.listdir():
        continue

    print(folder)

    path = config_file['project_folder_path'] + 'results/' + folder + '/'

    data = {}

    filenames = tqdm(os.listdir(path))

    commodities = config_file['available_commodity']
    transport_means = config_file['available_transport_means']

    routes = []
    starting_locations = []

    for f in filenames:

        if 'final' not in f:
            continue

        solution = pd.read_csv(path + f, index_col=0)
        solution = solution[solution.columns[0]]
        number = int(f.split('_')[0])

        status = solution.loc['status']
        starting_locations.append((solution.at['starting_longitude'], solution.at['starting_latitude']))
        if status == 'no benchmark':
            data[number] = {'costs': math.inf,
                            'start_commodity': None,
                            'second_commodity': None,
                            'latitude': solution.at['starting_latitude'],
                            'longitude': solution.at['starting_longitude'],
                            'sec_distance_mean_combination': None,
                            'transportation_costs': math.inf,
                            'conversion_costs': math.inf}
        else:
            # read strings as lists
            solution.loc['all_previous_transportation_costs']\
                = ast.literal_eval(solution.loc['all_previous_transportation_costs'])
            solution.loc['all_previous_conversion_costs']\
                = ast.literal_eval(solution.loc['all_previous_conversion_costs'])
            solution.loc['all_previous_distances'] \
                = ast.literal_eval(solution.loc['all_previous_distances'])
            solution.loc['all_previous_commodities'] \
                = ast.literal_eval(solution.loc['all_previous_commodities'])
            solution.loc['all_previous_total_costs'] \
                = ast.literal_eval(solution.loc['all_previous_total_costs'])
            solution.loc['all_previous_transport_means'] \
                = ast.literal_eval(solution.loc['all_previous_transport_means'])

            data[number] = {'costs': None,
                            'start_commodity': None,
                            'second_commodity': None,
                            'latitude': None,
                            'longitude': None,
                            'sec_distance_mean_combination': None}

            data[number]['latitude'] = float(solution.at['starting_latitude'])
            data[number]['longitude'] = float(solution.at['starting_longitude'])
            data[number]['start_commodity'] = solution.at['all_previous_commodities'][0]
            data[number]['costs'] = float(solution.at['current_total_costs'])
            data[number]['efficiency'] = float(solution.at['total_efficiency'])

            if float(solution.at['current_total_costs']) < float(min_costs['total_costs']):
                min_costs['total_costs'] = float(solution.at['current_total_costs'])

            if float(solution.at['current_total_costs']) > float(max_costs['total_costs']):
                max_costs['total_costs'] = float(solution.at['current_total_costs'])

            if float(sum(solution.at['all_previous_transportation_costs'])) < float(min_costs['transportation_costs']):
                min_costs['transportation_costs'] = sum(solution.at['all_previous_transportation_costs'])

            if float(sum(solution.at['all_previous_transportation_costs'])) > float(max_costs['transportation_costs']):
                max_costs['transportation_costs'] = sum(solution.at['all_previous_transportation_costs'])

            if float(sum(solution.at['all_previous_conversion_costs'])) < float(min_costs['conversion_costs']):
                min_costs['conversion_costs'] = sum(solution.at['all_previous_conversion_costs'])

            if float(sum(solution.at['all_previous_conversion_costs'])) > float(max_costs['conversion_costs']):
                max_costs['conversion_costs'] = sum(solution.at['all_previous_conversion_costs'])

            starting_commodity_f = solution.at['all_previous_commodities'][0]

            # production_costs_f = solution.at['all_previous_total_costs'][0]
            conversion_costs_f = sum(solution.at['all_previous_conversion_costs'])
            transportation_costs_f = sum(solution.at['all_previous_transportation_costs'])
            commodities_list = solution.at['all_previous_commodities']
            transportation_means_list = solution.at['all_previous_transport_means']

            production_costs_f = production_costs.at[int(number), 'Hydrogen_Gas']

            if (float(production_costs_f) + float(conversion_costs_f) + float(transportation_costs_f)) != float(solution.at['current_total_costs']):
                # conversion costs at start are not considered
                conversion_costs_f = float(solution.at['current_total_costs']) - float(production_costs_f) - float(transportation_costs_f)

            data[number]['transportation_costs'] = transportation_costs_f
            data[number]['conversion_costs'] = conversion_costs_f
            data[number]['production_costs'] = production_costs_f

            routes.append(ast.literal_eval(solution.loc['taken_routes']))

    data = pd.DataFrame.from_dict(data,
                                  columns=['costs', 'start_commodity', 'second_commodity', 'latitude',
                                           'longitude', 'efficiency', 'transportation_costs', 'conversion_costs',
                                           'production_costs'],
                                  orient='index')

    data['routes'] = routes

    data.to_csv(path_processed_results + folder + '_processed_results.csv')
