import pandas as pd
import multiprocessing
import itertools

import os
import math
import ast
import shapely
import numpy as np

import geopandas as gpd

from tqdm import tqdm
from shapely.geometry import Point

from data_processing.helpers_geometry import get_destination
from plotting.helpers_plotting import load_infrastructure_data, get_complete_infrastructure, create_weighted_routing_data_script
from data_processing.configuration import load_algorithm_configuration, load_plotting_configuration

# script to process results
# load configuration file
config_file = load_algorithm_configuration()
config_file_plotting = load_plotting_configuration(config_file)

path_production_costs = config_file['project_folder_path'] + 'start_destination_combinations.csv'
production_costs = pd.read_csv(path_production_costs, index_col=0)

# # convert polygon strings to shapely objects
if config_file['use_voronoi_cells']:
    production_costs['geometry'] \
        = production_costs['geometry'].apply(shapely.wkt.loads)

path_processed_data = config_file['project_folder_path'] + 'processed_data/'

infrastructure_data = load_infrastructure_data(path_processed_data)
destination = get_destination(config_file)
complete_infrastructure = get_complete_infrastructure(infrastructure_data, destination)

path_processed_results = config_file['project_folder_path'] + 'results/processed_results/'

if 'processed_results' not in os.listdir(config_file['project_folder_path'] + 'results/'):
    os.makedirs(path_processed_results)

min_costs = {'total_costs': np.inf, 'transportation_costs': np.inf, 'conversion_costs': np.inf}
max_costs = {'total_costs': 0, 'transportation_costs': 0, 'conversion_costs': 0}

results_to_process = config_file_plotting['process_results']
geometry_results_to_process = config_file_plotting['process_results']

water_availability = gpd.read_file(
            path_processed_data + "water_availability.gpkg",
            layer="ptx_water_available"
        )

for folder in results_to_process:

    destination_object = None

    path = config_file['project_folder_path'] + 'results/unprocessed_results/' + folder + '/'

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

        if destination_object is None:
            destination_object = solution.loc['destination']

        status = solution.loc['status']
        starting_locations.append((solution.at['starting_longitude'], solution.at['starting_latitude']))

        start_point = Point([solution.at['starting_longitude'], solution.at['starting_latitude']])
        if folder in config_file_plotting['scenarios_with_water_availability_consideration']:
            site = gpd.GeoDataFrame(
                geometry=[start_point],
                crs="EPSG:4326"
            )

            match = gpd.sjoin(
                site,
                water_availability,
                predicate="within",
                how="left"
            )

            is_available = match.index_right.notna().iloc[0]

            if not is_available:
                data[number] = {'costs': math.inf,
                                'start_commodity': None,
                                'second_commodity': None,
                                'latitude': solution.at['starting_latitude'],
                                'longitude': solution.at['starting_longitude'],
                                'sec_distance_mean_combination': None,
                                'transportation_costs': math.inf,
                                'conversion_costs': math.inf,
                                'efficiency': 0}

                routes.append([])

                continue

        if status == 'no benchmark':
            data[number] = {'costs': math.inf,
                            'start_commodity': None,
                            'second_commodity': None,
                            'latitude': solution.at['starting_latitude'],
                            'longitude': solution.at['starting_longitude'],
                            'sec_distance_mean_combination': None,
                            'transportation_costs': math.inf,
                            'conversion_costs': math.inf,
                            'efficiency': 0}

            routes.append([])
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
            # data[number]['efficiency'] = float(solution.at['total_efficiency'])

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
            # commodities_list = solution.at['all_previous_commodities']
            transportation_means_list = solution.at['all_previous_transport_means']

            production_costs_f = production_costs.at[int(number), 'Hydrogen_Gas']

            if (float(production_costs_f) + float(conversion_costs_f) + float(transportation_costs_f)) != float(solution.at['current_total_costs']):
                # conversion costs at start are not considered
                conversion_costs_f = float(solution.at['current_total_costs']) - float(production_costs_f) - float(transportation_costs_f)

            data[number]['transportation_costs'] = transportation_costs_f
            data[number]['conversion_costs'] = conversion_costs_f
            data[number]['production_costs'] = production_costs_f

            route = ast.literal_eval(solution.loc['taken_routes'])

            efficiency = 1
            commodities_list = []
            cost_route = [('production', float(production_costs_f))]
            distance = 0
            commodity = None
            pos = 0
            for r_segment in route:
                if len(r_segment) == 2:
                    efficiency *= r_segment[1]
                    commodity = r_segment[0]

                    if commodity != 'Hydrogen_Gas':
                        commodities_list.append(('Hydrogen_Gas', 0))
                        cost_route.append(('conversion', solution.loc['all_previous_total_costs'][pos] - float(production_costs_f)))

                    pos += 1

                if len(r_segment) == 3:  # conversion
                    efficiency *= r_segment[-1]

                    if r_segment[0] != r_segment[1]:
                        commodities_list.append((commodity, distance))
                        distance = 0
                        commodity = r_segment[1]

                        cost_route.append(('conversion', solution.loc['all_previous_total_costs'][pos] - solution.loc['all_previous_total_costs'][pos-1]))

                    pos += 1

                if len(r_segment) == 5:
                    efficiency *= r_segment[-1]
                    distance += r_segment[2]

                    cost_route.append(('transport', solution.loc['all_previous_total_costs'][pos] -
                                       solution.loc['all_previous_total_costs'][pos - 1]))
                    pos += 1

                    if r_segment == route[-1]:
                        commodities_list.append((commodity, distance))

            routes.append(route)

            data[number]['quantity'] = production_costs.loc[number, 'Hydrogen_Gas_Quantity']
            data[number]['efficiency'] = float(solution.at['total_efficiency']) * 100
            data[number]['commodities'] = commodities_list
            data[number]['cost_route'] = cost_route

            try:
                data[number]['solving_time'] = solution.at['solving_time']  # todo: remove except since not necessary anymore in new results
            except:
                data[number]['solving_time'] = 0

    data = pd.DataFrame.from_dict(data,
                                  columns=['costs', 'start_commodity', 'second_commodity', 'latitude',
                                           'longitude', 'efficiency', 'transportation_costs', 'conversion_costs',
                                           'production_costs', 'quantity', 'commodities', 'cost_route', 'solving_time'],
                                  orient='index')

    data['routes'] = routes

    data.sort_index(inplace=True)

    data.to_csv(path_processed_results + folder + '_processed_results.csv')
    destination_object = pd.Series(destination_object)
    destination_object.to_csv(path_processed_results + folder + '_destination.csv')

    # data =data.iloc[0:100]

    if folder in config_file_plotting['categorical_routes']:
        i = 0
        for i in range(50, 251, 50):
            category = str(i) + ' to ' + str(i + 50) + ' €/MWh'
            affected_index = data[(data['costs'] > i) & (data['costs'] <= i + 50)].index
            data.loc[affected_index, 'commodity'] = category

        affected_index = data[data['costs'] > i + 50].index
        data.loc[affected_index, 'commodity'] = '> ' + str(i + 50) + ' €/MWh'

        create_weighted_routing_data_script(data, complete_infrastructure, infrastructure_data, path_processed_results,
                                            folder, column_to_sort='commodity')
