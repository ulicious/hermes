from _helpers import calc_distance_list_to_single
import pandas as pd
from tqdm import tqdm
import math


def calculate_and_save_all_distances(options):
    path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'

    for i in tqdm(options.index.tolist()):
        distances = calc_distance_list_to_single(options['latitude'], options['longitude'],
                                                 options.at[i, 'latitude'], options.loc[i, 'longitude'])

        distances = pd.DataFrame(distances, index=options.index, columns=[i])

        distances.to_hdf(path_data + '/direct_distances/direct_distances_' + i + '.h5', i, mode='w', format='table')


def calculate_and_save_shortest_distances(options, data):
    path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'

    for i in tqdm(options.index.tolist()):

        latitude = options.at[i, 'latitude']
        longitude = options.at[i, 'longitude']

        graphs = []
        distances = []

        for m in ['Pipeline_Gas', 'Pipeline_Liquid']:

            networks = data[m].keys()

            for n in networks:
                geodata = data[m][n]['GeoData']

                geodata['direct_distance'] = calc_distance_list_to_single(geodata['latitude'],
                                                                          geodata['longitude'],
                                                                          latitude, longitude)

                distance_to_closest = geodata['direct_distance'].min()

                graphs.append(n)
                distances.append(distance_to_closest)

        distances = pd.DataFrame(distances, index=graphs, columns=[i])

        distances.to_hdf(path_data + '/shortest_distances/shortest_distances_' + i + '.h5', i, mode='w', format='table')


def calculate_and_save_shortest_distance(options, data):
    path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'

    minimal_values = {}
    minimal_value_nodes = {}
    for i in tqdm(options.index.tolist()):

        minimal_values[i] = math.inf

        latitude = options.at[i, 'latitude']
        longitude = options.at[i, 'longitude']

        graph = options.at[i, 'graph']

        for m in ['Pipeline_Gas', 'Pipeline_Liquid']:

            networks = data[m].keys()

            for n in networks:

                if n == graph:
                    continue

                geodata = data[m][n]['GeoData']

                geodata['direct_distance'] = calc_distance_list_to_single(geodata['latitude'],
                                                                          geodata['longitude'],
                                                                          latitude, longitude)

                distance_to_closest = geodata['direct_distance'].min()
                closest_node = geodata['direct_distance'].idxmin()

                if distance_to_closest < minimal_values[i]:
                    minimal_values[i] = distance_to_closest
                    minimal_value_nodes[i] = closest_node

            shipping_infrastructure = data['Shipping']['ports'].copy()
            if i in shipping_infrastructure.index:
                shipping_infrastructure = shipping_infrastructure.drop([i])

            shipping_infrastructure['direct_distance'] = calc_distance_list_to_single(shipping_infrastructure['latitude'],
                                                                                      shipping_infrastructure['longitude'],
                                                                                      latitude, longitude)

            distance_to_closest = shipping_infrastructure['direct_distance'].min()
            closest_node = shipping_infrastructure['direct_distance'].idxmin()

            if distance_to_closest < minimal_values[i]:
                minimal_values[i] = distance_to_closest
                minimal_value_nodes[i] = closest_node

    distances = pd.DataFrame({'minimal_distance': minimal_values.values(),
                              'closest_node': minimal_value_nodes.values()},
                             index=minimal_values.keys())
    distances.to_csv(path_data + 'minimal_distances.csv')
