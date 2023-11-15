import itertools
import math

import pandas as pd

from _helpers import calc_distance_list_to_list

path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/'
path_railroad_data = path_data + 'railway_data/'
path_gas_pipeline_data = path_data + 'gas_pipeline_data/'
path_oil_pipeline_data = path_data + 'oil_pipeline_data/'

shipping_distances = pd.read_excel(path_data + 'port_distances.xlsx', index_col=0)
ship_to_gas_distance_road = pd.read_csv(path_data + 'port_to_gas_pipeline_distances.csv', index_col=0)
ship_to_ship_road_distance = pd.read_csv(path_data + 'port_to_port_road_distance.csv', index_col=0)
node_to_node_road_distance = pd.read_csv(path_data + 'node_to_node_road_distance.csv', index_col=0)

all_distances = pd.DataFrame(None, index=ship_to_gas_distance_road.index,
                             columns=ship_to_gas_distance_road.index.tolist())

n = 0
combinations = list(itertools.combinations(all_distances.index, 2))
for combination in combinations:
    print(round(n / len(combinations) * 100))
    n += 1

    i = combination[0]
    j = combination[1]

    if i in ship_to_gas_distance_road.index:
        if j in ship_to_gas_distance_road.columns:
            if ship_to_gas_distance_road.loc[i, j] is not None:
                all_distances.loc[i, j] = ship_to_gas_distance_road.loc[i, j]

    if i in ship_to_gas_distance_road.columns:
        if j in ship_to_gas_distance_road.index:
            if ship_to_gas_distance_road.loc[j, i] is not None:
                all_distances.loc[j, i] = ship_to_gas_distance_road.loc[j, i]

    if i in ship_to_ship_road_distance.index:
        if j in ship_to_ship_road_distance.columns:
            if ship_to_ship_road_distance.loc[i, j] is not None:
                all_distances.loc[i, j] = ship_to_ship_road_distance.loc[i, j]

    if i in ship_to_ship_road_distance.columns:
        if j in ship_to_ship_road_distance.index:
            if ship_to_ship_road_distance.loc[j, i] is not None:
                all_distances.loc[j, i] = ship_to_ship_road_distance.loc[j, i]

    if i in node_to_node_road_distance.index:
        if j in node_to_node_road_distance.columns:
            if node_to_node_road_distance.loc[i, j] is not None:
                all_distances.loc[i, j] = node_to_node_road_distance.loc[i, j]

    if i in node_to_node_road_distance.columns:
        if j in node_to_node_road_distance.index:
            if node_to_node_road_distance.loc[j, i] is not None:
                all_distances.loc[j, i] = node_to_node_road_distance.loc[j, i]

all_distances.to_csv(path_data + 'all_distances_road.csv')
