import pandas as pd

from algorithm import start_algorithm

# todo: point has x,y --> lon, lat


# Define path of data
path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/Daten/'

# Define path of result


# load configuration


# load input data
location_data = pd.read_excel(path_data + 'start_destination_combinations.xlsx', index_col=0)
commodity_conversion_data = pd.read_excel(path_data + 'commodities_conversions.xlsx', index_col=0)
commodity_transportation_data = pd.read_excel(path_data + 'commodities_transportation.xlsx', index_col=0)
pipeline_gas_geodata = pd.read_csv(path_data + 'gas_pipeline_geodata.csv', index_col=0)
pipeline_gas_graphs = pd.read_csv(path_data + 'gas_pipeline_graphs.csv', index_col=0)
pipeline_gas_graphs_objects = pd.read_csv(path_data + 'gas_pipeline_graphs_objects.csv', index_col=0)
railroad_geodata = pd.read_csv(path_data + 'railroad_geodata.csv', index_col=0)
railroad_graphs = pd.read_csv(path_data + 'railroad_graphs.csv', index_col=0)
railroad_graphs_objects = pd.read_csv(path_data + 'railroad_graphs_objects.csv', index_col=0)
ports = pd.read_excel(path_data + 'ports_processed.xlsx', index_col=0)

# Define assumptions
configuration = {'tolerance_distance': 1000,
                 'to_final_destination_tolerance': 10000,
                 'find_only_closest': True,
                 'Pipeline_Gas_New': {'follow_existing_roads': True,
                                      'use_direct_path': True,
                                      'max_length_new_segment': 10000},
                 'Pipeline_Liquid_New': {'follow_existing_roads': True,
                                         'use_direct_path': True,
                                         'max_length_new_segment': 10000},
                 'Railroad_New': {'follow_existing_roads': True,
                                  'use_direct_path': True,
                                  'max_length_new_segment': 10000}}

start_algorithm(configuration, location_data, commodity_conversion_data, commodity_transportation_data,
                pipeline_gas_geodata, pipeline_gas_graphs,
                railroad_geodata, railroad_graphs, ports)
