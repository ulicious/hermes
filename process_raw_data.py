import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely import wkt
import shapely
import os

from vincenty import vincenty
from shapely.ops import unary_union

from methods_plotting import plot_lines, plot_lines_and_show_specific

if False:

    with open(path_data + 'seaports.geojson') as f:
        gj = geojson.load(f)
    features = gj['features']

    ports = pd.DataFrame(columns=['latitude', 'longitude', 'name', 'country', 'continent'])

    i = 0
    for port in features:
        ports.loc[i, 'longitude'] = port['geometry']['coordinates'][0]
        ports.loc[i, 'latitude'] = port['geometry']['coordinates'][1]
        ports.loc[i, 'name'] = port['properties']['name']
        ports.loc[i, 'country'] = port['properties']['country']
        ports.loc[i, 'continent'] = port['properties']['continent']

        i += 1
    ports.to_excel(path_data + 'ports_processed.xlsx')

    path_network_data


def get_geodata_and_graph_from_network_data(path_network_data, name_network):

    graph_number = 0
    node_number = 0
    edge_number = 0

    existing_nodes_dict = {}
    existing_edges_dict = {}
    existing_lines_dict = {}
    existing_graphs_dict = {}

    existing_nodes = []

    for file in os.listdir(path_network_data):

        railway_data = pd.read_csv(path_network_data + file, index_col=0, sep=';')
        lines = [shapely.wkt.loads(line) for line in railway_data['geometry']]

        # Split Multilinestring / Linestring into separate Linestrings if intersections exist
        result = unary_union(lines)
        existing_graphs_dict[graph_number] = result

        if isinstance(result, MultiLineString):

            for line in result.geoms:

                existing_lines_dict[edge_number] = line
                distance = 0

                coords = line.coords

                node_start = [round(coords.xy[0][0], 5), round(coords.xy[1][0], 5), graph_number]
                if node_start not in existing_nodes:
                    existing_nodes.append(node_start)
                    node_start_number = node_number
                    existing_nodes_dict[node_start_number] = node_start
                    node_number += 1

                else:
                    node_start_number = list(existing_nodes_dict.keys())[
                        list(existing_nodes_dict.values()).index(node_start)]

                node_end = [round(coords.xy[0][-1], 5), round(coords.xy[1][-1], 5), graph_number]
                if node_end not in existing_nodes:
                    existing_nodes.append(node_end)
                    node_end_number = node_number
                    existing_nodes_dict[node_end_number] = node_end
                    node_number += 1

                else:
                    node_end_number = list(existing_nodes_dict.keys())[
                        list(existing_nodes_dict.values()).index(node_end)]

                coords_before = None
                for i_x, x in enumerate(coords.xy[0]):
                    x = round(x, 5)
                    y = round(coords.xy[1][i_x], 5)

                    if coords_before is not None:

                        distance += vincenty((x, y), (coords_before[0], coords_before[1]))

                    coords_before = (x, y)

                existing_edges_dict[edge_number] = [graph_number, node_start_number, node_end_number, distance, line]
                existing_lines_dict[edge_number] = line
                edge_number += 1

            graph_number += 1

        else:
            existing_lines_dict[edge_number] = result
            coords = result.coords

            try:
                beeline_distance = vincenty((round(coords.xy[0][0], 5), round(coords.xy[1][0], 5)),
                                            (round(coords.xy[0][-1], 5), round(coords.xy[1][-1], 5)))
            except Exception:
                # Seems like some linestring are only a point. Skip
                continue

            if beeline_distance < 1000:
                continue

            distance = 0
            node_start = [round(coords.xy[0][0], 5), round(coords.xy[1][0], 5), graph_number]
            if node_start not in existing_nodes:
                existing_nodes.append(node_start)
                node_start_number = node_number
                existing_nodes_dict[node_start_number] = node_start
                node_number += 1
            else:
                node_start_number = list(existing_nodes_dict.keys())[
                    list(existing_nodes_dict.values()).index(node_start)]

            node_end = [round(coords.xy[0][-1], 5), round(coords.xy[1][-1], 5), graph_number]
            if node_end not in existing_nodes:
                existing_nodes.append(node_end)
                node_end_number = node_number
                existing_nodes_dict[node_end_number] = node_end
                node_number += 1
            else:
                node_end_number = list(existing_nodes_dict.keys())[
                    list(existing_nodes_dict.values()).index(node_end)]

            coords_before = None
            for i_x, x in enumerate(coords.xy[0]):
                x = round(x, 5)
                y = round(coords.xy[1][i_x], 5)

                if coords_before is not None:

                    distance += vincenty((x, y), (round(coords_before[0], 5), round(coords_before[1], 5)))

                coords_before = (x, y)

            existing_edges_dict[edge_number] = [graph_number, node_start_number, node_end_number, distance, result]
            existing_lines_dict[edge_number] = result
            edge_number += 1

            graph_number += 1

        print(graph_number)

    railway_graphs = pd.DataFrame.from_dict(existing_edges_dict, orient='index', columns=['graph',
                                                                                          'node_start',
                                                                                          'node_end',
                                                                                          'costs',
                                                                                          'line'])
    railway_geodata = pd.DataFrame.from_dict(existing_nodes_dict, orient='index', columns=['longitude',
                                                                                           'latitude',
                                                                                           'graph'])
    line_data = pd.DataFrame.from_dict(existing_lines_dict, orient='index', columns=['geometry'])

    line_data.to_csv(path_data + name_network + '_graphs_objects.csv')
    railway_graphs.to_csv(path_data + name_network + '_graphs.csv')
    railway_geodata.to_csv(path_data + name_network + '_geodata.csv')


path_data = '/home/localadmin/Dokumente/Daten_Transportmodell/'
path_railway_data = '/home/localadmin/Dokumente/Daten_Transportmodell/railway_data/'
path_gas_pipeline_data = '/home/localadmin/Dokumente/Daten_Transportmodell/gas_pipeline_data/'
path_oil_pipeline_data = '/home/localadmin/Dokumente/Daten_Transportmodell/oil_pipeline_data/'

# get_geodata_and_graph_from_network_data(path_railway_data, 'railroad')
get_geodata_and_graph_from_network_data(path_gas_pipeline_data, 'gas_pipeline')
get_geodata_and_graph_from_network_data(path_oil_pipeline_data, 'oil_pipeline')
