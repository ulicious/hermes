import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import LineString, Point

import matplotlib.lines as mlines
import networkx as nx
import searoute as sr

import math


def get_routes_figure(sub_axes, routes, starting_locations, line_styles, line_widths, commodity_colors, nice_name_dictionary,
                      infrastructure_data, complete_infrastructure, boundaries, fig_title=''):

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    keys = []
    line_networks = {}
    commodities = []

    # avoid plotting routes twice --> use lists to check if already in plotting data
    processed_combinations = []
    processed_nodes = []
    processed_coordinates = []

    for n, r in enumerate(routes):

        start_longitude = None
        start_latitude = None
        commodity = None
        for m, r_segment in enumerate(r):

            if m == 0:
                commodity = r_segment

                start_longitude = starting_locations[n][0]
                start_latitude = starting_locations[n][1]

                continue

            elif len(r_segment) == 2:
                # conversion
                commodity = r_segment[1]
            else:
                # transportation
                start = r_segment[0]
                if isinstance(r_segment[1], float):
                    transport_mean = r_segment[2]
                else:
                    transport_mean = r_segment[1]

                commodities.append(commodity)

                distance = r_segment[2]
                if distance == 0:
                    continue

                destination = r_segment[3]

                if (commodity, transport_mean) not in list(line_networks.keys()):
                    line_networks[(commodity, transport_mean)] = []
                    keys.append((commodity, transport_mean))

                if (transport_mean in ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid']) & (m == 1):
                    end_longitude = complete_infrastructure.at[destination, 'longitude']
                    end_latitude = complete_infrastructure.at[destination, 'latitude']

                    line = LineString(
                        [Point([start_longitude, start_latitude]), Point([end_longitude, end_latitude])])
                    line_networks[(commodity, transport_mean)].append(line)

                elif (transport_mean in ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid']) & (m != 1):
                    start_longitude = complete_infrastructure.at[start, 'longitude']
                    start_latitude = complete_infrastructure.at[start, 'latitude']

                    end_longitude = complete_infrastructure.at[destination, 'longitude']
                    end_latitude = complete_infrastructure.at[destination, 'latitude']

                    line = LineString(
                        [Point([start_longitude, start_latitude]), Point([end_longitude, end_latitude])])
                    line_networks[(commodity, transport_mean)].append(line)

                elif transport_mean == 'Shipping':
                    if (start, destination, commodity) not in processed_combinations:
                        # Now add shipping from start port to destination port
                        start_location = [complete_infrastructure.loc[start, 'longitude'],
                                          complete_infrastructure.loc[start, 'latitude']]
                        end_location = [complete_infrastructure.loc[destination, 'longitude'],
                                        complete_infrastructure.loc[destination, 'latitude']]

                        route = sr.searoute(start_location, end_location, append_orig_dest=True)

                        if route.geometry['coordinates'][0][0] > 180:
                            direction = 'right_to_left'
                        else:
                            direction = 'left_to_right'

                        last_coordinate = None
                        line_coordinates = []
                        split_line = False
                        for coordinate in route.geometry['coordinates']:

                            if last_coordinate is not None:
                                if (last_coordinate, coordinate, commodity) not in processed_coordinates:

                                    if coordinate == route.geometry['coordinates'][1]:
                                        line_coordinates.append(last_coordinate)

                                    if (last_coordinate[0] == 180) & (coordinate[0] == 180):
                                        # when route crosses the 180° longitude, coordinates get higher than 180
                                        # --> not allowed

                                        # plot line which is on left side of graph or vice versa
                                        line = LineString(line_coordinates)
                                        line_networks[(commodity, transport_mean)].append(line)

                                        # start new line
                                        line_coordinates = []

                                        split_line = True

                                    if not split_line:
                                        if direction == 'left_to_right':
                                            line_coordinates.append(coordinate)
                                        else:
                                            line_coordinates.append((coordinate[0] - 360, coordinate[1]))
                                    else:
                                        if direction == 'left_to_right':
                                            line_coordinates.append((coordinate[0] - 360, coordinate[1]))
                                        else:
                                            line_coordinates.append(coordinate)

                                    processed_coordinates.append((last_coordinate, coordinate, commodity))
                                    processed_coordinates.append((coordinate, last_coordinate, commodity))

                            last_coordinate = coordinate

                        if len(line_coordinates) < 2:
                            continue

                        line = LineString(line_coordinates)

                        line_networks[(commodity, transport_mean)].append(line)

                        processed_combinations.append((start, destination, commodity))

                else:
                    if (start, destination, commodity) not in processed_combinations:

                        graph = complete_infrastructure.at[start, 'graph']

                        graph_object = infrastructure_data[transport_mean][graph]['Graph']

                        path = nx.shortest_path(graph_object, start, destination)
                        if len(path) < 2:
                            continue

                        line_coordinates = []
                        last_node = None
                        for node in path:

                            if last_node is not None:
                                if (last_node, node, commodity) not in processed_nodes:

                                    if node == path[1]:
                                        node_point = Point([complete_infrastructure.loc[last_node, 'longitude'],
                                                            complete_infrastructure.loc[last_node, 'latitude']])
                                        line_coordinates.append(node_point)

                                    node_point = Point([complete_infrastructure.loc[node, 'longitude'],
                                                        complete_infrastructure.loc[node, 'latitude']])

                                    line_coordinates.append(node_point)

                                    processed_nodes.append((last_node, node, commodity))
                                    processed_nodes.append((node, last_node, commodity))

                                else:
                                    if len(line_coordinates) > 1:
                                        line = LineString(line_coordinates)
                                        line_networks[(commodity, transport_mean)].append(line)

                                    node_point = Point([complete_infrastructure.loc[node, 'longitude'],
                                                        complete_infrastructure.loc[node, 'latitude']])
                                    line_coordinates = [node_point]

                            last_node = node

                        if len(line_coordinates) > 1:
                            line = LineString(line_coordinates)
                            line_networks[(commodity, transport_mean)].append(line)

                        processed_combinations.append((start, destination, commodity))
                        processed_combinations.append((destination, start, commodity))

    transport_means = ['Road', 'Shipping', 'Pipeline']
    commodities = sorted(list(set(commodities)))
    # commodities = ['Hydrogen_Gas', 'Ammonia', 'DBT', 'MCH', 'Hydrogen_Liquid', 'Methanol', 'Methane_Gas', 'Methane_Liquid', 'FTF']

    handels_list_transport_means = []
    handels_list_commodities = []
    for t in transport_means:
        handels_list_transport_means.append(mlines.Line2D([], [], color='black',
                                                          linestyle=line_styles[t], markersize=5,
                                                          label=t))

    for c in commodities:
        handels_list_commodities.append(mlines.Line2D([], [], color=commodity_colors[c], marker='.',
                                                      linestyle='None', markersize=5,
                                                      label=nice_name_dictionary[c]))

    map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    antarctica = map_plot[map_plot['continent'] == 'Antarctica'].index[0]
    map_plot.drop([antarctica], inplace=True)
    map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])
    map_plot.plot(color='silver', ax=sub_axes)

    order_plotting = [('Methane_Liquid', 'Road'), ('DBT', 'Road'), ('MCH', 'Road'), ('FTF', 'Road'), ('Hydrogen_Liquid', 'Road'),
                      ('Methanol', 'Road'), ('Ammonia', 'Road'), ('Hydrogen_Gas', 'Road'),
                      ('Methane_Liquid', 'Shipping'), ('DBT', 'Shipping'), ('MCH', 'Shipping'), ('FTF', 'Shipping'),
                      ('Hydrogen_Liquid', 'Shipping'), ('Ammonia', 'Shipping'), ('Methanol', 'Shipping'),
                      ('FTF', 'Pipeline_Liquid'), ('FTF', 'New_Pipeline_Liquid'), ('Methane_Gas', 'Pipeline_Gas'),
                      ('Methane_Gas', 'New_Pipeline_Gas'), ('Hydrogen_Gas', 'Pipeline_Gas'),
                      ('Hydrogen_Gas', 'New_Pipeline_Gas')]

    for k in order_plotting:
        if k not in keys:
            continue

        commodity = k[0]
        transport_mean = k[1]

        alpha = 1
        # if commodity in ['Ammonia', 'Methane_Liquid']:
        #     alpha = 0.5

        line_gdf = gpd.GeoDataFrame(line_networks[k], columns=['geometry'])
        line_gdf.plot(color=commodity_colors[commodity], linestyle=line_styles[transport_mean],
                      linewidth=line_widths[k], ax=sub_axes, alpha=alpha)

    sub_axes.grid(visible=True, alpha=0.5)
    sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')

    sub_axes.set_ylabel('')
    sub_axes.set_xlabel('')
    sub_axes.set_yticklabels([])
    sub_axes.set_xticklabels([])
    sub_axes.set_xticks([])
    sub_axes.set_yticks([])

    sub_axes.set_ylim(boundaries['min_latitude'],
                      boundaries['max_latitude'])
    sub_axes.set_xlim(boundaries['min_longitude'],
                      boundaries['max_longitude'])

    return sub_axes, handels_list_transport_means, handels_list_commodities


def get_cost_figure(sub_axes, data, norm, cmap_chosen, boundaries, fig_title='', cost_type='total_costs',
                    plot_era=False, s=0.5):

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=sub_axes)

    data = data[data['costs'] != math.inf]
    if cost_type == 'total_costs':
        col = data.costs.map(norm).map(cmap_chosen)
    elif cost_type == 'transportation_costs':
        col = data.transportation_costs.map(norm).map(cmap_chosen)
    else:
        col = data.conversion_costs.map(norm).map(cmap_chosen)

    if not plot_era:
        data.plot(x="longitude", y="latitude", kind="scatter", c=col, ax=sub_axes, s=s, linewidths=0)
    else:
        data['color'] = col
        for ind in data.index:
            x = data.loc[ind, 'longitude']
            y = data.loc[ind, 'latitude']

            points = np.array([[x - 0.125, y - 0.125],
                               [x - 0.125, y + 0.125],
                               [x + 0.125, y + 0.125],
                               [x + 0.125, y - 0.125]])

            color = data.at[ind, 'color']
            sub_axes.add_patch(plt.Polygon(points, facecolor=color))

    sub_axes.grid(visible=True, alpha=0.5)
    sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')

    sub_axes.set_ylabel('')
    sub_axes.set_xlabel('')
    sub_axes.set_yticklabels([])
    sub_axes.set_xticklabels([])
    sub_axes.set_xticks([])
    sub_axes.set_yticks([])

    sub_axes.set_ylim(boundaries['min_latitude'],
                      boundaries['max_latitude'])
    sub_axes.set_xlim(boundaries['min_longitude'],
                      boundaries['max_longitude'])

    return sub_axes


def get_production_costs_figure(sub_axes, data, norm, cmap_chosen, boundaries, fig_title='',
                                plot_era=False, s=0.5):
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=sub_axes)

    col = data.production_costs.map(norm).map(cmap_chosen)
    if not plot_era:
        data.plot(x="longitude", y="latitude", kind="scatter", c=col, ax=sub_axes, s=s, linewidths=0)
    else:
        data['color'] = col
        for ind in data.index:
            x = data.loc[ind, 'longitude']
            y = data.loc[ind, 'latitude']

            points = np.array([[x - 0.125, y - 0.125],
                               [x - 0.125, y + 0.125],
                               [x + 0.125, y + 0.125],
                               [x + 0.125, y - 0.125]])

            color = data.at[ind, 'color']
            sub_axes.add_patch(plt.Polygon(points, facecolor=color))

    sub_axes.grid(visible=True, alpha=0.5)
    sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')

    sub_axes.set_ylabel('')
    sub_axes.set_xlabel('')
    sub_axes.set_yticklabels([])
    sub_axes.set_xticklabels([])
    sub_axes.set_xticks([])
    sub_axes.set_yticks([])

    sub_axes.set_ylim(boundaries['min_latitude'],
                      boundaries['max_latitude'])
    sub_axes.set_xlim(boundaries['min_longitude'],
                      boundaries['max_longitude'])

    return sub_axes


def get_energy_carrier_figure(sub_axes, data, boundaries, color_dictionary, nice_name_dictionary, fig_title='',
                              plot_era=False, s=0.5):

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=sub_axes)

    col = data.start_commodity.map(color_dictionary)
    data['color'] = col

    if not plot_era:
        data.plot(x="longitude", y="latitude", kind="scatter", c=col, ax=sub_axes, s=s, linewidths=0)
    else:
        data['color'] = col
        for ind in data.index:
            x = data.loc[ind, 'longitude']
            y = data.loc[ind, 'latitude']

            points = np.array([[x - 0.125, y - 0.125],
                               [x - 0.125, y + 0.125],
                               [x + 0.125, y + 0.125],
                               [x + 0.125, y - 0.125]])

            color = data.at[ind, 'color']
            sub_axes.add_patch(plt.Polygon(points, facecolor=color))

    sub_axes.grid(visible=True, alpha=0.5)
    sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')

    sub_axes.set_ylabel('')
    sub_axes.set_xlabel('')
    sub_axes.set_yticklabels([])
    sub_axes.set_xticklabels([])
    sub_axes.set_xticks([])
    sub_axes.set_yticks([])

    sub_axes.set_ylim(boundaries['min_latitude'],
                      boundaries['max_latitude'])
    sub_axes.set_xlim(boundaries['min_longitude'],
                      boundaries['max_longitude'])

    commodity_handles = []
    for commodity in data['start_commodity'].unique():
        commodity_handles.append(mlines.Line2D([], [], color=color_dictionary[commodity], marker='.',
                                               linestyle='None', markersize=5,
                                               label=nice_name_dictionary[commodity]))

    return sub_axes, commodity_handles


def get_infrastructure_figure(sub_axes, boundaries, link_to_data, fig_title=''):
    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    data_ports = 'ports.csv'
    data_pipeline_gas = 'gas_pipeline_graphs.csv'
    data_pipeline_oil = 'oil_pipeline_graphs.csv'

    data_ports = pd.read_csv(link_to_data + data_ports, index_col=0)
    data_pipeline_gas = gpd.read_file(link_to_data + data_pipeline_gas)
    data_pipeline_oil = gpd.read_file(link_to_data + data_pipeline_oil)

    for p in data_ports.index:
        data_ports.loc[p, 'geometry'] = Point([data_ports.loc[p, 'longitude'], data_ports.loc[p, 'latitude']])
    data_ports = gpd.GeoDataFrame(data_ports, geometry='geometry')

    data_pipeline_gas['line'] = data_pipeline_gas['line'].apply(shapely.wkt.loads)
    data_pipeline_gas = data_pipeline_gas.set_geometry('line')

    data_pipeline_oil['line'] = data_pipeline_oil['line'].apply(shapely.wkt.loads)
    data_pipeline_oil = data_pipeline_oil.set_geometry('line')

    # plot map on axis
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="lightgrey", ax=sub_axes)

    data_ports.plot(color="blue", ax=sub_axes, markersize=1, label='Port')
    data_pipeline_gas.plot(color="red", ax=sub_axes, linewidth=0.5, label='Gas Pipeline')
    data_pipeline_oil.plot(color="black", ax=sub_axes, linewidth=0.5, label='Oil Pipeline')

    sub_axes.grid(visible=True, alpha=0.5)
    sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')

    sub_axes.set_ylabel('')
    sub_axes.set_xlabel('')
    sub_axes.set_yticklabels([])
    sub_axes.set_xticklabels([])
    sub_axes.set_xticks([])
    sub_axes.set_yticks([])

    sub_axes.set_ylim(boundaries['min_latitude'],
                      boundaries['max_latitude'])
    sub_axes.set_xlim(boundaries['min_longitude'],
                      boundaries['max_longitude'])

    return sub_axes