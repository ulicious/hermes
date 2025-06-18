import random

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

import matplotlib.lines as mlines
import networkx as nx
import searoute as sr

import math


def get_routes_figure(sub_axes, routes, starting_locations, line_styles, line_widths, commodity_colors, nice_name_dictionary,
                      infrastructure_data, complete_infrastructure, boundaries, destination_location, fig_title='',
                      use_voronoi=False):

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

                        if route.geometry['coordinates'][0][0] < route.geometry['coordinates'][-1][0]:
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

                                    if (((last_coordinate[0] == 180) & (coordinate[0] == 180))
                                            | ((last_coordinate[0] == -180) & (coordinate[0] == -180))):
                                        # when route crosses the 180° longitude, coordinates get higher than 180
                                        # or vice versa
                                        # --> not allowed

                                        # finis line till 180 / -180
                                        if len(line_coordinates) > 1:
                                            line = LineString(line_coordinates)
                                            line_networks[(commodity, transport_mean)].append(line)

                                        # start new line
                                        line_coordinates = []

                                        split_line = True

                                    if not split_line:
                                        line_coordinates.append(coordinate)
                                    else:
                                        if direction == 'left_to_right':
                                            line_coordinates.append((coordinate[0] + 360, coordinate[1]))
                                        else:
                                            line_coordinates.append((coordinate[0] - 360, coordinate[1]))

                                    processed_coordinates.append((last_coordinate, coordinate, commodity))
                                    processed_coordinates.append((coordinate, last_coordinate, commodity))

                            last_coordinate = coordinate

                        if len(line_coordinates) > 1:
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

    # plot destination location / polygon
    if not use_voronoi:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, color='red', s=10)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, fc='none', ec='red', linewidth=0.5)

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


def get_cost_figure(sub_axes, data, norm, cmap_chosen, boundaries, destination_location,
                    fig_title='', cost_type='total_costs',
                    plot_era=False, use_voronoi=False, s=0.5, production_costs=None):

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

    data['color'] = col
    if use_voronoi:
        for color in data['color'].unique():
            affected_locations = data[data['color'] == color].index
            voronois = production_costs.loc[affected_locations, 'geometry'].tolist()
            voronois = gpd.GeoDataFrame(geometry=voronois)
            voronois.plot(ax=sub_axes, color=color, ec='black', linewidth=0.1)
    elif not plot_era:
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

    # plot destination location / polygon
    if not use_voronoi:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, color='red', s=s)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, fc='none', ec='red', linewidth=0.5)

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


def get_cost_and_quantity_figure(sub_axes, data, norm, cmap_chosen, boundaries, destination_location, production_costs,
                                 fig_title='', cost_type='total_costs', s=0.5):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    production_costs['quantity'] = [random.randint(0, 5) for u in production_costs.index]

    total_demand = 200

    data.sort_values(by=['costs'], inplace=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)

    for c in countries.index:
        poly = countries.loc[c, 'geometry']

        if isinstance(poly, Polygon):
            coords_3d = [(x, y, 0) for x, y in poly.exterior.coords]
            # Extract coordinates for plotting
            x, y, z = zip(*coords_3d)

            # Create the polygon for visualization
            verts = [list(zip(x, y, z))]
            poly_3d = Poly3DCollection(verts, color='silver')
            ax.add_collection3d(poly_3d)
        else:
            for sub_poly in poly.geoms:
                coords_3d = [(x, y, 0) for x, y in sub_poly.exterior.coords]
                x, y, z = zip(*coords_3d)

                # Create the polygon for visualization
                verts = [list(zip(x, y, z))]
                poly_3d = Poly3DCollection(verts, color='silver')
                ax.add_collection3d(poly_3d)

    # countries = gpd.GeoDataFrame(geometry=adjusted_countries)
    # countries.plot(ax=ax, color='silver')

    data = data[data['costs'] != math.inf]
    if cost_type == 'total_costs':
        col = data.costs.map(norm).map(cmap_chosen)
    elif cost_type == 'transportation_costs':
        col = data.transportation_costs.map(norm).map(cmap_chosen)
    else:
        col = data.conversion_costs.map(norm).map(cmap_chosen)

    data['color'] = col
    max_height = 0
    for i in data.index:
        voronoi = production_costs.loc[i, 'geometry']
        color = data.loc[i, 'color']

        # todo: adjust --> read from file and add efficiencies
        height = random.randint(0, 5) * data.loc[i, 'efficiency']
        if total_demand > 0:
            if total_demand < height:
                height = total_demand
                total_demand = 0
            else:
                total_demand -= height
        else:
            height = 0

        if height > max_height:
            max_height = height

        height = max(1, height)

        if isinstance(voronoi, Polygon):

            base_coords = [(x, y, 0) for x, y in voronoi.exterior.coords]
            top_coords = [(x, y, height) for x, y in voronoi.exterior.coords]

            sides = []
            for b in range(len(base_coords) - 1):  # Iterate over edges
                side = [base_coords[b], base_coords[b + 1], top_coords[b + 1], top_coords[b]]
                sides.append(side)

            # Add the base, top, and sides to the plot
            ax.add_collection3d(Poly3DCollection([base_coords], color=color))  # Base
            ax.add_collection3d(Poly3DCollection([top_coords], color=color))  # Top
            for side in sides:  # Sides
                ax.add_collection3d(Poly3DCollection([side], color=color))

        else:
            for geom in voronoi.geoms:
                base_coords = [(x, y, 0) for x, y in geom.exterior.coords]
                top_coords = [(x, y, height) for x, y in geom.exterior.coords]

                sides = []
                for b in range(len(base_coords) - 1):  # Iterate over edges
                    side = [
                        base_coords[b],
                        base_coords[b + 1],
                        top_coords[b + 1],
                        top_coords[b],
                    ]
                    sides.append(side)

                # Add the base, top, and sides to the plot
                ax.add_collection3d(Poly3DCollection([base_coords], color=color))  # Base
                ax.add_collection3d(Poly3DCollection([top_coords], color=color))  # Top
                for side in sides:  # Sides
                    ax.add_collection3d(Poly3DCollection([side], color=color))

    # voronois = gpd.GeoDataFrame(geometry=voronois)
    # voronois.plot(ax=ax, color=colors, ec='black', linewidth=0.1)

    # # plot destination location / polygon
    # if isinstance(destination_location, Point):
    #     destination_location = gpd.GeoDataFrame(geometry=[destination_location])
    #     destination_location.plot(ax=ax, color='red', s=s)
    # else:
    #     destination_location = gpd.GeoDataFrame(geometry=[destination_location])
    #     destination_location.plot(ax=ax, fc='none', ec='red', linewidth=0.5)

    # ax.grid(visible=True, alpha=0.5)
    # ax.text(0.6, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='left')

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylim(boundaries['min_latitude'],
                boundaries['max_latitude'])
    ax.set_xlim(boundaries['min_longitude'],
                boundaries['max_longitude'])
    ax.set_zlim(0, 50)

    # sub_axes.grid(visible=True, alpha=0.5)
    # sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')
    #
    # sub_axes.set_ylabel('')
    # sub_axes.set_xlabel('')
    # sub_axes.set_yticklabels([])
    # sub_axes.set_xticklabels([])
    # sub_axes.set_xticks([])
    # sub_axes.set_yticks([])
    #
    # sub_axes.set_ylim(boundaries['min_latitude'],
    #                   boundaries['max_latitude'])
    # sub_axes.set_xlim(boundaries['min_longitude'],
    #                   boundaries['max_longitude'])

    plt.show()

    return None


def get_supply_curves(data, production_costs):

    data['quantity'] = [random.randint(0, 100) for u in data.index]

    data.sort_values(by=['costs'], inplace=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    # countries = gpd.GeoDataFrame(geometry=adjusted_countries)
    # countries.plot(ax=ax, color='silver')

    # data = data[data['costs'] != math.inf]
    # if cost_type == 'total_costs':
    #     col = data.costs.map(norm).map(cmap_chosen)
    # elif cost_type == 'transportation_costs':
    #     col = data.transportation_costs.map(norm).map(cmap_chosen)
    # else:
    #     col = data.conversion_costs.map(norm).map(cmap_chosen)

    # data['color'] = col

    data.reset_index(drop=True, inplace=True)

    max_height = 0
    bottom = np.zeros(len(data.index))
    for column in ['production_costs', 'conversion_costs', 'transportation_costs']:
        ax.bar(data.index, data[column], bottom=bottom, label=column, width=data['quantity'], alpha=0.5)
        bottom += data[column]

    ax.plot(data['costs'], color='red', linewidth=2)

    # ax.set_ylabel('')
    # ax.set_xlabel('')
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # ax.set_ylim(boundaries['min_latitude'],
    #             boundaries['max_latitude'])
    # ax.set_xlim(boundaries['min_longitude'],
    #             boundaries['max_longitude'])
    # ax.set_zlim(0, 50)

    # sub_axes.grid(visible=True, alpha=0.5)
    # sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')
    #
    # sub_axes.set_ylabel('')
    # sub_axes.set_xlabel('')
    # sub_axes.set_yticklabels([])
    # sub_axes.set_xticklabels([])
    # sub_axes.set_xticks([])
    # sub_axes.set_yticks([])
    #
    # sub_axes.set_ylim(boundaries['min_latitude'],
    #                   boundaries['max_latitude'])
    # sub_axes.set_xlim(boundaries['min_longitude'],
    #                   boundaries['max_longitude'])

    plt.show()

    return None


def get_production_costs_figure(sub_axes, data, norm, cmap_chosen, boundaries, destination_location,
                                fig_title='', plot_era=False, use_voronoi=False, s=0.5, production_costs=None):
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=sub_axes)

    col = data.production_costs.map(norm).map(cmap_chosen)
    data['color'] = col
    if use_voronoi:
        for color in data['color'].unique():
            affected_locations = data[data['color'] == color].index
            voronois = production_costs.loc[affected_locations, 'geometry'].tolist()
            voronois = gpd.GeoDataFrame(geometry=voronois)
            voronois.plot(ax=sub_axes, color=color, ec='black', linewidth=0.1)
    elif not plot_era:
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

    # plot destination location / polygon
    if not use_voronoi:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, color='red', s=s)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, fc='none', ec='red', linewidth=0.5)

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


def get_energy_carrier_figure(sub_axes, data, boundaries, color_dictionary, nice_name_dictionary, destination_location,
                              fig_title='', plot_era=False, use_voronoi=False, s=0.5, production_costs=None):

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=sub_axes)

    col = data.start_commodity.map(color_dictionary)
    data['color'] = col

    if use_voronoi:
        for color in data['color'].unique():
            affected_locations = data[data['color'] == color].index
            voronois = production_costs.loc[affected_locations, 'geometry'].tolist()
            voronois = gpd.GeoDataFrame(geometry=voronois)
            voronois.plot(ax=sub_axes, color=color, ec='black', linewidth=0.1)
    elif not plot_era:
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

    # plot destination location / polygon
    if not use_voronoi:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, color='red', s=s)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=sub_axes, fc='none', ec='red', linewidth=0.5)

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
