import os
import random
import ast
import multiprocessing
import itertools
import shapely

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mlp
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib as mpl

from math import sqrt
from tqdm import tqdm
from shapely.geometry import LineString, Point, Polygon, MultiLineString, MultiPolygon, box
from shapely.ops import linemerge
from joblib import Parallel, delayed
import plotly.graph_objects as go
from matplotlib.ticker import FixedLocator

import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import networkx as nx
import searoute as sr

import math

from data_processing.natural_earth_data import load_world, load_world_lowres


# from plotting.helpers_plotting import get_geometry_segments


def _load_plot_world(high_resolution=False):
    if high_resolution:
        world = load_world()
    else:
        world = load_world_lowres()

    rename_columns = {}
    if 'CONTINENT' in world.columns and 'continent' not in world.columns:
        rename_columns['CONTINENT'] = 'continent'
    if 'NAME_EN' in world.columns and 'name' not in world.columns:
        rename_columns['NAME_EN'] = 'name'
    elif 'NAME' in world.columns and 'name' not in world.columns:
        rename_columns['NAME'] = 'name'
    if rename_columns:
        world = world.rename(columns=rename_columns)

    return world


def _filter_world_to_boundaries(world, boundaries):
    boundary_box = box(boundaries['min_longitude'], boundaries['min_latitude'],
                       boundaries['max_longitude'], boundaries['max_latitude'])
    return world[world.geometry.intersects(boundary_box)].copy()


def _read_csv_or_empty(path, columns=None, index_col=0):
    if os.path.exists(path):
        return pd.read_csv(path, index_col=index_col)
    return pd.DataFrame(columns=columns or [])


def _read_geodata_or_empty(path, columns=None):
    if os.path.exists(path):
        return gpd.read_file(path)
    return gpd.GeoDataFrame(columns=columns or [], geometry='geometry', crs='EPSG:4326')


def get_routes_figure(data, line_styles, line_widths, commodity_colors, nice_name_dictionary,
                      infrastructure_data, complete_infrastructure, boundaries, destination_location, fig_title='',
                      add_legend=True,
                      ax=None, width=15.69, height=9, return_fig=False, save=False, path_saving='',
                      return_handles=False, existing_commodities=None, existing_transport_means=None,
                      add_fig_title=False, fig=None):

    plt.rcParams.update({'font.size': 9,
                         'font.family': 'Times New Roman'})
    centimeter_to_inch = 1 / 2.54

    if existing_commodities is None:
        existing_commodities = []

    if existing_transport_means is None:
        existing_transport_means = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * centimeter_to_inch, height * centimeter_to_inch))

    keys = []
    line_networks = {}
    commodities = []

    # avoid plotting routes twice --> use lists to check if already in plotting data
    processed_combinations = []
    processed_nodes = []
    processed_coordinates = []

    for i in data.index:
        routes = data.loc[i, 'routes']
        routes = ast.literal_eval(routes)

        start_longitude = data.loc[i, 'longitude']
        start_latitude = data.loc[i, 'latitude']

        commodity = None

        for m, r_segment in enumerate(routes):

            if m == 0:
                commodity = r_segment[0]
                commodities.append(commodity)
                continue

            elif (len(r_segment) == 2) | (len(r_segment) == 3):
                # conversion
                commodity = r_segment[1]
                commodities.append(commodity)
            else:
                # transportation
                start = r_segment[0]
                if isinstance(r_segment[1], float):
                    transport_mean = r_segment[2]
                else:
                    transport_mean = r_segment[1]

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

    new_transport_means = []
    new_commodities = []

    for t in transport_means:

        new_transport_means.append(mlines.Line2D([], [], color='black',
                                                 linestyle=line_styles[t], markersize=5, label=t))

        exists = False
        for i in existing_transport_means:
            if i._label == t:
                exists = True

        if not exists:
            existing_transport_means.append(mlines.Line2D([], [], color='black',
                                                          linestyle=line_styles[t], markersize=5,
                                                          label=t))

    for c in commodities:

        new_commodities.append(mlines.Line2D([], [], color=commodity_colors[c], marker='.',
                                             linestyle='None', markersize=5, label=nice_name_dictionary[c]))

        exists = False
        for i in existing_commodities:
            if i._label == nice_name_dictionary[c]:
                exists = True

        if not exists:
            existing_commodities.append(mlines.Line2D([], [], color=commodity_colors[c], marker='.',
                                                      linestyle='None', markersize=5,
                                                      label=nice_name_dictionary[c]))

    map_plot = _load_plot_world()
    antarctica = map_plot[map_plot['continent'] == 'Antarctica'].index[0]
    map_plot.drop([antarctica], inplace=True)
    map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])
    map_plot.plot(color='silver', ax=ax)

    order_plotting = [('Methane_Liquid', 'Road'), ('DBT', 'Road'), ('MCH', 'Road'), ('FTF', 'Road'), ('Hydrogen_Liquid', 'Road'),
                      ('Methanol', 'Road'), ('Ammonia', 'Road'), ('Hydrogen_Gas', 'Road'),
                      ('Methane_Liquid', 'Shipping'), ('DBT', 'Shipping'), ('MCH', 'Shipping'), ('FTF', 'Shipping'),
                      ('Hydrogen_Liquid', 'Shipping'), ('Ammonia', 'Shipping'), ('Methanol', 'Shipping'),
                      ('FTF', 'Pipeline_Liquid'), ('FTF', 'New_Pipeline_Liquid'), ('Methane_Gas', 'Pipeline_Gas'),
                      ('Methane_Gas', 'New_Pipeline_Gas'), ('Hydrogen_Gas', 'Pipeline_Gas'),
                      ('Hydrogen_Gas', 'New_Pipeline_Gas')]

    all_networks = []
    for k in order_plotting:
        if k not in keys:
            continue

        commodity = k[0]
        transport_mean = k[1]

        alpha = 1

        line_gdf = gpd.GeoDataFrame(line_networks[k], columns=['geometry'])
        line_gdf.plot(color=commodity_colors[commodity], linestyle=line_styles[transport_mean],
                      linewidth=line_widths[k], ax=ax, alpha=alpha,
                      path_effects=[pe.Stroke(linewidth=line_widths[k]*1.05, foreground='black'), pe.Normal()])
        line_gdf['color'] = commodity_colors[commodity]
        line_gdf['style'] = line_styles[transport_mean]
        line_gdf['width'] = line_widths[k]

        all_networks.append(line_gdf)

    # plot destination location / polygon
    if isinstance(destination_location, Point):
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=ax, color='red', s=10)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=ax, fc='none', ec='red', linewidth=0.5)

    ax.grid(visible=True, alpha=0.5)

    if add_fig_title:
        ax.text(0.4, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='left')

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

    if return_fig:
        if add_legend:
            ax.legend(handles=new_transport_means, loc='upper center', ncols=3,
                      bbox_to_anchor=(0.5, 0.2), title='Transport Mean',
                      labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

            ax.legend(handles=new_commodities, loc='upper center', ncol=2,
                      bbox_to_anchor=(0.5, 0.), title='Commodity',
                      labelspacing=0.1, handletextpad=0.1, columnspacing=0.25)

        if return_handles:
            return ax, existing_commodities, existing_transport_means
        else:
            return ax

    if add_legend:
        fig.legend(handles=new_transport_means, loc='upper center', ncols=3,
                   bbox_to_anchor=(0.5, 0.3), title='Transport Mean',
                   labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

        if len(commodities) <= 4:
            ncols = len(commodities)
        else:
            ncols = math.ceil(len(commodities) / 2)

        fig.legend(handles=new_commodities, loc='upper center', ncol=ncols,
                   bbox_to_anchor=(0.5, 0.175), title='Commodity',
                   labelspacing=0.1, handletextpad=0.1, columnspacing=0.25)

    if save:
        if fig is not None:
            fig.tight_layout()
            plt.subplots_adjust(bottom=0.3)

            fig.savefig(path_saving + fig_title + '.png', bbox_inches='tight', dpi=600)
            fig.savefig(path_saving + fig_title + '.svg', bbox_inches='tight')

            plt.close(fig)

            all_networks = pd.concat(all_networks)
            all_networks.to_excel(path_saving + fig_title + '.xlsx', index=True)


def get_weighted_routes(commodity_data, boundaries, line_styles, color_dictionary,
                        nice_name_dictionary, destination_location=None, fig_title='', add_legend=True, fig=None,
                        ax=None, width=15.69, height=9, return_fig=False, save=False, path_saving='',
                        return_handles=False, existing_commodities=None, existing_transport_means=None,
                        add_fig_title=False, ignore_commodity=False, country_comparison=False, color=None,
                        column='commodity', max_quantity=None, min_quantity=None):

    plt.rcParams.update({'font.size': 9,
                         'legend.title_fontsize': 9,
                         'font.family': 'Times New Roman'})

    if ignore_commodity:
        replacement_dict = {}
        new_keys = {k for k in commodity_data['geometry'].unique()}  # Identify new keys
        new_mapping = {key: i for i, key in enumerate(new_keys)}  # Create mappings for new keys
        replacement_dict.update(new_mapping)  # Update the replacement_dict in one step

        commodity_data['geometry'] = commodity_data['geometry'].map(replacement_dict)

        commodity_data = commodity_data.groupby(['geometry', 'transport_mean']).agg({"quantity": "sum"})
        commodity_data.reset_index(drop=False, inplace=True)
        commodity_data['commodity'] = 'None'

        reversed_dict = dict((v, k) for k, v in replacement_dict.items())
        commodity_data['geometry'] = commodity_data['geometry'].map(reversed_dict)

        commodity_data = commodity_data.sort_values(by=['quantity'], ascending=False)

    colors = ['yellowgreen', 'indianred', 'royalblue', 'gold', 'chocolate', 'hotpink']
    for n, c in enumerate(commodity_data[column].unique()):
        if c not in [*color_dictionary.keys()]:
            color_dictionary[c] = colors[n]
            nice_name_dictionary[c] = c

    if color is not None:
        for key in [*color_dictionary.keys()]:
            color_dictionary[key] = color

    # commodity_data = commodity_data.iloc[0:10]
    commodity_data = commodity_data[commodity_data['quantity'] > 0]

    if existing_commodities is None:
        existing_commodities = []

    if existing_transport_means is None:
        existing_transport_means = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(width / 2.54, height / 2.54))

    if max_quantity is None:
        max_quantity = commodity_data['quantity'].max()
    if min_quantity is None:
        min_quantity = commodity_data['quantity'].min()
    scale = 5 / max_quantity

    transport_means = ['Road', 'Shipping', 'Pipeline']
    commodities = commodity_data[column].unique().tolist()

    new_transport_means = []
    new_commodities = []

    for t in transport_means:

        new_transport_means.append(mlines.Line2D([], [], color='black',
                                                 linestyle=line_styles[t], markersize=5, label=t))

        exists = False
        for i in existing_transport_means:
            if i._label == t:
                exists = True

        if not exists:
            existing_transport_means.append(mlines.Line2D([], [], color='black',
                                                          linestyle=line_styles[t], markersize=5,
                                                          label=t))

    for c in commodities:

        new_commodities.append(mlines.Line2D([], [], color=color_dictionary[c], marker='.',
                                             linestyle='None', markersize=5, label=nice_name_dictionary[c]))

        exists = False
        for i in existing_commodities:
            if i._label == nice_name_dictionary[c]:
                exists = True

        if not exists:
            existing_commodities.append(mlines.Line2D([], [], color=color_dictionary[c], marker='.',
                                                      linestyle='None', markersize=5,
                                                      label=nice_name_dictionary[c]))

    ranges = [25, 50, 100]
    powers = range(12)
    target_quantity = 1
    for p in powers:
        for r in ranges:
            if 0.5 < max_quantity / (r * 10**p) < 1:
                target_quantity = r * 10**p

    sizes = []
    for s in [target_quantity * 0.25, target_quantity * 0.5, target_quantity]:
        sizes.append(mlines.Line2D([], [], color='black', linestyle='-', markersize=5,
                                   label=str(int(s / 1000000)), linewidth=sqrt(s * scale)))

    # plot results
    map_plot = _load_plot_world()
    antarctica = map_plot[map_plot['continent'] == 'Antarctica'].index[0]
    map_plot.drop([antarctica], inplace=True)

    map_plot = gpd.GeoDataFrame(geometry=map_plot['geometry'])
    map_plot.plot(color='gainsboro', ax=ax, alpha=0.75)

    # cmap = mpl.colormaps['coolwarm']
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["royalblue", "violet", "darkred"])
    norm = mpl.colors.Normalize(vmin=min_quantity, vmax=max_quantity)
    col = commodity_data.quantity.map(norm).map(cmap)
    commodity_data['color'] = col

    commodity_data.sort_values(by=['quantity'], inplace=True, ascending=True)
    for quantity in tqdm(commodity_data['quantity'].unique()):
        sub_data_q = commodity_data[commodity_data['quantity'] == quantity]

        if sub_data_q.empty:
            continue

        quantity_scaled = sqrt(quantity * scale)

        for c in sub_data_q[column].unique():
            sub_data_c = sub_data_q[sub_data_q[column] == c]

            if sub_data_c.empty:
                continue

            # color = color_dictionary[c]
            color = sub_data_q['color'].values[0]

            for t in sub_data_c['transport_mean'].unique():
                sub_data_t = sub_data_c[sub_data_c['transport_mean'] == t]

                if sub_data_t.empty:
                    continue

                if t == 'Road':
                    linestyles = 'dotted'
                else:
                    linestyles = '-'

                for n, g in enumerate([LineString, Point]):
                    sub_data_g = sub_data_t[sub_data_t['geometry'].apply(lambda x: isinstance(x, g))]

                    if sub_data_g.empty:
                        continue

                    geometries = sub_data_g['geometry']

                    if n == 0:

                        # simplified_geometries = []
                        # for geom in geometries:
                        #     simplified_geometries.append(geom.simplify(1, preserve_topology=True))
                        # geometries = simplified_geometries

                        multiline_geometries = MultiLineString(geometries.values.tolist())
                        multiline_geometries = shapely.line_merge(multiline_geometries)
                        # geometries = [multiline_geometries.simplify(0.1, preserve_topology=True)]
                        geometries = [multiline_geometries]

                        line_gdf = gpd.GeoDataFrame(geometry=geometries)
                        quantity = sqrt(quantity * scale)

                        # quantity = max(0.2, quantity)

                        # line_gdf.plot(color=color, linestyle=linestyles, linewidth=0.5, ax=ax)
                        line_gdf.plot(color=color, linestyle=linestyles, linewidth=quantity, ax=ax)

                        # line_gdf.plot(color=color, linestyle=linestyles,
                        #               linewidth=quantity, ax=ax,
                        #               path_effects=[pe.Stroke(linewidth=quantity_scaled, foreground='black'),
                        #                             pe.Normal()])

                    else:
                        point_gdf = gpd.GeoDataFrame(geometry=geometries)
                        ax.scatter(point_gdf.geometry.x, point_gdf.geometry.y, color=color, s=quantity_scaled,
                                   linewidths=0, marker='o')
                        # point_gdf.plot(color=color, ax=ax, markersize=quantity_scaled, edgecolor=None, marker='o')

    # plot destination location / polygon
    if destination_location is not None:
        if isinstance(destination_location, Point):
            destination_location_gdf = gpd.GeoDataFrame(geometry=[destination_location])
            destination_location_gdf.plot(ax=ax, color='red', s=10)
        else:
            destination_location_gdf = gpd.GeoDataFrame(geometry=[destination_location])
            destination_location_gdf.plot(ax=ax, fc='none', ec='red', linewidth=0.5)

    ax.grid(visible=True, alpha=0.5)
    if add_fig_title:
        ax.text(0.5, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='center')

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_ylim(boundaries['min_latitude'], boundaries['max_latitude'])
    ax.set_xlim(boundaries['min_longitude'], boundaries['max_longitude'])

    if country_comparison:
        legend_label = 'Country'
    else:
        legend_label = 'Commodity'

    if return_fig:
        if add_legend:
            ax.legend(handles=new_transport_means, loc='upper center', ncols=3,
                      bbox_to_anchor=(0.25, 0.2), title='Transport Mean',
                      labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

            ax.legend(handles=sizes, loc='upper center', ncols=3,
                      bbox_to_anchor=(0.75, 0.2), title='Quantity [TWh]',
                      labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

            if len(commodities) < 3:
                ncols = len(commodities)
            else:
                ncols = 3

            ax.legend(handles=new_commodities, loc='upper center', ncol=ncols,
                      bbox_to_anchor=(0.5, 0.1), title=legend_label,
                      labelspacing=0.1, handletextpad=0.1, columnspacing=0.25)

        if return_handles:
            return ax, existing_commodities, existing_transport_means
        else:
            return ax

    if add_legend:

        if False:

            fig.legend(handles=new_transport_means, loc='upper left', ncols=3,
                       bbox_to_anchor=(0.1, 0.395), title='Transport Mean',
                       labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1,
                       fontsize=9)

            fig.legend(handles=sizes, loc='upper right', ncols=3,
                       bbox_to_anchor=(0.9, 0.395), title='Quantity [TWh]',
                       labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1,
                       fontsize=9)

        else:
            shrink = 0.5

            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax,  orientation='horizontal', anchor=(0.5, 0), shrink=shrink, aspect=30, pad=0)

            cbar.ax.tick_params(labelsize=9)
            cbar.set_label('Quantity [TWh]', rotation=0, labelpad=5, fontsize=9)

            ticks = np.asarray(cbar.get_ticks(), dtype=float)
            vmin, vmax = norm.vmin, norm.vmax

            ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

            if len(ticks) >= 2:
                tick_dist = np.median(np.diff(ticks))

                # Minimum
                if ticks[0] - vmin > 0.5 * tick_dist:
                    ticks = np.insert(ticks, 0, vmin)
                else:
                    ticks[0] = vmin

                # Maximum
                if vmax - ticks[-1] > 0.5 * tick_dist:
                    ticks = np.append(ticks, vmax)
                else:
                    ticks[-1] = vmax

            else:
                ticks = np.array([vmin, vmax])

            ticks = np.unique(np.round(ticks, 0))

            cbar.set_ticks(ticks)

        if not ignore_commodity:

            if len(commodities) <= 4:
                ncols = len(commodities)
            else:
                ncols = math.ceil(len(commodities) / 2)

            fig.legend(handles=new_commodities, loc='upper center', ncol=ncols,
                       bbox_to_anchor=(0.5, 0.27), title=legend_label,
                       labelspacing=0.1, handletextpad=0.1, columnspacing=0.25,
                       fontsize=9)

    if save:
        if fig is not None:

            fig.tight_layout()
            plt.subplots_adjust(bottom=0.4)

            if ignore_commodity:
                name = path_saving + fig_title + '_no_com'
            else:
                name = path_saving + fig_title

            fig.savefig(name + '.png', bbox_inches='tight', dpi=600)
            fig.savefig(name + '.svg', bbox_inches='tight')

            plt.close(fig)


def get_number_figure(data, norm, cmap_chosen, boundaries, destination_location, ax=None,
                      width=15.69, height=9,
                      fig_title='', add_fig_title=False, column='costs', limit_scale=False, add_colorbar=True,
                      plot_era=False, use_voronoi=False, s=0.5, production_costs=None,
                      return_fig=False, save=False, save_path='', fig=None):

    centimeter_to_inch = 1 / 2.54
    mlp.rcParams.update({'font.size': 9,
                         'font.family': 'Times New Roman'})

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * centimeter_to_inch, height * centimeter_to_inch))

    countries = _load_plot_world()
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=ax)

    data = data[data['costs'] != math.inf]
    col = data[column].map(norm).map(cmap_chosen)

    if 'costs' in column:
        unit = '€ / MWh'
    else:
        unit = '%'

    data['color'] = col
    col = col.values.tolist()
    if use_voronoi:
        voronois = production_costs.loc[data.index, 'geometry'].tolist()
        voronois = gpd.GeoDataFrame(geometry=voronois)
        voronois.plot(ax=ax, color=col, ec='black', linewidth=0.01)
        # for color in data['color'].unique():
        #     affected_locations = data[data['color'] == color].index
        #     voronois = production_costs.loc[affected_locations, 'geometry'].tolist()
        #     voronois = gpd.GeoDataFrame(geometry=voronois)
        #     voronois.plot(ax=ax, color=color, ec='black', linewidth=0.01)
    elif not plot_era:
        data.plot(x="longitude", y="latitude", kind="scatter", c=col, ax=ax, s=s, linewidths=0)
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
            ax.add_patch(plt.Polygon(points, facecolor=color))

    # plot destination location / polygon
    if destination_location is not None:
        if isinstance(destination_location, Point):
            destination_location = gpd.GeoDataFrame(geometry=[destination_location])
            destination_location.plot(ax=ax, color='red', s=s)
        else:
            destination_location = gpd.GeoDataFrame(geometry=[destination_location])
            destination_location.plot(ax=ax, fc='none', ec='red', linewidth=0.5)

    ax.grid(visible=True, alpha=0.5)

    if add_fig_title:
        ax.text(0.5, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='center', size=9)

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

    if add_colorbar:
        if return_fig:
            shrink = 1
        else:
            shrink = 0.5

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        sm = plt.cm.ScalarMappable(cmap=cmap_chosen, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3%", pad=0.05)

        # pos = cax.get_position()
        #
        # cax.set_position([
        #     pos.x0 + pos.width * 0.05,  # etwas nach rechts
        #     pos.y0,
        #     pos.width * 0.9,  # 90% Breite
        #     pos.height
        # ])

        if limit_scale:
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='max', anchor=(0.5, 0), shrink=shrink,
                                aspect=30, pad=0)
        else:
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', anchor=(0.5, 0), shrink=shrink, aspect=30,
                                pad=0)

        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(unit, rotation=0, labelpad=5, fontsize=9)

        ticks = np.asarray(cbar.get_ticks(), dtype=float)
        vmin, vmax = norm.vmin, norm.vmax

        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

        if len(ticks) >= 2:
            tick_dist = np.median(np.diff(ticks))

            # Minimum
            if ticks[0] - vmin > 0.5 * tick_dist:
                ticks = np.insert(ticks, 0, vmin)
            else:
                ticks[0] = vmin

            # Maximum
            if vmax - ticks[-1] > 0.5 * tick_dist:
                ticks = np.append(ticks, vmax)
            else:
                ticks[-1] = vmax

        else:
            ticks = np.array([vmin, vmax])

        ticks = np.unique(np.round(ticks, 0))

        cbar.set_ticks(ticks)

        labels = cbar.ax.get_xticklabels()
        labels[0].set_horizontalalignment('left')

        if not limit_scale:
            labels[-1].set_horizontalalignment('right')

        # cbar.ax.xaxis.set_major_locator(FixedLocator(ticks))
        # cbar.update_ticks()

    if return_fig:
        return ax

    if save:
        if fig is not None:

            fig.tight_layout()
            # plt.subplots_adjust(bottom=0.2)

            # fig.savefig(save_path + fig_title + '.png', dpi=600)
            fig.savefig(save_path + fig_title + '.png', bbox_inches='tight', dpi=600)
            fig.savefig(save_path + fig_title + '.svg', bbox_inches='tight')

            plt.close(fig)

            data[['latitude', 'longitude', column]].to_excel(save_path + fig_title + '.xlsx')


def get_used_locations_figure(data, boundaries, destination_location, quantity, ax=None,
                              width=15.69, height=8, production_costs=None, add_legend=True,
                              fig_title='', add_fig_title=False, use_voronoi=False, s=0.5,
                              return_fig=False, save=False, save_path='', fig=None):

    centimeter_to_inch = 1 / 2.54
    mlp.rcParams.update({'font.size': 9,
                         'font.family': 'Times New Roman'})

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * centimeter_to_inch, height * centimeter_to_inch))

    countries = _load_plot_world()
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=ax)

    data.sort_values(by=['costs'], inplace=True)

    colors = ['yellowgreen', 'gold', 'indianred', 'royalblue']

    processed_locations = []
    for n, q in enumerate(quantity):
        used_quantity = 0
        used_locations = []

        for i in data.index:

            if i in processed_locations:
                continue

            loc_quantity = data.loc[i, 'quantity']
            used_locations.append(i)
            processed_locations.append(i)

            if used_quantity + loc_quantity >= q:

                col = colors[n]
                if use_voronoi:
                    voronois = production_costs.loc[used_locations, 'geometry'].tolist()
                    voronois = gpd.GeoDataFrame(geometry=voronois)
                    voronois.plot(ax=ax, color=col, ec='black', linewidth=0.05)

                else:
                    data.loc[used_locations, :].plot(x="longitude", y="latitude", kind="scatter", c=col,
                                                     ax=ax, s=s, linewidths=0)

                break
            else:
                used_quantity += loc_quantity

    # plot destination location / polygon
    if isinstance(destination_location, Point):
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=ax, color='red', s=s)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=ax, fc='none', ec='red', linewidth=0.5)

    ax.grid(visible=True, alpha=0.5)

    if add_fig_title:
        ax.text(0.4, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='left', size=9)

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

    quantity_handles = []
    for n, q in enumerate(quantity):

        quantity_handles.append(mlines.Line2D([], [], color=colors[n], marker='.',
                                              linestyle='None', markersize=5, label=str(q / 1000000)))

    if add_legend:
        fig.legend(handles=quantity_handles, loc='upper center', ncol=len(quantity_handles),
                   bbox_to_anchor=(0.5, 0.27), title='TWh',
                   labelspacing=0.1, handletextpad=0.1, columnspacing=0.25,
                   fontsize=9)

    if return_fig:
        return ax

    if save:
        if fig is not None:
            fig.savefig(save_path + fig_title + '.png', bbox_inches='tight', dpi=600)
            fig.savefig(save_path + fig_title + '.svg', bbox_inches='tight')

            plt.close(fig)


def get_cost_and_quantity_figure(sub_axes, data, norm, cmap_chosen, boundaries, destination_location, production_costs,
                                 fig_title='', cost_type='total_costs', s=0.5, return_fig=False, save=False,
                                 path_saving=''):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    countries = _load_plot_world()
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

    return None


def get_supply_curves(data, color_dictionary, nice_name_dictionary,
                      add_legend=True, return_fig=False, save=False, fig_title='', add_fig_title=False,
                      path_saving='', width=15.69, height=12, country=None, production_costs=None, ax=None, fig=None,
                      ylim=None, current_ax=None):

    mlp.rcParams.update({
        'font.size': 9,
        'font.family': 'Times New Roman'
    })

    mpl.rcParams['font.family'] = 'Times New Roman'

    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    # ------------------------------------------------------------
    # Create figure with two vertically stacked subplots
    # Top: cost supply curve
    # Bottom: commodity band
    # ------------------------------------------------------------

    cost_routes = []
    commodity_routes = []

    if fig is None:
        fig, (ax, ax_band) = plt.subplots(
            2, 1,
            figsize=(width / 2.54, height / 2.54),
            gridspec_kw={
                'height_ratios': [5, 1],
                'hspace': 0.75
            }
        )
    else:

        pos = ax.get_position()

        ax.set_position([
            pos.x0,
            pos.y0 + pos.height * 0.3,  # optional: keep upper edge fixed
            pos.width,
            pos.height * 0.7
        ])

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # Create band axis
        ax_band = inset_axes(
            ax,
            width="100%",
            height="20%",
            loc='lower left',
            bbox_to_anchor=(0, -0.6, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

    if country is not None:
        country_locations = production_costs[
            production_costs['country_start'] == country
            ].index.tolist()

        country_locations = list(
            set(country_locations).intersection(data.index.tolist())
        )

        data = data.loc[country_locations, :]

    data.sort_values(by=['costs'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    data['quantity'] = data['quantity'] * data['efficiency']

    data['quantity'] = data['quantity'] / 1000000

    max_value = data['quantity'].max()
    exponent = int(math.floor(math.log10(max_value)))

    divisor = 10 ** exponent

    data['quantity'] = data['quantity'] / divisor

    total_quantity = data['quantity'].sum()

    # ------------------------------------------------------------
    # Define cost columns and bar widths
    # ------------------------------------------------------------

    dynamic_widths = data['quantity']

    # ------------------------------------------------------------
    # Calculate bar start positions
    # ------------------------------------------------------------

    spacing = 0

    spacing_list = [0]
    last = 0

    for i in data.index[:-1]:
        spacing_list.append(data.loc[i, 'quantity'] + last + spacing)
        last = data.loc[i, 'quantity'] + last + spacing

    start_positions_with_spacing = spacing_list

    # ------------------------------------------------------------
    # Plot stacked cost bars
    # ------------------------------------------------------------

    bottom = np.zeros(len(dynamic_widths))

    for i, (start_pos, width) in enumerate(zip(start_positions_with_spacing, dynamic_widths)):

        cost_route = ast.literal_eval(data.loc[i, 'cost_route'])

        cost_routes.append(cost_route)

        for n in cost_route:
            cost_type = n[0]
            costs = n[1]

            if cost_type == 'production':
                color = 'cornflowerblue'
                label = 'Production Costs'
            elif cost_type == 'conversion':
                color = 'lightcoral'
                label = 'Conversion Costs'
            else:
                color = 'khaki'
                label = 'Transport Costs'

            used_labels = set()
            if label not in used_labels:
                current_label = label
                used_labels.add(label)
            else:
                current_label = None

            ax.bar(
                start_pos,
                costs,
                bottom=bottom[i],
                width=width,
                label=current_label,
                align='edge',
                color=color
            )

            bottom[i] += costs

    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', which='both', labelbottom=True)
    ax.set_xlabel(rf'Potential Quantity [10$^{{{exponent}}}$ TWh]', fontdict={'fontsize': 9}, fontname='Times New Roman')

    # ------------------------------------------------------------
    # Plot commodity band in separate subplot
    # ------------------------------------------------------------

    all_commodities = []

    for i, (start_pos, width) in enumerate(
            zip(start_positions_with_spacing, dynamic_widths)
    ):

        # initial band: Hydrogen gas
        bottom_band = 0

        ax_band.broken_barh(
            [(start_pos, width)],
            (bottom_band, 0.1),
            facecolors=color_dictionary.get('Hydrogen_Gas', 'lightgrey'),
            edgecolors='none',
            alpha=0.85
        )

        bottom_band += 0.1

        commodities_and_distances = ast.literal_eval(data.loc[i, 'commodities'])
        commodity_routes.append(commodities_and_distances)

        total_distance = 0
        for entry in commodities_and_distances:
            total_distance += entry[1]

        for entry in commodities_and_distances:

            commodity = entry[0]
            distance = entry[1]

            all_commodities.append(commodity)

            ax_band.broken_barh(
                [(start_pos, width)],
                (bottom_band, distance / total_distance * 0.9),
                facecolors=color_dictionary.get(commodity, 'lightgrey'),
                edgecolors='none',
                alpha=0.85
            )

            bottom_band += distance / total_distance * 0.89

    # ax_band.text(
    #     -0.01,
    #     1.0,
    #     'Start',
    #     transform=ax_band.transAxes,
    #     va='center',
    #     ha='right',
    #     fontsize=9
    # )
    #
    # ax_band.text(
    #     -0.01,
    #     0.0,
    #     'Destination',
    #     transform=ax_band.transAxes,
    #     va='center',
    #     ha='right',
    #     fontsize=9
    # )

    # ------------------------------------------------------------
    # Format main axis
    # ------------------------------------------------------------

    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    ax.set_xlim([0, total_quantity])

    if ylim is not None:
        ax.set_ylim([0, ylim])
    else:
        ax.set_ylim([0, bottom.max()])

    if current_ax is None:
        ax.set_ylabel('Costs [€ / MWh]', fontdict={'fontsize': 9})
    else:
        if current_ax in [1, 3]:
            ax.set_yticklabels([])

    # ------------------------------------------------------------
    # Format commodity band axis
    # ------------------------------------------------------------

    ax_band.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )

    ax_band.set_xlim([0, total_quantity])
    ax_band.set_ylim([0, 1])

    ax_band.set_yticks([])
    ax_band.set_xticks([])

    ax_band.set_ylabel('')
    ax_band.set_xlabel('')

    for spine in ax_band.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------
    # Add scenario title
    # ------------------------------------------------------------

    if add_fig_title:
        ax.text(
            0.5,
            0.08,
            fig_title,
            transform=ax.transAxes,
            va='center',
            ha='center',
            color='white',
            fontsize=9,
            fontweight='bold',
            zorder=10
        )

    # ------------------------------------------------------------
    # Add legends
    # ------------------------------------------------------------

    if add_legend:

        # Cost legend below the main plot
        handles, labels = ax.get_legend_handles_labels()

        labels = [
            'Production Costs',
            'Conversion Costs',
            'Transport Costs'
        ]

        cost_handles = [
            mlines.Line2D([], [], color='cornflowerblue', linewidth=6, label='Production Costs'),
            mlines.Line2D([], [], color='lightcoral', linewidth=6, label='Conversion Costs'),
            mlines.Line2D([], [], color='khaki', linewidth=6, label='Transport Costs')
        ]

        ax.legend(
            handles=cost_handles,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.2),
            ncols=3,
            loc='upper center'
        )

        # Commodity legend below the commodity band
        all_commodities = set(all_commodities)

        commodity_handles = []

        for c in color_dictionary.keys():
            if c in all_commodities:
                commodity_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=color_dictionary[c],
                        marker='s',
                        linestyle='None',
                        markersize=7,
                        label=nice_name_dictionary[c]
                    )
                )

        if len(commodity_handles) > 0:

            if len(commodity_handles) < 5:
                ncols = len(commodity_handles)
            else:
                ncols = 5

            ax_band.legend(
                handles=commodity_handles,
                loc='upper center',
                ncol=ncols,
                bbox_to_anchor=(0.5, -0.1),
                title='Commodities',
                labelspacing=0.1,
                handletextpad=0.1,
                columnspacing=0.25,
                fontsize=9,
                title_fontsize=9
            )

    # ------------------------------------------------------------
    # Return or save figure
    # ------------------------------------------------------------

    data = {'cost_routes': cost_routes,
            'commodity_routes': commodity_routes}
    data = pd.DataFrame(data, columns=['cost_routes', 'commodity_routes'])

    if return_fig:
        return fig

    if save:
        if fig is not None:
            fig.tight_layout()
            plt.subplots_adjust(bottom=0.28)

            fig.savefig(
                path_saving + fig_title + '.png',
                bbox_inches='tight',
                dpi=600
            )

            fig.savefig(
                path_saving + fig_title + '.svg',
                bbox_inches='tight'
            )

        data.to_excel(path_saving + fig_title + '.xlsx')


def get_production_costs_figure(sub_axes, data, norm, cmap_chosen, boundaries, destination_location,
                                fig_title='', plot_era=False, use_voronoi=False, s=0.5, production_costs=None):
    countries = _load_plot_world()
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


def get_energy_carrier_figure(data, boundaries, color_dictionary, nice_name_dictionary, destination_location, ax=None,
                              fig=None, width=15.69, height=8, add_fig_title=False, add_legend=True,
                              fig_title='', plot_era=False, use_voronoi=False, s=0.5, production_costs=None,
                              return_fig=False, save=False, path_saving='', return_handles=True,
                              existing_commodities=None):

    if existing_commodities is None:
        existing_commodities = []

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})
    centimeter_to_inch = 1 / 2.54

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * centimeter_to_inch, height * centimeter_to_inch))

    countries = _load_plot_world()
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="silver", ax=ax)

    data = data[data['costs'] != math.inf]
    col = data.start_commodity.map(color_dictionary)
    data['color'] = col

    if use_voronoi:
        voronois = production_costs.loc[data.index, 'geometry'].tolist()
        voronois = gpd.GeoDataFrame(geometry=voronois)
        voronois.plot(ax=ax, color=col.values.tolist(), ec='black', linewidth=0.01)
        # for color in data['color'].unique():
        #     affected_locations = data[data['color'] == color].index
        #     voronois = production_costs.loc[affected_locations, 'geometry'].tolist()
        #     voronois = gpd.GeoDataFrame(geometry=voronois)
        #     voronois.plot(ax=ax, color=color, ec='black', linewidth=0.1)
    elif not plot_era:
        data.plot(x="longitude", y="latitude", kind="scatter", c=col, ax=ax, s=s, linewidths=0)
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
            ax.add_patch(plt.Polygon(points, facecolor=color))

    # plot destination location / polygon
    if isinstance(destination_location, Point):
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=ax, color='red', s=s)
    else:
        destination_location = gpd.GeoDataFrame(geometry=[destination_location])
        destination_location.plot(ax=ax, fc='none', ec='red', linewidth=0.5)

    ax.grid(visible=True, alpha=0.5)

    if add_fig_title:
        ax.text(0.5, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='center', size=9)

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

    for commodity in data['start_commodity'].unique():

        exists = False
        for i in existing_commodities:
            if i._label == nice_name_dictionary[commodity]:
                exists = True

        if not exists:
            existing_commodities.append(mlines.Line2D([], [], color=color_dictionary[commodity],
                                                      marker='.', linestyle='None', markersize=5,
                                                      label=nice_name_dictionary[commodity]))

    if add_legend:
        if return_fig:
            bbox_to_anchor = (0.25, 0.)
        else:
            bbox_to_anchor = (0.5, 0.)

        # commodity legend
        ax.legend(handles=existing_commodities, loc='upper center', ncols=2, bbox_to_anchor=bbox_to_anchor,
                  labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=0.5, fontsize=9)

    if return_fig:
        if return_handles:
            return ax, existing_commodities
        else:
            return ax

    if save:
        if fig is not None:
            fig.tight_layout()
            fig.savefig(path_saving + fig_title + '.png', bbox_inches='tight', dpi=600)
            fig.savefig(path_saving + fig_title + '.svg', bbox_inches='tight')

            data[['latitude', 'longitude', 'start_commodity']].to_excel(path_saving + fig_title + '.xlsx')


def get_infrastructure_figure(boundaries, path_data, ax=None, fig=None, fig_title='', width=15.69, height=9,
                               return_fig=False, save=False, plot_legend=True, path_saving='',
                               country_edgecolor=None, country_linewidth=0.2, high_resolution_map=False):
    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})
    centimeter_to_inch = 1 / 2.54

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * centimeter_to_inch, height * centimeter_to_inch))

    data_ports = 'ports.csv'
    data_pipeline_gas = 'gas_pipeline_graphs.csv'
    data_pipeline_oil = 'oil_pipeline_graphs.csv'

    data_ports = _read_csv_or_empty(path_data + data_ports,
                                    columns=['latitude', 'longitude', 'name', 'country', 'continent'])
    data_pipeline_gas = _read_geodata_or_empty(path_data + data_pipeline_gas,
                                               columns=['graph', 'node_start', 'node_end', 'distance', 'line',
                                                        'geometry'])
    data_pipeline_oil = _read_geodata_or_empty(path_data + data_pipeline_oil,
                                               columns=['graph', 'node_start', 'node_end', 'distance', 'line',
                                                        'geometry'])

    if data_ports.empty:
        data_ports = gpd.GeoDataFrame(data_ports, geometry=[], crs='EPSG:4326')
    else:
        data_ports['geometry'] = [
            Point([data_ports.loc[p, 'longitude'], data_ports.loc[p, 'latitude']])
            for p in data_ports.index
        ]
        data_ports = gpd.GeoDataFrame(data_ports, geometry='geometry')

    if 'line' in data_pipeline_gas.columns and not data_pipeline_gas.empty:
        data_pipeline_gas['line'] = data_pipeline_gas['line'].apply(shapely.wkt.loads)
        data_pipeline_gas = data_pipeline_gas.set_geometry('line')
    else:
        data_pipeline_gas = gpd.GeoDataFrame(columns=list(data_pipeline_gas.columns) + ['geometry'],
                                             geometry='geometry', crs='EPSG:4326')

    if 'line' in data_pipeline_oil.columns and not data_pipeline_oil.empty:
        data_pipeline_oil['line'] = data_pipeline_oil['line'].apply(shapely.wkt.loads)
        data_pipeline_oil = data_pipeline_oil.set_geometry('line')
    else:
        data_pipeline_oil = gpd.GeoDataFrame(columns=list(data_pipeline_oil.columns) + ['geometry'],
                                             geometry='geometry', crs='EPSG:4326')

    # plot map on axis
    countries = _load_plot_world(high_resolution=high_resolution_map)
    countries = _filter_world_to_boundaries(countries, boundaries)
    antarctica = countries[countries['continent'] == 'Antarctica'].index
    countries.drop(antarctica, inplace=True)
    countries.plot(color="lightgrey", edgecolor=country_edgecolor, linewidth=country_linewidth, ax=ax)

    if not data_ports.empty:
        data_ports.plot(color="blue", ax=ax, markersize=1, label='Port')
    if not data_pipeline_gas.empty:
        data_pipeline_gas.plot(color="red", ax=ax, linewidth=0.5, label='Gas Pipeline')
    if not data_pipeline_oil.empty:
        data_pipeline_oil.plot(color="black", ax=ax, linewidth=0.5, label='Oil Pipeline')

    ax.grid(visible=True, alpha=0.5)
    ax.text(0.6, 0.05, fig_title, transform=ax.transAxes, va='bottom', ha='left')

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

    if plot_legend:
        # infrastructure legend
        handles_list_infrastructure = []
        if not data_ports.empty:
            handles_list_infrastructure.append(mlines.Line2D([], [], color='blue', marker='.',
                                                             linestyle='None', markersize=5,
                                                             label='Port'))
        if not data_pipeline_gas.empty:
            handles_list_infrastructure.append(mlines.Line2D([], [], color='red',
                                                             linestyle='-', markersize=5,
                                                             label='Gas Pipeline'))
        if not data_pipeline_oil.empty:
            handles_list_infrastructure.append(mlines.Line2D([], [], color='black',
                                                             linestyle='-', markersize=5,
                                                             label='Oil Pipeline'))

        if handles_list_infrastructure:
            ax.legend(handles=handles_list_infrastructure, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0),
                      labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=0.5, fontsize=9)

    if return_fig:
        return ax

    if save:
        if fig is not None:

            fig.tight_layout()
            plt.subplots_adjust(bottom=0.1)

            fig.savefig(path_saving + 'infrastructure.png', bbox_inches='tight', dpi=600)
            fig.savefig(path_saving + 'infrastructure.svg', bbox_inches='tight')

            plt.close(fig)


def get_tight_boundaries_for_start_locations_infrastructure_destination(start_locations, infrastructure_boundaries,
                                                                        destination_location,
                                                                        padding=0.1,
                                                                        padding_fraction=None,
                                                                        min_padding=None):
    """Create tight map boundaries with an absolute degree padding around all input geometries."""
    bounds = []

    if 'geometry' not in start_locations.columns:
        raise ValueError("Missing column 'geometry' in start locations. Voronoi cells are required for this plot.")

    start_geometries = start_locations['geometry'].dropna().copy()
    if not start_geometries.empty:
        start_geometries = start_geometries.apply(lambda geometry: shapely.wkt.loads(geometry)
                                                  if isinstance(geometry, str) else geometry)
        bounds.append(tuple(gpd.GeoSeries(start_geometries).total_bounds))

    if infrastructure_boundaries is not None:
        bounds.append((infrastructure_boundaries['min_longitude'],
                       infrastructure_boundaries['min_latitude'],
                       infrastructure_boundaries['max_longitude'],
                       infrastructure_boundaries['max_latitude']))

    if destination_location is not None and not destination_location.is_empty:
        bounds.append(destination_location.bounds)

    if not bounds:
        raise ValueError('Could not derive plot boundaries. Start locations, infrastructure bounds and destination are empty.')

    min_longitude = min(bound[0] for bound in bounds)
    min_latitude = min(bound[1] for bound in bounds)
    max_longitude = max(bound[2] for bound in bounds)
    max_latitude = max(bound[3] for bound in bounds)

    return {'min_latitude': max(min_latitude - padding, -90),
            'max_latitude': min(max_latitude + padding, 90),
            'min_longitude': max(min_longitude - padding, -180),
            'max_longitude': min(max_longitude + padding, 180)}


def get_start_locations_infrastructure_destination_figure(start_locations, boundaries, path_data, destination_location,
                                                          ax=None, fig=None, fig_title='start_locations_infrastructure_destination',
                                                          width=15.69, height=9, return_fig=False, save=False,
                                                          path_saving='', plot_legend=True):
    """Plot Voronoi start cells, infrastructure and destination without requiring optimization results."""
    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})
    centimeter_to_inch = 1 / 2.54

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * centimeter_to_inch, height * centimeter_to_inch))

    get_infrastructure_figure(boundaries, path_data, ax=ax, fig=fig, fig_title='', return_fig=True,
                              plot_legend=False, country_edgecolor='darkgrey', country_linewidth=0.25,
                              high_resolution_map=True)

    if 'geometry' not in start_locations.columns:
        raise ValueError("Missing column 'geometry' in start locations. Voronoi cells are required for this plot.")

    start_locations = start_locations.copy()
    start_locations['geometry'] = start_locations['geometry'].apply(lambda geometry: shapely.wkt.loads(geometry)
                                                                    if isinstance(geometry, str) else geometry)
    start_locations = start_locations[start_locations['geometry'].notna()]
    if start_locations.empty:
        raise ValueError('No start-location geometries found. The plot requires Voronoi cells in start_destination_combinations.csv.')

    start_locations = gpd.GeoDataFrame(start_locations, geometry='geometry')

    random_generator = random.Random(42)
    pastel_colors = [(0.55 + 0.45 * random_generator.random(),
                      0.55 + 0.45 * random_generator.random(),
                      0.55 + 0.45 * random_generator.random())
                     for _ in range(len(start_locations))]
    start_locations.plot(ax=ax, color=pastel_colors, edgecolor='white', linewidth=0.25, alpha=0.65,
                         label='Start location Voronoi cell')

    destination = gpd.GeoDataFrame(geometry=[destination_location])
    if isinstance(destination_location, Point):
        destination.plot(ax=ax, color='forestgreen', markersize=25, label='Destination')
    elif isinstance(destination_location, (Polygon, MultiPolygon)):
        destination.plot(ax=ax, facecolor='white', edgecolor='forestgreen', linewidth=1.25, label='Destination')
    else:
        destination.boundary.plot(ax=ax, color='forestgreen', linewidth=1.25, label='Destination')

    ax.set_ylim(boundaries['min_latitude'],
                boundaries['max_latitude'])
    ax.set_xlim(boundaries['min_longitude'],
                boundaries['max_longitude'])

    if plot_legend:
        handles_list = [mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=5,
                                      label='Port'),
                        mlines.Line2D([], [], color='red', linestyle='-', markersize=5,
                                      label='Gas Pipeline'),
                        mlines.Line2D([], [], color='black', linestyle='-', markersize=5,
                                      label='Oil Pipeline'),
                        mlines.Line2D([], [], color='forestgreen', linestyle='-', markersize=5,
                                      label='Destination')]
        ax.legend(handles=handles_list, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0),
                  labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=0.5, fontsize=9)

    if return_fig:
        return ax

    if save and fig is not None:
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        fig.savefig(path_saving + fig_title + '.png', bbox_inches='tight', dpi=600)
        fig.savefig(path_saving + fig_title + '.svg', bbox_inches='tight')

        plt.close(fig)


def get_commodity_transport_mean_histogram(data, color_dictionary, nice_names, path_saving, scenario_name):
    centimeter_to_inch = 1 / 2.54
    plt.rcParams.update({'font.size': 9,
                         'font.family': 'Times New Roman'})
    sorted_means = ['Road', 'Shipping', 'Pipeline_Gas', 'New_Pipeline_Gas', 'Pipeline_Liquid', 'New_Pipeline_Liquid']

    routes = data['routes'].tolist()

    commodity_data = {}
    for n, r in enumerate(routes):
        r = ast.literal_eval(r)

        commodity = None
        for m, r_segment in enumerate(r):

            if m == 0:
                commodity = r_segment[0]
                continue

            elif len(r_segment) == 3:
                # conversion
                commodity = r_segment[1]
                continue

            else:
                # transportation
                transport_mean = r_segment[1]
                distance = r_segment[2]
                if distance == 0:
                    continue

            if commodity not in list(commodity_data.keys()):
                commodity_data[commodity] = {}

            if transport_mean not in list(commodity_data[commodity].keys()):
                commodity_data[commodity][transport_mean] = []

            commodity_data[commodity][transport_mean].append(distance / 1000)

    num_commodities = len(commodity_data.keys())
    num_columns_total = 0
    for k1 in list(commodity_data.keys()):
        if len(commodity_data[k1].keys()) > num_columns_total:
            num_columns_total = len(commodity_data[k1].keys())

    fig_total, ax_total = plt.subplots(figsize=(15.69 * centimeter_to_inch, num_commodities * 3.553 * centimeter_to_inch))
    spec = gridspec.GridSpec(num_commodities, 14, figure=fig_total, wspace=8, hspace=0.75)
    plt.setp(ax_total.spines.values(), visible=False)
    ax_total.set_yticks([])
    ax_total.set_xticks([])

    data_dict = {}

    n = 0
    handels_list = []
    for k1 in list(color_dictionary.keys()):
        if k1 not in commodity_data.keys():
            continue

        handels_list.append(mlines.Line2D([], [], color=color_dictionary[k1], marker='.',
                                          linestyle='None', markersize=5,
                                          label=nice_names[k1]))

        number_columns = len(list(commodity_data[k1].keys()))
        if number_columns == 0:
            continue

        length = int(12 / number_columns)

        fig, ax = plt.subplots(nrows=1, ncols=number_columns, sharex='col', figsize=(15.69 * centimeter_to_inch,
                                                                                     4 * centimeter_to_inch))

        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.05)
        fig.suptitle('Distance [km]', x=0.5, y=0)

        i = 0
        col = 0
        for k2 in sorted_means:
            if k2 not in list(commodity_data[k1].keys()):
                continue

            data_dict[k1 + '_' + k2] = {}

            if number_columns == 1:
                sub_ax = ax
            else:
                sub_ax = ax[i]

            sub_ax.set_title(nice_names[k2])

            max_value = math.ceil(max(commodity_data[k1][k2]))
            min_value = math.floor(min(commodity_data[k1][k2]))

            ax_gs = fig_total.add_subplot(spec[n, 1+col:1+col+length])

            if len(list(set(commodity_data[k1][k2]))) == 1:
                # all values are equal

                max_value = int(max_value + 25)
                min_value = int(min_value - 25)

                if (max_value - min_value) % 50 != 0:
                    max_value += (max_value - min_value) % 50

                step_size = math.ceil((max_value - min_value) / 50)

                plot_old = sns.histplot(data=commodity_data[k1][k2], ax=sub_ax,
                                        bins=range(min_value, max_value, step_size), color=color_dictionary[k1])
                sub_ax.set_xlim(min_value, max_value)

                plot_new = sns.histplot(data=commodity_data[k1][k2], ax=ax_gs,
                                        bins=range(min_value, max_value, step_size), color=color_dictionary[k1])
                ax_gs.set_xlim(min_value, max_value)

            else:

                difference = math.ceil(max_value) - math.floor(min_value)

                if difference % 50 != 0:
                    to_add = 50 - difference % 50

                    if min_value - to_add / 2 < 0:
                        min_value = 0
                        to_add_from_min = abs(min_value - to_add / 2)
                        max_value += int(to_add / 2) + to_add_from_min
                    else:
                        max_value += int(to_add / 2)
                        min_value -= int(to_add / 2)

                step_size = math.ceil((max_value - min_value) / 50)
                if step_size < 2:
                    max_value += 25
                    min_value -= 25

                    step_size = math.ceil((max_value - min_value) / 50)

                bins = range(int(min_value), int(max_value), int(step_size))

                plot_old = sns.histplot(data=commodity_data[k1][k2], ax=sub_ax, bins=bins,
                                        color=color_dictionary[k1])
                plot_new = sns.histplot(data=commodity_data[k1][k2], ax=ax_gs, bins=bins,
                                        color=color_dictionary[k1])

                ax_gs.set_xlim(min_value, max_value)

            max_bin = 0
            for bar, b0, b1 in zip(plot_old.containers[0], bins[:-1], bins[1:]):
                if bar.get_height() > max_bin:
                    max_bin = bar.get_height()

                data_dict[k1 + '_' + k2][str(b0) + '_to_' + str(b1)] = bar.get_height()

            if i == 0:
                plot_old.set_ylabel('')
                plot_new.set_ylabel('')
                plot_old.set_ylabel('# occurrences')
                ax_gs.set_ylabel(nice_names[k1] + '\n# occurrences')
            else:
                plot_old.set_ylabel('')
                plot_new.set_ylabel('')
                ax_gs.set_ylabel('')

            ax_gs.set_title(nice_names[k2], fontsize=9)
            ax_gs.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
            ax_gs.tick_params(axis='both', which='major', labelsize=7)

            i += 1
            col += length

        n += 1

        # Save the plot as PNG and SVG
        if True:
            fig.savefig(path_saving + scenario_name + '_' + k1 + '_distances_histogram.png', bbox_inches='tight', dpi=600)
            fig.savefig(path_saving + scenario_name + '_' + k1 + '_distances_histogram.svg', bbox_inches='tight')

        plt.close(fig)
    fig_total.align_ylabels()

    plt.figtext(0.5, 0.05, 'Distance [km]', wrap=True, horizontalalignment='center', fontsize=9)

    fig_total.savefig(path_saving + scenario_name + '_distances_histogram.png', bbox_inches='tight', dpi=600)
    fig_total.savefig(path_saving + scenario_name + '_distances_histogram.svg', bbox_inches='tight')

    plt.close(fig_total)

    with pd.ExcelWriter(path_saving + scenario_name + '_distances_histogram.xlsx') as writer:

        for sheet_name, nested_dict in data_dict.items():
            # Inneres Dict -> DataFrame
            df = pd.DataFrame.from_dict(
                nested_dict,
                orient="index"
            )

            df.index.name = 'bin_distance_lb_to_ub'

            df.columns = ['occurence']

            # In eigenes Sheet schreiben
            df.to_excel(writer, sheet_name=sheet_name)


def get_calculation_time(result_files, results, path_saving, fig_title, nice_name_dictionary=None, width=15.69):

    centimeter_to_inch = 1 / 2.54
    plt.rcParams.update({'font.size': 9,
                         'font.family': 'Times New Roman'})
    if nice_name_dictionary is None:
        nice_name_dictionary = {}

    desired_height_figure = (len(results) * 0.75 + 1) * centimeter_to_inch
    figsize = (width * centimeter_to_inch, desired_height_figure)

    all_runtimes = []
    for n, r in enumerate(results):

        result_file = result_files[n]
        if r not in [*nice_name_dictionary.keys()]:
            result_file['scenario'] = r
        else:
            result_file['scenario'] = nice_name_dictionary[r]

        all_runtimes.append(result_file)

    all_runtimes = pd.concat(all_runtimes)
    all_runtimes['solving_time'] = all_runtimes['solving_time'] / 60

    # sns.set_theme(palette="pastel")

    flierprops = dict(marker='x', markerfacecolor='None', markersize=1, markeredgecolor='black')

    # Draw a nested boxplot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot = sns.boxplot(x="solving_time", y="scenario", data=all_runtimes, ax=ax, flierprops=flierprops)

    plt.subplots_adjust(left=0.215, bottom=0.2)
    plot.set_xscale("symlog")
    plot.set_xlabel("Minutes")
    plot.set_ylabel("Scenario")

    plot.set_xlim([0, all_runtimes['solving_time'].max() * 1.05])

    fig.savefig(path_saving + fig_title + '.png', bbox_inches='tight', dpi=600)


import plotly.graph_objects as go
from matplotlib.colors import to_rgba


import plotly.graph_objects as go
from matplotlib.colors import to_rgba


def get_sankey_diagram(ranked_routes, commodity_colors, nice_name_dictionary, path_saving, fig_title):

    PLOT_WIDTH_CM = 15.69
    PLOT_HEIGHT_CM = 25.0
    TOP_MARGIN_CM = 0.7
    LEGEND_HEIGHT_CM = 1.0
    ROUTE_GAP_CM = 0.7

    # Plotly Sankey uses 0 at the top and 1 at the bottom of each trace domain.
    SANKY_START_WITHIN_ROUTE = 0.25
    SANKY_HEIGHT_WITHIN_ROUTE = 0.75
    MAIN_FLOW_Y_IN_SANKY = 0.0
    LOSS_FLOW_Y_IN_SANKY = 5 / 6
    SECTION_LABEL_Y_WITHIN_ROUTE = 0.25
    COUNT_LABEL_Y_WITHIN_ROUTE = -0.3
    FINAL_EFFICIENCY_LABEL_Y_WITHIN_ROUTE = -0.3
    START_BLOCK_MAX_ROW_SHARE = 2 / 3

    NODE_ALPHA = 0.95
    FLOW_ALPHA = 0.50

    def color_with_alpha(color, alpha):
        r, g, b, _ = to_rgba(color)
        return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})"

    LOSS_NODE_COLOR = color_with_alpha('grey', NODE_ALPHA)
    LOSS_FLOW_COLOR = color_with_alpha('grey', FLOW_ALPHA)

    START_COMMODITY = ranked_routes["start_commodity"]
    routes = sorted(ranked_routes["routes"], key=lambda route: int(route["rank"]))

    labels = []
    node_colors = []
    node_x = []
    node_y = []
    sources = []
    targets = []
    values = []
    link_colors = []
    annotations = []
    shapes = []

    route_count = len(routes)
    plot_area_height_cm = PLOT_HEIGHT_CM - TOP_MARGIN_CM - LEGEND_HEIGHT_CM
    route_height_cm = (
        plot_area_height_cm - ROUTE_GAP_CM * max(route_count - 1, 0)
    ) / route_count

    plot_width_px = int(PLOT_WIDTH_CM / 2.54 * 96)
    plot_height_px = int(PLOT_HEIGHT_CM / 2.54 * 96)

    used_commodities = []

    for route_index, route in enumerate(routes):
        route_top_cm = TOP_MARGIN_CM + route_index * (route_height_cm + ROUTE_GAP_CM)
        route_bottom_cm = route_top_cm + route_height_cm
        sankey_top_cm = route_top_cm + route_height_cm * SANKY_START_WITHIN_ROUTE
        sankey_height_cm = route_height_cm * SANKY_HEIGHT_WITHIN_ROUTE

        sankey_row_top = sankey_top_cm / PLOT_HEIGHT_CM
        sankey_row_height = sankey_height_cm / PLOT_HEIGHT_CM

        main_flow_y = sankey_row_top + sankey_row_height * MAIN_FLOW_Y_IN_SANKY
        loss_flow_y = sankey_row_top + sankey_row_height * LOSS_FLOW_Y_IN_SANKY

        paper_row_top = 1 - route_top_cm / PLOT_HEIGHT_CM
        paper_row_bottom = 1 - route_bottom_cm / PLOT_HEIGHT_CM
        paper_row_height = paper_row_top - paper_row_bottom
        main_flow_label_y = paper_row_top - paper_row_height * SECTION_LABEL_Y_WITHIN_ROUTE
        count_label_y = paper_row_top - paper_row_height * COUNT_LABEL_Y_WITHIN_ROUTE
        final_efficiency_label_y = (
            paper_row_top - paper_row_height * FINAL_EFFICIENCY_LABEL_Y_WITHIN_ROUTE
        )

        ordered_sections = route["sections"]

        example_route = ast.literal_eval(route["example_route"])
        transport_distances = [
            float(step[2])
            for step in example_route
            if isinstance(step, tuple) and len(step) == 5
        ]

        total_available_weight = 10
        min_section_weight = 1

        conversion_count = sum(
            1
            for section in ordered_sections
            if section["section_type"] == "conversion"
        )

        transport_count = sum(
            1
            for section in ordered_sections
            if section["section_type"] == "transport"
        )

        available_transport_weight = total_available_weight - conversion_count * min_section_weight

        if transport_count > 0 and available_transport_weight < transport_count * min_section_weight:
            raise ValueError(
                "There are not enough remaining weights to give every transport section "
                "a minimum weight of 1."
            )

        total_transport_distance = sum(transport_distances)

        section_weights = []
        transport_index = 0

        for section in ordered_sections:
            if section["section_type"] == "conversion":
                section_weights.append(min_section_weight)

            elif section["section_type"] == "transport":
                if total_transport_distance > 0:
                    remaining_transport_weight = (
                            available_transport_weight - transport_count * min_section_weight
                    )

                    section_weights.append(
                        min_section_weight
                        + remaining_transport_weight
                        * transport_distances[transport_index]
                        / total_transport_distance
                    )
                else:
                    section_weights.append(available_transport_weight / transport_count)

                transport_index += 1

        total_weight = sum(section_weights) or 1

        base_color = commodity_colors[START_COMMODITY]

        current_node = len(labels)
        labels.append("")
        node_colors.append(color_with_alpha(base_color, NODE_ALPHA))
        node_x.append(0.01)
        node_y.append(main_flow_y)

        spacer_value = 100.0 * (1 / START_BLOCK_MAX_ROW_SHARE - 1)
        spacer_source = len(labels)
        labels.append("")
        node_colors.append("rgba(0, 0, 0, 0)")
        node_x.append(0.01)
        node_y.append(sankey_row_top + sankey_row_height * 0.985)

        spacer_target = len(labels)
        labels.append("")
        node_colors.append("rgba(0, 0, 0, 0)")
        node_x.append(0.99)
        node_y.append(sankey_row_top + sankey_row_height * 0.985)

        sources.append(spacer_source)
        targets.append(spacer_target)
        values.append(spacer_value)
        link_colors.append("rgba(0, 0, 0, 0)")

        current_value = 100.0
        cumulative_weight = 0

        SANKY_DOMAIN_X0 = 0.02
        SANKY_DOMAIN_X1 = 1.0

        if not ordered_sections:
            final_node = len(labels)
            labels.append("Final efficiency<br>100.00%")
            node_colors.append(color_with_alpha(base_color, NODE_ALPHA))
            node_x.append(0.99)
            node_y.append(main_flow_y)
            sources.append(current_node)
            targets.append(final_node)
            values.append(100.0)
            link_colors.append(color_with_alpha(base_color, FLOW_ALPHA))

        for step_position, section in enumerate(ordered_sections, start=1):
            kind = section["section_type"]
            label = section["section_label"]

            if 'Conversion' in label:
                label = 'Conv.'
            else:
                label = nice_name_dictionary[label.split(' ')[-1]]

            label = label.replace(' ', '<br>')

            to_commodity = section["to_commodity"]

            used_commodities.append(to_commodity)

            avg_efficiency = float(section["avg_section_efficiency"])
            commodity_color = commodity_colors[to_commodity]

            step_weight = section_weights[step_position - 1]
            previous_step_x = cumulative_weight / total_weight
            cumulative_weight += step_weight
            retained_value = current_value * avg_efficiency
            loss_value = max(current_value - retained_value, 0.0)
            next_node = len(labels)
            step_x = cumulative_weight / total_weight
            # process_label_x = previous_step_x + step_weight / total_weight / 2
            loss_label_x = (previous_step_x + 0.1)
            process_label_x_in_sankey = previous_step_x + step_weight / total_weight / 2
            process_label_x = (SANKY_DOMAIN_X0 + process_label_x_in_sankey * (SANKY_DOMAIN_X1 - SANKY_DOMAIN_X0))

            labels.append(
                ""
                if step_position == len(ordered_sections)
                else ""
            )
            node_colors.append(color_with_alpha(commodity_color, NODE_ALPHA))
            node_x.append(min(step_x, 0.99))
            node_y.append(main_flow_y)

            sources.append(current_node)
            targets.append(next_node)
            values.append(retained_value)
            link_colors.append(color_with_alpha(commodity_color, FLOW_ALPHA))

            if loss_value > 0:
                loss_node = len(labels)
                labels.append(f"Loss<br>{loss_value:.1f}%")
                node_colors.append(LOSS_NODE_COLOR)
                loss_x = 0.01 + 0.98 * max(min(loss_label_x, 0.96), 0.04)
                node_x.append(loss_x)
                node_y.append(loss_flow_y)

                sources.append(current_node)
                targets.append(loss_node)
                values.append(loss_value)
                link_colors.append(LOSS_FLOW_COLOR)

            annotations.append(
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": process_label_x,
                    "y": main_flow_label_y,
                    "text": f"<b>{label}</b>",
                    "showarrow": False,
                    "align": "center",
                    'xanchor': 'center',
                    "font": {"size": 7, "color": "rgba(25, 25, 25, 0.95)"},
                    "yanchor": "middle",
                }
            )

            current_node = next_node
            current_value = retained_value

        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 1,
                "y": final_efficiency_label_y,
                "text": f"<b>Final efficiency: {current_value:.1f}%</b>",
                "showarrow": False,
                "align": "right",
                "font": {"size": 9, "color": "rgba(25, 25, 25, 0.95)"},
                "xanchor": "right",
                "yanchor": "top",
            }
        )

        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.05,
                "y": count_label_y,
                "text": f"n={route['count']}",
                "showarrow": False,
                "align": "center",
                "font": {"size": 10, "color": "rgba(70, 70, 70, 0.9)"},
                "xanchor": "center",
                "yanchor": "top",
            }
        )

    traces = [
        go.Sankey(
            arrangement="fixed",
            domain={"x": [0.02, 1.0], "y": [0.0, 1.0]},
            node={
                "label": labels,
                "color": node_colors,
                "pad": 10,
                "thickness": 13,
                "line": {"color": "rgba(0, 0, 0, 0)", "width": 0},
                "x": node_x,
                "y": node_y,
            },
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": link_colors,
            },
            textfont={"size": 10},
            name="",
        )
    ]

    used_commodities = set(used_commodities)

    legend_items = [
        commodity
        for commodity in nice_name_dictionary.keys()
        if commodity in used_commodities and commodity != "UNKNOWN_COMMODITY"
    ]

    legend_items.append("__CONVERSION__")

    legend_step_x = 0.25
    legend_item_width = 0.13

    if len(legend_items) <= 4:
        legend_rows = [legend_items]
    else:
        first_row_count = (len(legend_items) + 1) // 2

        if len(legend_items) - first_row_count == 1:
            first_row_count -= 1

        legend_rows = [
            legend_items[:first_row_count],
            legend_items[first_row_count:],
        ]

    legend_y_values = [-0.035, -0.075]

    for row_index, legend_row in enumerate(legend_rows):
        row_width = (len(legend_row) - 1) * legend_step_x + legend_item_width
        row_start_x = 0.5 - row_width / 2

        for legend_index, commodity in enumerate(legend_row):
            x0 = row_start_x + legend_index * legend_step_x
            x1 = x0 + 0.018
            y_center = legend_y_values[row_index]

            if commodity == "__CONVERSION__":
                annotations.append(
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": x0,
                        "y": y_center,
                        "text": "Conv. = Conversion",
                        "showarrow": False,
                        "align": "left",
                        "font": {"size": 11, "color": "black"},
                        "xanchor": "left",
                        "yanchor": "middle",
                    }
                )
                continue

            color = color_with_alpha(commodity_colors[commodity], NODE_ALPHA)

            shapes.append(
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": y_center - 0.01,
                    "y1": y_center + 0.01,
                    "fillcolor": color,
                    "line": {"color": "rgba(40, 40, 40, 0.25)", "width": 0.5},
                }
            )

            annotations.append(
                {
                    "xref": "paper",
                    "yref": "paper",
                    "x": x1 + 0.006,
                    "y": y_center,
                    "text": nice_name_dictionary[commodity],
                    "showarrow": False,
                    "align": "left",
                    "font": {"size": 11},
                    "xanchor": "left",
                    "yanchor": "middle",
                }
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=plot_height_px,
        width=plot_width_px,
        margin={"l": 12, "r": 28, "t": 10, "b": 70},
        font={"size": 11},
        annotations=annotations,
        shapes=shapes,
    )

    fig.write_html(path_saving + "/" + fig_title + ".html", include_plotlyjs="cdn")

    fig.write_image(
        path_saving + "/" + fig_title + ".png",
        width=plot_width_px,
        height=plot_height_px,
        scale=3,
    )

    fig.write_image(
        path_saving + "/" + fig_title + ".svg",
        width=plot_width_px,
        height=plot_height_px,
    )
