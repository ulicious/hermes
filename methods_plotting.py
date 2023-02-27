import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#import pygal
import pandas as pd
from shapely.wkt import loads
from textwrap import wrap


def plot_line(line, direct_plot=True):
    # Read world map
    map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

    colors_used = []
    for ind in map_plot.index:
        colors_used.append('grey')

    colors_used.append('red')

    line_gdf = gpd.GeoDataFrame(line, index=range(1), columns=['geometry'])
    map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

    map_plot.plot(color=colors_used, linewidth=0.8)

    plt.xlabel('Längengrad', fontsize=4)
    plt.ylabel('Breitengrad', fontsize=4)

    if direct_plot:
        plt.show()


def plot_lines(lines, direct_plot=True, different_colors=False):
    # Read world map
    map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

    colors_used = []
    for ind in map_plot.index:
        colors_used.append('grey')

    for line in lines:
        line_gdf = gpd.GeoDataFrame(line, index=range(1), columns=['geometry'])

        if different_colors:
            colors_used.append(['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0])
        else:
            colors_used.append('red')

        map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

    map_plot.plot(color=colors_used, linewidth=0.8)

    plt.xlabel('Längengrad', fontsize=4)
    plt.ylabel('Breitengrad', fontsize=4)

    if direct_plot:
        plt.show()


def plot_lines_and_show_specific(lines, line_specific, direct_plot=True):
    # Read world map
    map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

    colors_used = []
    for ind in map_plot.index:
        colors_used.append('lightgrey')

    for line in lines:
        line_gdf = gpd.GeoDataFrame(line, index=range(1), columns=['geometry'])

        colors_used.append('grey')

        map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

    line_gdf = gpd.GeoDataFrame(line_specific, index=range(1), columns=['geometry'])
    colors_used.append('red')

    map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

    map_plot.plot(color=colors_used, linewidth=0.8)

    plt.xlabel('Längengrad', fontsize=4)
    plt.ylabel('Breitengrad', fontsize=4)

    if direct_plot:
        plt.show()


def plot_solution(s, direct_plot=True):
    # Read world map
    map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

    colors_used = []
    for ind in map_plot.index:
        colors_used.append('grey')

    colors = {'Road': 'red',
              'Shipping': 'darkblue',
              'Railroad': 'yellow',
              'Pipeline_Gas': 'purple',
              'Pipeline_Gas_New': 'indigo',
              'Pipeline_Liquid': 'green',
              'Pipeline_Liquid_New': 'turquoise'}

    lines = s.get_result_lines()
    means_of_transport = s.get_used_transport_means()

    for i, line in enumerate(lines):
        colors_used.append(colors[means_of_transport[i]])

    line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
    map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

    map_plot.plot(color=colors_used, linewidth=0.8)

    plt.xlabel('Längengrad', fontsize=4)
    plt.ylabel('Breitengrad', fontsize=4)

    if direct_plot:
        plt.show()


def plot_solution_path(solutions_dict, final_solution):
    import cv2
    import os

    path_data = 'C:/Users/mt5285/Documents/pictures_plot/'

    kept_solution = None
    all_past_solutions = []

    for i in [*solutions_dict.keys()]:

        fig = plt.figure()

        map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

        colors_used = []
        for ind in map_plot.index:
            colors_used.append('grey')

        colors = {'Road': 'red',
                  'Shipping': 'darkblue',
                  'Railroad': 'yellow',
                  'Pipeline_Gas': 'purple',
                  'Pipeline_Gas_New': 'indigo',
                  'Pipeline_Liquid': 'green',
                  'Pipeline_Liquid_New': 'turquoise'}

        # First, plot all solutions of past iterations in grey to keep track of them
        if all_past_solutions:
            for s in all_past_solutions:

                lines = s.get_result_lines()
                for j in lines:
                    colors_used.append('lightgrey')

                line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
                map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

            if kept_solution is not None:
                lines = kept_solution.get_result_lines()
                means_of_transport = kept_solution.get_used_transport_means()

                for j, line in enumerate(lines):
                    colors_used.append(colors[means_of_transport[j]])

                line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
                map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        # Second, plot solutions of current iteration to show progress
        for s in solutions_dict[i]:

            all_past_solutions.append(s)

            lines = s.get_result_lines()
            means_of_transport = s.get_used_transport_means()

            for j, line in enumerate(lines):
                colors_used.append(colors[means_of_transport[j]])

            line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
            map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        # Third, always plot final solution
        kept_solution = final_solution[i]
        lines = kept_solution.get_result_lines()
        means_of_transport = kept_solution.get_used_transport_means()

        for j, line in enumerate(lines):
            colors_used.append(colors[means_of_transport[j]])

        line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
        map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        plt.xlabel('Längengrad', fontsize=4)
        plt.ylabel('Breitengrad', fontsize=4)

        map_plot.plot(color=colors_used, linewidth=0.8)

        plt.savefig(path_data + 'iteration_' + str(i) + '.png', dpi=300)
        plt.close(fig)

    video_name = 'video.avi'

    images = [img for img in os.listdir(path_data) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path_data, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(path_data + video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(path_data, image)))

    cv2.destroyAllWindows()
    video.release()
