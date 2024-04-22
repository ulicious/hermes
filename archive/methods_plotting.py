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

import warnings
warnings.filterwarnings('ignore')


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

    lines = s.get_result_line()
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


def plot_solution_path(k, path_plot, solutions_dict, final_solution):
    import cv2
    import os

    if not os.path.exists(path_plot + '/' + str(k) + '/'):
        os.makedirs(path_plot + '/' + str(k) + '/')

    all_past_solutions = []

    for i in [*solutions_dict.keys()]:

        fig = plt.figure()

        map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

        colors_used = []
        for ind in map_plot.index:
            colors_used.append('grey')

        # First, plot all solutions of past iterations in grey to keep track of them
        if all_past_solutions:
            for s in all_past_solutions:

                result_line = s.get_result_line()

                lines = []
                for key in [*result_line.keys()]:
                    lines.append(result_line[key]['line'])
                    colors_used.append('lightgrey')

                line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
                map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        # Second, plot solutions of current iteration to show progress
        for s in solutions_dict[i]:

            if s is None:
                continue

            all_past_solutions.append(s)

            result_line = s.get_result_line()

            lines = []
            for key in [*result_line.keys()]:
                lines.append(result_line[key]['line'])
                colors_used.append(result_line[key]['color'])

            line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
            map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        # Third, always plot final solution
        kept_solution = final_solution[i]
        result_line = kept_solution.get_result_line()

        lines = []
        for key in [*result_line.keys()]:
            lines.append(result_line[key]['line'])
            colors_used.append(result_line[key]['color'])

        line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
        map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        plt.xlabel('Längengrad', fontsize=4)
        plt.ylabel('Breitengrad', fontsize=4)

        map_plot.plot(color=colors_used, linewidth=0.8)

        plt.savefig(path_plot + '/' + str(k) + '/iteration_' + str(i) + '.png', dpi=300)
        plt.close(fig)

    # print last solution as single solution. All else is grey
    if True:

        fig = plt.figure()

        map_plot = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        map_plot = gpd.GeoDataFrame(map_plot['geometry'], columns=['geometry'])

        colors_used = []
        for ind in map_plot.index:
            colors_used.append('grey')

        # plot all past solutions in light grey
        if all_past_solutions:
            for s in all_past_solutions:

                result_line = s.get_result_line()

                lines = []
                for key in [*result_line.keys()]:
                    lines.append(result_line[key]['line'])
                    colors_used.append('lightgrey')

                line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
                map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        # plot final solution
        result_line = kept_solution.get_result_line()

        lines = []
        for key in [*result_line.keys()]:
            lines.append(result_line[key]['line'])
            colors_used.append(result_line[key]['color'])

        line_gdf = gpd.GeoDataFrame(lines, index=range(len(lines)), columns=['geometry'])
        map_plot = pd.concat([map_plot, line_gdf], ignore_index=True)

        plt.xlabel('Längengrad', fontsize=4)
        plt.ylabel('Breitengrad', fontsize=4)

        map_plot.plot(color=colors_used, linewidth=0.8)

        plt.savefig(path_plot + '/' + str(k) + '/iteration_' + str(i+1) + '.png', dpi=300)
        plt.close(fig)

    video_name = 'video.avi'

    images = [img for img in os.listdir(path_plot + '/' + str(k) + '/') if img.endswith(".png")]
    frame = cv2.imread(os.path.join(path_plot + '/' + str(k) + '/', images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(path_plot + '/' + str(k) + '/' + video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(path_plot + '/' + str(k) + '/', image)))

    cv2.destroyAllWindows()
    video.release()
