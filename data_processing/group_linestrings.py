import itertools
import os
import math

import pandas as pd
import random
import shapely
import geopandas as gpd
from shapely import geometry, ops
from shapely.geometry import Point, Polygon

from joblib import Parallel, delayed
from tqdm import tqdm

from algorithm.methods_geographic import calc_distance_single_to_single, calc_distance_list_to_list

import warnings
warnings.filterwarnings('ignore')


def close_gaps(line_combinations, gap_distance=20000):

    # todo: this is way to simple. A lot of unnecessary lines are added

    """
    checks all combinations of lines if the distance between the two lines is less than 20,000. If so, a new line is
    added to close the gap

    @param list line_combinations:
    @param int gap_distance:
    @return: list of old lines including gap closing lines
    """

    new_lines = []
    for lines in tqdm(line_combinations):
        l1 = lines[0]
        l2 = lines[1]

        closest_points = ops.nearest_points(l1, l2)
        distance = calc_distance_single_to_single(closest_points[0].y, closest_points[0].x,
                                                  closest_points[1].y, closest_points[1].x)
        if 100 <= distance <= gap_distance:
            new_line_points = ops.nearest_points(l1, l2)
            l3 = geometry.LineString(new_line_points)

            new_lines.append(l3)

    return new_lines


def find_group_in_data(data):

    """
    Finds connected groups of lines within the given list of geometric line data.

    @param list data: A list containing geometric line data. The first two elements of the list represent
                      the initial groups of lines (group_one and group_two, respectively). Subsequent elements
                      are ignored.
    @return: A tuple containing two lists:
             - group_one: A list representing the first group of connected lines found in the data.
             - group_two: A list representing the residual lines.

    @rtype (list, list)
    """

    # input list with lines
    group_one = data[0]
    group_two = data[1]

    old_group = None
    while group_one != old_group:  # while loop stops as soon as group one has not changed since last loop

        old_group = group_one.copy()

        # each time an element has been added to a group, we restart the for loop as new element might be connected
        # to not connected elements
        broken = False

        # there might be a third line which connects l1 and l2
        l1 = None
        l2 = None
        l3 = None
        for l1 in group_one:
            for l2 in group_two:

                if l1 == l2:
                    # if same line --> ignore
                    continue

                if (l1.intersects(l2)) | (l1.equals(l2)):
                    # The different elements do not exist in both groups but at least two intersect
                    broken = True
                    break

            if broken:
                break

        if broken:
            # there has been an intersection or new line
            try:
                if l3 is None:
                    # lines did intersect
                    multi_line = geometry.MultiLineString([l1, l2])
                else:
                    # lines did not intersect but are close together --> new line added
                    multi_line = geometry.MultiLineString([l1, l2, l3])

                merged_line = ops.linemerge(multi_line)

            except:
                # l1 or l2 are several LineStrings and need to be split first
                def segments(curve):
                    return list(map(geometry.LineString, zip(curve.coords[:-1],
                                                             curve.coords[1:])))

                try:
                    line_segments_1 = segments(l1)
                    line_segments_2 = segments(l2)

                except:
                    # checks for undetected MultiLineStrings
                    data_new = gpd.GeoDataFrame([l1], columns=['geometry'])
                    data_new.set_geometry('geometry')
                    data_new_exploded = data_new.explode(ignore_index=True)

                    line_segments_1 = []
                    for line in data_new_exploded['geometry'].tolist():
                        new_line = segments(line)
                        for nl in new_line:
                            line_segments_1.append(nl)

                    data_new = gpd.GeoDataFrame([l2], columns=['geometry'])
                    data_new.set_geometry('geometry')
                    data_new_exploded = data_new.explode(ignore_index=True)

                    line_segments_2 = []
                    for line in data_new_exploded['geometry'].tolist():
                        new_line = segments(line)
                        for nl in new_line:
                            line_segments_2.append(nl)

                if l3 is None:
                    # lines did intersect
                    lines_list = line_segments_1 + line_segments_2
                else:
                    # lines did not intersect but are close together --> add new line
                    lines_list = line_segments_1 + line_segments_2 + [l3]

                multi_line = geometry.MultiLineString(lines_list)
                merged_line = ops.linemerge(multi_line)

            # remove lines from groups as merged line replaces them
            if l1 in group_one:
                group_one.remove(l1)

            if l2 in group_one:
                group_one.remove(l2)

            if l1 in group_two:
                group_two.remove(l1)

            if l2 in group_two:
                group_two.remove(l2)

            # add merged line to group one
            group_one.append(merged_line)

    return group_one, group_two


def group_LineStrings(name, num_cores, path_to_file, path_processed_data, use_minimal_example=False):

    """
    Reads global energy monitor data, filters pipeline data based on status, removes rows with no geodata information,
    constructs a GeoDataFrame, sorts lines into groups if they intersect, and saves processed data in a folder.

    @param str path_to_file: Path to the file containing network.
    @param str path_processed_data: Path to the directory containing the raw data files.
    @param str name: Name of the file containing the energy monitor data (without file extension).
    @param int num_cores: Number of CPU cores to use for parallel processing (default is 4).
    @param bool use_minimal_example: Indicates if only subset of data should be used
    """

    # read global energy monitor data
    data = pd.read_excel(path_to_file)

    # filter pipeline data based on status
    data = data.loc[data['Status'].isin(['Operating', 'Construction'])]

    # remove rows which have no geodata information
    data = data[data['WKTFormat'].notna()]
    empty_rows = data[data['WKTFormat'] == '--'].index.tolist()
    data.drop(empty_rows, inplace=True)

    # drop duplicates
    data.drop_duplicates(subset=['WKTFormat'], inplace=True)

    # construct geodataframe
    data_new = gpd.GeoDataFrame(pd.Series(data['WKTFormat'].tolist()).apply(shapely.wkt.loads), columns=['geometry'])
    data_new.set_geometry('geometry')

    data_new_exploded = data_new.explode(ignore_index=True)

    def divide_data(lines_local, chunk_size=100):

        """
        shuffles all lines and puts them into equal size lists

        @param list lines_local: list with all lines
        @param int chunk_size: number of lists which will be returned
        
        @return: nested list with all lists containing the lines
        @rtype: list
        """

        # Adjust chunk size s.t. multiprocessing is optimally used
        if len(lines_local) / chunk_size < 120:
            chunk_size = max(math.ceil(len(lines_local) / 120), 50)

        divided_data = []
        random.shuffle(lines_local)
        for i in range(0, len(lines_local), chunk_size):
            divided_data.append(lines_local[i:i+chunk_size])

        return divided_data

    single_lines = data_new_exploded['geometry'].tolist()

    df_sl = pd.DataFrame(single_lines, columns=['single_lines'])
    df_sl = df_sl.drop_duplicates(['single_lines'])
    single_lines = [i for i in df_sl['single_lines'].tolist()]

    data_new = gpd.GeoDataFrame(pd.Series(single_lines), columns=['geometry'])
    data_new.set_geometry('geometry')

    if use_minimal_example:
        # If the minimal example is applied, we set a frame on top of Europe and only consider pipelines within this frame

        x_split_point_left = -21
        x_split_point_right = 45

        y_split_point_top = 71
        y_split_point_bottom = 35

        frame_polygon = Polygon([Point(x_split_point_left, y_split_point_top),
                                 Point(x_split_point_right, y_split_point_top),
                                 Point(x_split_point_right, y_split_point_bottom),
                                 Point(x_split_point_left, y_split_point_bottom)])

        new_single_lines = []
        for line in single_lines:

            if line.intersects(frame_polygon):
                new_single_lines.append(line.intersection(frame_polygon))

        single_lines = new_single_lines

    # close gaps
    combinations = list(itertools.combinations(single_lines, 2))
    new_lines = close_gaps(combinations)

    single_lines += new_lines

    # find connections and intersections of data
    runs = 0
    while runs < 10:
        # the while loop is started as long as no changes have occurred within 10 runs

        old_sl = single_lines.copy()
        single_lines = divide_data(single_lines)  # shuffle data

        # process data
        inputs = []
        for s in single_lines:
            inputs.append([s.copy(), s.copy()])

        inputs = tqdm(inputs)
        results = Parallel(n_jobs=num_cores)(delayed(find_group_in_data)(i) for i in inputs)

        single_lines = []
        for result in results:
            single_lines = single_lines + result[0]

        if len(single_lines) == len(old_sl):
            # if no changes have occurred, increase run
            runs += 1
        else:
            # if lines have been connected, run is reset
            runs = 0

    # save processed data in folder (each network own folder)
    name_folder = path_processed_data + name + '_network_data/'
    if name + '_network_data' not in os.listdir(path_processed_data):
        os.mkdir(name_folder)

    network_number = 0
    for line_1 in single_lines:

        # todo: some lines have no length. Will be deleted later. However, could be done here

        new_gpd = gpd.GeoDataFrame()

        try:
            # line consists of several line segments
            i = 0
            for line_2 in line_1.geoms:
                new_gpd.loc[i, 'geometry'] = line_2
                i += 1
        except:
            new_gpd.loc[0, 'geometry'] = line_1

        # add geometries to geopandas dataframe and save them
        new_gpd.set_geometry('geometry')
        new_gpd.to_csv(name_folder + 'sorted_' + name + '_networks_' + str(network_number) + '.csv', sep=';')
        network_number += 1
