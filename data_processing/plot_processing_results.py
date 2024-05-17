import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiLineString
import yaml
import os
import random


def plot_original_pipeline_data():

    def plot_geometry_list():

        """
        method to plot shapely.geometry.LineString
        """

        def generate_random_colors(n):
            colors = []
            for _ in range(n):
                # Generate random RGB values
                r = random.random()
                g = random.random()
                b = random.random()
                # Append the color in hexadecimal format
                colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
            return colors

        c = generate_random_colors(len(lines))

        gdf = gpd.GeoDataFrame(geometry=lines)

        gdf.plot(color=c)

        plt.show()

    # read global energy monitor data
    data = pd.read_excel(path_raw_data + 'network_pipelines_gas.xlsx')

    # filter pipeline data based on status
    data = data.loc[data['Status'].isin(['Operating', 'Construction'])]

    # remove rows which have no geodata information
    data = data[data['WKTFormat'].notna()]
    empty_rows = data[data['WKTFormat'] == '--'].index.tolist()
    data.drop(empty_rows, inplace=True)

    lines = data['WKTFormat'].tolist()
    lines = [li for li in lines if li != '--']

    # construct geodataframe
    data_new = gpd.GeoDataFrame(pd.Series(lines).apply(shapely.wkt.loads), columns=['geometry'])
    data_new.set_geometry('geometry')
    data_new_exploded = data_new.explode(ignore_index=True)

    single_lines = data_new_exploded['geometry'].tolist()

    df_sl = pd.DataFrame(single_lines, columns=['single_lines'])
    df_sl = df_sl.drop_duplicates(['single_lines'])
    single_lines = [i for i in df_sl['single_lines'].tolist()]

    if config_file['use_minimal_example']:
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

        lines = new_single_lines

    plot_geometry_list()


def get_infrastructure_figure(sub_axes, boundaries, link_to_data, fig_title=''):

    """
    plots

    @param sub_axes:
    @param boundaries:
    @param link_to_data:
    @param fig_title:
    @return:
    """

    plt.rcParams.update({'font.size': 11,
                         'font.family': 'Times New Roman'})

    data_ports = 'ports.csv'
    data_pipeline_gas_lines = 'gas_pipeline_graphs.csv'
    data_pipeline_oil_lines = 'oil_pipeline_graphs.csv'
    data_pipeline_gas_nodes = 'gas_pipeline_node_locations.csv'
    data_pipeline_oil_nodes = 'oil_pipeline_node_locations.csv'

    data_ports = pd.read_csv(link_to_data + data_ports, index_col=0)
    data_pipeline_gas_lines = gpd.read_file(link_to_data + data_pipeline_gas_lines)
    data_pipeline_oil_lines = gpd.read_file(link_to_data + data_pipeline_oil_lines)
    data_pipeline_gas_nodes = gpd.read_file(link_to_data + data_pipeline_gas_nodes)
    data_pipeline_oil_nodes = gpd.read_file(link_to_data + data_pipeline_oil_nodes)

    for p in data_ports.index:
        data_ports.loc[p, 'geometry'] = Point([data_ports.loc[p, 'longitude'], data_ports.loc[p, 'latitude']])
    data_ports = gpd.GeoDataFrame(data_ports, geometry='geometry')

    data_pipeline_gas_lines['line'] = data_pipeline_gas_lines['line'].apply(shapely.wkt.loads)
    data_pipeline_gas_lines = data_pipeline_gas_lines.set_geometry('line')

    data_pipeline_oil_lines['line'] = data_pipeline_oil_lines['line'].apply(shapely.wkt.loads)
    data_pipeline_oil_lines = data_pipeline_oil_lines.set_geometry('line')

    gas_nodes = []
    for i in data_pipeline_gas_nodes.index:
        gas_nodes.append(Point([data_pipeline_gas_nodes.at[i, 'longitude'], data_pipeline_gas_nodes.at[i, 'latitude']]))
        # plt.text(x=float(data_pipeline_gas_nodes.at[i, 'longitude']), y=float(data_pipeline_gas_nodes.at[i, 'latitude']), s=data_pipeline_gas_nodes.at[i, 'field_1'])
    data_pipeline_gas_nodes['geometry'] = gas_nodes

    oil_nodes = []
    for i in data_pipeline_oil_nodes.index:
        oil_nodes.append(Point([data_pipeline_oil_nodes.at[i, 'longitude'], data_pipeline_oil_nodes.at[i, 'latitude']]))
    data_pipeline_oil_nodes['geometry'] = oil_nodes

    # plot map on axis
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    antarctica = countries[countries['continent'] == 'Antarctica'].index[0]
    countries.drop([antarctica], inplace=True)
    countries.plot(color="lightgrey", ax=sub_axes)

    data_ports.plot(color="blue", ax=sub_axes, markersize=1, label='Port')
    data_pipeline_gas_lines.plot(color="red", ax=sub_axes, linewidth=0.5, label='Gas Pipeline')
    data_pipeline_oil_lines.plot(color="black", ax=sub_axes, linewidth=0.5, label='Oil Pipeline')
    data_pipeline_gas_nodes.plot(color='orange', ax=sub_axes)
    data_pipeline_oil_nodes.plot(color='grey', ax=sub_axes)

    sub_axes.grid(visible=True, alpha=0.5)
    sub_axes.text(0.6, 0.05, fig_title, transform=sub_axes.transAxes, va='bottom', ha='left')

    # sub_axes.set_ylabel('')
    # sub_axes.set_xlabel('')
    # sub_axes.set_yticklabels([])
    # sub_axes.set_xticklabels([])
    # sub_axes.set_xticks([])
    # sub_axes.set_yticks([])

    sub_axes.set_ylim(boundaries['min_latitude'],
                      boundaries['max_latitude'])
    sub_axes.set_xlim(boundaries['min_longitude'],
                      boundaries['max_longitude'])

    return sub_axes


def plot_unprocessed_pipelines():

    def plot_geometry_list():

        """
        method to plot shapely.geometry.LineString

        @param lines: list of LineStrings
        @param same_colors: boolean if same colors are applied to all lines
        @param wait_plot: boolean if figure should be plotted immediately
        @param ax: matplotlib axis if axis defined outside of method should be used
        """

        def generate_random_colors(n):
            colors = []
            for _ in range(n):
                # Generate random RGB values
                r = random.random()
                g = random.random()
                b = random.random()
                # Append the color in hexadecimal format
                colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
            return colors

        c = generate_random_colors(len(lines))[0]

        gdf = gpd.GeoDataFrame(geometry=lines)

        gdf.plot(color=c, ax=ax)

    networks = os.listdir(path_processed_data + 'gas_network_data/')
    fig, ax = plt.subplots()
    for n in networks:

        network_csv = pd.read_csv(path_processed_data + 'gas_network_data/' + n, index_col=0, sep=';')
        lines = [shapely.wkt.loads(line) for line in network_csv['geometry']]

        plot_geometry_list()

    plt.show()

# load configuration file
path_config = os.path.dirname(os.getcwd()) + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_processed_data = config_file['project_folder_path'] + 'processed_data/'

if not config_file['use_minimal_example']:
    # use boundaries from config file
    min_lat = config_file['minimal_latitude']
    max_lat = config_file['maximal_latitude']
    min_lon = config_file['minimal_longitude']
    max_lon = config_file['maximal_longitude']
else:
    # if minimal example, set boundaries to Europe
    min_lat, max_lat = 35, 71
    min_lon, max_lon = -25, 45

boundaries = {'min_latitude': min_lat - 2,
              'max_latitude': max_lat + 2,
              'min_longitude': min_lon - 2,
              'max_longitude': max_lon + 2}

path_raw_data = config_file['project_folder_path'] + 'raw_data/'
path_techno_economic_data = config_file['project_folder_path'] + 'raw_data/'

# plot original data
# plot_original_pipeline_data()

# plot unprocessed pipelines
plot_unprocessed_pipelines()

# plot processed pipelines and ports
fig, ax = plt.subplots()
# ax = get_infrastructure_figure(ax, boundaries, path_processed_data)

plt.show()
