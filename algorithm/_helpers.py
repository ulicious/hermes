import random
import shapely

import geopandas as gpd
import matplotlib.pyplot as plt


def plot_geometry_list(lines, same_colors=False, wait_plot=False, ax=None):

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

    if same_colors:
        c = generate_random_colors(1)[0]
    else:
        c = generate_random_colors(len(lines))

    gdf = gpd.GeoDataFrame(geometry=lines)

    if ax is not None:
        gdf.plot(color=c, ax=ax)
    else:
        gdf.plot(color=c)

    if not wait_plot:
        plt.show()


def plot_geometry(geometry):
    """
    method to plot any shapely.geometry object

    @param geometry: shapely.geometry object
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

    c = generate_random_colors(len([geometry]))

    gdf = gpd.GeoDataFrame(geometry=[geometry])
    gdf.plot(color=c)
    plt.show()


def plot_subgraphs(graph_data, subgraph_nodes):

    """
    method to plot subgraphs of a graph. Mainly used for debugging of network issues

    @param graph_data: DataFrame with graph information (nodes, lines)
    @param subgraph_nodes: nodes of different subgraphs which seem disconnected from graph
    """

    fig, ax = plt.subplots()

    for nodes in subgraph_nodes:
        edge_index = graph_data[(graph_data['node_start'].isin(nodes)) | graph_data['node_end'].isin(nodes)].index
        lines = graph_data.loc[edge_index, 'line'].apply(shapely.wkt.loads).tolist()

        plot_geometry_list(lines, same_colors=True, wait_plot=True, ax=ax)

    plt.show()