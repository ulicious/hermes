import geopandas as gpd
import matplotlib.pyplot as plt


def create_random_colors(n):
    import random

    return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]


def plot_points_and_areas(p1=None, a1=None, p2=None, a2=None):

    fig, ax = plt.subplots()

    if a1 is not None:
        print(type(a1))
        print(a1)

        a1_gdf = gpd.GeoDataFrame(geometry=a1)
        a1_gdf.plot(ax=ax)

    if a2 is not None:
        a2_gdf = gpd.GeoDataFrame(geometry=a2)
        a2_gdf.plot(ax=ax, color='yellow')

    if p1 is not None:
        p1_gdf = gpd.GeoDataFrame(geometry=p1)
        p1_gdf.plot(ax=ax, color='red')

    if p2 is not None:
        p2_gdf = gpd.GeoDataFrame(geometry=p2)
        p2_gdf.plot(ax=ax, color='green')

    plt.show()
