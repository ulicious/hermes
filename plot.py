import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
#import pygal
import pandas as pd
from shapely.wkt import loads
from textwrap import wrap

# ----------------------------------------------------------Erstellung Routen-Plots--------------------------------------------------------------------------------


def plot(continents, lines, name, directory, transport_modes, route_sections, country_start, country_dest):        # continents, lines, transport_modes as list

    # Read world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    if continents == ['world']:
        map = world
    else:
        pass

    # Select regions
        map = world.loc[world['continent'] == continents[0]]
        if len(continents) > 1:
            for c in range(1, len(continents)):
                continent = world.loc[world['continent'] == continents[c]]
                map = pd.concat([map, continent])
        map = map.copy()
        map.reset_index(drop=True, inplace=True)

    # Insert lines
    for l in range(len(lines)):
        route = 'route ' + str(l)
        map.at[route, 'geometry'] = loads(str(lines[l]))
        map.at[route, 'name'] = transport_modes[l]

    # set colors of plot
    new_pipe = Line2D([0], [0], color='limegreen', label='neue Pipeline', lw=1)
    ex_pipe = Line2D([0], [0], color='gold', label='bestehende Pipelines', lw=1)
    truck = Line2D([0], [0], color='black', label='Straßentransport', lw=1)
    ship = Line2D([0], [0], color='salmon', label='Schiffsroute', lw=1)
    start = Patch(color='cadetblue', label='Exportland ' + country_start)
    end = Patch(color='steelblue', label='Importland ' + country_dest)
    colors = []
    legend_list = []
    for r in range(0, len(map['name'])):
        # new pipeline:
        if map.iloc[r, 2] == 'new pipeline to destination' or map.iloc[r, 2] == 'new pipeline to port' or map.iloc[r, 2] == 'new pipeline to ex pipeline' \
                or map.iloc[r, 2] == 'new pipeline from existing pipeline to port' or map.iloc[r, 2] =='new pipeline from existing pipeline to destination':
            colors.append('limegreen')
            legend_list.append(new_pipe)
        # existing pipeline:
        elif map.iloc[r, 2] == 'pipeline usage to port' or map.iloc[r, 2] == 'pipeline usage as long as possible':
            colors.append('gold')
            legend_list.append(ex_pipe)
        # truck transport:
        elif map.iloc[r, 2] == 'truck transport to destination' or map.iloc[r, 2] == 'truck to port' or map.iloc[r, 2] == 'truck to ex pipeline' \
                or map.iloc[r, 2] == 'truck transport from existing pipeline to port' or map.iloc[r, 2] == 'truck transport from existing pipeline to destination':
            colors.append('black')
            legend_list.append(truck)
        # shipping:
        elif map.iloc[r, 2] == 'shipping':
            colors.append('salmon')
            legend_list.append(ship)
        elif map.iloc[r, 3] == country_start:
            colors.append('cadetblue')
            legend_list.append(start)
        elif map.iloc[r, 3] == country_dest:
            colors.append('steelblue')
            legend_list.append(end)
        else:
            colors.append('lightsteelblue')

    res = []
    [res.append(x) for x in legend_list if x not in res]
    legend_list = res.copy()
    map.plot(color=colors, linewidth=0.8)
    name = directory + '/' + name

    if '[' in route_sections:
        index_start_route = route_sections.index('[')
        index_end_route = route_sections.index(']')
        if len(route_sections[index_start_route:index_end_route+1]) > 100:
            pipeline_route = '\n'.join(wrap(route_sections[index_start_route:index_end_route+1], 170))
            route_sections = route_sections[:index_start_route - 1] + '\n' + pipeline_route + '\n' + route_sections[index_end_route + 1:]

    plt.title(r"$\bf{" + 'Route: ' + "}$" + str(route_sections), fontsize=4)
    plt.xlabel('Längengrad', fontsize=4)
    plt.ylabel('Breitengrad', fontsize=4)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.legend(handles=legend_list, fontsize=4)

    plt.savefig(name, bbox_inches='tight', dpi=300)
    # plt.show()


# Hilfsfunktion für schnelle Testplots
def just_plot(line):
    # Read world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Add searoute to world map
    world.at['searoute', 'geometry'] = line

    # set colors of plot
    colors = []
    for row in world.index:
        if row != world.index[-1]:
            colors.append('b')
        else:
            colors.append('r')
    # plot
    world.plot(color=colors)
    plt.show()

