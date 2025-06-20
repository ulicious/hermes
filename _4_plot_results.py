import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import math
import ast
import shapely
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import geopandas as gpd
import cartopy.io.shapereader as shpreader

from shapely.geometry import Point

from plotting.helpers_plotting import load_data, get_complete_infrastructure
from plotting.get_figures import get_routes_figure, get_cost_figure, get_production_costs_figure, get_infrastructure_figure, \
    get_energy_carrier_figure, get_cost_and_quantity_figure, get_supply_curves

# load configuration file
path_config = os.getcwd() + '/algorithm_configuration.yaml'
yaml_file = open(path_config)
config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_config = os.getcwd() + '/plotting_configuration.yaml'
yaml_file = open(path_config)
config_file_plotting = yaml.load(yaml_file, Loader=yaml.FullLoader)

path_production_costs = config_file['project_folder_path'] + 'start_destination_combinations.csv'
production_costs = pd.read_csv(path_production_costs, index_col=0)

# # convert polygon strings to shapely objects
if config_file['use_voronoi_cells']:
    production_costs['geometry'] \
        = production_costs['geometry'].apply(shapely.wkt.loads)

path_processed_data = config_file['project_folder_path'] + 'processed_data/'

use_voronoi = config_file['use_voronoi_cells']
# create shapely object of destination
if config_file['destination_type'] == 'location':
    destination_location = Point(config_file['destination_location'])
else:
    destination_continent = config_file['destination_continent']

    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    world = gpd.read_file(country_shapefile)

    state_shapefile = shpreader.natural_earth(resolution='10m', category='cultural',
                                              name='admin_1_states_provinces')
    states = gpd.read_file(state_shapefile)

    country_states = config_file['destination_polygon']

    first = True
    for c in [*country_states.keys()]:
        if country_states[c]:
            for s in country_states[c]:
                if first:
                    destination_location = states[states['name'] == s]['geometry'].values[0]
                    first = False
                else:
                    destination_location.union(states[states['name'] == s]['geometry'].values[0])
        else:
            if first:
                destination_location = world[world['NAME_EN'] == c]['geometry'].values[0]
                first = False
            else:
                destination_location.union(world[world['NAME_EN'] == c]['geometry'].values[0])

infrastructure_data, destination = load_data(path_processed_data, config_file)
complete_infrastructure = get_complete_infrastructure(infrastructure_data, destination)

path_saving = config_file['project_folder_path'] + 'results/plots/'

min_costs = {'total_costs': np.inf, 'transportation_costs': np.inf, 'conversion_costs': np.inf}
max_costs = {'total_costs': 0, 'transportation_costs': 0, 'conversion_costs': 0}

path = config_file['project_folder_path'] + 'results/location_results/'

data = {}

filenames = os.listdir(path)

commodities = config_file['available_commodity']
transport_means = config_file['available_transport_means']

# commodity_data will store all transportation distances
commodity_distances = {}
for c in commodities:
    commodity_distances[c] = {'Road': [],
                              'Shipping': [],
                              'Pipeline_Gas': [],
                              'Pipeline_Liquid': [],
                              'New_Pipeline_Gas': [],
                              'New_Pipeline_Liquid': []}

common_combinations = {}

solutions_routes = {}

df_comparison = pd.DataFrame(index=filenames,
                             columns=['produced_energy_carrier', 'production_costs', 'conversion_costs', 'conversions',
                                      'transportation_costs', 'road_distance', 'ship_distance',
                                      'gas_pipeline_distance'])

commodity_data = {}

routes = []
starting_locations = []

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

for f in filenames:

    if 'final' not in f:
        continue

    solution = pd.read_csv(path + f, index_col=0)
    solution = solution[solution.columns[0]]
    number = int(f.split('_')[0])

    status = solution.loc['status']
    starting_locations.append((solution.at['starting_longitude'], solution.at['starting_latitude']))
    if status == 'no benchmark':
        data[number] = {'costs': math.inf,
                        'start_commodity': None,
                        'second_commodity': None,
                        'latitude': solution.at['starting_latitude'],
                        'longitude': solution.at['starting_longitude'],
                        'sec_distance_mean_combination': None,
                        'transportation_costs': math.inf,
                        'conversion_costs': math.inf}
    else:
        # read strings as lists
        solution.loc['all_previous_transportation_costs']\
            = ast.literal_eval(solution.loc['all_previous_transportation_costs'])
        solution.loc['all_previous_conversion_costs']\
            = ast.literal_eval(solution.loc['all_previous_conversion_costs'])
        solution.loc['all_previous_distances'] \
            = ast.literal_eval(solution.loc['all_previous_distances'])
        solution.loc['all_previous_commodities'] \
            = ast.literal_eval(solution.loc['all_previous_commodities'])
        solution.loc['all_previous_total_costs'] \
            = ast.literal_eval(solution.loc['all_previous_total_costs'])
        solution.loc['all_previous_transport_means'] \
            = ast.literal_eval(solution.loc['all_previous_transport_means'])

        data[number] = {'costs': None,
                        'start_commodity': None,
                        'second_commodity': None,
                        'latitude': None,
                        'longitude': None,
                        'sec_distance_mean_combination': None}

        data[number]['latitude'] = float(solution.at['starting_latitude'])
        data[number]['longitude'] = float(solution.at['starting_longitude'])
        data[number]['start_commodity'] = solution.at['all_previous_commodities'][0]
        data[number]['costs'] = float(solution.at['current_total_costs'])
        data[number]['efficiency'] = float(solution.at['total_efficiency'])

        if float(solution.at['current_total_costs']) < float(min_costs['total_costs']):
            min_costs['total_costs'] = float(solution.at['current_total_costs'])

        if float(solution.at['current_total_costs']) > float(max_costs['total_costs']):
            max_costs['total_costs'] = float(solution.at['current_total_costs'])

        if float(sum(solution.at['all_previous_transportation_costs'])) < float(min_costs['transportation_costs']):
            min_costs['transportation_costs'] = sum(solution.at['all_previous_transportation_costs'])

        if float(sum(solution.at['all_previous_transportation_costs'])) > float(max_costs['transportation_costs']):
            max_costs['transportation_costs'] = sum(solution.at['all_previous_transportation_costs'])

        if float(sum(solution.at['all_previous_conversion_costs'])) < float(min_costs['conversion_costs']):
            min_costs['conversion_costs'] = sum(solution.at['all_previous_conversion_costs'])

        if float(sum(solution.at['all_previous_conversion_costs'])) > float(max_costs['conversion_costs']):
            max_costs['conversion_costs'] = sum(solution.at['all_previous_conversion_costs'])

        starting_commodity_f = solution.at['all_previous_commodities'][0]

        # production_costs_f = solution.at['all_previous_total_costs'][0]
        conversion_costs_f = sum(solution.at['all_previous_conversion_costs'])
        transportation_costs_f = sum(solution.at['all_previous_transportation_costs'])
        commodities_list = solution.at['all_previous_commodities']
        transportation_means_list = solution.at['all_previous_transport_means']

        production_costs_f = production_costs.at[int(number), 'Hydrogen_Gas']

        if (float(production_costs_f) + float(conversion_costs_f) + float(transportation_costs_f)) != float(solution.at['current_total_costs']):
            # conversion costs at start are not considered
            conversion_costs_f = float(solution.at['current_total_costs']) - float(production_costs_f) - float(transportation_costs_f)

        data[number]['transportation_costs'] = transportation_costs_f
        data[number]['conversion_costs'] = conversion_costs_f
        data[number]['production_costs'] = production_costs_f

        routes.append(ast.literal_eval(solution.loc['taken_routes']))

boundaries = {'min_latitude': min_lat - 2,
              'max_latitude': max_lat + 2,
              'min_longitude': min_lon - 2,
              'max_longitude': max_lon + 2}

centimeter_to_inch = 1 / 2.54
plt.rcParams.update({'font.size': 9,
                     'font.family': 'Times New Roman'})

data = pd.DataFrame.from_dict(data,
                              columns=['costs', 'start_commodity', 'second_commodity', 'latitude',
                                       'longitude', 'efficiency', 'transportation_costs', 'conversion_costs', 'production_costs'],
                              orient='index')

min_costs = min(data['costs'].min(), data['production_costs'].min())
max_costs = max(data['costs'].max(), data['production_costs'].max())

cmap_chosen = mpl.colormaps['viridis_r']
norm = mpl.colors.Normalize(vmin=min_costs, vmax=max_costs)
sm = plt.cm.ScalarMappable(cmap=cmap_chosen, norm=norm)

height = 14
fig, ax = plt.subplots(2, 2, figsize=(15.69 * centimeter_to_inch, height * centimeter_to_inch))

plot_routes = False

# get_supply_curves(data, production_costs)

print('production costs')
production_costs_axis = ax[(0, 0)]
production_costs_axis = get_production_costs_figure(production_costs_axis, data, norm, cmap_chosen, boundaries,
                                                    destination_location, use_voronoi=use_voronoi,
                                                    production_costs=production_costs, s=10)

print('total costs')
total_costs_axis = ax[(0, 1)]
total_costs_axis = get_cost_figure(total_costs_axis, data, norm, cmap_chosen, boundaries, destination_location,
                                   cost_type='total_costs', use_voronoi=use_voronoi,
                                   production_costs=production_costs,  s=10)

color_dictionary = config_file_plotting['commodity_colors']
nice_name_dictionary = config_file_plotting['nice_name_dictionary']
transport_mean_line_styles = config_file_plotting['transport_mean_line_styles']

line_widths = config_file_plotting['line_widths']
line_widths_new = {}
for commodity in line_widths.keys():
    for transport_mean in line_widths[commodity].keys():
        line_widths_new[(commodity, transport_mean)] = line_widths[commodity][transport_mean]
line_widths = line_widths_new

# test_ax = get_cost_and_quantity_figure(None, data, norm, cmap_chosen, boundaries, destination_location,
#                                        production_costs, cost_type='total_costs', s=10)

if plot_routes:
    print('route')
    routes_axis = ax[(1, 0)]
    routes_axis, handles_list_transport_means, handles_list_commodities\
        = get_routes_figure(routes_axis, routes, starting_locations, transport_mean_line_styles, line_widths,
                            color_dictionary, nice_name_dictionary, infrastructure_data, complete_infrastructure,
                            boundaries, destination_location, use_voronoi=use_voronoi)

else:  # todo: man könnte jedem polygon einen z wert zuordnen und dann daraus ein 3d graph machen
    print('commodities')
    commodity_axis = ax[(1, 0)]
    commodity_axis, commodity_handles \
        = get_energy_carrier_figure(commodity_axis, data, boundaries, color_dictionary, nice_name_dictionary,
                                    destination_location, s=10, use_voronoi=use_voronoi,
                                    production_costs=production_costs)

    if True:
        # if we choose commodities, we still want the routes graph but don't plot in common graph
        fig_routes, routes_axis = plt.subplots()
        routes_axis, handles_list_transport_means, handles_list_commodities \
            = get_routes_figure(routes_axis, routes, starting_locations, transport_mean_line_styles, line_widths,
                                color_dictionary, nice_name_dictionary, infrastructure_data, complete_infrastructure,
                                boundaries, destination_location, use_voronoi=use_voronoi)

        fig_routes.legend(handles=handles_list_transport_means, loc='upper center', ncols=3,
                          bbox_to_anchor=(0.5, 0.22), title='Transport Mean',
                          labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

        fig_routes.legend(handles=handles_list_commodities, loc='upper center', ncol=2,
                          bbox_to_anchor=(0.5, 0.12), title='Commodity',
                          labelspacing=0.1, handletextpad=0.1, columnspacing=0.25)

        fig_routes.savefig(path_saving + '_routes.png', bbox_inches='tight', dpi=600)
        fig_routes.savefig(path_saving + '_routes.svg', bbox_inches='tight')

print('infrastructure')
infrastructure_axis = ax[(1, 1)]
infrastructure_axis = get_infrastructure_figure(infrastructure_axis, boundaries, path_processed_data)

# color bar
height_bar = 7.5 / height * 0.02
cbar_ax = fig.add_axes([0.05, 0.54, 0.9, height_bar])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('€ / MWh', rotation=0, labelpad=5)

if plot_routes:
    # route legend
    fig.legend(handles=handles_list_transport_means, loc='upper center', ncols=3,
               bbox_to_anchor=(0.5, 0.19), title='Transport Mean',
               labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

    fig.legend(handles=handles_list_commodities, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.12), title='Commodity',
               labelspacing=0.1, handletextpad=0.1, columnspacing=0.25)

else:
    # commodity legend
    fig.legend(handles=commodity_handles, loc='upper center', ncols=2,
               bbox_to_anchor=(0.255, 0.15),
               labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=0.5)

# infrastructure legend
handels_list_infrastructure = [mlines.Line2D([], [], color='blue', marker='.',
                                             linestyle='None', markersize=5,
                                             label='Port'),
                               mlines.Line2D([], [], color='red',
                                             linestyle='-', markersize=5,
                                             label='Gas Pipeline'),
                               mlines.Line2D([], [], color='black',
                                             linestyle='-', markersize=5,
                                             label='Oil Pipeline')]

fig.legend(handles=handels_list_infrastructure, loc='upper center', ncol=3, bbox_to_anchor=(0.74, 0.15),
           labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=0.5)

fig.tight_layout()
plt.subplots_adjust(bottom=0.3)

fig.savefig(path_saving + '_all_graphs.png', bbox_inches='tight', dpi=600)
fig.savefig(path_saving + '_all_graphs.svg', bbox_inches='tight')

plt.close(fig)
