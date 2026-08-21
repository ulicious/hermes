import os
import time
import matplotlib.pyplot as plt
import shapely

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.lines as mlines

from math import sqrt

from plotting.get_figures import get_number_figure, get_energy_carrier_figure, get_infrastructure_figure, \
    get_routes_figure, get_weighted_routes, get_commodity_transport_mean_histogram, get_supply_curves, \
    get_used_locations_figure, get_calculation_time, get_sankey_diagram, \
    get_start_locations_infrastructure_destination_figure, \
    get_tight_boundaries_for_start_locations_infrastructure_destination, get_water_availability_figure, \
    safe_output_path, resolve_plot_boundaries, get_configured_colormap, get_plot_color_config, \
    get_hydrogen_potential_plot_scale
from plotting.helpers_plotting import load_infrastructure_data, load_first_available_destination, \
    get_complete_infrastructure, load_result, plot_comparison_plot, match_routing_results
from data_processing.configuration import load_algorithm_configuration, load_plotting_configuration, get_raw_data_path

time_start = time.time()


def check_required_files_exist(required_files, purpose):
    missing_files = [file for file in required_files if not os.path.exists(file)]
    if missing_files:
        missing_files_text = '\n'.join('- ' + file for file in missing_files)
        raise FileNotFoundError(f'Missing data for {purpose}:\n{missing_files_text}')


def plot_h2_potential(data, boundaries, fig_title, geometry_source=None):
    required_columns = {'Hydrogen_Gas_Quantity'}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            'Missing data for H2-potential plotting:\n- column '
            + '\n- column '.join(repr(column) for column in missing_columns)
        )
    geometry_available = 'geometry' in data.columns
    geometry_source_available = (
        geometry_source is not None
        and not geometry_source.empty
        and 'geometry' in geometry_source.columns
    )
    if not geometry_available and not geometry_source_available:
        raise ValueError("H2-potential plotting requires a 'geometry' column.")

    potential_data = data.copy()
    if not geometry_available:
        valid_locations = potential_data.index.intersection(geometry_source.index)
        potential_data = potential_data.loc[valid_locations].copy()
    potential_data['Hydrogen_Gas_Quantity'] = pd.to_numeric(
        potential_data['Hydrogen_Gas_Quantity'], errors='coerce'
    )
    valid_potential = (
        np.isfinite(potential_data['Hydrogen_Gas_Quantity'])
        & (potential_data['Hydrogen_Gas_Quantity'] >= 0)
    )
    if geometry_available:
        valid_potential &= potential_data['geometry'].apply(
            lambda geometry: hasattr(geometry, 'is_empty') and not geometry.is_empty
        )
    potential_data = potential_data[valid_potential].copy()
    if potential_data.empty:
        raise ValueError('H2-potential plotting requires at least one valid cell quantity and geometry.')

    potential_divisor, potential_unit = get_hydrogen_potential_plot_scale(
        potential_data['Hydrogen_Gas_Quantity'].max()
    )
    potential_column = 'Hydrogen_Gas_Quantity_plot'
    potential_data[potential_column] = (
        potential_data['Hydrogen_Gas_Quantity'] / potential_divisor
    )
    maximum_display_value = potential_data[potential_column].max()
    only_zero_potential = np.isclose(maximum_display_value, 0)
    potential_norm = mpl.colors.Normalize(
        vmin=0,
        vmax=1 if only_zero_potential else maximum_display_value,
    )
    get_number_figure(
        potential_data,
        potential_norm,
        cmap,
        boundaries,
        destination_location=None,
        column=potential_column,
        use_voronoi=True,
        production_costs=geometry_source,
        save=True,
        save_path=path_saving,
        fig_title=fig_title,
        plot_colors=plot_colors,
        unit=potential_unit,
        colorbar_ticks=[0] if only_zero_potential else None,
        export_column='Hydrogen_Gas_Quantity',
    )


# get general configuration
config_file_general = load_algorithm_configuration()

project_folder_path = config_file_general['project_folder_path']
path_data = os.path.join(project_folder_path, 'processed_data')
path_files = os.path.join(project_folder_path, 'results', 'processed_results')
path_saving = os.path.join(project_folder_path, 'results', 'plots')

config_file_plotting = load_plotting_configuration(config_file_general)

start_destination_combinations_file = os.path.join(config_file_general['project_folder_path'],
                                                   'start_destination_combinations.csv')
required_infrastructure_files = [os.path.join(path_data, file) for file in [
    'gas_pipeline_node_locations.csv',
    'gas_pipeline_graphs.csv',
    'oil_pipeline_node_locations.csv',
    'oil_pipeline_graphs.csv',
    'ports.csv']]
missing_infrastructure_files = [file for file in required_infrastructure_files if not os.path.exists(file)]
infrastructure_data_available = not missing_infrastructure_files
if missing_infrastructure_files:
    missing_infrastructure_files_text = '\n'.join('- ' + file for file in missing_infrastructure_files)
    print('Optional infrastructure plotting data missing; continuing without these layers:\n'
          + missing_infrastructure_files_text)
infrastructure_data = load_infrastructure_data(path_data)
destination = None
complete_infrastructure = None
if os.path.exists(start_destination_combinations_file):
    production_costs = pd.read_csv(start_destination_combinations_file, index_col=0)
    if 'geometry' in production_costs.columns:
        production_costs['geometry'] = production_costs['geometry'].apply(shapely.wkt.loads)
else:
    production_costs = pd.DataFrame()

cmap = get_configured_colormap(config_file_plotting)
plot_colors = get_plot_color_config(config_file_plotting)

extent_result_frames = []
extent_result_names = config_file_plotting.get('process_results', [])
for result_name in extent_result_names:
    result_file = os.path.join(path_files, result_name + '_processed_results.csv')
    if os.path.exists(result_file):
        extent_result_frames.append(pd.read_csv(result_file, index_col=0))

if extent_result_frames:
    extent_result_data = pd.concat(extent_result_frames, ignore_index=True)
else:
    extent_result_data = pd.DataFrame()

extent_destination = None
if str(config_file_plotting.get('plot_extent', 'default')).strip().lower() == 'results':
    try:
        extent_destination = load_first_available_destination(path_files, extent_result_names)
    except FileNotFoundError:
        pass

global_plot_boundaries = resolve_plot_boundaries(
    config_file_plotting,
    data=extent_result_data,
    destination_location=extent_destination,
)

color_dictionary = plot_colors['commodity_colors']
nice_name_dictionary = config_file_plotting['nice_name_dictionary']
transport_mean_line_styles = config_file_plotting['transport_mean_line_styles']

line_widths = config_file_plotting['line_widths']
line_widths_new = {}
for commodity in line_widths.keys():
    for transport_mean in line_widths[commodity].keys():
        line_widths_new[(commodity, transport_mean)] = line_widths[commodity][transport_mean]
line_widths = line_widths_new

# get results to process
production_plot_results = config_file_plotting['production_costs_plot']
conversion_plot_results = config_file_plotting['conversion_costs_plot']
transport_plot_results = config_file_plotting['transport_costs_plot']
total_supply_costs_plot_results = config_file_plotting['total_supply_costs_plot']
profit_plot_results = config_file_plotting['profit_plot']
all_costs_plot_results = config_file_plotting['all_costs_plot']
commodity_plot_results = config_file_plotting['commodity_plot']
efficiency_plot_results = config_file_plotting['efficiency_plot']
potential_plot_results = config_file_plotting.get('potential_plot', [])
full_plot_results = config_file_plotting['full_plot']

commodity_transport_mean_results = config_file_plotting['commodity_transport_mean_plot']
routes_plot_results = config_file_plotting['routes_plot']
sankey_plot_results = config_file_plotting['sankey_plot']
weighted_routes_plot_results = config_file_plotting['weighted_routes_plot']
compare_costs_and_quantities_results = config_file_plotting['compare_costs_and_quantities_plot']
supply_curve_results = config_file_plotting['supply_curve_plot']['results']

routes_comparison_plot_results = config_file_plotting['routes_comparison_plot']
matched_supply_routes_plot = config_file_plotting['matched_supply_routes_plot']

all_results = list(set(production_plot_results + conversion_plot_results + transport_plot_results
                       + total_supply_costs_plot_results + profit_plot_results
                       + all_costs_plot_results + commodity_plot_results
                       + efficiency_plot_results + potential_plot_results
                       + sankey_plot_results + routes_plot_results
                       + full_plot_results + weighted_routes_plot_results + commodity_transport_mean_results
                       + supply_curve_results))

route_infrastructure_plots_requested = (
    bool(routes_plot_results)
    or bool(routes_comparison_plot_results)
    or bool(matched_supply_routes_plot)
)

if (config_file_plotting['start_locations_infrastructure_destination_plot']
        or (route_infrastructure_plots_requested and infrastructure_data_available)):
    destination = load_first_available_destination(path_files, all_results)

if route_infrastructure_plots_requested and infrastructure_data_available:
    complete_infrastructure = get_complete_infrastructure(infrastructure_data, destination)
elif route_infrastructure_plots_requested:
    print('Route infrastructure plots require optional infrastructure data and will be skipped.')
    routes_plot_results = []
    routes_comparison_plot_results = []
    matched_supply_routes_plot = []

# Start plotting

# infrastructure
if config_file_plotting['infrastructure_plot']:
    get_infrastructure_figure(
        global_plot_boundaries, path_data, save=True, path_saving=path_saving,
        plot_colors=plot_colors)

# water availability
if config_file_plotting['water_availability_plot']:
    get_water_availability_figure(
        global_plot_boundaries, path_data, save=True, path_saving=path_saving,
        plot_colors=plot_colors)

# start locations, infrastructure and destination
if config_file_plotting['start_locations_infrastructure_destination_plot']:
    check_required_files_exist([start_destination_combinations_file], 'start-location plotting')
    if 'geometry' not in production_costs.columns:
        raise ValueError("Missing data for start-location plotting:\n- column 'geometry' in "
                         + start_destination_combinations_file)
    get_start_locations_infrastructure_destination_figure(
        production_costs, global_plot_boundaries, path_data, destination,
        save=True, path_saving=path_saving, plot_colors=plot_colors)

# H2 potential in the original input-data grid cells
if config_file_plotting.get('h2_potential_grid_cell_plot', False):
    location_data_file = get_raw_data_path(config_file_general, 'location_data')
    check_required_files_exist([location_data_file], 'H2 grid-cell potential plotting')
    grid_cell_potential_data = pd.read_csv(
        location_data_file,
        sep=';',
        usecols=['latitude', 'longitude', 'Hydrogen_Gas_Quantity', 'polygon'],
    )
    grid_cell_potential_data['geometry'] = grid_cell_potential_data['polygon'].apply(
        lambda geometry: shapely.wkt.loads(geometry) if isinstance(geometry, str) and geometry else None
    )
    plot_h2_potential(
        grid_cell_potential_data,
        global_plot_boundaries,
        'h2_potential_grid_cells',
    )

# result plots
for r in all_results:
    print(r)

    with_routes = r in weighted_routes_plot_results
    data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination_location \
        = load_result(r, path_files, config_file_plotting, production_costs, with_routes=with_routes)
    scenario_boundaries = resolve_plot_boundaries(
        config_file_plotting,
        data=data,
        destination_location=destination_location,
    )

    # nice names for results can be undefined --> will be r if not defined
    if r not in [*nice_name_dictionary.keys()]:
        nice_name_dictionary[r] = r

    if r in production_plot_results:
        get_number_figure(data.copy(), norm_prod, cmap, scenario_boundaries, destination_location, column='production_costs',
                          use_voronoi=True, production_costs=production_costs, limit_scale=config_file_plotting['limit_scale'],
                          save=True, save_path=path_saving, fig_title=r + '_production_costs',
                          plot_colors=plot_colors)

    if r in conversion_plot_results:
        get_number_figure(data.copy(), norm_conv, cmap, scenario_boundaries, destination_location, column='conversion_costs',
                          use_voronoi=True, production_costs=production_costs,
                          save=True, save_path=path_saving, fig_title=r + '_conversion_costs',
                          plot_colors=plot_colors)

    if r in transport_plot_results:
        get_number_figure(data.copy(), norm_trans, cmap, scenario_boundaries, destination_location, column='transportation_costs',
                          use_voronoi=True, production_costs=production_costs,
                          limit_scale=config_file_plotting['limit_scale'], save=True, save_path=path_saving,
                          fig_title=r + '_transport_costs', plot_colors=plot_colors)

    if r in total_supply_costs_plot_results:
        get_number_figure(data.copy(), norm_total, cmap, scenario_boundaries, destination_location,
                          use_voronoi=True, production_costs=production_costs,
                          limit_scale=config_file_plotting['limit_scale'],
                          save=True, save_path=path_saving, fig_title=r + '_total_costs',
                          plot_colors=plot_colors)

    if r in profit_plot_results:
        get_number_figure(data.copy(), norm_adjusted_costs, cmap, scenario_boundaries, destination_location,
                          column='adjusted_costs',
                          use_voronoi=True, production_costs=production_costs,
                          limit_scale=config_file_plotting['limit_scale'],
                          save=True, save_path=path_saving, fig_title=r + '_profit',
                          plot_colors=plot_colors)

    if r in commodity_plot_results:
        get_energy_carrier_figure(data.copy(), scenario_boundaries, color_dictionary, nice_name_dictionary, destination_location,
                                  use_voronoi=True, production_costs=production_costs,
                                  save=True, path_saving=path_saving, fig_title=r + '_energy_carrier',
                                  plot_colors=plot_colors)

    if r in efficiency_plot_results:
        get_number_figure(data.copy(), norm_efficiency, cmap, scenario_boundaries, destination_location, column='efficiency',
                          use_voronoi=True, production_costs=production_costs,
                          save=True, save_path=path_saving, fig_title=r + '_efficiency',
                          plot_colors=plot_colors)

    if r in potential_plot_results:
        plot_h2_potential(
            data.copy(),
            scenario_boundaries,
            r + '_h2_potential_voronoi',
            geometry_source=production_costs,
        )

    if r in routes_plot_results:
        get_routes_figure(data.copy(), transport_mean_line_styles, line_widths, color_dictionary, nice_name_dictionary,
                          infrastructure_data, complete_infrastructure, scenario_boundaries, destination_location,
                          save=True, path_saving=path_saving, fig_title=r + '_routes',
                          plot_colors=plot_colors)

    if r in sankey_plot_results:
        get_sankey_diagram(ranked_routes, color_dictionary, nice_name_dictionary, path_saving, r + '_sankey')

    if r in weighted_routes_plot_results:
        for commodity in weighted_routes['commodity'].unique():
            sub_data = weighted_routes[weighted_routes['commodity'] == commodity]
            get_weighted_routes(sub_data, scenario_boundaries,
                                transport_mean_line_styles, color_dictionary,
                                nice_name_dictionary, destination_location=destination_location, save=True,
                                path_saving=path_saving, fig_title=r + '_' + str(commodity) + '_weighted_routes',
                                country_comparison=True, plot_colors=plot_colors)

        get_weighted_routes(weighted_routes, scenario_boundaries,
                            transport_mean_line_styles, color_dictionary,
                            nice_name_dictionary, destination_location=destination_location, save=True,
                            path_saving=path_saving, fig_title=r + '_weighted_routes',
                            country_comparison=True, plot_colors=plot_colors)

        get_weighted_routes(weighted_routes, scenario_boundaries,
                            transport_mean_line_styles, color_dictionary,
                            nice_name_dictionary, destination_location=destination_location, save=True,
                            path_saving=path_saving, ignore_commodity=True,
                            fig_title=r + '_weighted_routes_no_commodity', plot_colors=plot_colors)

    if r in commodity_transport_mean_results:
        get_commodity_transport_mean_histogram(data.copy(), color_dictionary, nice_name_dictionary, path_saving, r)

    if r in supply_curve_results:
        for c in config_file_plotting['supply_curve_plot']['countries']:
            get_supply_curves(data.copy(), color_dictionary, nice_name_dictionary,
                              add_legend=True, save=True, fig_title=r + '_' + c + '_supply_curve',
                              path_saving=path_saving, country=c, production_costs=production_costs,
                              plot_colors=plot_colors)

    if r in all_costs_plot_results:
        diff_lat = scenario_boundaries['max_latitude'] - scenario_boundaries['min_latitude']
        ratio_lat_lon = diff_lat / (
            scenario_boundaries['max_longitude'] - scenario_boundaries['min_longitude'])

        plot_width = 15.69  # todo: sollte auch als parameter im plot configuration sein
        distance_between = 0.25  # todo: sollte auch als parameter im plot configuration sein
        subplot_width = (plot_width - 3 * distance_between) / 2
        subplot_height = subplot_width * ratio_lat_lon + 1.5

        plot_height = 2 * subplot_height + distance_between * 3
        fig = plt.figure(figsize=(plot_width / 2.54, plot_height / 2.54))

        relative_distance_between_height = distance_between / plot_height
        relative_distance_between_width = distance_between / plot_width
        relative_subplot_height = subplot_height / plot_height
        relative_subplot_width = subplot_width / plot_width

        ax1 = fig.add_axes((relative_distance_between_width,
                            2 * relative_distance_between_height + relative_subplot_height,
                            relative_subplot_width, relative_subplot_height))  # [left, bottom, width, height]

        ax2 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                            2 * relative_distance_between_height + relative_subplot_height,
                            relative_subplot_width, relative_subplot_height))

        ax3 = fig.add_axes((relative_distance_between_width,
                            relative_distance_between_height,
                            relative_subplot_width, relative_subplot_height))

        ax4 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                            relative_distance_between_height,
                            relative_subplot_width, relative_subplot_height))

        ax1 = get_number_figure(data.copy(), norm_prod, cmap, scenario_boundaries, destination_location,
                                use_voronoi=True, column='production_costs', production_costs=production_costs,
                                limit_scale=config_file_plotting['limit_scale'], ax=ax1, return_fig=True, fig=fig,
                                fig_title='H2 Production Costs', add_fig_title=True, plot_colors=plot_colors)

        ax2 = get_number_figure(data.copy(), norm_conv, cmap, scenario_boundaries, destination_location,
                                column='conversion_costs', use_voronoi=True, production_costs=production_costs,
                                ax=ax2, return_fig=True, fig=fig, fig_title='Conversion Costs', add_fig_title=True,
                                plot_colors=plot_colors)

        ax3 = get_number_figure(data.copy(), norm_trans, cmap, scenario_boundaries, destination_location,
                                column='transportation_costs', use_voronoi=True, production_costs=production_costs,
                                ax=ax3, limit_scale=config_file_plotting['limit_scale'], return_fig=True, fig=fig,
                                fig_title='Transport Costs', add_fig_title=True, plot_colors=plot_colors)

        ax4 = get_number_figure(data.copy(), norm_total, cmap, scenario_boundaries, destination_location, ax=ax4,
                                use_voronoi=True, production_costs=production_costs,
                                limit_scale=config_file_plotting['limit_scale'], return_fig=True, fig=fig,
                                fig_title='Total Supply Costs', add_fig_title=True, plot_colors=plot_colors)

        fig.savefig(safe_output_path(path_saving, r + '_all_costs.png'), bbox_inches='tight', dpi=600)
        fig.savefig(safe_output_path(path_saving, r + '_all_costs.svg'), bbox_inches='tight')

    if r in full_plot_results:
        diff_lat = scenario_boundaries['max_latitude'] - scenario_boundaries['min_latitude']
        ratio_lat_lon = diff_lat / (
            scenario_boundaries['max_longitude'] - scenario_boundaries['min_longitude'])

        plot_width = 15.69  # todo: sollte auch als parameter im plot configuration sein
        distance_between = 0.25  # todo: sollte auch als parameter im plot configuration sein
        subplot_width = (plot_width - 3 * distance_between) / 2
        subplot_height = subplot_width * ratio_lat_lon + 1.5

        plot_height = 2 * subplot_height + distance_between * 3
        fig = plt.figure(figsize=(plot_width / 2.54, plot_height / 2.54))

        relative_distance_between_height = distance_between / plot_height
        relative_distance_between_width = distance_between / plot_width
        relative_subplot_height = subplot_height / plot_height
        relative_subplot_width = subplot_width / plot_width

        ax1 = fig.add_axes((relative_distance_between_width,
                            2 * relative_distance_between_height + relative_subplot_height,
                            relative_subplot_width, relative_subplot_height))  # [left, bottom, width, height]

        ax2 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                            2 * relative_distance_between_height + relative_subplot_height,
                            relative_subplot_width, relative_subplot_height))

        ax3 = fig.add_axes((relative_distance_between_width,
                            relative_distance_between_height,
                            relative_subplot_width, relative_subplot_height))

        ax4 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                            relative_distance_between_height,
                            relative_subplot_width, relative_subplot_height))

        ax1 = get_number_figure(data.copy(), norm_prod, cmap, scenario_boundaries, destination_location,
                                column='production_costs', use_voronoi=True,
                                production_costs=production_costs, limit_scale=config_file_plotting['limit_scale'],
                                ax=ax1, return_fig=True, fig=fig, fig_title='H2 Production Costs',
                                add_fig_title=True, plot_colors=plot_colors)

        ax2 = get_number_figure(data.copy(), norm_total, cmap, scenario_boundaries, destination_location,
                                use_voronoi=True, production_costs=production_costs,
                                limit_scale=config_file_plotting['limit_scale'],
                                ax=ax2, return_fig=True, fig=fig, fig_title='Total Supply Costs',
                                add_fig_title=True, plot_colors=plot_colors)

        ax3 = get_energy_carrier_figure(data.copy(), scenario_boundaries, color_dictionary, nice_name_dictionary,
                                        destination_location, use_voronoi=True,
                                        production_costs=production_costs, ax=ax3, fig=fig,
                                        plot_colors=plot_colors)

        ax4 = get_infrastructure_figure(
            scenario_boundaries, path_data, ax=ax4, return_fig=True, fig=fig,
            plot_colors=plot_colors)

        fig.savefig(safe_output_path(path_saving, r + '_mixed_overview.png'), bbox_inches='tight', dpi=600)
        fig.savefig(safe_output_path(path_saving, r + '_mixed_overview.svg'), bbox_inches='tight')

plot_comparison_plot('costs', config_file_plotting['conversion_costs_comparison_plot'],
                     path_files, path_saving,
                     config_file_plotting, production_costs, cmap, global_plot_boundaries,
                     nice_name_dictionary=nice_name_dictionary,
                     cost_type='conversion_costs')

plot_comparison_plot('costs', config_file_plotting['transport_costs_comparison_plot'],
                     path_files, path_saving,
                     config_file_plotting, production_costs, cmap, global_plot_boundaries,
                     nice_name_dictionary=nice_name_dictionary,
                     cost_type='transportation_costs')

plot_comparison_plot('costs', config_file_plotting['total_supply_costs_comparison_plot'],
                     path_files, path_saving,
                     config_file_plotting, production_costs, cmap, global_plot_boundaries,
                     nice_name_dictionary=nice_name_dictionary,
                     cost_type='')

plot_comparison_plot('costs', config_file_plotting['profit_comparison_plot'],
                     path_files, path_saving,
                     config_file_plotting, production_costs, cmap, global_plot_boundaries,
                     nice_name_dictionary=nice_name_dictionary,
                     cost_type='adjusted_costs')

plot_comparison_plot('energy_carrier', config_file_plotting['commodity_comparison_plot'],
                     path_files, path_saving,
                     config_file_plotting, production_costs, cmap, global_plot_boundaries,
                     color_dictionary=color_dictionary, nice_name_dictionary=nice_name_dictionary)

plot_comparison_plot('routes', routes_comparison_plot_results,
                     path_files, path_saving,
                     config_file_plotting, production_costs, cmap, global_plot_boundaries,
                     color_dictionary=color_dictionary, nice_name_dictionary=nice_name_dictionary,
                     transport_mean_line_styles=transport_mean_line_styles, line_widths=line_widths,
                     infrastructure_data=infrastructure_data, complete_infrastructure=complete_infrastructure)

for country in config_file_plotting['supply_curve_comparison_plot']['countries']:
    plot_comparison_plot('supply_curves', config_file_plotting['supply_curve_comparison_plot']['results'],
                         path_files, path_saving,
                         config_file_plotting, production_costs, cmap, global_plot_boundaries,
                         color_dictionary=color_dictionary, nice_name_dictionary=nice_name_dictionary,
                         transport_mean_line_styles=transport_mean_line_styles, line_widths=line_widths,
                         infrastructure_data=infrastructure_data, complete_infrastructure=complete_infrastructure,
                         country=country, distance_between=1, subplot_height=4)

for n, comparison in enumerate(matched_supply_routes_plot):
    results_data = []
    for r in comparison:
        data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination_location \
            = load_result(r, path_files, config_file_plotting, production_costs, with_routes=False)

        results_data.append(data.copy())

    matched_df = match_routing_results(results_data, comparison,
                                       complete_infrastructure, infrastructure_data)
    matched_boundaries = resolve_plot_boundaries(
        config_file_plotting,
        route_geometries=matched_df['geometry'],
    )

    get_weighted_routes(matched_df, matched_boundaries, transport_mean_line_styles, color_dictionary,
                        nice_name_dictionary, fig_title=str(n) + '_matched_weighted_routes',
                        save=True, path_saving=path_saving, country_comparison=True,
                        plot_colors=plot_colors)

for r in compare_costs_and_quantities_results:
    weighted_routes_file = os.path.join(path_files, r + '_routes_and_quantities.csv')
    if not os.path.exists(weighted_routes_file):
        raise FileNotFoundError(
            "Cannot create 'compare_costs_and_quantities_plot' for "
            + str(r)
            + ' because the required categorized routes file is missing:\n'
            + weighted_routes_file
            + "\n\nGenerate it first by adding the scenario to both 'process_results' and "
              "'categorical_routes' in 4_plotting_configuration.yaml and running "
              "scripts._5_process_plot_data."
        )

    output_path = safe_output_path(path_saving, r + '_distribution_cost_and_quantities.png')

    # mpl.use('TkAgg')

    data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination_location \
        = load_result(r, path_files, config_file_plotting, production_costs, with_routes=True)

    export_routes = weighted_routes.copy()
    export_routes['geometry'] = export_routes['geometry'].apply(lambda geometry: geometry.wkt)
    summary_by_cost_category = (
        export_routes.groupby('commodity', as_index=False)
        .agg(
            quantity_MWh=('quantity_MWh', 'sum'),
            quantity_TWh=('quantity_TWh', 'sum'),
            number_of_route_segments=('quantity_MWh', 'size')
        )
    )
    summary_by_transport_mean = (
        export_routes.groupby(['commodity', 'transport_mean'], as_index=False)
        .agg(
            quantity_MWh=('quantity_MWh', 'sum'),
            quantity_TWh=('quantity_TWh', 'sum'),
            number_of_route_segments=('quantity_MWh', 'size')
        )
    )
    excel_output_path = safe_output_path(path_saving, r + '_distribution_cost_and_quantities.xlsx')
    with pd.ExcelWriter(excel_output_path) as writer:
        export_routes.to_excel(writer, sheet_name='routes', index=False)
        summary_by_cost_category.to_excel(writer, sheet_name='summary_by_cost', index=False)
        summary_by_transport_mean.to_excel(writer, sheet_name='summary_by_transport', index=False)

    scenario_boundaries = resolve_plot_boundaries(
        config_file_plotting,
        data=data,
        destination_location=destination_location,
        route_geometries=weighted_routes['geometry'],
    )

    diff_lat = scenario_boundaries['max_latitude'] - scenario_boundaries['min_latitude']
    ratio_lat_lon = diff_lat / (
        scenario_boundaries['max_longitude'] - scenario_boundaries['min_longitude'])

    plot_width = 15.69
    distance_between = 0.25
    subplot_width = (plot_width - 3 * distance_between) / 2
    individual_legend_height = 0
    subplot_height = subplot_width * ratio_lat_lon + individual_legend_height
    legend_height = 0.4

    plot_height = 3 * subplot_height + 4 * distance_between + legend_height
    fig = plt.figure(figsize=(plot_width / 2.54, plot_height / 2.54))

    relative_distance_between_height = distance_between / plot_height
    relative_distance_between_width = distance_between / plot_width
    relative_subplot_height = subplot_height / plot_height
    relative_subplot_width = subplot_width / plot_width
    relative_legend_height = legend_height / plot_height

    # top left
    ax1 = fig.add_axes((relative_distance_between_width,
                        3 * relative_distance_between_height + 2 * relative_subplot_height + relative_legend_height,
                        relative_subplot_width, relative_subplot_height))

    # top right
    ax2 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                        3 * relative_distance_between_height + 2 * relative_subplot_height + relative_legend_height,
                        relative_subplot_width, relative_subplot_height))

    # mid left
    ax3 = fig.add_axes((relative_distance_between_width,
                        2 * relative_distance_between_height + relative_subplot_height + relative_legend_height,
                        relative_subplot_width, relative_subplot_height))

    # mid right
    ax4 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                        2 * relative_distance_between_height + relative_subplot_height + relative_legend_height,
                        relative_subplot_width, relative_subplot_height))

    # bottom left
    ax5 = fig.add_axes((relative_distance_between_width,
                        relative_distance_between_height + relative_legend_height,
                        relative_subplot_width, relative_subplot_height))

    # # bottom right
    ax6 = fig.add_axes((2 * relative_distance_between_width + relative_subplot_width,
                        relative_distance_between_height + relative_legend_height,
                        relative_subplot_width, relative_subplot_height))

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    # weighted_routes = weighted_routes.iloc[0:1000]
    colors = plot_colors['category_colors']['cost_categories']
    weighted_routes['commodity'] = (
        weighted_routes['commodity'].astype(str)
        .str.replace('€/MWh', '€ MWh$^{-1}$', regex=False)
        .str.replace('€ / MWh', '€ MWh$^{-1}$', regex=False)
        .str.replace('€ MWh⁻¹', '€ MWh$^{-1}$', regex=False)
    )
    max_quantity = weighted_routes['quantity'].max()
    min_quantity = weighted_routes['quantity'].min()

    range_order = ['50 to 100 € MWh$^{-1}$', '100 to 150 € MWh$^{-1}$', '150 to 200 € MWh$^{-1}$',
                   '200 to 250 € MWh$^{-1}$', '250 to 300 € MWh$^{-1}$', '> 300 € MWh$^{-1}$']
    for n, commodity in enumerate(range_order):

        ax = axes[n]
        color = colors[n]

        sub_data = weighted_routes[weighted_routes['commodity'] == commodity]
        ax = get_weighted_routes(sub_data, scenario_boundaries,
                                 transport_mean_line_styles, color_dictionary,
                                 nice_name_dictionary, destination_location=destination_location,
                                 fig_title=commodity, add_fig_title=True, ax=ax,
                                 return_fig=True, add_legend=False, color=color, max_quantity=max_quantity,
                                 min_quantity=min_quantity, plot_colors=plot_colors)

        # if n == 1:
        #     break

    powers = range(15)
    target_quantity = 1
    for p in powers:
        if 1 > max_quantity / 10 ** p:
            target_quantity = 10 ** p

            if 1 > max_quantity / (10 ** p / 2):
                target_quantity = 10 ** p / 2

                if 1 > max_quantity / (10 ** p / 4):
                    target_quantity = 10 ** p / 4

                    if 1 > max_quantity / (10 ** p / 8):
                        target_quantity = 10 ** p / 8

            break

    if False:
        scale = 5 / target_quantity

        sizes = []
        for s in [target_quantity * 0.25, target_quantity * 0.5, target_quantity]:
            sizes.append(mlines.Line2D([], [], color='black', linestyle='-', markersize=5,
                                       label=str(int(s / 1000000)), linewidth=sqrt(s * scale)))

        fig.legend(handles=sizes, loc='upper center', ncols=3,
                   bbox_to_anchor=(0.5, relative_legend_height), title='Quantity [TWh]',
                   handlelength=1, frameon=False)
    else:
        import matplotlib
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # cmap = mpl.colormaps['coolwarm']
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "",
            plot_colors['colorbar']['quantity_colormap'],
        )
        norm = matplotlib.colors.Normalize(vmin=min_quantity / 1000000, vmax=max_quantity / 1000000)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        fig.subplots_adjust(bottom=0.8)
        cbar_ax = fig.add_axes([0.025, 0., 0.95, 0.02])

        shrink = 0.5
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', anchor=(0.5, 0), shrink=shrink, aspect=30, pad=0)
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

        labels = cbar.ax.get_xticklabels()
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')

    plt.subplots_adjust(bottom=relative_legend_height)

    # mpl.rcParams['path.simplify'] = True
    # mpl.rcParams['path.simplify_threshold'] = 0.1
    mpl.rcParams['agg.path.chunksize'] = 20000

    fig.savefig(output_path, bbox_inches='tight', dpi=600)

    plt.close(fig)

for n, comparison in enumerate(config_file_plotting['solving_time_plot']):
    all_data = []
    for r in comparison:
        data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination_location \
            = load_result(r, path_files, config_file_plotting, production_costs, with_routes=False)
        all_data.append(data)

    get_calculation_time(all_data, comparison, path_saving, str(n) + '_processing_times', nice_name_dictionary)


if time.time() - time_start < 60:
    print('total processing time [s]: ' + str(time.time() - time_start))
elif time.time() - time_start < 3600:
    print('total processing time [m]: ' + str((time.time() - time_start) / 60))
else:
    print('total processing time [h]: ' + str((time.time() - time_start) / 60 / 60))
