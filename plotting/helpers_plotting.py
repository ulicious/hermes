import shapely
import math
import itertools
import multiprocessing
import ast
import os

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import searoute as sr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from tqdm import tqdm
from shapely import wkt
from shapely.geometry import MultiLineString, Point, LineString
from collections import defaultdict
from statistics import mean

from plotting.get_figures import get_number_figure, get_routes_figure, get_energy_carrier_figure, get_weighted_routes, \
    get_supply_curves, safe_output_path, resolve_plot_boundaries
from data_processing.configuration import CONVERSION_CONFIG, load_yaml


def _read_csv_or_empty(path, columns=None, index_col=0, dtype=None):
    """Read a CSV if available, otherwise return an empty frame with expected columns."""
    if os.path.exists(path):
        return pd.read_csv(path, index_col=index_col, dtype=dtype)
    return pd.DataFrame(columns=columns or [])


def _load_strike_prices_from_result_path(path_files):
    project_folder_path = os.path.abspath(os.path.join(path_files, os.pardir, os.pardir))
    conversion_config = load_yaml(os.path.join(project_folder_path, CONVERSION_CONFIG))
    return conversion_config.get('strike_prices', {})


def _get_final_route_commodity(route, fallback_commodity=None):
    if isinstance(route, str):
        try:
            route = ast.literal_eval(route)
        except (ValueError, SyntaxError):
            return fallback_commodity

    if not isinstance(route, list):
        return fallback_commodity

    commodity = fallback_commodity
    for route_segment in route:
        if not isinstance(route_segment, (list, tuple)):
            continue

        if len(route_segment) == 2:
            commodity = route_segment[0]
        elif len(route_segment) == 3:
            commodity = route_segment[1]

    return commodity


def get_geometry_plot_point(geometry):
    """Return a stable point for plotting infrastructure tables from point or area geometries."""
    if isinstance(geometry, Point):
        return geometry

    if geometry is None or geometry.is_empty:
        raise ValueError('Destination geometry is empty and cannot be added to complete infrastructure.')

    return geometry.representative_point()


def load_destination(path_files, result_name):
    destination = pd.read_csv(os.path.join(path_files, result_name + '_destination.csv'), index_col=0)
    return wkt.loads(destination.values[0][0])


def load_first_available_destination(path_files, preferred_results=None):
    if preferred_results is not None:
        for result_name in preferred_results:
            destination_file = os.path.join(path_files, result_name + '_destination.csv')
            if os.path.exists(destination_file):
                return load_destination(path_files, result_name)

    destination_files = sorted(
        file for file in os.listdir(path_files)
        if file.endswith('_destination.csv')
    )
    if not destination_files:
        raise FileNotFoundError('Missing destination data in processed results.')

    result_name = destination_files[0][:-len('_destination.csv')]
    return load_destination(path_files, result_name)


def load_infrastructure_data(path_data):
    node_columns = ['latitude', 'longitude', 'graph', 'continent']
    graph_columns = ['graph', 'node_start', 'node_end', 'distance', 'line']
    port_columns = ['latitude', 'longitude', 'name', 'country', 'continent',
                    'longitude_on_coastline', 'latitude_on_coastline']

    pipeline_gas_node_locations = _read_csv_or_empty(
        os.path.join(path_data, 'gas_pipeline_node_locations.csv'), columns=node_columns,
        dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_gas_graphs = _read_csv_or_empty(
        os.path.join(path_data, 'gas_pipeline_graphs.csv'), columns=graph_columns)
    pipeline_liquid_node_locations = _read_csv_or_empty(
        os.path.join(path_data, 'oil_pipeline_node_locations.csv'), columns=node_columns,
        dtype={'latitude': np.float16, 'longitude': np.float16})
    pipeline_liquid_graphs = _read_csv_or_empty(
        os.path.join(path_data, 'oil_pipeline_graphs.csv'), columns=graph_columns)
    ports = _read_csv_or_empty(os.path.join(path_data, 'ports.csv'), columns=port_columns)

    data = {'Shipping': {'ports': ports}}

    data = process_network_data(data, 'Pipeline_Gas', pipeline_gas_node_locations, pipeline_gas_graphs)

    data = process_network_data(data, 'Pipeline_Liquid', pipeline_liquid_node_locations, pipeline_liquid_graphs)

    return data


def process_network_data(data, name, geo_data, graph_data):

    """
    Function is used to create different data structures for the network data
    @param dict data: dictionary for all data
    @param str name: name of network
    @param pandas.DataFrame geo_data: geo data of network (locations of nodes)
    @param pandas.DataFrame graph_data: information on lines of network

    @return: different data structures
    """

    data[name] = {}

    if geo_data.empty or graph_data.empty:
        return data
    if 'graph' not in geo_data.columns or 'line' not in graph_data.columns:
        return data

    graph_data = graph_data[graph_data['line'].notna()].copy()
    if graph_data.empty:
        return data

    graph_data = gpd.GeoDataFrame(graph_data)
    graph_data['line'] = graph_data['line'].apply(shapely.wkt.loads)

    for g in geo_data['graph'].unique():
        graph = nx.Graph()
        edges_graph = graph_data[graph_data['graph'] == g].index
        lines = []
        for edge in edges_graph:

            node_start = graph_data.loc[edge, 'node_start']
            node_end = graph_data.loc[edge, 'node_end']
            distance = graph_data.loc[edge, 'distance']

            # graph.add_edge(node_start, node_end, distance)
            graph.add_edge(node_start, node_end, weight=distance)
            lines.append(graph_data.loc[edge, 'line'])

        nodes_graph_original = geo_data[geo_data['graph'] == g].index
        graph_object = MultiLineString(lines)
        data[name][g] = {'Graph': graph,
                         'GraphData': graph_data,
                         'GraphObject': graph_object,
                         'GeoData': geo_data.loc[nodes_graph_original]}

    return data


def get_complete_infrastructure(data, final_destination):

    options = pd.DataFrame()
    # Check final destination and add to option outside tolerance if applicable
    destination_point = get_geometry_plot_point(final_destination)
    options.loc['Destination', 'latitude'] = destination_point.y
    options.loc['Destination', 'longitude'] = destination_point.x

    options_to_concat = []
    for m in ['Shipping', 'Pipeline_Gas', 'Pipeline_Liquid']:

        # get all options of current mean of transport
        if m == 'Shipping':

            # get all options of current mean of transport
            options_shipping = data[m]['ports']
            if options_shipping.empty:
                continue
            options_shipping['graph'] = None

            options_to_concat.append(options_shipping)

        else:
            networks = data[m].keys()
            for n in networks:
                options_network = data[m][n]['GeoData'].copy()
                if not options_network.empty:
                    options_to_concat.append(options_network)

    if options_to_concat:
        options = pd.concat([options] + options_to_concat)

    # create common infrastructure column
    options['infrastructure'] = options.index
    if 'graph' in options.columns:
        graph_df = options[options['graph'].apply(lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x) if isinstance(x, (list, float)) else False)]
        options.loc[graph_df.index, 'infrastructure'] = options.loc[graph_df.index, 'infrastructure']

    return options


def create_weighted_routing_data_script(data, complete_infrastructure, infrastructure_data, path_processed_results,
                                        folder, column_to_sort=None):

    if column_to_sort is None:

        longitudes = data['longitude'].tolist()
        latitudes = data['latitude'].tolist()
        routes = data['routes'].tolist()
        quantities = data['quantity'].tolist()

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=100, maxtasksperchild=1)

        # Create an iterable of tuples, each containing the task ID and shared_dict
        task_args = list(zip(routes,
                             quantities,
                             longitudes,
                             latitudes,
                             itertools.repeat(complete_infrastructure),
                             itertools.repeat(infrastructure_data),
                             range(len(data.index))))

        # Start processing tasks and ensure parallelism
        geometry_results = []
        for result in tqdm(list(pool.map(get_geometry_segments, task_args))):
            geometry_results.append(result)

        # Close and join the worker pool
        pool.close()
        pool.join()

        result_dfs = []
        replacement_dict = {}
        n = 0
        for r in tqdm(geometry_results):
            if [*r[0].keys()]:
                keys = list(r[0].keys())

                new_keys = {k[0] for k in keys if k[0] not in replacement_dict}  # Identify new keys
                new_mapping = {key: i for i, key in enumerate(new_keys, start=n)}  # Create mappings for new keys
                replacement_dict.update(new_mapping)  # Update the replacement_dict in one step
                n += len(new_keys)  # Increment `n` by the number of new keys

                # Create a DataFrame directly
                result_df = pd.DataFrame({
                    'geometry': [replacement_dict[k[0]] for k in keys],
                    'commodity': [k[1] for k in keys],
                    'quantity': list(r[0].values()),
                    'transport_mean': list(r[1].values())
                })

                result_dfs.append(result_df)

        result_dfs = pd.concat(result_dfs)

        result_dfs = result_dfs.groupby(['geometry', 'commodity', 'transport_mean']).agg({"quantity": "sum"})
        result_dfs.reset_index(drop=False, inplace=True)

        reversed_dict = dict((v, k) for k, v in replacement_dict.items())
        result_dfs['geometry'] = result_dfs['geometry'].map(reversed_dict)
        result_dfs['quantity_MWh'] = result_dfs['quantity']

        result_dfs.to_csv(safe_output_path(
            path_processed_results, folder + '_routes_and_quantities.csv'))

    else:
        individual_data_dfs = []
        for value in data[column_to_sort].unique():

            subdata = data[data[column_to_sort] == value]

            if subdata.empty:
                continue

            longitudes = subdata['longitude'].tolist()
            latitudes = subdata['latitude'].tolist()
            routes = subdata['routes'].tolist()
            quantities = subdata['quantity'].tolist()

            # # Create a pool of worker processes
            pool = multiprocessing.Pool(processes=100, maxtasksperchild=1)
            #
            # Create an iterable of tuples, each containing the task ID and shared_dict
            task_args = list(zip(routes,
                                 quantities,
                                 longitudes,
                                 latitudes,
                                 itertools.repeat(complete_infrastructure),
                                 itertools.repeat(infrastructure_data),
                                 range(len(subdata.index))))

            task_args = tqdm(task_args)

            # Start processing tasks and ensure parallelism
            geometry_results = []
            for result in list(pool.map(get_geometry_segments, task_args)):
                geometry_results.append(result)

            # Close and join the worker pool
            pool.close()
            pool.join()

            # from joblib import Parallel, delayed
            # geometry_results = Parallel(n_jobs=100)(delayed(get_geometry_segments)(i) for i in task_args)

            result_dfs = []
            replacement_dict = {}
            n = 0
            for gr in tqdm(geometry_results):
                if [*gr[0].keys()]:
                    keys = list(gr[0].keys())

                    new_keys = {k[0] for k in keys if k[0] not in replacement_dict}  # Identify new keys
                    new_mapping = {key: i for i, key in enumerate(new_keys, start=n)}  # Create mappings for new keys
                    replacement_dict.update(new_mapping)  # Update the replacement_dict in one step
                    n += len(new_keys)  # Increment `n` by the number of new keys

                    # Create a DataFrame directly
                    result_df = pd.DataFrame({
                        'geometry': [replacement_dict[k[0]] for k in keys],
                        'commodity': [k[1] for k in keys],
                        'quantity': list(gr[0].values()),
                        'transport_mean': list(gr[1].values())
                    })

                    result_dfs.append(result_df)

            result_df = pd.concat(result_dfs)

            result_df = result_df.groupby(['geometry', 'commodity', 'transport_mean']).agg({"quantity": "sum"})
            result_df.reset_index(drop=False, inplace=True)

            reversed_dict = dict((v, k) for k, v in replacement_dict.items())
            result_df['geometry'] = result_df['geometry'].map(reversed_dict)

            result_df['commodity'] = value

            individual_data_dfs.append(result_df)

        weighted_routes = pd.concat(individual_data_dfs)
        weighted_routes.reset_index(drop=True, inplace=True)
        weighted_routes['quantity_MWh'] = weighted_routes['quantity']

        weighted_routes.to_csv(safe_output_path(
            path_processed_results, folder + '_routes_and_quantities.csv'))


def get_geometry_segments(args):

    number_used_infrastructure_local = {}
    type_used_infrastructure_local = {}

    r_local = args[0]
    if isinstance(r_local, str):
        r_local = ast.literal_eval(r_local)

    quantity_local = args[1]

    if quantity_local == 0:
        return {}, {}

    start_longitude = args[2]
    start_latitude = args[3]

    complete_infrastructure = args[4]
    infrastructure_data = args[5]

    commodity_local = None
    for m, r_segment in enumerate(r_local):

        if m == 0:
            # start
            commodity_local = r_segment[0]
            efficiency_factor = float(r_segment[1])
            if not 0 <= efficiency_factor <= 1:
                raise ValueError('Route efficiency factors must be between 0 and 1.')
            quantity_local *= efficiency_factor

            continue

        elif len(r_segment) == 3:
            # conversion
            commodity_local = r_segment[1]
            efficiency_factor = float(r_segment[2])
            if not 0 <= efficiency_factor <= 1:
                raise ValueError('Route efficiency factors must be between 0 and 1.')
            quantity_local *= efficiency_factor
        else:
            # transportation
            start = r_segment[0]
            if isinstance(r_segment[1], float):
                transport_mean_local = r_segment[2]
            else:
                transport_mean_local = r_segment[1]

            distance = r_segment[2]
            if distance == 0:
                if len(r_segment) >= 5:
                    efficiency_factor = float(r_segment[-1])
                    if not 0 <= efficiency_factor <= 1:
                        raise ValueError('Route efficiency factors must be between 0 and 1.')
                    quantity_local *= efficiency_factor
                continue

            destination = r_segment[3]

            if (transport_mean_local in ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid']) & (m == 1):
                end_longitude = complete_infrastructure.at[destination, 'longitude']
                end_latitude = complete_infrastructure.at[destination, 'latitude']

                line = LineString(
                    [Point([start_longitude, start_latitude]), Point([end_longitude, end_latitude])])

                if line not in [*number_used_infrastructure_local.keys()]:
                    number_used_infrastructure_local[(line, commodity_local)] = quantity_local
                    type_used_infrastructure_local[(line, commodity_local)] = transport_mean_local
                else:
                    number_used_infrastructure_local[(line, commodity_local)] += quantity_local

            elif (transport_mean_local in ['Road', 'New_Pipeline_Gas', 'New_Pipeline_Liquid']) & (m != 1):
                start_longitude = complete_infrastructure.at[start, 'longitude']
                start_latitude = complete_infrastructure.at[start, 'latitude']

                end_longitude = complete_infrastructure.at[destination, 'longitude']
                end_latitude = complete_infrastructure.at[destination, 'latitude']

                line = LineString(
                    [Point([start_longitude, start_latitude]), Point([end_longitude, end_latitude])])

                if (line, commodity_local) not in [*number_used_infrastructure_local.keys()]:
                    number_used_infrastructure_local[(line, commodity_local)] = quantity_local
                    type_used_infrastructure_local[(line, commodity_local)] = transport_mean_local
                else:
                    number_used_infrastructure_local[(line, commodity_local)] += quantity_local

            elif transport_mean_local == 'Shipping':
                # Now add shipping from start port to destination port
                start_location = [complete_infrastructure.loc[start, 'longitude'],
                                  complete_infrastructure.loc[start, 'latitude']]
                end_location = [complete_infrastructure.loc[destination, 'longitude'],
                                complete_infrastructure.loc[destination, 'latitude']]

                # ports
                if (Point(start_location), commodity_local) not in [*number_used_infrastructure_local.keys()]:
                    number_used_infrastructure_local[(Point(start_location), commodity_local)] = quantity_local
                    type_used_infrastructure_local[(Point(start_location), commodity_local)] = transport_mean_local
                else:
                    number_used_infrastructure_local[(Point(start_location), commodity_local)] += quantity_local

                if (Point(end_location), commodity_local) not in [*number_used_infrastructure_local.keys()]:
                    number_used_infrastructure_local[(Point(end_location), commodity_local)] = quantity_local
                    type_used_infrastructure_local[(Point(end_location), commodity_local)] = transport_mean_local
                else:
                    number_used_infrastructure_local[(Point(end_location), commodity_local)] += quantity_local

                # routes
                route = sr.searoute(start_location, end_location, append_orig_dest=True)

                last_coordinate = None
                adjustment = 0
                for coordinate in route.geometry['coordinates']:

                    if last_coordinate is not None:

                        if (last_coordinate[0] + adjustment == 180) & (coordinate[0] + adjustment == 180):
                            # coordinates move above 180 longitude
                            adjustment = -360
                            last_coordinate[0] = last_coordinate[0] + adjustment
                        elif (last_coordinate[0] + adjustment == -180) & (coordinate[0] + adjustment == -180):
                            # coordinates move below -180 longitude
                            adjustment = 360
                            last_coordinate[0] = last_coordinate[0] + adjustment

                        coordinate[0] = coordinate[0] + adjustment

                        line = LineString([last_coordinate, coordinate])
                        reverse_line = shapely.reverse(line)

                        line_not_in_dict = (line, commodity_local) not in [*number_used_infrastructure_local.keys()]
                        reverse_line_not_in_dict = (reverse_line, commodity_local) not in [*number_used_infrastructure_local.keys()]

                        if line_not_in_dict | reverse_line_not_in_dict:  # new line
                            number_used_infrastructure_local[(line, commodity_local)] = quantity_local
                            type_used_infrastructure_local[(line, commodity_local)] = transport_mean_local
                        else:
                            if not line_not_in_dict:  # line is in dict
                                number_used_infrastructure_local[(line, commodity_local)] += quantity_local
                            else:
                                number_used_infrastructure_local[(reverse_line, commodity_local)] += quantity_local

                    last_coordinate = coordinate

            else:
                graph = complete_infrastructure.at[start, 'graph']

                graph_object = infrastructure_data[transport_mean_local][graph]['Graph']

                path = nx.shortest_path(graph_object, start, destination)
                if len(path) < 2:
                    if len(r_segment) >= 5:
                        efficiency_factor = float(r_segment[-1])
                        if not 0 <= efficiency_factor <= 1:
                            raise ValueError('Route efficiency factors must be between 0 and 1.')
                        quantity_local *= efficiency_factor
                    continue

                last_node = None
                for node in path:

                    node = Point([complete_infrastructure.loc[node, 'longitude'],
                                  complete_infrastructure.loc[node, 'latitude']])

                    if last_node is not None:
                        line = LineString([last_node, node])
                        reverse_line = shapely.reverse(line)

                        line_not_in_dict = (line, commodity_local) not in [*number_used_infrastructure_local.keys()]
                        reverse_line_not_in_dict = (reverse_line, commodity_local) not in [
                            *number_used_infrastructure_local.keys()]

                        if line_not_in_dict | reverse_line_not_in_dict:  # new line
                            number_used_infrastructure_local[(line, commodity_local)] = quantity_local
                            type_used_infrastructure_local[(line, commodity_local)] = transport_mean_local
                        else:
                            if not line_not_in_dict:  # line is in dict
                                number_used_infrastructure_local[(line, commodity_local)] += quantity_local
                            else:
                                number_used_infrastructure_local[(reverse_line, commodity_local)] += quantity_local

                    last_node = node

            if len(r_segment) >= 5:
                efficiency_factor = float(r_segment[-1])
                if not 0 <= efficiency_factor <= 1:
                    raise ValueError('Route efficiency factors must be between 0 and 1.')
                quantity_local *= efficiency_factor

    return number_used_infrastructure_local, type_used_infrastructure_local


def get_ranked_routes(data):

    start_commodity = 'Hydrogen_Gas'
    number_sankeys = 10

    route_groups = defaultdict(
        lambda: {
            "count": 0,
            "csv_efficiencies": [],
            "calculated_efficiencies": [],
            "quantities": [],
            "example_route": "",
            "sections": defaultdict(list),
        }
    )

    total_rows = 0
    skipped_rows = 0

    for i in data.index:
        total_rows += 1
        row = data.loc[i, :]

        try:
            raw_route = ast.literal_eval(row["routes"])
            if not isinstance(raw_route, list):
                raise ValueError("Route is not a list")

            steps = []
            signature_parts = [f"COMMODITY:{start_commodity}"]
            current_commodity = start_commodity

            for raw_step in raw_route:
                if not isinstance(raw_step, tuple):
                    raise ValueError(f"Route step is not a tuple: {raw_step!r}")

                if len(raw_step) == 2:
                    to_commodity, efficiency = raw_step
                    from_commodity = current_commodity
                    to_commodity = str(to_commodity)
                    current_commodity = to_commodity

                    if from_commodity == to_commodity:
                        continue

                    efficiency = float(efficiency)
                    if not math.isfinite(efficiency):
                        raise ValueError(f"Invalid efficiency: {efficiency!r}")

                    label = f"Conversion: {from_commodity} -> {to_commodity}"
                    signature_parts.append(f"CONVERSION:{from_commodity}->{to_commodity}")
                    steps.append(
                        {
                            "index": len(steps) + 1,
                            "kind": "conversion",
                            "label": label,
                            "efficiency": efficiency,
                            "from_commodity": from_commodity,
                            "to_commodity": to_commodity,
                        }
                    )

                elif len(raw_step) == 3:
                    from_commodity, to_commodity, efficiency = raw_step
                    from_commodity = str(from_commodity)
                    to_commodity = str(to_commodity)
                    current_commodity = to_commodity

                    if from_commodity == to_commodity:
                        continue

                    efficiency = float(efficiency)
                    if not math.isfinite(efficiency):
                        raise ValueError(f"Invalid efficiency: {efficiency!r}")

                    label = f"Conversion: {from_commodity} -> {to_commodity}"
                    signature_parts.append(f"CONVERSION:{from_commodity}->{to_commodity}")
                    steps.append(
                        {
                            "index": len(steps) + 1,
                            "kind": "conversion",
                            "label": label,
                            "efficiency": efficiency,
                            "from_commodity": from_commodity,
                            "to_commodity": to_commodity,
                        }
                    )

                elif len(raw_step) == 5:
                    _from_node, transport_mean, _distance, _to_node, efficiency = raw_step
                    transport_mean = str(transport_mean)
                    transported_commodity = current_commodity or "UNKNOWN_COMMODITY"

                    efficiency = float(efficiency)
                    if not math.isfinite(efficiency):
                        raise ValueError(f"Invalid efficiency: {efficiency!r}")

                    label = f"Transport: {transported_commodity} via {transport_mean}"
                    signature_parts.append(
                        f"TRANSPORT:{transported_commodity}|{transport_mean}"
                    )
                    steps.append(
                        {
                            "index": len(steps) + 1,
                            "kind": "transport",
                            "label": label,
                            "efficiency": efficiency,
                            "from_commodity": transported_commodity,
                            "to_commodity": transported_commodity,
                        }
                    )

                else:
                    raise ValueError(
                        f"Unexpected route step length {len(raw_step)}: {raw_step!r}"
                    )

            calculated_route_efficiency = math.prod(
                step["efficiency"] for step in steps
            ) * 100
            route_signature = " | ".join(signature_parts)

        except (SyntaxError, ValueError, TypeError) as error:
            skipped_rows += 1
            print(f"Skipping row {total_rows}: {error}")
            continue

        group = route_groups[route_signature]
        group["count"] += 1
        group["example_route"] = group["example_route"] or row["routes"]
        group["calculated_efficiencies"].append(calculated_route_efficiency)

        if row.get("efficiency"):
            group["csv_efficiencies"].append(float(row["efficiency"]))
        if row.get("quantity"):
            group["quantities"].append(float(row["quantity"]))

        for step in steps:
            group["sections"][
                (
                    step["index"],
                    step["kind"],
                    step["label"],
                    step["from_commodity"],
                    step["to_commodity"],
                )
            ].append(step["efficiency"])

    top_routes = sorted(
        route_groups.items(),
        key=lambda item: (
            item[1]["count"],
            mean(item[1]["quantities"]) if item[1]["quantities"] else float("nan"),
        ),
        reverse=True,
    )[:number_sankeys]

    route_data = []

    for rank, (signature, group) in enumerate(top_routes, start=1):
        sections = []

        for (
            step_index,
            kind,
            label,
            from_commodity,
            to_commodity,
        ), efficiencies in sorted(group["sections"].items()):
            sections.append(
                {
                    "section_index": step_index,
                    "section_type": kind,
                    "section_label": label,
                    "from_commodity": from_commodity,
                    "to_commodity": to_commodity,
                    "avg_section_efficiency": mean(efficiencies),
                    "min_section_efficiency": min(efficiencies),
                    "max_section_efficiency": max(efficiencies),
                    "section_observations": len(efficiencies),
                }
            )

        route_data.append(
            {
                "rank": rank,
                "count": group["count"],
                "share_of_rows": group["count"] / max(total_rows - skipped_rows, 1),
                "avg_route_efficiency_from_csv_percent": (
                    mean(group["csv_efficiencies"])
                    if group["csv_efficiencies"]
                    else float("nan")
                ),
                "avg_route_efficiency_calculated_percent": (
                    mean(group["calculated_efficiencies"])
                    if group["calculated_efficiencies"]
                    else float("nan")
                ),
                "avg_quantity": (
                    mean(group["quantities"]) if group["quantities"] else float("nan")
                ),
                "avg_input_quantity_MWh": (
                    mean(group["quantities"]) if group["quantities"] else float("nan")
                ),
                "avg_input_quantity_TWh": (
                    mean(group["quantities"]) / 1_000_000
                    if group["quantities"]
                    else float("nan")
                ),
                "route_signature": signature,
                "example_route": group["example_route"],
                "sections": sections,
            }
        )

    ranked_routes = {
        "start_commodity": start_commodity,
        "number_sankeys": number_sankeys,
        "rows_read": total_rows,
        "rows_skipped": skipped_rows,
        "unique_route_clusters": len(route_groups),
        "routes": route_data,
    }

    return ranked_routes


def load_result(r, path_files, config_file_plotting, production_costs, with_routes=True):
    data = pd.read_csv(os.path.join(path_files, r + '_processed_results.csv'), index_col=0)
    if 'input_quantity_MWh' not in data.columns and 'quantity' in data.columns:
        data['input_quantity_MWh'] = data['quantity']
    strike_prices = _load_strike_prices_from_result_path(path_files)

    destination = load_destination(path_files, r)

    # overwrite production costs with h2 production costs
    data['production_costs'] = production_costs['Hydrogen_Gas']
    data['final_commodity'] = data.apply(
        lambda row: _get_final_route_commodity(row.get('routes'), row.get('start_commodity')),
        axis=1
    )
    data['commodity_price'] = data['final_commodity'].map(strike_prices).fillna(0)
    data['adjusted_costs'] = data['costs'] - data['commodity_price']

    min_prod_costs = data['production_costs'].min()
    max_prod_costs = data['production_costs'].max()

    if config_file_plotting['limit_scale']:
        Q1 = data.loc[np.isfinite(data['production_costs']), 'production_costs'].quantile(0.25)
        Q3 = data.loc[np.isfinite(data['production_costs']), 'production_costs'].quantile(0.75)
        IQR = Q3 - Q1

        max_prod_costs = Q3 + 1.5 * IQR

    norm_prod = mpl.colors.Normalize(vmin=min_prod_costs, vmax=max_prod_costs)

    min_conv_costs = max(0, data['conversion_costs'].min())
    max_conv_costs = data.loc[np.isfinite(data['conversion_costs']), 'conversion_costs'].max()

    norm_conv = mpl.colors.Normalize(vmin=min_conv_costs, vmax=max_conv_costs)

    min_trans_costs = max(0, data['transportation_costs'].min())
    max_trans_costs = data.loc[np.isfinite(data['transportation_costs']), 'transportation_costs'].max()

    if config_file_plotting['limit_scale']:
        Q1 = data.loc[np.isfinite(data['transportation_costs']), 'transportation_costs'].quantile(0.25)
        Q3 = data.loc[np.isfinite(data['transportation_costs']), 'transportation_costs'].quantile(0.75)
        IQR = Q3 - Q1

        max_trans_costs = Q3 + 1.5 * IQR

    norm_trans = mpl.colors.Normalize(vmin=min_trans_costs, vmax=max_trans_costs)

    min_total_costs = max(0, data['costs'].min())
    max_total_costs = data.loc[np.isfinite(data['costs']), 'costs'].max()

    if config_file_plotting['limit_scale']:
        Q1 = data.loc[np.isfinite(data['costs']), 'costs'].quantile(0.25)
        Q3 = data.loc[np.isfinite(data['costs']), 'costs'].quantile(0.75)
        IQR = Q3 - Q1

        max_total_costs = Q3 + 1.5 * IQR

    norm_total = mpl.colors.Normalize(vmin=min_total_costs, vmax=max_total_costs)

    finite_adjusted_costs = data.loc[np.isfinite(data['adjusted_costs']), 'adjusted_costs']
    if finite_adjusted_costs.empty:
        min_adjusted_costs = 0
        max_adjusted_costs = 0
    else:
        min_adjusted_costs = finite_adjusted_costs.min()
        max_adjusted_costs = finite_adjusted_costs.max()

        if config_file_plotting['limit_scale']:
            Q1 = finite_adjusted_costs.quantile(0.25)
            Q3 = finite_adjusted_costs.quantile(0.75)
            IQR = Q3 - Q1

            max_adjusted_costs = Q3 + 1.5 * IQR

    norm_adjusted_costs = mpl.colors.Normalize(vmin=min_adjusted_costs, vmax=max_adjusted_costs)

    min_efficiency = data['efficiency'].min()
    max_efficiency = data['efficiency'].max()

    if config_file_plotting['limit_scale']:
        Q1 = data['efficiency'].quantile(0.25)
        Q3 = data['efficiency'].quantile(0.75)
        IQR = Q3 - Q1

        max_efficiency = min(100, Q3 + 1.5 * IQR)

    norm_efficiency = mpl.colors.Normalize(vmin=min_efficiency, vmax=max_efficiency)

    min_all_costs = min(min_prod_costs, min_conv_costs, min_trans_costs, min_total_costs)
    max_all_costs = max(max_prod_costs, max_conv_costs, max_trans_costs, max_total_costs)
    norm_all = mpl.colors.Normalize(vmin=min_all_costs, vmax=max_all_costs)

    starting_locations = zip(data['longitude'].tolist(), data['latitude'].tolist())

    if with_routes:
        weighted_routes = pd.read_csv(
            os.path.join(path_files, r + '_routes_and_quantities.csv'), index_col=0)
        if 'quantity_MWh' in weighted_routes.columns:
            weighted_routes['quantity'] = weighted_routes['quantity_MWh']
        else:
            weighted_routes['quantity_MWh'] = weighted_routes['quantity']
        weighted_routes['quantity_TWh'] = weighted_routes['quantity_MWh'] / 1_000_000
        weighted_routes = weighted_routes.sort_values(by=['quantity'], ascending=False)
        weighted_routes['geometry'] = weighted_routes['geometry'].apply(shapely.wkt.loads)
    else:
        weighted_routes = pd.DataFrame()

    ranked_routes = get_ranked_routes(data)

    return data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination


def plot_comparison_plot(plot_type, comparisons, path_files, path_saving, config_file_plotting, production_costs, cmap,
                         boundaries, color_dictionary=None, nice_name_dictionary=None,
                         cost_type=None, transport_mean_line_styles=None, line_widths=None,
                         infrastructure_data=None, complete_infrastructure=None, country=None,
                         plot_width=15.69, subplot_height=None, distance_between=0.25, distance_left=0):

    mpl.rcParams.update({'font.size': 9,
                         'font.family': 'Times New Roman'})

    diff_lat = boundaries['max_latitude'] - boundaries['min_latitude']
    ratio_lat_lon = diff_lat / (boundaries['max_longitude'] - boundaries['min_longitude'])

    subplot_width = (plot_width - 3 * distance_between - distance_left) / 2
    if subplot_height is None:
        subplot_height = subplot_width * ratio_lat_lon

    if cost_type is None:
        saving_name = plot_type
    else:
        saving_name = cost_type

        if saving_name == '':
            saving_name = 'total_supply_costs'

    if country is not None:
        saving_name = country + '_' + saving_name

    for n, comparison in enumerate(comparisons):
        len_comp = len(comparison)

        if len_comp == 2:
            plot_height = subplot_height + distance_between * 2
        elif len_comp == 4:
            plot_height = 2 * subplot_height + distance_between * 3
        else:
            print('Comparison is not 2x2 or 4x4')
            continue

        # add space for legend
        if plot_type == 'costs':
            legend_height = 1.5
        elif plot_type == 'energy_carrier':
            legend_height = 1
        else:
            legend_height = 2.5

        plot_height += legend_height
        fig = plt.figure(figsize=(plot_width / 2.54, plot_height / 2.54))

        relative_distance_between_height = distance_between / plot_height
        relative_distance_between_width = distance_between / plot_width
        relative_distance_left = distance_left / plot_width
        relative_subplot_height = subplot_height / plot_height
        relative_subplot_width = subplot_width / plot_width
        legend_height_relative = legend_height / plot_height

        if len_comp == 2:
            ax1 = fig.add_axes((relative_distance_left + relative_distance_between_width,
                                legend_height_relative + relative_distance_between_height,
                                relative_subplot_width, relative_subplot_height))  # [left, bottom, width, height]

            ax2 = fig.add_axes((relative_distance_left + 2 * relative_distance_between_width + relative_subplot_width,
                                legend_height_relative + relative_distance_between_height,
                                relative_subplot_width, relative_subplot_height))

            axes = [ax1, ax2]

        else:
            ax1 = fig.add_axes((relative_distance_left + relative_distance_between_width,
                                legend_height_relative + 2 * relative_distance_between_height + relative_subplot_height,
                                relative_subplot_width, relative_subplot_height))  # [left, bottom, width, height]

            ax2 = fig.add_axes((relative_distance_left + 2 * relative_distance_between_width + relative_subplot_width,
                                legend_height_relative + 2 * relative_distance_between_height + relative_subplot_height,
                                relative_subplot_width, relative_subplot_height))

            ax3 = fig.add_axes((relative_distance_left + relative_distance_between_width,
                                legend_height_relative + relative_distance_between_height,
                                relative_subplot_width, relative_subplot_height))

            ax4 = fig.add_axes((relative_distance_left + 2 * relative_distance_between_width + relative_subplot_width,
                                legend_height_relative + relative_distance_between_height,
                                relative_subplot_width, relative_subplot_height))

            axes = [ax1, ax2, ax3, ax4]

        # costs plot have same colormap norm for better comparison
        norm = None
        max_total_costs = None
        if plot_type in ['costs', 'supply_curves']:

            all_data = pd.DataFrame()
            for r in comparison:
                data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination_location \
                    = load_result(r, path_files, config_file_plotting, production_costs, with_routes=False)

                all_data = pd.concat([all_data, data])

            if (cost_type is None) | (cost_type == ''):
                cost_type = 'costs'

            min_costs = all_data[cost_type].min()
            max_costs = all_data[cost_type].max()

            if config_file_plotting['limit_scale']:
                Q1 = all_data[cost_type].quantile(0.25)
                Q3 = all_data[cost_type].quantile(0.75)
                IQR = Q3 - Q1

                max_costs = Q3 + 1.5 * IQR

            norm = mpl.colors.Normalize(vmin=min_costs, vmax=max_costs)

        transport_means = []
        commodities = []
        all_data = {}

        for m, r in enumerate(comparison):
            data, weighted_routes, norm_prod, norm_conv, norm_trans, norm_total, norm_adjusted_costs, norm_efficiency, norm_all, ranked_routes, starting_locations, destination_location \
                = load_result(r, path_files, config_file_plotting, production_costs, with_routes=False)
            current_boundaries = resolve_plot_boundaries(
                config_file_plotting,
                data=data,
                destination_location=destination_location,
            )

            current_ax = axes[m]

            if cost_type == 'conversion_costs':
                limit_scale = False
            elif cost_type == 'transportation_costs':
                limit_scale = True
            else:
                limit_scale = True

            if r not in [*nice_name_dictionary.keys()]:
                nice_name_dictionary[r] = r

            if plot_type == 'costs':
                current_ax = get_number_figure(data, norm, cmap, current_boundaries, destination_location,
                                               column=cost_type, use_voronoi=True,
                                               production_costs=production_costs,
                                               ax=current_ax, return_fig=True, fig=fig, add_colorbar=False,
                                               fig_title=nice_name_dictionary[r], add_fig_title=True,
                                               limit_scale=limit_scale)

                all_data[r + '_' + cost_type] = data[['latitude', 'longitude', cost_type]]
                all_data[r + '_' + cost_type].columns = ['latitude', 'longitude', r]


            elif plot_type == 'efficiency':
                current_ax = get_number_figure(data, norm, cmap, current_boundaries, destination_location,
                                               column=plot_type, use_voronoi=True,
                                               production_costs=production_costs,
                                               ax=current_ax, return_fig=True, fig=fig, add_colorbar=False,
                                               fig_title=nice_name_dictionary[r], add_fig_title=True,
                                               limit_scale=limit_scale)

                all_data[r + '_efficiency'] = data[['latitude', 'longitude', 'efficiency']]
                all_data[r + '_efficiency'].columns = ['latitude', 'longitude', r]

            elif plot_type == 'energy_carrier':
                current_ax, commodities \
                    = get_energy_carrier_figure(data, current_boundaries, color_dictionary, nice_name_dictionary,
                                                destination_location, use_voronoi=True,
                                                production_costs=production_costs, ax=current_ax, fig=fig,
                                                fig_title=nice_name_dictionary[r], add_fig_title=True,
                                                add_legend=False, return_fig=True, return_handles=True,
                                                existing_commodities=commodities)

                all_data[r + '_commodity'] = data[['latitude', 'longitude', 'start_commodity']]
                all_data[r + '_commodity'].columns = ['latitude', 'longitude', r]

            elif plot_type == 'routes':
                current_ax, commodities, transport_means \
                    = get_routes_figure(data, transport_mean_line_styles, line_widths, color_dictionary,
                                        nice_name_dictionary, infrastructure_data, complete_infrastructure,
                                        current_boundaries,
                                        destination_location, fig_title=nice_name_dictionary[r], ax=current_ax,
                                        return_fig=True,  add_legend=False, return_handles=True,
                                        existing_commodities=commodities, existing_transport_means=transport_means,
                                        add_fig_title=True)

            elif plot_type == 'supply_curves':

                data = data[data['costs'].notna()]  # eher inf, oder?
                data = data[data['cost_route'].notna()]

                current_ax = get_supply_curves(data.copy(), color_dictionary, nice_name_dictionary, add_legend=False, add_fig_title=True, return_fig=True,
                                               fig_title=nice_name_dictionary[r], country=country, ax=current_ax,
                                               fig=fig, production_costs=production_costs, ylim=max_total_costs,
                                               current_ax=m)

                country_locations = production_costs[
                    production_costs['country_start'] == country
                    ].index.tolist()

                country_locations = list(
                    set(country_locations).intersection(data.index.tolist())
                )

                data_country = data.loc[country_locations, :].copy()
                data_country['input_quantity_MWh'] = data_country['quantity']
                data_country['efficiency_percent'] = data_country['efficiency']
                data_country['delivered_quantity_TWh'] = (
                    data_country['input_quantity_MWh']
                    * data_country['efficiency_percent']
                    / 100
                    / 1_000_000
                )

                for entry in data_country['commodities']:
                    values = ast.literal_eval(entry)
                    for v in values:
                        commodities.append(v[0])

                commodities = list(set(commodities))

                all_data[r + '_cost_routes'] = data_country[['latitude', 'longitude', 'cost_route']]
                all_data[r + '_cost_routes'].columns = ['latitude', 'longitude', r + '_cost_route']

                all_data[r + '_commodity_routes'] = data_country[['latitude', 'longitude', 'commodities']]
                all_data[r + '_commodity_routes'].columns = ['latitude', 'longitude', r + '_commodities']

                all_data[r + '_quantity'] = data_country[[
                    'latitude',
                    'longitude',
                    'input_quantity_MWh',
                    'efficiency_percent',
                    'delivered_quantity_TWh',
                ]]
                all_data[r + '_quantity'].columns = [
                    'latitude',
                    'longitude',
                    r + '_input_quantity_MWh',
                    r + '_efficiency_percent',
                    r + '_delivered_quantity_TWh',
                ]

        if plot_type in ['costs', 'efficiency']:

            height = 0.2 / plot_height

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

            cbar_ax = fig.add_axes((0.05, legend_height_relative - height, 0.9, height))  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='max')
            cbar.set_label('€ / MWh', rotation=0, labelpad=5)

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

        if plot_type == 'energy_carrier':
            # commodity legend
            if len(commodities) <= 4:
                ncols = len(commodities)
            else:
                ncols = math.ceil(len(commodities) / 2)

            fig.legend(handles=commodities, ncols=ncols, bbox_to_anchor=(0.5, legend_height_relative * 1.25), loc='upper center',
                       labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=0.5, fontsize=9)

        if plot_type == 'routes':

            # commodity legend
            if len(commodities) <= 4:
                ncols = len(commodities)
            else:
                ncols = math.ceil(len(commodities) / 2)

            fig.legend(handles=transport_means, loc='upper center', ncols=3,
                       bbox_to_anchor=(0.5, legend_height_relative * 1.075), title='Transport Mean',
                       labelspacing=0.1, handletextpad=0.1, columnspacing=0.25, handlelength=1)

            fig.legend(handles=commodities, loc='upper center', ncol=ncols,
                       bbox_to_anchor=(0.5, legend_height_relative * 1.075 - 0.19), title='Commodity',
                       labelspacing=0.1, handletextpad=0.1, columnspacing=0.25)

        if plot_type == 'supply_curves':
            # Add labels, legend, and titles
            labels = ['Production Costs', 'Conversion Costs', 'Transport Costs']
            colors = {'Production Costs': 'cornflowerblue', 'Conversion Costs': 'lightcoral', 'Transport Costs': 'gold'}

            cost_handles = [
                mlines.Line2D([], [], color='cornflowerblue', linewidth=6, label='Production Costs', markersize=5),
                mlines.Line2D([], [], color='lightcoral', linewidth=6, label='Conversion Costs', markersize=5),
                mlines.Line2D([], [], color='khaki', linewidth=6, label='Transport Costs', markersize=5)
            ]

            commodity_handles = []

            for c in color_dictionary.keys():
                if c in commodities:
                    commodity_handles.append(
                        mlines.Line2D(
                            [],
                            [],
                            color=color_dictionary[c],
                            marker='s',
                            linestyle='None',
                            markersize=5,
                            label=nice_name_dictionary[c]
                        )
                    )

            if len(commodity_handles) < 3:
                ncols = len(commodity_handles)
            else:
                ncols = 3

            if len_comp == 2:
                fig.text(-0.01, 0.725, 'Costs [€ / MWh]', va='center', ha='left', fontdict={'fontsize': 9}, rotation=90)  # 46

                fig.legend(handles=cost_handles, fontsize=9, bbox_to_anchor=(0.475, 0.365), ncols=3,
                           loc='upper center')

                fig.legend(
                    handles=commodity_handles,
                    loc='upper center',
                    ncol=ncols,
                    bbox_to_anchor=(0.475, 0.295),
                    title='Commodities',
                    labelspacing=0.1,
                    handletextpad=0.1,
                    columnspacing=0.25,
                    fontsize=9,
                    title_fontsize=9
                )
            else:

                fig.text(-0.01, 0.455, 'Costs [€ / MWh]', va='center', ha='left', fontdict={'fontsize': 9}, rotation=90) # 46
                fig.text(-0.01, 0.82, 'Costs [€ / MWh]', va='center', ha='left', fontdict={'fontsize': 9}, rotation=90) # oben 815

                fig.legend(handles=cost_handles, fontsize=9, bbox_to_anchor=(0.475, 0.2), ncols=3,
                           loc='upper center')

                fig.legend(
                    handles=commodity_handles,
                    loc='upper center',
                    ncol=ncols,
                    bbox_to_anchor=(0.475, 0.15),
                    title='Commodities',
                    labelspacing=0.1,
                    handletextpad=0.1,
                    columnspacing=0.25,
                    fontsize=9,
                    title_fontsize=9
                )

            # fig.text(0.5, 0.2, 'Potential Quantities [TWh]', va='bottom', ha='center', fontdict={'fontsize': 9})

            plt.subplots_adjust(left=0.2)

            # fig.set_ylabel('€ / MWh', fontdict={'fontsize': 9})
            # fig.set_xlabel('TWh', fontdict={'fontsize': 9})

        comparison_filename = str(n) + '_' + saving_name + '_comparison'
        fig.savefig(safe_output_path(path_saving, comparison_filename + '.png'),
                    bbox_inches='tight', dpi=600)
        fig.savefig(safe_output_path(path_saving, comparison_filename + '.svg'),
                    bbox_inches='tight')

        plt.close(fig)

        # all_data = pd.DataFrame(all_data, columns=[*all_data.keys()])
        all_data_df = None
        for df_1 in all_data.values():
            if all_data_df is None:
                all_data_df = df_1
            else:
                all_data_df = pd.merge(
                    df_1, all_data_df,
                    on=["latitude", "longitude"],
                    how="outer"
                )

        all_data_df.to_excel(safe_output_path(path_saving, comparison_filename + '.xlsx'))


def match_routing_results(result_list, result_names, complete_infrastructure, infrastructure_data):

    common_index = result_list[0].index.tolist()
    for r in result_list:
        common_index = list(set(common_index).intersection(r.index.tolist()))

    comparison_df = pd.DataFrame(0, index=common_index, columns=result_names)
    for n, r in enumerate(result_names):
        comparison_df[r] = result_list[n].loc[common_index, 'costs']

    comparison_df['lowest_value_col'] = comparison_df.idxmin(axis=1)
    country_matching = []
    for n, r in enumerate(result_names):
        result_df = result_list[n]

        lowest_val_index = comparison_df[comparison_df['lowest_value_col'] == r].index

        # matching.append(result_df.loc[lowest_val_index, :])
        matching_df = result_df.loc[lowest_val_index, :]

        longitudes = matching_df['longitude'].tolist()
        latitudes = matching_df['latitude'].tolist()
        routes = matching_df['routes'].tolist()
        quantities = matching_df['quantity'].tolist()

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=100, maxtasksperchild=1)

        # Create an iterable of tuples, each containing the task ID and shared_dict
        task_args = zip(routes,
                        quantities,
                        longitudes,
                        latitudes,
                        itertools.repeat(complete_infrastructure),
                        itertools.repeat(infrastructure_data),
                        range(len(matching_df.index)))

        # Start processing tasks and ensure parallelism
        geometry_results = []
        for result in tqdm(list(pool.map(get_geometry_segments, task_args))):
            geometry_results.append(result)

        # Close and join the worker pool
        pool.close()
        pool.join()

        result_dfs = []
        replacement_dict = {}
        n = 0
        for gr in tqdm(geometry_results):
            if [*gr[0].keys()]:
                keys = list(gr[0].keys())

                new_keys = {k[0] for k in keys if k[0] not in replacement_dict}  # Identify new keys
                new_mapping = {key: i for i, key in enumerate(new_keys, start=n)}  # Create mappings for new keys
                replacement_dict.update(new_mapping)  # Update the replacement_dict in one step
                n += len(new_keys)  # Increment `n` by the number of new keys

                # Create a DataFrame directly
                result_df = pd.DataFrame({
                    'geometry': [replacement_dict[k[0]] for k in keys],
                    'commodity': [k[1] for k in keys],
                    'quantity': list(gr[0].values()),
                    'transport_mean': list(gr[1].values())
                })

                result_dfs.append(result_df)

        result_dfs = pd.concat(result_dfs)

        result_dfs = result_dfs.groupby(['geometry', 'commodity', 'transport_mean']).agg({"quantity": "sum"})
        result_dfs.reset_index(drop=False, inplace=True)

        reversed_dict = dict((v, k) for k, v in replacement_dict.items())
        result_dfs['geometry'] = result_dfs['geometry'].map(reversed_dict)

        result_dfs['commodity'] = r

        country_matching.append(result_dfs)

    country_matching = pd.concat(country_matching)

    return country_matching
