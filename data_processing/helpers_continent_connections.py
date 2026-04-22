import itertools
import json
import math

import geopandas as gpd
import pandas as pd
import cartopy.io.shapereader as shpreader


def _load_world():
    country_shapefile = shpreader.natural_earth(
        resolution='10m',
        category='cultural',
        name='admin_0_countries_deu',
    )
    world = gpd.read_file(country_shapefile)[['CONTINENT', 'geometry']].copy()
    return world.set_crs('EPSG:4326', allow_override=True)


def _initialize_connections(continent_names):
    return {continent: {continent} for continent in continent_names}


def _register_connections(connections, continent_names):
    continent_names = {continent for continent in continent_names if isinstance(continent, str) and continent}
    if not continent_names:
        return

    for continent in continent_names:
        connections.setdefault(continent, set()).add(continent)

    for continent_a, continent_b in itertools.combinations(continent_names, 2):
        connections[continent_a].add(continent_b)
        connections[continent_b].add(continent_a)


def _build_reachability(connections):
    reachability = {}

    for start_continent in connections:
        visited = set()
        stack = [start_continent]

        while stack:
            continent = stack.pop()
            if continent in visited:
                continue

            visited.add(continent)
            stack.extend(connections.get(continent, set()) - visited)

        reachability[start_continent] = sorted(visited)

    return reachability


def _get_pipeline_continents_by_graph(node_locations, world):
    if node_locations.empty:
        return {}

    nodes = node_locations.copy()
    nodes['longitude'] = pd.to_numeric(nodes['longitude'], errors='coerce')
    nodes['latitude'] = pd.to_numeric(nodes['latitude'], errors='coerce')
    nodes.dropna(subset=['longitude', 'latitude', 'graph'], inplace=True)

    if nodes.empty:
        return {}

    geometry = gpd.points_from_xy(nodes['longitude'], nodes['latitude'])
    nodes_gdf = gpd.GeoDataFrame(nodes[['graph']].copy(), geometry=geometry, crs='EPSG:4326')

    joined = gpd.sjoin(nodes_gdf, world[['CONTINENT', 'geometry']], how='left', predicate='within')
    joined.dropna(subset=['CONTINENT'], inplace=True)

    if joined.empty:
        return {}

    return joined.groupby('graph')['CONTINENT'].agg(lambda values: set(values)).to_dict()


def build_continent_connectivity(landmasses, gas_nodes, oil_nodes):
    world = _load_world()
    continent_names = sorted(world['CONTINENT'].dropna().unique().tolist())
    connections = _initialize_connections(continent_names)

    landmasses_gdf = landmasses.copy()
    if landmasses_gdf.crs is None:
        landmasses_gdf = landmasses_gdf.set_crs('EPSG:4326', allow_override=True)

    for geometry in landmasses_gdf['geometry']:
        if geometry is None or geometry.is_empty:
            continue

        touching_continents = world.loc[world.intersects(geometry), 'CONTINENT'].dropna().unique().tolist()
        _register_connections(connections, touching_continents)

    for graph_continents in _get_pipeline_continents_by_graph(gas_nodes, world).values():
        _register_connections(connections, graph_continents)

    for graph_continents in _get_pipeline_continents_by_graph(oil_nodes, world).values():
        _register_connections(connections, graph_continents)

    direct_connections = {
        continent: sorted(neighbours)
        for continent, neighbours in connections.items()
    }
    reachable_continents = _build_reachability(connections)

    return {
        'direct_connections': direct_connections,
        'reachable_continents': reachable_continents,
    }


def save_continent_connectivity(path_file, continent_connectivity):
    with open(path_file, 'w', encoding='utf-8') as file:
        json.dump(continent_connectivity, file, indent=2, sort_keys=True)
