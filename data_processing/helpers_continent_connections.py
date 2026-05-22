import json
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd


GEOGRAPHIC_LAND_CONTINENT_CONNECTIONS = {
    'Africa': {'Africa', 'Asia', 'Europe'},
    'Antarctica': {'Antarctica'},
    'Asia': {'Africa', 'Asia', 'Europe'},
    'Europe': {'Africa', 'Asia', 'Europe'},
    'North America': {'North America', 'South America'},
    'Oceania': {'Oceania'},
    'Seven seas (open ocean)': {'Seven seas (open ocean)'},
    'South America': {'North America', 'South America'},
}


def _load_world():
    country_shapefile = shpreader.natural_earth(
        resolution='10m',
        category='cultural',
        name='admin_0_countries_deu',
    )
    world = gpd.read_file(country_shapefile)[['CONTINENT', 'geometry']].copy()
    return world.set_crs('EPSG:4326', allow_override=True)


def _register_connection(connections, continent_a, continent_b):
    if not continent_a or not continent_b:
        return

    connections.setdefault(continent_a, set()).add(continent_a)
    connections.setdefault(continent_b, set()).add(continent_b)
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
    if node_locations is None or node_locations.empty:
        return {}

    nodes = node_locations.copy()
    nodes['longitude'] = pd.to_numeric(nodes['longitude'], errors='coerce')
    nodes['latitude'] = pd.to_numeric(nodes['latitude'], errors='coerce')
    nodes.dropna(subset=['longitude', 'latitude', 'graph'], inplace=True)

    if nodes.empty:
        return {}

    nodes_gdf = gpd.GeoDataFrame(
        nodes[['graph']].copy(),
        geometry=gpd.points_from_xy(nodes['longitude'], nodes['latitude']),
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(nodes_gdf, world[['CONTINENT', 'geometry']], how='left', predicate='within')
    joined.dropna(subset=['CONTINENT'], inplace=True)

    if joined.empty:
        return {}

    return joined.groupby('graph')['CONTINENT'].agg(lambda values: sorted(set(values))).to_dict()


def build_continent_connectivity(landmasses, gas_nodes, oil_nodes):
    connections = {
        continent: set(neighbours)
        for continent, neighbours in GEOGRAPHIC_LAND_CONTINENT_CONNECTIONS.items()
    }

    world = _load_world()

    for graph_continents in _get_pipeline_continents_by_graph(gas_nodes, world).values():
        for index, continent_a in enumerate(graph_continents):
            for continent_b in graph_continents[index + 1:]:
                _register_connection(connections, continent_a, continent_b)

    for graph_continents in _get_pipeline_continents_by_graph(oil_nodes, world).values():
        for index, continent_a in enumerate(graph_continents):
            for continent_b in graph_continents[index + 1:]:
                _register_connection(connections, continent_a, continent_b)

    return {
        'reachable_continents': _build_reachability(connections),
    }


def save_continent_connectivity(path_file, continent_connectivity):
    with open(path_file, 'w', encoding='utf-8') as file:
        json.dump(continent_connectivity, file, indent=2, sort_keys=True)
