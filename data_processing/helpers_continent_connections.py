import json
import geopandas as gpd
import pandas as pd

from data_processing.natural_earth_data import load_world

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


def _load_world(path_raw_data=None):
    world = load_world(path_raw_data)[['CONTINENT', 'geometry']].copy()
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

    nodes = add_continents_to_pipeline_nodes(node_locations, world=world)
    nodes['longitude'] = pd.to_numeric(nodes['longitude'], errors='coerce')
    nodes['latitude'] = pd.to_numeric(nodes['latitude'], errors='coerce')
    nodes.dropna(subset=['longitude', 'latitude', 'graph'], inplace=True)

    if nodes.empty:
        return {}

    nodes = nodes[nodes['continent'].notna() & (nodes['continent'].astype(str).str.lower() != 'nan')]
    if nodes.empty:
        return {}

    return nodes.groupby('graph')['continent'].agg(lambda values: sorted(set(values))).to_dict()


def add_continents_to_pipeline_nodes(node_locations, path_raw_data=None, world=None, nearest_tolerance_m=100000):
    """Attach continent metadata to pipeline nodes with one vectorized spatial lookup."""
    if node_locations is None:
        return pd.DataFrame(columns=['longitude', 'latitude', 'graph', 'continent'])

    nodes = node_locations.copy()
    if 'continent' not in nodes.columns:
        nodes['continent'] = pd.NA

    if nodes.empty or not {'longitude', 'latitude'}.issubset(nodes.columns):
        return nodes

    missing_continent = nodes['continent'].isna() | (nodes['continent'].astype(str).str.lower() == 'nan')
    if not missing_continent.any():
        return nodes

    lookup = nodes.loc[missing_continent, ['longitude', 'latitude']].copy()
    lookup['longitude'] = pd.to_numeric(lookup['longitude'], errors='coerce')
    lookup['latitude'] = pd.to_numeric(lookup['latitude'], errors='coerce')
    lookup.dropna(subset=['longitude', 'latitude'], inplace=True)

    if lookup.empty:
        return nodes

    if world is None:
        world = _load_world(path_raw_data)
    else:
        world = world[['CONTINENT', 'geometry']].copy()
        world = world.set_crs('EPSG:4326', allow_override=True)

    points = gpd.GeoDataFrame(
        lookup.copy(),
        geometry=gpd.points_from_xy(lookup['longitude'], lookup['latitude']),
        crs='EPSG:4326',
    )

    joined = gpd.sjoin(points, world[['CONTINENT', 'geometry']], how='left', predicate='within')
    joined = joined[joined['CONTINENT'].notna()]
    if not joined.empty:
        joined = joined[~joined.index.duplicated(keep='first')]
        nodes.loc[joined.index, 'continent'] = joined['CONTINENT'].astype(str)

    unresolved = lookup.index[nodes.loc[lookup.index, 'continent'].isna()
                              | (nodes.loc[lookup.index, 'continent'].astype(str).str.lower() == 'nan')]
    if len(unresolved) > 0:
        try:
            nearest_points = points.loc[unresolved].copy()
            nearest = gpd.sjoin_nearest(
                nearest_points.to_crs('EPSG:3857'),
                world[['CONTINENT', 'geometry']].to_crs('EPSG:3857'),
                how='left',
                distance_col='distance_to_nearest_land',
            )
            nearest = nearest[
                (nearest['distance_to_nearest_land'] <= nearest_tolerance_m)
                & nearest['CONTINENT'].notna()
            ]
            if not nearest.empty:
                nearest = nearest[~nearest.index.duplicated(keep='first')]
                nodes.loc[nearest.index, 'continent'] = nearest['CONTINENT'].astype(str)
        except Exception:
            pass

    return nodes


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
