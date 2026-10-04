"""Shared origin-polygon semantics for routing and benchmark preparation."""

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.wkt import loads


def get_start_polygon_nodes(location_data, infrastructure, configuration):
    """Include polygon boundaries, exclude holes and synthetic destination nodes.

    Missing/point geometry retains the point-start behavior.
    """
    if not configuration.get('zero_distance_to_infrastructure_in_start_polygon', False):
        return pd.Index([])
    geometry = location_data.loc['Start'].get('geometry')
    if isinstance(geometry, str):
        geometry = loads(geometry)
    if getattr(geometry, 'geom_type', None) not in ('Polygon', 'MultiPolygon'):
        return pd.Index([])
    return pd.Index([node for node, row in infrastructure.iterrows()
                     if node not in ('Start', 'Destination')
                     and geometry.covers(Point(row['longitude'], row['latitude']))])


def apply_start_polygon_distances(values, target_nodes, origin_nodes, configuration,
                                  start_polygon_nodes=(), iteration=None):
    """Return an origin-by-target matrix; only Start in iteration 0 is changed."""
    values = np.asarray(values)
    if iteration == 0 and configuration.get(
            'zero_distance_to_infrastructure_in_start_polygon', False):
        values = values.copy()
        origins = np.asarray(origin_nodes, dtype=object) == 'Start'
        targets = pd.Index(target_nodes).isin(start_polygon_nodes)
        values[np.ix_(origins, targets)] = 0
    return values
