import random
import math

from shapely.geometry import Polygon, Point, MultiPolygon
import geopandas as gpd

from data_processing.natural_earth_data import load_states, load_world


def _load_world():
    return load_world()


def _load_states():
    return load_states()


def _get_destination_countries_from_config(config_file):
    if 'destination_countries' not in config_file:
        raise KeyError("Missing required configuration key 'destination_countries'.")

    return config_file['destination_countries']


def _unique_preserve_order(values):
    return list(dict.fromkeys(values))


def create_rectangle(minimal_latitude, maximal_latitude, minimal_longitude, maximal_longitude):
    """Create a rectangular lon/lat polygon from latitude and longitude bounds."""
    return Polygon([Point([minimal_longitude, minimal_latitude]),
                    Point([minimal_longitude, maximal_latitude]),
                    Point([maximal_longitude, maximal_latitude]),
                    Point([maximal_longitude, minimal_latitude])])


def get_boundaries_from_config(config_file, prefix, use_minimal_example=None):
    """
    Read latitude/longitude bounds from explicitly prefixed config keys.
    """
    if use_minimal_example is None:
        use_minimal_example = config_file.get('use_minimal_example', False)
    if use_minimal_example:
        return 35, 71, -25, 45

    required_keys = [
        prefix + 'minimal_latitude',
        prefix + 'maximal_latitude',
        prefix + 'minimal_longitude',
        prefix + 'maximal_longitude',
    ]
    missing_keys = [key for key in required_keys if key not in config_file]
    if missing_keys:
        raise KeyError('Missing required configuration keys: ' + ', '.join(missing_keys))

    return (config_file[prefix + 'minimal_latitude'], config_file[prefix + 'maximal_latitude'],
            config_file[prefix + 'minimal_longitude'], config_file[prefix + 'maximal_longitude'])


def create_country_state_polygon(country_states, world=None, states=None,
                                 use_biggest_landmass=False):
    """Create one polygon from a {country: [states]} mapping."""
    if not country_states:
        raise ValueError('No countries configured for polygon creation.')

    if world is None:
        world = _load_world()
    if states is None:
        states = _load_states()

    first = True
    combined_location = None
    selected_countries = []
    selected_continents = []

    for country in sorted(country_states.keys()):
        country_data = world[world['NAME_EN'] == country]
        if country_data.empty:
            raise ValueError('Country not found in Natural Earth data: ' + str(country))

        selected_countries.append(country)
        selected_continents += country_data['CONTINENT'].dropna().tolist()

        if country_states[country]:
            for state in sorted(country_states[country]):
                state_data = states[states['name'] == state]
                if state_data.empty:
                    raise ValueError('State not found in Natural Earth data: ' + str(state))

                geom = state_data['geometry'].values[0]
                combined_location = geom if first else combined_location.union(geom)
                first = False
        else:
            geom = country_data['geometry'].values[0]
            combined_location = geom if first else combined_location.union(geom)
            first = False

    combined_location = combined_location.buffer(0)

    if use_biggest_landmass and len(country_states) == 1 and isinstance(combined_location, MultiPolygon):
        combined_location = max(combined_location.geoms, key=lambda geom: geom.area)

    return {'location': combined_location,
            'countries': _unique_preserve_order(selected_countries),
            'continents': _unique_preserve_order(selected_continents),
            'country_states': country_states}


def get_start_location_information(config_file, world=None, states=None):
    """Build the configured start-location sampling polygon."""
    if world is None:
        world = _load_world()
    if states is None:
        states = _load_states()

    if config_file.get('use_minimal_example', False):
        europe = world[world['CONTINENT'] == 'Europe']
        return {'location': europe.unary_union.buffer(0),
                'countries': europe['NAME_EN'].dropna().tolist(),
                'continents': ['Europe'],
                'country_states': {}}

    area_type = config_file.get('start_location_area_type', 'rectangle')
    if area_type == 'countries':
        return create_country_state_polygon(
            config_file.get('start_location_countries', {}),
            world=world, states=states,
            use_biggest_landmass=config_file.get('use_biggest_landmass', False))

    if area_type != 'rectangle':
        raise ValueError("start_location_area_type must be 'rectangle' or 'countries'.")

    min_lat, max_lat, min_lon, max_lon = get_boundaries_from_config(
        config_file, prefix='start_location_')
    rectangle = create_rectangle(min_lat, max_lat, min_lon, max_lon)
    countries = world[world.geometry.intersects(rectangle)]
    return {'location': rectangle,
            'countries': countries['NAME_EN'].dropna().tolist(),
            'continents': countries['CONTINENT'].dropna().tolist(),
            'country_states': {}}


def get_destination_information(config_file, world=None, states=None):
    """
    Build destination geometry and derive destination countries and continents from the same source.

    @param dict config_file: dictionary with configurations
    @param geopandas.GeoDataFrame world: optional country geometries
    @param geopandas.GeoDataFrame states: optional state geometries
    @return: dict with location, countries, continents and country-state mapping
    """

    if world is None:
        world = _load_world()

    if config_file['destination_type'] == 'location':
        destination_location = Point(config_file['destination_location'])
        matching_countries = world[world['geometry'].contains(destination_location)]

        if matching_countries.empty:
            matching_countries = world[world['geometry'].intersects(destination_location)]

        if matching_countries.empty:
            raise ValueError('Destination location is not inside a Natural Earth country geometry.')

        destination_countries = matching_countries['NAME_EN'].dropna().tolist()
        destination_continents = matching_countries['CONTINENT'].dropna().tolist()

        return {'location': destination_location,
                'countries': _unique_preserve_order(destination_countries),
                'continents': _unique_preserve_order(destination_continents),
                'country_states': {}}

    return create_country_state_polygon(
        _get_destination_countries_from_config(config_file),
        world=world, states=states,
        use_biggest_landmass=config_file['use_biggest_landmass'])


def create_polygons(hydrogen_costs_and_quantities):
    """
    creates era5 polygon based on coordinates and attaches them to cost DataFrame

    @param hydrogen_costs_and_quantities: DataFrame containing latitude and longitude
    @return: hydrogen_costs_and_quantities DataFrame
    """

    for i in hydrogen_costs_and_quantities.index:
        lon = hydrogen_costs_and_quantities.at[i, 'longitude']
        lat = hydrogen_costs_and_quantities.at[i, 'latitude']

        polygon = Polygon([Point([lon + 0.125, lat + 0.125]),
                           Point([lon + 0.125, lat - 0.125]),
                           Point([lon - 0.125, lat - 0.125]),
                           Point([lon - 0.125, lat + 0.125]), ])

        hydrogen_costs_and_quantities.at[i, 'polygon'] = polygon

    return hydrogen_costs_and_quantities


def round_to_quarter(coordinate):
    """
    levelized costs are ordered in a grid. Find the closest grid to a given coordinate

    @param float coordinate: given latitude or longitude
    @return: the closest grid coordinate
    """

    # Find the fractional part
    fraction = coordinate % 1

    # Round the fractional part to the nearest .25, .5, .75, or 0
    if fraction <= 0.125:
        return int(coordinate) + 0.00
    elif fraction <= 0.375:
        return int(coordinate) + 0.25
    elif fraction <= 0.625:
        return int(coordinate) + 0.50
    elif fraction <= 0.875:
        return int(coordinate) + 0.75
    else:
        return int(coordinate) + 1.00


def randlatlon1(min_latitude, max_latitude, min_longitude, max_longitude):

    """
    Creates latitude and longitude and checks if within boundaries

    @param float min_latitude: minimal value for latitude
    @param float max_latitude: maximal value for latitude
    @param float min_longitude: minimal value for longitude
    @param float max_longitude: maximal value for longitude
    @return: tuple with longitude, latitude
    """

    def is_within_boundaries(lat, lon):

        """
        check if latitude and longitude is within boundaries

        @param float lat: latitude value
        @param float lon: longitude value
        @return: boolean value if in boundaries
        """

        return min_latitude <= lat <= max_latitude and min_longitude <= lon <= max_longitude

    while True:
        pi = math.pi
        cf = 180.0 / pi  # radians to degrees Correction Factor

        # get a random Gaussian 3D vector:
        gx = random.gauss(0.0, 1.0)
        gy = random.gauss(0.0, 1.0)
        gz = random.gauss(0.0, 1.0)

        # normalize to an equidistributed (x,y,z) point on the unit sphere:
        norm2 = gx*gx + gy*gy + gz*gz
        norm1 = 1.0 / math.sqrt(norm2)
        x = gx * norm1
        y = gy * norm1
        z = gz * norm1

        radLat = math.asin(z)      # latitude in radians
        radLon = math.atan2(y, x)   # longitude in radians

        Lat = round(cf*radLat, 5)
        Lon = round(cf*radLon, 5)

        if is_within_boundaries(Lat, Lon):
            return Lon, Lat


def get_destination(config_file):
    # create shapely object of destination

    import logging
    logging.getLogger("fiona").setLevel(logging.ERROR)
    logging.getLogger("fiona._env").setLevel(logging.ERROR)

    return get_destination_information(config_file)['location']
