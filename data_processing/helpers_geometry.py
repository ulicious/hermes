import random
import math

from shapely.geometry import Polygon, Point, MultiPolygon
import geopandas as gpd
import cartopy.io.shapereader as shpreader


def _load_world():
    country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    return gpd.read_file(country_shapefile)


def _load_states():
    state_shapefile = shpreader.natural_earth(resolution='10m', category='cultural',
                                              name='admin_1_states_provinces')
    return gpd.read_file(state_shapefile)


def _get_destination_countries_from_config(config_file):
    if 'destination_countries' not in config_file:
        raise KeyError("Missing required configuration key 'destination_countries'.")

    return config_file['destination_countries']


def _unique_preserve_order(values):
    return list(dict.fromkeys(values))


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

    destination_countries = _get_destination_countries_from_config(config_file)
    if not destination_countries:
        raise ValueError('No destination countries configured.')

    if states is None:
        states = _load_states()

    first = True
    destination_location = None
    destination_continents = []
    for c in sorted(destination_countries.keys()):
        country_data = world[world['NAME_EN'] == c]
        if country_data.empty:
            raise ValueError('Destination country not found in Natural Earth data: ' + str(c))

        destination_continents += country_data['CONTINENT'].dropna().tolist()

        if destination_countries[c]:
            for s in sorted(destination_countries[c]):
                state_data = states[states['name'] == s]
                if state_data.empty:
                    raise ValueError('Destination state not found in Natural Earth data: ' + str(s))

                if first:
                    destination_location = state_data['geometry'].values[0]
                    first = False
                else:
                    destination_location = destination_location.union(state_data['geometry'].values[0])
        else:
            if first:
                destination_location = country_data['geometry'].values[0]
                first = False
            else:
                destination_location = destination_location.union(country_data['geometry'].values[0])

    destination_location = destination_location.buffer(0)

    if config_file['use_biggest_landmass']:
        if len([*destination_countries.keys()]) == 1:
            if isinstance(destination_location, MultiPolygon):
                largest_area = 0
                chosen_geom = None
                for geom in destination_location.geoms:
                    if geom.area > largest_area:
                        largest_area = geom.area
                        chosen_geom = geom

                destination_location = chosen_geom

    return {'location': destination_location,
            'countries': _unique_preserve_order([*destination_countries.keys()]),
            'continents': _unique_preserve_order(destination_continents),
            'country_states': destination_countries}


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
