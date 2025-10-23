import random
import math

from shapely.geometry import Polygon, Point, MultiPolygon
import geopandas as gpd
import cartopy.io.shapereader as shpreader


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
    if config_file['destination_type'] == 'location':
        destination_location = Point(config_file['destination_location'])
    else:
        country_shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
        world = gpd.read_file(country_shapefile)

        state_shapefile = shpreader.natural_earth(resolution='10m', category='cultural',
                                                  name='admin_1_states_provinces')
        states = gpd.read_file(state_shapefile)

        country_states = config_file['destination_polygon']

        first = True
        destination_location = None
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

        if config_file['use_biggest_landmass']:  # use only biggest land area
            if len([*country_states.keys()]) == 1:
                if isinstance(destination_location, MultiPolygon):
                    largest_area = 0
                    chosen_geom = None
                    for geom in destination_location.geoms:
                        if geom.area > largest_area:
                            largest_area = geom.area
                            chosen_geom = geom

                    destination_location = chosen_geom

    return destination_location
