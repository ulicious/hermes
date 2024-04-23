import geopandas as gpd
import cartopy.io.shapereader as shpreader

from shapely.geometry import MultiPolygon


def get_landmass_polygons_and_coastlines(use_minimal_example=False):

    """
    Create large polygons based on connected country polygons

    @param use_minimal_example: boolean if only Europe should be considered

    @return: multipolygons for landmass and linestrings of coastlines
    """

    # Load the shapefile data for country boundaries with 10m resolution
    world_high_res = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')

    # Read the shapefile data
    reader = shpreader.Reader(world_high_res)

    # Extract the polygons for the specified country
    polygons = []
    for country in reader.records():

        if use_minimal_example:
            if country.attributes['CONTINENT'] != 'Europe':
                continue

        country_polygon = country.geometry

        if isinstance(country_polygon, MultiPolygon):
            for p in country_polygon.geoms:
                polygons.append(p)
        else:
            polygons.append(country_polygon)

    # Combine polygons that touch each other into MultiPolygons
    merged_polygons = []
    coastlines = []
    while len(polygons) > 0:
        merged_polygon = polygons.pop(0)

        broken = True
        while broken:
            broken = False
            for polygon in polygons:

                if merged_polygon.touches(polygon):
                    merged_polygon = merged_polygon.union(polygon)
                    polygons.remove(polygon)
                    broken = True
                    break

        merged_polygons.append(merged_polygon)
        coastlines.append(merged_polygon.boundary)

    polygons = gpd.GeoDataFrame(geometry=merged_polygons)
    coastlines = gpd.GeoDataFrame(geometry=coastlines)

    return polygons, coastlines
