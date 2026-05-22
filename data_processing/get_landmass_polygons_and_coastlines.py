import os

os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"

import geopandas as gpd
import cartopy.io.shapereader as shpreader

from shapely.geometry import MultiPolygon

import zipfile
import fiona
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


def create_water_availability_polygon(BASE_DIR):

    BASE_DIR = Path(BASE_DIR)

    aqueduct_zip = BASE_DIR / "water.zip"
    coastline_zip = BASE_DIR / "ne_10m_coastline.zip"

    aqueduct_extract_dir = BASE_DIR / "aqueduct_data"
    coastline_extract_dir = BASE_DIR / "coastline_data"

    coastline_shp = coastline_extract_dir / "ne_10m_coastline.shp"

    risk_column = "bws_score"
    threshold = 3.0
    coast_distance_km = 50
    metric_crs = "EPSG:3857"

    def unzip_file(zip_path, extract_to):
        if not zip_path.exists():
            raise FileNotFoundError(f"File not found:\n{zip_path}")

        if extract_to.exists() and any(extract_to.iterdir()):
            return

        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)

    def repair_geometries(gdf):
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        return gdf

    unzip_file(aqueduct_zip, aqueduct_extract_dir)
    unzip_file(coastline_zip, coastline_extract_dir)

    gdb_paths = list(aqueduct_extract_dir.rglob("*.gdb"))

    if not gdb_paths:
        raise FileNotFoundError("No .gdb file found.")

    gdb_path = gdb_paths[0]

    gdf = gpd.read_file(
        gdb_path,
        layer="annual",
        GEOMETRY_NAME="geometry",
        promote_to_multi=True
    )

    gdf = gdf[[risk_column, "geometry"]].copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf[risk_column].notna()].copy()
    gdf = repair_geometries(gdf)

    coastline = gpd.read_file(coastline_shp)

    if coastline.crs is None:
        coastline = coastline.set_crs("EPSG:4326")

    coastline = repair_geometries(coastline)

    gdf_metric = gdf.to_crs(metric_crs)
    gdf_metric = repair_geometries(gdf_metric)

    coastline_metric = coastline.to_crs(metric_crs)
    coastline_metric = repair_geometries(coastline_metric)

    coastline_union = unary_union(coastline_metric.geometry)

    coastal_buffer = coastline_union.buffer(
        coast_distance_km * 1000
    )

    high_water_stress = gdf_metric[risk_column] >= threshold

    near_coast = gdf_metric.geometry.intersects(
        coastal_buffer
    )

    water_available = (
        ~high_water_stress | near_coast
    )

    gdf_available = gdf_metric.loc[
        water_available,
        [risk_column, "geometry"]
    ].copy()

    gdf_available["near_coast_50km"] = near_coast.loc[
        water_available
    ].values

    gdf_available["ptx_status"] = "available"
    gdf_available["threshold"] = threshold
    gdf_available["coast_distance_km"] = coast_distance_km

    gdf_available = repair_geometries(gdf_available)

    gdf_available = gdf_available.to_crs("EPSG:4326")

    return gdf_available


def get_landmass_polygons_and_coastlines(path_raw_data, use_minimal_example=False):

    """
    Create large polygons based on connected country polygons

    @param bool use_minimal_example: boolean if only Europe should be considered

    @return: geopandas.GeoDataFrame multipolygons for landmass and linestrings of coastlines
    """

    # # Load the shapefile data for country boundaries with 10m resolution
    # world_high_res = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries_deu')
    #
    # # Read the shapefile data
    # reader = shpreader.Reader(world_high_res)
    #
    # # Extract the polygons for the specified country
    # polygons = []
    # for country in reader.records():
    #
    #     if use_minimal_example:
    #         if country.attributes['CONTINENT'] != 'Europe':
    #             continue
    #
    #     country_polygon = country.geometry
    #
    #     if isinstance(country_polygon, MultiPolygon):
    #         for p in country_polygon.geoms:
    #             polygons.append(p)
    #     else:
    #         polygons.append(country_polygon)
    #
    # # Combine polygons that touch each other into MultiPolygons
    # merged_polygons = []
    # coastlines = []
    # while len(polygons) > 0:
    #     merged_polygon = polygons.pop(0)
    #
    #     broken = True
    #     while broken:
    #         broken = False
    #         for polygon in polygons:
    #
    #             if merged_polygon.touches(polygon):
    #                 merged_polygon = merged_polygon.union(polygon)
    #                 polygons.remove(polygon)
    #                 broken = True
    #                 break
    #
    #     merged_polygons.append(merged_polygon)
    #     coastlines.append(merged_polygon.boundary)
    #
    # polygons = gpd.GeoDataFrame(geometry=merged_polygons)
    # coastlines = gpd.GeoDataFrame(geometry=coastlines)

    print('start water')

    water_availability = create_water_availability_polygon(path_raw_data)

    polygons = None
    coastlines = None

    return polygons, coastlines, water_availability
