import logging
import os
import urllib.request
import zipfile

import geopandas as gpd

from data_processing.configuration import load_algorithm_configuration


logger = logging.getLogger(__name__)


NATURAL_EARTH_DATASETS = [
    ('10m', 'cultural', 'admin_0_countries_deu'),
    ('10m', 'cultural', 'admin_1_states_provinces'),
    ('10m', 'physical', 'land'),
    ('10m', 'physical', 'coastline'),
    ('110m', 'cultural', 'admin_0_countries'),
]


def get_configured_raw_data_path():
    """Return the configured raw_data folder for scripts that do not get it explicitly."""
    config_file = load_algorithm_configuration()
    return os.path.join(config_file['project_folder_path'], 'raw_data')


def get_natural_earth_folder(path_raw_data=None):
    if path_raw_data is None:
        path_raw_data = get_configured_raw_data_path()
    return os.path.join(path_raw_data, 'natural_earth')


def get_natural_earth_dataset_folder(path_raw_data, resolution, category, name):
    return os.path.join(get_natural_earth_folder(path_raw_data), resolution + '_' + category, name)


def get_natural_earth_url(resolution, category, name):
    return 'https://naturalearth.s3.amazonaws.com/{resolution}_{category}/ne_{resolution}_{name}.zip'.format(
        resolution=resolution,
        category=category,
        name=name,
    )


def get_natural_earth_shapefile(path_raw_data, resolution, category, name):
    dataset_folder = get_natural_earth_dataset_folder(path_raw_data, resolution, category, name)
    shapefile_name = 'ne_{resolution}_{name}.shp'.format(resolution=resolution, name=name)
    return os.path.join(dataset_folder, shapefile_name)


def download_natural_earth_data(path_raw_data, force_update=False, datasets=None):
    """Download and extract all Natural Earth datasets used by the project into raw_data."""
    if datasets is None:
        datasets = NATURAL_EARTH_DATASETS

    natural_earth_folder = get_natural_earth_folder(path_raw_data)
    os.makedirs(natural_earth_folder, exist_ok=True)

    for resolution, category, name in datasets:
        shapefile_path = get_natural_earth_shapefile(path_raw_data, resolution, category, name)
        if os.path.exists(shapefile_path) and not force_update:
            logger.info('Natural Earth data already available: %s', shapefile_path)
            continue

        dataset_folder = get_natural_earth_dataset_folder(path_raw_data, resolution, category, name)
        os.makedirs(dataset_folder, exist_ok=True)

        zip_path = os.path.join(dataset_folder, 'ne_{resolution}_{name}.zip'.format(
            resolution=resolution,
            name=name,
        ))
        url = get_natural_earth_url(resolution, category, name)

        logger.info('Download Natural Earth data: %s', url)
        urllib.request.urlretrieve(url, zip_path)

        logger.info('Extract Natural Earth data to %s', dataset_folder)
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(dataset_folder)

        if not os.path.exists(shapefile_path):
            raise FileNotFoundError('Natural Earth download did not create expected shapefile: ' + shapefile_path)


def read_natural_earth(path_raw_data=None, resolution='10m', category='cultural', name='admin_0_countries_deu'):
    if path_raw_data is None:
        path_raw_data = get_configured_raw_data_path()

    shapefile_path = get_natural_earth_shapefile(path_raw_data, resolution, category, name)
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(
            'Missing Natural Earth shapefile:\n'
            + shapefile_path
            + '\nRun _1_script_process_raw_data.py once to download Natural Earth data into raw_data.'
        )

    return gpd.read_file(shapefile_path)


def load_world(path_raw_data=None):
    return read_natural_earth(path_raw_data, resolution='10m', category='cultural', name='admin_0_countries_deu')


def load_states(path_raw_data=None):
    return read_natural_earth(path_raw_data, resolution='10m', category='cultural', name='admin_1_states_provinces')


def load_land(path_raw_data=None):
    return read_natural_earth(path_raw_data, resolution='10m', category='physical', name='land')


def load_coastline(path_raw_data=None):
    return read_natural_earth(path_raw_data, resolution='10m', category='physical', name='coastline')


def load_world_lowres(path_raw_data=None):
    world = read_natural_earth(path_raw_data, resolution='110m', category='cultural', name='admin_0_countries')
    rename_columns = {}
    if 'CONTINENT' in world.columns and 'continent' not in world.columns:
        rename_columns['CONTINENT'] = 'continent'
    if 'NAME' in world.columns and 'name' not in world.columns:
        rename_columns['NAME'] = 'name'
    if rename_columns:
        world = world.rename(columns=rename_columns)
    return world
