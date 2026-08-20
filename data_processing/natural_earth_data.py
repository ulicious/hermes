import os

import geopandas as gpd

from data_processing.configuration import load_algorithm_configuration


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


def get_natural_earth_shapefile(path_raw_data, resolution, category, name):
    dataset_folder = get_natural_earth_dataset_folder(path_raw_data, resolution, category, name)
    shapefile_name = 'ne_{resolution}_{name}.shp'.format(resolution=resolution, name=name)
    return os.path.join(dataset_folder, shapefile_name)


def validate_natural_earth_data(path_raw_data, datasets=None):
    """Require the Natural Earth files installed by the Zenodo-backed setup."""
    if datasets is None:
        datasets = NATURAL_EARTH_DATASETS
    missing = [
        get_natural_earth_shapefile(path_raw_data, resolution, category, name)
        for resolution, category, name in datasets
        if not os.path.isfile(get_natural_earth_shapefile(path_raw_data, resolution, category, name))
    ]
    if missing:
        raise FileNotFoundError(
            'Missing Natural Earth data installed by the HERMES setup:\n'
            + '\n'.join(missing)
            + '\nRun _run_workflow.py with RUN_SETUP_PROJECT_FOLDER = True.'
        )


def read_natural_earth(path_raw_data=None, resolution='10m', category='cultural', name='admin_0_countries_deu'):
    if path_raw_data is None:
        path_raw_data = get_configured_raw_data_path()

    shapefile_path = get_natural_earth_shapefile(path_raw_data, resolution, category, name)
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(
            'Missing Natural Earth shapefile:\n'
            + shapefile_path
            + '\nRun _run_workflow.py with RUN_SETUP_PROJECT_FOLDER = True to download the versioned Zenodo dataset.'
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
