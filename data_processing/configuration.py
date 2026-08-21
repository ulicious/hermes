import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, 'configs')

ZENODO_DATA_RECORD_ID = '22031725'
ZENODO_DATA_DOI = '10.5281/zenodo.22031725'
ZENODO_DATA_BASE_URL = 'https://zenodo.org/api/records/{}/files'.format(ZENODO_DATA_RECORD_ID)
ZENODO_DATA_FILES = {
    'country_data.csv': 'b83d231131455c66dcb86ffc7581a4ac',
    'location_data.csv': '28dd310b4276d8f04b9ae6a23478ba39',
    'network_pipelines_gas.xlsx': '03452282e65fe8fb8a8fe9934c2d6f6c',
    'network_pipelines_oil.xlsx': '9970c32b11ee79341df6f4be94dc6009',
    'seaports.geojson': 'cf7cec9a71fbdd429f40dd78dbb02542',
    'water.zip': 'e922f99c19605acf089272245a38405f',
    'natural_earth.zip': '53ff58b372937e390f89dc480cea6e09',
}

ALGORITHM_CONFIG = '1_algorithm_configuration.yaml'
TRANSPORTATION_CONFIG = '2_techno_economic_data_transportation.yaml'
CONVERSION_CONFIG = '3_techno_economic_data_conversion.yaml'
PLOTTING_CONFIG = '4_plotting_configuration.yaml'
RAW_DATA_CONFIG_KEYS = (
    'country_data',
    'location_data',
    'network_pipelines_gas',
    'network_pipelines_oil',
    'seaports',
)

CONFIG_FILENAMES = [
    ALGORITHM_CONFIG,
    TRANSPORTATION_CONFIG,
    CONVERSION_CONFIG,
    PLOTTING_CONFIG,
]

LEGACY_CONFIG_FILENAMES = [
    'algorithm_configuration.yaml',
    'techno_economic_data_transportation.yaml',
    'techno_economic_data_conversion.yaml',
    'plotting_configuration.yaml',
    '_1_algorithm_configuration.yaml',
    '_5_plotting_configuration.yaml',
]

BOOLEAN_CONFIG_KEYS = [
    'use_minimal_example',
    'use_low_storage',
    'use_low_memory',
    'create_mip_data',
    'start_locations_update_only_conversion_costs_and_efficiency',
    'use_voronoi_cells',
    'weight_hydrogen_costs_by_quantity',
    'each_country_at_least_one_location',
    'create_locations_for_islands',
    'low_temp_heat_available_at_start',
    'mid_temp_heat_available_at_start',
    'high_temp_heat_available_at_start',
    'infrastructure_enforce_update_of_data',
    'infrastructure_update_only_conversion_costs_and_efficiency',
    'low_temp_heat_available_at_ports',
    'mid_temp_heat_available_at_ports',
    'high_temp_heat_available_at_ports',
    'low_temp_heat_available_at_pipelines',
    'mid_temp_heat_available_at_pipelines',
    'high_temp_heat_available_at_pipelines',
    'use_biggest_landmass',
    'build_new_infrastructure',
    'H2_ready_infrastructure',
    'low_temp_heat_available_at_destination',
    'mid_temp_heat_available_at_destination',
    'high_temp_heat_available_at_destination',
    'consider_commodity_prices',
    'print_runtime_information',
    'print_benchmark_info',
]

PROJECT_STRUCTURE = [
    'algorithm_configurations',
    'raw_data',
    'processed_data',
    os.path.join('processed_data', 'inner_infrastructure_distances'),
    os.path.join('processed_data', 'mip_data'),
    'results',
    os.path.join('results', 'location_results'),
    os.path.join('results', 'plots'),
    os.path.join('results', 'processed_results'),
    os.path.join('results', 'unprocessed_results'),
    os.path.join('results', 'algorithm_tracking'),
]


def load_yaml(path_file):
    with open(path_file, encoding='utf-8') as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'y', 'on'}:
            return True
        if normalized in {'false', '0', 'no', 'n', 'off', ''}:
            return False
    return bool(value)


def normalize_algorithm_configuration(config_file):
    for key in BOOLEAN_CONFIG_KEYS:
        if key in config_file:
            config_file[key] = _as_bool(config_file[key])
    for key in RAW_DATA_CONFIG_KEYS:
        if key not in config_file:
            raise KeyError(
                "Missing required raw-data filename '" + key
                + "' in " + ALGORITHM_CONFIG + '.'
            )
        filename = config_file[key]
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                "Raw-data filename '" + key + "' in " + ALGORITHM_CONFIG
                + ' must be a non-empty string.'
            )
        filename = filename.strip()
        if filename != os.path.basename(filename):
            raise ValueError(
                "Raw-data filename '" + key + "' must be a filename without a directory: "
                + filename
            )
        config_file[key] = filename
    return config_file


def get_raw_data_path(config_file, key):
    if key not in RAW_DATA_CONFIG_KEYS:
        raise KeyError('Unknown raw-data configuration key: ' + str(key))
    return os.path.join(
        config_file['project_folder_path'],
        'raw_data',
        config_file[key],
    )


def get_config_folder(project_folder_path):
    return project_folder_path


def get_config_path(project_folder_path, filename):
    return os.path.join(get_config_folder(project_folder_path), filename)


def _path_is_inside_folder(path_file, path_folder):
    path_file = os.path.abspath(path_file)
    path_folder = os.path.abspath(path_folder)
    return os.path.commonpath([path_file, path_folder]) == path_folder


def _load_project_yaml(config_file, filename):
    project_folder_path = os.path.abspath(config_file['project_folder_path'])
    config_path = os.path.abspath(os.path.join(get_config_folder(project_folder_path), filename))
    if not _path_is_inside_folder(config_path, project_folder_path):
        raise ValueError(
            'Configuration file is outside the project folder:\n'
            + config_path
            + '\nProject folder:\n'
            + project_folder_path
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError('Missing configuration file:\n' + config_path)
    return load_yaml(config_path)


def _template_config_path(filename):
    return os.path.join(CONFIG_TEMPLATE_FOLDER, filename)


def _copy_file(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)
    if not os.path.exists(target):
        raise FileNotFoundError('File copy failed:\n' + source + '\n->\n' + target)


def _format_yaml_single_quoted(value):
    return "'" + str(value).replace("'", "''") + "'"


def _set_project_folder_path(path_algorithm_config, project_folder_path):
    with open(path_algorithm_config, encoding='utf-8') as file:
        lines = file.readlines()

    replacement = 'project_folder_path: ' + _format_yaml_single_quoted(project_folder_path) + '  # full path of folder\n'
    for index, line in enumerate(lines):
        if line.strip().startswith('project_folder_path:'):
            lines[index] = replacement
            break
    else:
        lines.insert(0, replacement)

    with open(path_algorithm_config, 'w', encoding='utf-8') as file:
        file.writelines(lines)


def create_project_folder_structure(project_folder_path):
    os.makedirs(project_folder_path, exist_ok=True)
    for folder in PROJECT_STRUCTURE:
        os.makedirs(os.path.join(project_folder_path, folder), exist_ok=True)


def copy_config_files(project_folder_path):
    create_project_folder_structure(project_folder_path)
    config_folder = get_config_folder(project_folder_path)
    for filename in CONFIG_FILENAMES:
        _copy_file(
            _template_config_path(filename),
            os.path.join(config_folder, filename),
        )
    _set_project_folder_path(
        os.path.join(config_folder, ALGORITHM_CONFIG),
        project_folder_path,
    )
    return config_folder


def remove_legacy_config_files(project_folder_path):
    legacy_folders = [
        project_folder_path,
        os.path.join(project_folder_path, 'config'),
    ]
    legacy_filenames = CONFIG_FILENAMES + LEGACY_CONFIG_FILENAMES
    removed_files = []
    for folder in legacy_folders:
        for filename in legacy_filenames:
            path_file = os.path.join(folder, filename)
            if os.path.exists(path_file):
                os.remove(path_file)
                removed_files.append(path_file)
    return removed_files


def _md5(path_file):
    digest = hashlib.md5()
    with open(path_file, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _download_zenodo_file(filename, expected_md5, destination):
    if os.path.isfile(destination) and _md5(destination) == expected_md5:
        print('Zenodo input already available: ' + filename)
        return

    url = ZENODO_DATA_BASE_URL + '/' + filename + '/content'
    temporary_path = destination + '.part'
    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    print('Downloading HERMES input data: ' + filename)
    try:
        with urllib.request.urlopen(url) as response, open(temporary_path, 'wb') as target:
            shutil.copyfileobj(response, target)
        actual_md5 = _md5(temporary_path)
        if actual_md5 != expected_md5:
            raise ValueError(
                'Checksum mismatch for downloaded file ' + filename
                + ': expected ' + expected_md5 + ', got ' + actual_md5
            )
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _extract_natural_earth(path_raw_data):
    archive_path = os.path.join(path_raw_data, 'natural_earth.zip')
    expected_shapefiles = [
        os.path.join(path_raw_data, 'natural_earth', folder, name, 'ne_' + prefix + '_' + name + '.shp')
        for folder, name, prefix in [
            ('10m_cultural', 'admin_0_countries_deu', '10m'),
            ('10m_cultural', 'admin_1_states_provinces', '10m'),
            ('10m_physical', 'land', '10m'),
            ('10m_physical', 'coastline', '10m'),
            ('110m_cultural', 'admin_0_countries', '110m'),
        ]
    ]
    if all(os.path.isfile(path) for path in expected_shapefiles):
        return

    destination_root = os.path.abspath(path_raw_data)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = os.path.abspath(os.path.join(destination_root, member.filename))
            if os.path.commonpath([member_path, destination_root]) != destination_root:
                raise ValueError('Unsafe path in natural_earth.zip: ' + member.filename)
        archive.extractall(destination_root)

    missing = [path for path in expected_shapefiles if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            'natural_earth.zip does not contain the expected HERMES directory structure:\n'
            + '\n'.join(missing)
        )


def download_raw_data(project_folder_path):
    path_raw_data = os.path.join(project_folder_path, 'raw_data')
    os.makedirs(path_raw_data, exist_ok=True)
    for filename, expected_md5 in ZENODO_DATA_FILES.items():
        _download_zenodo_file(
            filename,
            expected_md5,
            os.path.join(path_raw_data, filename),
        )
    _extract_natural_earth(path_raw_data)


def setup_project_folder(project_folder_path):
    project_folder_path = os.path.abspath(project_folder_path)
    create_project_folder_structure(project_folder_path)
    remove_legacy_config_files(project_folder_path)
    copy_config_files(project_folder_path)
    download_raw_data(project_folder_path)
    return project_folder_path


def _project_folder_from_cli():
    arguments = sys.argv[1:]
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == '--project-folder':
            value_index = index + 1
            if value_index < len(arguments):
                return arguments[value_index]
        if argument.startswith('--project-folder='):
            return argument.split('=', 1)[1]
        if argument in {'--algorithm-config'}:
            index += 2
            continue
        index += 1

    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {'--algorithm-config', '--project-folder'}:
            index += 2
            continue
        if argument.startswith('--algorithm-config=') or argument.startswith('--project-folder='):
            index += 1
            continue
        if not argument.startswith('-'):
            return argument
        index += 1
    return None


def resolve_project_folder_path(project_folder_path=None):
    if project_folder_path is not None:
        return project_folder_path
    if os.environ.get('HERMES_PROJECT_FOLDER'):
        return os.environ['HERMES_PROJECT_FOLDER']
    cli_project_folder = _project_folder_from_cli()
    if cli_project_folder is not None:
        return cli_project_folder
    return os.getcwd()


def _algorithm_config_path_from_cli():
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == '--algorithm-config':
            value_index = index + 1
            if value_index < len(arguments):
                return arguments[value_index]
        if argument.startswith('--algorithm-config='):
            return argument.split('=', 1)[1]
    return None


def resolve_algorithm_config_path(project_folder_path, algorithm_config_path=None):
    if algorithm_config_path is None:
        algorithm_config_path = os.environ.get('HERMES_ALGORITHM_CONFIG')
    if algorithm_config_path is None:
        algorithm_config_path = _algorithm_config_path_from_cli()
    if algorithm_config_path is None:
        algorithm_config_path = os.path.join(get_config_folder(project_folder_path), ALGORITHM_CONFIG)
    elif not os.path.isabs(algorithm_config_path):
        algorithm_config_path = os.path.join(project_folder_path, algorithm_config_path)
    return os.path.abspath(algorithm_config_path)


def _ensure_trailing_separator(path_folder):
    if path_folder.endswith(('/', '\\')):
        return path_folder
    return path_folder + os.sep


def load_algorithm_configuration(project_folder_path=None, algorithm_config_path=None):
    project_folder_path = resolve_project_folder_path(project_folder_path)
    project_folder_path = os.path.abspath(project_folder_path)
    project_folder_path = _ensure_trailing_separator(project_folder_path)
    config_path = resolve_algorithm_config_path(project_folder_path, algorithm_config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            'Missing configuration file:\n'
            + config_path
            + '\nRun _run_workflow.py with RUN_SETUP_PROJECT_FOLDER = True first.'
        )
    config_file = normalize_algorithm_configuration(load_yaml(config_path))
    config_file['project_folder_path'] = project_folder_path
    config_file['_configuration_path'] = config_path
    return config_file


def load_plotting_configuration(config_file=None):
    if config_file is None:
        config_file = load_algorithm_configuration()
    return _load_project_yaml(config_file, PLOTTING_CONFIG)


def validate_plotting_result_cases(config_file, plotting_config):
    results_folder = os.path.join(
        config_file['project_folder_path'],
        'results',
        'unprocessed_results',
    )
    configured_cases = plotting_config.get('process_results', [])

    if not isinstance(configured_cases, list):
        raise TypeError(
            "'process_results' in "
            + PLOTTING_CONFIG
            + ' must be a YAML list.'
        )

    missing_cases = [
        case_name
        for case_name in configured_cases
        if not os.path.isdir(os.path.join(results_folder, case_name))
    ]
    if missing_cases:
        formatted_cases = '\n'.join('  - ' + str(case_name) for case_name in missing_cases)
        raise FileNotFoundError(
            'Cannot process plotting results because configured case folders are missing:\n'
            + formatted_cases
            + '\n\nExpected in:\n'
            + results_folder
            + '\n\nUpdate process_results in '
            + PLOTTING_CONFIG
            + ' or create the missing result folders.'
        )


def load_technology_data(config_file):
    conversion_data = _load_project_yaml(config_file, CONVERSION_CONFIG)
    transportation_data = _load_project_yaml(config_file, TRANSPORTATION_CONFIG)
    return conversion_data, transportation_data
