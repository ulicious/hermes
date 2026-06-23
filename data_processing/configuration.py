import os
import shutil
import sys

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_CONFIG_FOLDER = os.path.join(PROJECT_ROOT, 'data')

ALGORITHM_CONFIG = '1_algorithm_configuration.yaml'
TRANSPORTATION_CONFIG = '2_techno_economic_data_transportation.yaml'
CONVERSION_CONFIG = '3_techno_economic_data_conversion.yaml'
PLOTTING_CONFIG = '4_plotting_configuration.yaml'
LOCATION_DATA_FILE = 'location_data.csv'
COUNTRY_DATA_FILE = 'country_data.csv'

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
    return config_file


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
    return os.path.join(TEMPLATE_CONFIG_FOLDER, filename)


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


def copy_provided_raw_data(project_folder_path):
    path_raw_data = os.path.join(project_folder_path, 'raw_data')
    os.makedirs(path_raw_data, exist_ok=True)
    for filename in os.listdir(TEMPLATE_CONFIG_FOLDER):
        if filename in CONFIG_FILENAMES + LEGACY_CONFIG_FILENAMES:
            continue
        source = os.path.join(TEMPLATE_CONFIG_FOLDER, filename)
        if os.path.isfile(source):
            _copy_file(source, os.path.join(path_raw_data, filename))


def setup_project_folder(project_folder_path):
    project_folder_path = os.path.abspath(project_folder_path)
    create_project_folder_structure(project_folder_path)
    remove_legacy_config_files(project_folder_path)
    copy_config_files(project_folder_path)
    copy_provided_raw_data(project_folder_path)
    return project_folder_path


def _project_folder_from_cli():
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == '--project-folder':
            value_index = index + 2
            if value_index < len(sys.argv):
                return sys.argv[value_index]
        if argument.startswith('--project-folder='):
            return argument.split('=', 1)[1]
    for argument in arguments:
        if argument.startswith('-'):
            continue
        return argument
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


def _ensure_trailing_separator(path_folder):
    if path_folder.endswith(('/', '\\')):
        return path_folder
    return path_folder + os.sep


def load_algorithm_configuration(project_folder_path=None):
    project_folder_path = resolve_project_folder_path(project_folder_path)
    project_folder_path = os.path.abspath(project_folder_path)
    project_folder_path = _ensure_trailing_separator(project_folder_path)
    config_path = os.path.join(get_config_folder(project_folder_path), ALGORITHM_CONFIG)
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


def load_technology_data(config_file):
    conversion_data = _load_project_yaml(config_file, CONVERSION_CONFIG)
    transportation_data = _load_project_yaml(config_file, TRANSPORTATION_CONFIG)
    return conversion_data, transportation_data
