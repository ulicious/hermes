import os
import shutil
import sys

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_CONFIG_FOLDER = os.path.join(PROJECT_ROOT, 'data')

ALGORITHM_CONFIG = 'algorithm_configuration.yaml'
PLOTTING_CONFIG = 'plotting_configuration.yaml'
CONVERSION_CONFIG = 'techno_economic_data_conversion.yaml'
TRANSPORTATION_CONFIG = 'techno_economic_data_transportation.yaml'
LOCATION_DATA_FILE = 'location_data.csv'
COUNTRY_DATA_FILE = 'country_data.csv'

CONFIG_FILENAMES = [
    ALGORITHM_CONFIG,
    PLOTTING_CONFIG,
    CONVERSION_CONFIG,
    TRANSPORTATION_CONFIG,
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


def get_config_folder(project_folder_path):
    return project_folder_path


def get_config_path(project_folder_path, filename):
    return os.path.join(get_config_folder(project_folder_path), filename)


def _template_config_path(filename):
    return os.path.join(TEMPLATE_CONFIG_FOLDER, filename)


def _copy_file(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)


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


def copy_provided_raw_data(project_folder_path):
    path_raw_data = os.path.join(project_folder_path, 'raw_data')
    os.makedirs(path_raw_data, exist_ok=True)
    for filename in os.listdir(TEMPLATE_CONFIG_FOLDER):
        if filename in CONFIG_FILENAMES:
            continue
        source = os.path.join(TEMPLATE_CONFIG_FOLDER, filename)
        if os.path.isfile(source):
            _copy_file(source, os.path.join(path_raw_data, filename))


def setup_project_folder(project_folder_path):
    project_folder_path = os.path.abspath(project_folder_path)
    create_project_folder_structure(project_folder_path)
    copy_config_files(project_folder_path)
    copy_provided_raw_data(project_folder_path)
    return project_folder_path


def _project_folder_from_cli():
    for index, argument in enumerate(sys.argv[1:]):
        if argument == '--project-folder':
            value_index = index + 2
            if value_index < len(sys.argv):
                return sys.argv[value_index]
        if argument.startswith('--project-folder='):
            return argument.split('=', 1)[1]
    return None


def resolve_project_folder_path(project_folder_path=None):
    if project_folder_path is not None:
        return project_folder_path
    if os.environ.get('HERMES_PROJECT_FOLDER'):
        return os.environ['HERMES_PROJECT_FOLDER']
    cli_project_folder = _project_folder_from_cli()
    if cli_project_folder is not None:
        return cli_project_folder
    template_config = load_yaml(_template_config_path(ALGORITHM_CONFIG))
    return template_config['project_folder_path']


def load_algorithm_configuration(project_folder_path=None):
    project_folder_path = resolve_project_folder_path(project_folder_path)
    config_path = os.path.join(get_config_folder(project_folder_path), ALGORITHM_CONFIG)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            'Missing configuration file:\n'
            + config_path
            + '\nRun _0_setup_project_folder.py with the desired project folder first.'
        )
    return load_yaml(config_path)


def load_plotting_configuration(config_file=None):
    if config_file is None:
        config_file = load_algorithm_configuration()
    config_folder = get_config_folder(config_file['project_folder_path'])
    return load_yaml(os.path.join(config_folder, PLOTTING_CONFIG))


def load_technology_data(config_file):
    config_folder = get_config_folder(config_file['project_folder_path'])
    conversion_data = load_yaml(os.path.join(config_folder, CONVERSION_CONFIG))
    transportation_data = load_yaml(os.path.join(config_folder, TRANSPORTATION_CONFIG))
    return conversion_data, transportation_data
