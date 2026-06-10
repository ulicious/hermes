import argparse
import os

from data_processing.configuration import (
    CONFIG_FILENAMES,
    get_config_path,
    setup_project_folder,
)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m scripts._0_setup_project_folder',
        description='Create the HERMES project folder structure and copy input/config files.'
    )
    parser.add_argument(
        'project_folder',
        help='Path to the project folder where HERMES should store configs, raw data, processed data, and results.',
    )
    args = parser.parse_args()

    project_folder = setup_project_folder(args.project_folder)
    print('HERMES project folder prepared:')
    print(project_folder)
    print('Configuration files copied to:')
    for filename in CONFIG_FILENAMES:
        path_file = get_config_path(project_folder, filename)
        if not os.path.exists(path_file):
            raise FileNotFoundError('Expected copied configuration file is missing: ' + path_file)
        print(path_file)
    print('Raw data files were copied to:')
    print(os.path.join(project_folder, 'raw_data'))


if __name__ == '__main__':
    main()
