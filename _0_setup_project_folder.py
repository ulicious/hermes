import argparse
import os

from data_processing.configuration import CONFIG_FILENAMES, setup_project_folder


def main():
    parser = argparse.ArgumentParser(
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
        print(os.path.join(project_folder, filename))
    print('Raw data files were copied to:')
    print(os.path.join(project_folder, 'raw_data'))


if __name__ == '__main__':
    main()
