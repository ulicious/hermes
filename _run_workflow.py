import os
import subprocess
import sys


# Defaults to the folder from which the workflow is started.
PROJECT_FOLDER = os.getcwd()

# Select which workflow steps should run.
RUN_SETUP_PROJECT_FOLDER = False
RUN_PROCESS_RAW_DATA = False
RUN_CREATE_START_LOCATIONS = False
RUN_MAIN_ALGORITHM = False
RUN_MIP_OPTIMIZATION = False
RUN_PROCESS_PLOT_DATA = False
RUN_PLOT_RESULTS = True
RUN_ANALYZE_ALGORITHM_TRACKING = False

# Optional scenario mode for the main algorithm.
# Put alternative algorithm configuration YAML files in this folder.
# Each file is run for all locations and writes to results/<config filename>/.
RUN_ALGORITHM_CONFIG_BATCH = False
ALGORITHM_CONFIG_BATCH_FOLDER = os.path.join(PROJECT_FOLDER, 'algorithm_configurations')


WORKFLOW_STEPS = [
    (RUN_SETUP_PROJECT_FOLDER, 'scripts._0_setup_project_folder'),
    (RUN_PROCESS_RAW_DATA, 'scripts._1_script_process_raw_data'),
    (RUN_CREATE_START_LOCATIONS, 'scripts._2_create_random_locations'),
    (RUN_MAIN_ALGORITHM, 'scripts._3_main'),
    (RUN_MIP_OPTIMIZATION, 'scripts._4_mip_optimization'),
    (RUN_PROCESS_PLOT_DATA, 'scripts._5_process_plot_data'),
    (RUN_PLOT_RESULTS, 'scripts._6_plot_results'),
    (RUN_ANALYZE_ALGORITHM_TRACKING, 'scripts._7_analyze_algorithm_tracking'),
]


def run_step(module_name, extra_arguments=None):
    path_repo = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, '-m', module_name, PROJECT_FOLDER]
    if extra_arguments:
        command.extend(extra_arguments)
    print('')
    print('Start ' + module_name)
    print(' '.join(command))
    subprocess.run(command, check=True, cwd=path_repo)


def get_algorithm_config_batch_files():
    if not os.path.isdir(ALGORITHM_CONFIG_BATCH_FOLDER):
        raise FileNotFoundError(
            'Algorithm configuration batch folder not found:\n'
            + ALGORITHM_CONFIG_BATCH_FOLDER
        )

    config_files = []
    for filename in os.listdir(ALGORITHM_CONFIG_BATCH_FOLDER):
        if filename.lower().endswith(('.yaml', '.yml')):
            config_files.append(os.path.join(ALGORITHM_CONFIG_BATCH_FOLDER, filename))

    config_files.sort(key=lambda path_file: os.path.basename(path_file).lower())
    if not config_files:
        raise FileNotFoundError(
            'No YAML configuration files found in:\n'
            + ALGORITHM_CONFIG_BATCH_FOLDER
        )
    return config_files


def run_algorithm_config_batch():
    config_files = get_algorithm_config_batch_files()
    print('')
    print('Start algorithm configuration batch')
    print('Configuration folder: ' + ALGORITHM_CONFIG_BATCH_FOLDER)
    for config_path in config_files:
        print('')
        print('Run algorithm configuration: ' + os.path.basename(config_path))
        run_step('scripts._3_main', ['--algorithm-config', config_path])


def main():
    if not PROJECT_FOLDER:
        raise ValueError('Set PROJECT_FOLDER in _run_workflow.py before running the workflow.')

    selected_steps = [module_name for enabled, module_name in WORKFLOW_STEPS if enabled]
    if not selected_steps and not RUN_ALGORITHM_CONFIG_BATCH:
        print('No workflow steps selected. Set at least one RUN_* flag to True.')
        return

    for module_name in selected_steps:
        run_step(module_name)

    if RUN_ALGORITHM_CONFIG_BATCH:
        run_algorithm_config_batch()

    print('')
    print('Selected HERMES workflow steps finished successfully.')


if __name__ == '__main__':
    main()
