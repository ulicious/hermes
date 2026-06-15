import os
import subprocess
import sys


# Adjust this path to your HERMES project folder before running the workflow.
PROJECT_FOLDER = 'C:/Users/mt5285/Documents/Transportmodell/'

# Select which workflow steps should run.
RUN_SETUP_PROJECT_FOLDER = False
RUN_PROCESS_RAW_DATA = False
RUN_CREATE_START_LOCATIONS = False
RUN_MAIN_ALGORITHM = False
RUN_MIP_OPTIMIZATION = False
RUN_PROCESS_PLOT_DATA = False
RUN_PLOT_RESULTS = True
RUN_ANALYZE_ALGORITHM_TRACKING = False


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


def run_step(module_name):
    path_repo = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, '-m', module_name, PROJECT_FOLDER]
    print('')
    print('Start ' + module_name)
    print(' '.join(command))
    subprocess.run(command, check=True, cwd=path_repo)


def main():
    if not PROJECT_FOLDER:
        raise ValueError('Set PROJECT_FOLDER in _run_workflow.py before running the workflow.')

    selected_steps = [module_name for enabled, module_name in WORKFLOW_STEPS if enabled]
    if not selected_steps:
        print('No workflow steps selected. Set at least one RUN_* flag to True.')
        return

    for module_name in selected_steps:
        run_step(module_name)

    print('')
    print('Selected HERMES workflow steps finished successfully.')


if __name__ == '__main__':
    main()
