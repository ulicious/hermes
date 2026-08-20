##############
Advanced usage
##############

This page describes alternatives to the central runner workflow. They are not
required for a normal first run.

Run individual stages
#####################

Run module commands from the repository root and quote paths containing spaces:

.. code-block:: bash

   python -m scripts._0_setup_project_folder "/path/to/project"
   python -m scripts._1_script_process_raw_data "/path/to/project"
   python -m scripts._2_create_random_locations "/path/to/project"
   python -m scripts._3_main "/path/to/project"
   python -m scripts._3_export_infrastructure "/path/to/project"
   python -m scripts._4_mip_optimization "/path/to/project"
   python -m scripts._5_process_plot_data "/path/to/project"
   python -m scripts._6_plot_results "/path/to/project"
   python -m scripts._7_analyze_algorithm_tracking "/path/to/project"

Most stages use the shared configuration loader and also accept
``--project-folder "/path/to/project"``. The setup stage requires its positional
project-folder argument.

Project-folder environment variable
###################################

Bash:

.. code-block:: bash

   export HERMES_PROJECT_FOLDER="/path/to/project"
   python -m scripts._3_main

PowerShell:

.. code-block:: powershell

   $env:HERMES_PROJECT_FOLDER = "C:\path\to\project"
   python -m scripts._3_main

When the shared loader is used, an explicit Python function argument has
priority over the environment variable, which has priority over the CLI path.
Without any of these, the current working directory is used.

Alternative algorithm configurations
####################################

Run one alternative configuration with:

.. code-block:: bash

   python -m scripts._3_main "/path/to/project" \
       --algorithm-config "algorithm_configurations/scenario_a.yaml"

A relative configuration path is resolved against the project folder. Results
are written to a directory named exactly like the configuration file, including
its extension:

.. code-block:: text

   PROJECT_FOLDER/results/scenario_a.yaml/
       location_results/
       algorithm_tracking/

Batch execution
###############

Place alternative ``.yaml`` or ``.yml`` files in
``PROJECT_FOLDER/algorithm_configurations/`` and set:

.. code-block:: python

   RUN_ALGORITHM_CONFIG_BATCH = True

The runner processes the files in case-insensitive filename order. When normal
``RUN_*`` switches are enabled at the same time, those stages run before the
batch. Each scenario uses its own result directory and skips locations already
represented by a per-location result or status file there.

The batch directory can be changed in ``_run_workflow.py`` through
``ALGORITHM_CONFIG_BATCH_FOLDER``.
