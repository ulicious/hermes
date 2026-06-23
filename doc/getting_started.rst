..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _getting_started:

###############
Getting Started
###############

.. _installation:

Installation
############

1. Clone Git Project
====================

Use your integrated development environment (IDE) and clone the GitHub repository via terminal:

1. Navigate to the target directory
2. Use following command in your terminal

.. code-block:: none

    git clone https://github.com/ulicious/hermes

Alternatively, if your IDE has a version control integration, you can further clone the git project without a terminal command. Mostly, this is possible when creating new projects. For detailed instructions, please see the documentation of your IDE.

2. Install requirements
=======================

To install all required packages of HERMES, use following commands in the terminal of your IDE.

Using conda
-----------

If you have conda installed, you can use the transport_model environment. First, install the environment:

.. code-block:: none

    conda env create -f doc/environment.yml

and afterwards, choose the Python .exe file in the created folder of the environment.

Using pip
---------

First, choose a python interpreter and afterwards, install all requirements with the following command in the IDE terminal.

.. code-block:: none

    pip install -r requirements.txt

3. Setting up folder structure
==============================

The code expects a separate project folder referenced by ``project_folder_path``
in ``1_algorithm_configuration.yaml``. Template configuration files are stored
in the repository's ``data/`` directory. Create the project folder before data
processing by using the central workflow runner. Open ``_run_workflow.py``, set
``PROJECT_FOLDER`` to the desired working folder, set
``RUN_SETUP_PROJECT_FOLDER = True``, and run:

.. code-block:: none

    python _run_workflow.py

The setup step creates the required folder structure, copies editable
configuration files directly into ``PROJECT FOLDER/``, copies provided input
data into ``PROJECT FOLDER/raw_data/``, and writes the given project folder path
into the copied ``1_algorithm_configuration.yaml``. If setup is run
again, the copied files are overwritten.

The setup step creates missing folders automatically. The resulting structure
should look as follows:

.. code-block:: none

    PROJECT FOLDER/
        1_algorithm_configuration.yaml
        2_techno_economic_data_transportation.yaml
        3_techno_economic_data_conversion.yaml
        4_plotting_configuration.yaml
        algorithm_configurations/  # optional alternative algorithm YAML files
        raw_data/
        processed_data/
            inner_infrastructure_distances/
            mip_data/  # only if create_mip_data = True
        start_destination_combinations.csv
        results/
            location_results/
            algorithm_tracking/
            plots/

File copying is handled only by the setup step. The preprocessing step reads
the files from the project folder and does not copy repository files.

If an older project folder still contains numbered configuration files or a
``config/`` subfolder from a previous layout, the setup step removes the known
obsolete configuration files and writes the current files directly into
``PROJECT FOLDER/``.

The setup step writes the path towards ``PROJECT FOLDER`` into
:ref:`general_configuration`. The workflow scripts are stored in ``scripts/``
and are normally started through ``_run_workflow.py``. Advanced users can still
run a single step from the repository root via module call, for example
``python -m scripts._3_main "PROJECT FOLDER"``. To run the main algorithm with
an alternative algorithm configuration file, use
``python -m scripts._3_main "PROJECT FOLDER" --algorithm-config "path/to/config.yaml"``.
The ``HERMES_PROJECT_FOLDER`` environment variable can also be used.

Only this central project-folder path is stored as a path in
``1_algorithm_configuration.yaml``. All subfolders are derived by the code from
that project folder. Raw-data file names are fixed by the code as well and are
not configured in the YAML file. The standard raw input files copied by
the setup step include:

- ``location_data.csv``
- ``country_data.csv``
- ``network_pipelines_gas.xlsx``
- ``network_pipelines_oil.xlsx``
- ``seaports.geojson``
- ``water.zip``

.. _usage:

Usage
#####

The following article will describe the necessary steps to run the HERMES model

Adjust parameters if desired
============================

Most settings are controlled through the configuration files in
``PROJECT FOLDER/``. Before running the workflow, review at least:

- ``project_folder_path`` and the destination settings
- preprocessing and memory settings such as ``use_low_storage``, ``use_low_memory``, and ``create_mip_data``
- start-location settings such as ``number_locations``, ``location_creation_type``, ``use_voronoi_cells``, and island handling
- algorithm settings such as ``target_commodity``, distance tolerances, and infrastructure switches
- the techno-economic YAML files in ``PROJECT FOLDER/``

Settings such as raw-data file names or whether repository files should be
copied are not part of ``1_algorithm_configuration.yaml``. File copying is the
responsibility of the setup step; later workflow steps only read from the
project folder.

For a full explanation of the available parameters, see
:ref:`parameter_explanation_algorithm`, :ref:`parameter_explanation_conversion`,
:ref:`parameter_explanation_transport`, and :ref:`parameter_explanation_plotting`.

Run Python code
===============

Recommended: use the central workflow runner. Open ``_run_workflow.py``, set
``PROJECT_FOLDER`` and the ``RUN_*`` booleans at the top of the file, then run:

.. code-block:: none

    python _run_workflow.py

The runner starts the selected workflow scripts in order and passes the project
folder automatically. It can start setup, raw-data processing, start-location
creation, the main algorithm, MIP optimization, plot-data processing, plotting,
and algorithm-tracking analysis. Keep only the desired ``RUN_*`` flags set to
``True``.

To run several algorithm scenarios without repeatedly editing
``1_algorithm_configuration.yaml``, set ``RUN_ALGORITHM_CONFIG_BATCH = True`` in
``_run_workflow.py`` and place alternative algorithm configuration YAML files in
``PROJECT FOLDER/algorithm_configurations/``. The runner executes
``scripts._3_main`` once per YAML file for all not-yet-processed start
locations. Each scenario writes its outputs to a folder named exactly like the
configuration file:

.. code-block:: none

    PROJECT FOLDER/
        results/
            scenario_a.yaml/
                location_results/
                algorithm_tracking/

The default ``1_algorithm_configuration.yaml`` still writes to the standard
``PROJECT FOLDER/results/location_results/`` and
``PROJECT FOLDER/results/algorithm_tracking/`` folders.

Advanced users can also run individual workflow modules from the repository
root:

0. ``python -m scripts._0_setup_project_folder "PROJECT FOLDER"``: creates the project folder structure and copies configuration and input files
1. ``python -m scripts._1_script_process_raw_data "PROJECT FOLDER"``: preprocesses raw infrastructure data, ports, network distances, continent connectivity, and conversion costs at infrastructure nodes
2. ``python -m scripts._2_create_random_locations "PROJECT FOLDER"``: creates start locations and attaches location-specific production and conversion data
3. ``python -m scripts._3_main "PROJECT FOLDER"``: runs the routing algorithm for all not-yet-processed start locations
4. ``python -m scripts._4_mip_optimization "PROJECT FOLDER"``: optionally runs the MIP validation workflow
5. ``python -m scripts._5_process_plot_data "PROJECT FOLDER"``: optionally prepares result data for plotting
6. ``python -m scripts._6_plot_results "PROJECT FOLDER"``: optionally creates plots
7. ``python -m scripts._7_analyze_algorithm_tracking "PROJECT FOLDER"``: optionally analyzes algorithm tracking logs

The algorithm creates one result file per start location in
``PROJECT FOLDER/results/location_results/``. When ``--algorithm-config`` or
``RUN_ALGORITHM_CONFIG_BATCH`` is used with an alternative algorithm
configuration file, the result files are written to
``PROJECT FOLDER/results/<configuration filename>/location_results/`` instead.

If desired, the plot-data processing and plotting steps can be used afterwards
to create standard plots in ``PROJECT FOLDER/results/plots/``.

Things to consider
==================

- If techno-economic assumptions change, rerun the raw-data processing and
  start-location creation steps. The flags
  ``infrastructure_update_only_conversion_costs_and_efficiency`` and
  ``start_locations_update_only_conversion_costs_and_efficiency`` can be used
  to skip full infrastructure preprocessing or start-location regeneration.
- ``infrastructure_enforce_update_of_data`` forces regeneration of already
  processed infrastructure files.
- ``use_low_storage`` reduces disk usage by skipping precomputed inner-network
  distances, but increases runtime during routing.
- ``use_low_memory`` disables the parallel high-memory code path and usually
  increases runtime.
- ``create_mip_data`` writes additional processed files for later mixed-integer
  validation and therefore increases preprocessing effort.
- The minimal example overwrites the geographic bounds and restricts the workflow
  to Europe.
