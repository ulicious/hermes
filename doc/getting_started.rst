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
in ``_1_algorithm_configuration.yaml``. Template configuration files are stored
in the repository's ``data/`` directory. Create the project folder before data
processing by running:

.. code-block:: none

    python _0_setup_project_folder.py "PROJECT FOLDER"

The setup script creates the required folder structure, copies configuration
files into ``PROJECT FOLDER/config/``, copies provided input data into
``PROJECT FOLDER/raw_data/``, and writes the given project folder path into the
copied ``_1_algorithm_configuration.yaml``. If the setup script is run again,
the copied files are overwritten.

The setup script creates missing folders automatically. The resulting structure
should look as follows:

.. code-block:: none

    PROJECT FOLDER/
        config/
            _1_algorithm_configuration.yaml
            _5_plotting_configuration.yaml
            techno_economic_data_conversion.yaml
            techno_economic_data_transportation.yaml
        raw_data/
        processed_data/
            inner_infrastructure_distances/
            mip_data/  # only if create_mip_data = True
        start_destination_combinations.csv
        results/
            location_results/
            plots/

File copying is handled only by ``_0_setup_project_folder.py``. The preprocessing
script ``_1_script_process_raw_data.py`` reads the files from the project folder
and does not copy repository files.

The setup script writes the path towards ``PROJECT FOLDER`` into
:ref:`general_configuration`.
If the project folder differs from the template default, pass it to later scripts
with ``--project-folder "PROJECT FOLDER"`` or set the ``HERMES_PROJECT_FOLDER``
environment variable.

.. _usage:

Usage
#####

The following article will describe the necessary steps to run the HERMES model

Adjust parameters if desired
============================

Most settings are controlled through the configuration files in
``PROJECT FOLDER/config/``. Before running the workflow, review at least:

- ``project_folder_path`` and the destination settings
- preprocessing and memory settings such as ``use_low_storage``, ``use_low_memory``, and ``create_mip_data``
- start-location settings such as ``number_locations``, ``location_creation_type``, ``use_voronoi_cells``, and island handling
- algorithm settings such as ``target_commodity``, distance tolerances, and infrastructure switches
- the techno-economic YAML files in ``PROJECT FOLDER/config/``

For a full explanation of the available parameters, see
:ref:`parameter_explanation_algorithm`, :ref:`parameter_explanation_conversion`,
:ref:`parameter_explanation_transport`, and :ref:`parameter_explanation_plotting`.

Run Python code
===============

Run following python files consecutively:

0. ``_0_setup_project_folder.py "PROJECT FOLDER"``: creates the project folder structure and copies configuration and input files
1. ``_1_script_process_raw_data.py --project-folder "PROJECT FOLDER"``: preprocesses raw infrastructure data, ports, network distances, continent connectivity, and conversion costs at infrastructure nodes
2. ``_2_create_random_locations.py --project-folder "PROJECT FOLDER"``: creates start locations and attaches location-specific production and conversion data
3. ``_3_main.py --project-folder "PROJECT FOLDER"``: runs the routing algorithm for all not-yet-processed start locations

The algorithm creates one result file per start location in
``PROJECT FOLDER/results/location_results/``.

If desired, ``_4_plot_results.py`` can be used afterwards to create standard plots
in ``PROJECT FOLDER/results/plots/``.

Things to consider
==================

- If techno-economic assumptions change, rerun ``_1_script_process_raw_data.py`` and
  ``_2_create_random_locations.py``. The flags
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
