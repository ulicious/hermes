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
in ``algorithm_configuration.yaml``. HERMES creates missing folders automatically,
but the resulting structure should look as follows:

.. code-block:: none

    PROJECT FOLDER/
        raw_data/
        processed_data/
            inner_infrastructure_distances/
            mip_data/  # only if create_mip_data = True
        start_destination_combinations.csv
        results/
            location_results/
            plots/

If ``use_provided_data`` is set to ``True``, the files from the repository's
``data/`` directory are copied into ``PROJECT FOLDER/raw_data/`` during preprocessing.

Please indicate the path towards ``PROJECT FOLDER`` in :ref:`general_configuration`.

.. _usage:

Usage
#####

The following article will describe the necessary steps to run the HERMES model

Adjust parameters if desired
============================

Most settings are controlled through ``algorithm_configuration.yaml`` and
``plotting_configuration.yaml``. Before running the workflow, review at least:

- ``project_folder_path`` and the destination settings
- preprocessing and memory settings such as ``use_low_storage``, ``use_low_memory``, and ``create_mip_data``
- start-location settings such as ``number_locations``, ``location_creation_type``, ``use_voronoi_cells``, and island handling
- algorithm settings such as ``target_commodity``, distance tolerances, and infrastructure switches
- the techno-economic YAML files in ``data/`` or in ``PROJECT FOLDER/raw_data/``

For a full explanation of the available parameters, see
:ref:`parameter_explanation_algorithm`, :ref:`parameter_explanation_conversion`,
:ref:`parameter_explanation_transport`, and :ref:`parameter_explanation_plotting`.

Run Python code
===============

Run following python files consecutively:

1. ``_1_script_process_raw_data.py``: preprocesses raw infrastructure data, ports, network distances, continent connectivity, and conversion costs at infrastructure nodes
2. ``_2_create_random_locations.py``: creates start locations and attaches location-specific production and conversion data
3. ``_3_main.py``: runs the routing algorithm for all not-yet-processed start locations

The algorithm creates one result file per start location in
``PROJECT FOLDER/results/location_results/``.

If desired, ``_4_plot_results.py`` can be used afterwards to create standard plots
in ``PROJECT FOLDER/results/plots/``.

Things to consider
==================

- If techno-economic assumptions change, rerun ``_1_script_process_raw_data.py`` and
  ``_2_create_random_locations.py``. The flag
  ``update_only_conversion_costs_and_efficiency`` can be used to skip the full
  infrastructure preprocessing.
- ``enforce_update_of_data`` forces regeneration of already processed files.
- ``use_low_storage`` reduces disk usage by skipping precomputed inner-network
  distances, but increases runtime during routing.
- ``use_low_memory`` disables the parallel high-memory code path and usually
  increases runtime.
- ``create_mip_data`` writes additional processed files for later mixed-integer
  validation and therefore increases preprocessing effort.
- The minimal example overwrites the geographic bounds and restricts the workflow
  to Europe.
