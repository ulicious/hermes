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

First, choose a python interpreter and afterwards, install all requirements with following command in the IDE terminal.

.. code-block:: none

    pip install -r doc/requirements.txt

3. Setting up folder structure
==============================

Processed data and results need to be stored. Therefore, following folder structure needs to be implemented

.. code-block:: none

    PROJECT FOLDER/
        processed_data/
        raw_data/
        results/
            location_results/
            plots/

Please indicate the path towards the PROJECT_FOLDER in :ref:`general_configuration`.

.. _usage:

Usage
#####

The following article will describe the necessary steps to run the HERMES model

Adjust parameters if desired
============================

All parameters are set based on the parameters used in the publication. Some of these parameters directly affect the creation of random locations and the raw data processing. Therefore, please adjust parameters if desired. These include:

- algorithm_configuration: General configurations affecting the infrastructure processing, the algorithm, and scenario assumptions
- data/techno_economic_data_conversion: Techno-economic data of conversions (investments, feedstock demand and costs etc.)
- data/techno_economic_data_transportation: Techno-economic data of transport (which commodity can be transported by which transport mean and at which costs)

For the explanation of the different parameters, please see: :ref:`parameter_explanation_algorithm`, :ref:`parameter_explanation_conversion` and :ref:`parameter_explanation_transport`

Run Python code
===============

Run following python files consecutively:

1. _1_script_process_raw_data.py: Processing of raw data
2. _2_create_random_locations.py: Creates random locations
3. _3_main.py: Main algorithm to calculate most cost-efficient transport routes

The algorithm will process the random locations and creates a result file for each location in /PROJECT_FOLDER/results/location_results/

If desired, _4_plot_results will create some standard plots for the results

Things to consider
==================

- If techno-economic data and assumptions are changed, conversion costs need to be updated
    - run "1_script_process_raw_data" and "2_create_random_locations" with the setting update_only_conversion_costs_and_efficiency = True
- Data processing is quite time-consuming and heavily depends on the resources of your machine
- The processed data will take quite some storage space (distances are not calculate if 'use_low_storage' = True)
    - Minimal example: 11 MB (without distances) | ~500 MB (with distances)
    - Full approach: 55 MB (without distances) | ~5 GB (with distances)
- The computational expenses heavily rely on the data and setting