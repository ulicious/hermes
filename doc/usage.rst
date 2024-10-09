..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _usage:

#####
Usage
#####

The following article will describe the necessary steps to run the HERMES model

1. Setting up folder structure
##############################

Processed data and results need to be stored. Therefore, following folder structure needs to be implemented

.. code-block:: none

    PROJECT FOLDER/
        processed_data/
        raw_data/
        results/
            location_results/
            plots/

Please indicate the path towards the PROJECT_FOLDER in algorithm_configuration -> general_configurations -> project_folder_path

2. Adjust parameters if desired
###############################

All parameters are set based on the parameters used in the publication. Some of these parameters directly affect the creation of random locations and the raw data processing. Therefore, please adjust parameters if desired. These include:

1. algorithm_configuration -> general configurations
2. algorithm_configuration -> general information
3. algorithm_configuration -> parameters and assumptions for start - destination - combination file
4. algorithm_configuration -> parameters and assumptions for infrastructure processing
5. data/techno_economic_data_conversion

For the explanation of the different parameters, please see: :ref:`parameter_explanation_algorithm`, :ref:`parameter_explanation_conversion` and :ref:`parameter_explanation_transport`

3. Run Python code
##################

Run following python files consecutively:

1. _1_script_process_raw_data.py: Processing of raw data
2. _2_create_random_locations.py: Creates random locations
3. _3_main.py: Main algorithm to calculate most cost-efficient transport routes

The algorithm will process the random locations and creates a result file for each location in /PROJECT_FOLDER/results/location_results/

If desired, _4_plot_results will create some standard plots for the results