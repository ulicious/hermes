..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _usage:

#####
Usage
#####

The following article will describe the necessary steps to run the HERMES model

2. Adjust parameters if desired
===============================

All parameters are set based on the parameters used in the publication. Some of these parameters directly affect the creation of random locations and the raw data processing. Therefore, please adjust parameters if desired. These include:

1. algorithm_configuration -> general configurations
2. algorithm_configuration -> general information
3. algorithm_configuration -> parameters and assumptions for start - destination - combination file
4. algorithm_configuration -> parameters and assumptions for infrastructure processing
5. data/techno_economic_data_conversion

For the explanation of the different parameters, please see: :ref:`parameter_explanation_algorithm`, :ref:`parameter_explanation_conversion` and :ref:`parameter_explanation_transport`

3. Run Python code
==================

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
- Data processing is quite time-consuming and heavily depends on the resources of you computer
- The processed data will take quite some storage space (distances are not calculate if 'use_low_storage' = True)
  - Minimal example: 11 MB (without distances) | ~500 MB (with distances)
  - Full approach: 55 MB (without distances) | ~5 GB (with distances)
- The computational expenses heavily rely on the data and setting