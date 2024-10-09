..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _parameter_explanation_algorithm:

###############################
Parameter Explanation
###############################

Several parameters affect the HERMES model. Following article will describe all parameters in detail.


General configuration
====

Settings regarding general configuration include overarching settings like paths or computational resources

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/general_configuration.csv
   :delim: ";"

Available commodities and transport means
====

These two lists indicate which commodities and transport means can be used. This will affect raw data processing an the algorithm

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/availability.csv
   :delim: ";"

General raw data processing assumptions
====

Includes assumptions regarding costs and updates of cost calculations

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/raw_data_processing.csv
   :delim: ";"

Setting and assumptions affecting starting locations
====

Affects number of starting locations, limits locations (continents / coordinates) and indicate if heat is available at start

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/random_locations.csv
   :delim: ";"

Setting and assumptions affecting infrastructure processing
====

Affects number of access points in pipelines and indicates if heat is available at infrastructure

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/infrastructure_processing.csv
   :delim: ";"

Setting and assumptions affecting main algorithm
====

Affects main algorithm regarding tolerances, maximal distances of road and new pipelines, heat availability at destination etc.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/algorithm.csv
   :delim: ";"
