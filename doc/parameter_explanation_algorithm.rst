..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _parameter_explanation_algorithm:

###################
Algorithm Parameter
###################

Several parameters affect the HERMES model. Following article will describe all parameters in detail.

.. _general_configuration:

General configuration
=====================

Settings regarding general configuration include overarching settings like paths or computational resources, available commodities and transport means, and how recalculations of infrastructure and location conversion costs are processed.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/general_configuration.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

.. _locations:

Setting and assumptions affecting starting locations
====================================================

Affects number of starting locations, limits locations (continents / coordinates) and indicate if heat is available at start

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/random_locations.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

.. _infrastructure:

Setting and assumptions affecting infrastructure processing
===========================================================

Affects number of access points in pipelines and indicates if heat is available at infrastructure

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/infrastructure_processing.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

.. _algorithm:

Setting and assumptions affecting main algorithm
================================================

Affects main algorithm regarding tolerances, maximal distances of road and new pipelines, heat availability at destination etc.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/algorithm.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;
