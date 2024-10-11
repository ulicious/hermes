..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _parameters:

#####################
Parameter Explanation
#####################

.. _parameter_explanation_algorithm:

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

.. _parameter_explanation_conversion:

Conversion Parameter
####################

Cost type
=========

The first thing to adjust is the cost type of each feedstock and the capital costs.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/conversion_cost_parameters_cost_type.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

Possible are:

- 'location': costs will be looked up for each location (from location_data)
- 'uniform': as set in techno_economic_data_conversion.yaml (see below)
- ['COUNTRY_NAME_1', 'COUNTRY_NAME_2', ...]: costs will be looked up for specific countries in list, for all other based on location data
- 'all_countries': always country (from country_data)

These cost types must be defined for each feedstock and the capital costs:

- Hydrogen_Gas
- Electricity
- CO2
- Low_Temperature_Heat
- Mid_Temperature_Heat
- High_Temperature_Heat
- Nitrogen
- interest_rate

Uniform Costs
=============

Uniform costs can be adjusted as well.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/conversion_cost_parameters.csv
   :width: 100
   :widths: 30, 10, 60
   :delim: ;

Conversion Specific Assumptions
===============================

Next to feedstock and capital costs, conversion specific parameters are implemented. For each commodity, the target commodities, which the initial commodity can be converted into, must be specified. Furthermore, for each target commodity, the techno-economic parameters must be specified. The structure for each initial commodity looks as following:

.. code-block:: none

    initial commodity:
        target commodities: list with names of target commodities
        target commodity 1:
            techno economic parameters target commodity 1 (see below)
        target commodity 2:
            techno economic parameters target commodity 2 (see below)

And following parameters are necessary.

.. _tea_parameters_conversion:

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/target_commodity.csv
   :width: 100
   :widths: 20, 20, 60
   :delim: ;

.. _parameter_explanation_transport:

Transport Parameter
###################

Transport assumptions and setting decide on availability of different transport means and costs for transport for each commodity. For each commodity, following structure needs to be set up:

.. code-block:: none

    commodity:
        available transport means: list of transport means usable by commodity
        transport mean 1: costs
        transport mean 2: costs
        ...

All costs in EURO / MWh / 1000 km

