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

These settings define paths, computational limits, the globally available
commodities and transport means, and whether preprocessing artefacts should be
reused or rebuilt.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/general_configuration.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

.. _locations:

Setting and assumptions affecting starting locations
====================================================

Affects the destination definition, the generation of start locations, the use of
Voronoi cells, and whether heat is available at the origin.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/random_locations.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

.. _infrastructure:

Setting and assumptions affecting infrastructure processing
===========================================================

Affects preprocessing of ports and pipeline networks, the spacing of pipeline
access points, and whether heat is available at infrastructure nodes.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/infrastructure_processing.csv
   :width: 100
   :widths: 20, 10, 50, 20
   :delim: ;

.. _algorithm:

Setting and assumptions affecting main algorithm
================================================

Affects routing tolerances, maximum distances for additional transport segments,
destination-side heat assumptions, and whether commodity strike prices are added
to the optimization target.

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

The transport input data is stored in ``techno_economic_data_transportation.yaml``.
For each commodity, the file defines the admissible transport means together with
the corresponding cost and loss assumptions.

.. _parameter_explanation_plotting:

Plotting Parameter
###################

``plotting_configuration.yaml`` controls which result folders are processed and
how plots are styled. The current plotting workflow expects the following
structure:

.. code-block:: none

    PROJECT FOLDER/
        results/
            location_results/  # per-location result csv files written by _3_main.py
            plots/  # figures written by _4_plot_results.py

The ``process_results`` setting is used to name result sets that should be
processed for later comparison workflows.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/plotting_processing_parameters.csv
   :width: 100
   :widths: 20, 20, 40, 20
   :delim: ;

Single-result plotting options:

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/plotting_single_parameters.csv
   :width: 100
   :widths: 20, 20, 60
   :delim: ;

Comparison plotting options:

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/plotting_comparison_parameters.csv
   :width: 100
   :widths: 20, 20, 40, 20
   :delim: ;

Styling parameters in the current plotting configuration:

.. code-block:: none

    commodity_colors:  # adjust commodity colors
        commodity_1: color_1
        commodity_2: color_2
        ...

    nice_name_dictionary:  # to define nice names of results. If result not in dictionary, then nice name is result name
        result_1: nice_name_result_1
        result_2: nice_name_result_2
        ...

    transport_mean_styles:  # set line styles of transport routes based on transport means
        ...

    line_widths: # sets line widths of transport routes based on transport means
        ...

    infrastructure_colors:  # defines colors in infrastructure plot
        ...
