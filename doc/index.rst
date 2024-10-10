.. HERMES: Hydrogen Economy Routing Model for cost-efficient Supply documentation master file, created by
   sphinx-quickstart on Tue Oct  8 11:11:51 2024.

HERMES: Hydrogen Economy Routing Model for cost-efficient Supply
================================================================

HERMES is a multi commodity multi mean of transport algorithm,
capable to find the most cost-effective transportation route from pre-defined starting points to a desired location.
The algorithm derives possible solutions for transportation from a starting point to the final destination.
Based on the underlying infrastructure data, the algorithm iteratively explores infrastructure nodes and calculates
costs for each reached node. It terminates as soon as it reaches the final destination and exploration
to other nodes is not possible anymore

Important things to consider:
=============================

- If techno-economic data and assumptions are changed, conversion costs need to be updated
  - run "1_script_process_raw_data" and "2_create_random_locations" with the setting update_only_conversion_costs_and_efficiency = True
- Data processing is quite time-consuming and heavily depends on the resources of you computer
- The processed data will take quite some storage space (distances are not calculate if 'use_low_storage' = True)
  - Minimal example: 11 MB (without distances) | ~500 MB (with distances)
  - Full approach: 55 MB (without distances) | ~5 GB (with distances)
- The computational expenses heavily rely on the data and setting

Citation
========
Soon

Big thanks to:
==============

- genthalili's SeaRoute package (https://github.com/genthalili/searoute-py/tree/main)
- NetworkX: Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started:

   introduction
   installation
   usage

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Parameters:

   parameter_explanation_algorithm
   parameter_explanation_conversion
   parameter_explanation_transport

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Methodology Deepdive:

   calculation_conversion_costs
   infrastructure_processing
   location_creation
   benchmarking
   cost_approximation

.. role:: underline
    :class: underline
