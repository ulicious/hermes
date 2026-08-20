HERMES: Hydrogen Economy Routing Model for cost-efficient Supply
================================================================

HERMES is a multi-commodity routing model for hydrogen and hydrogen-derived
energy carriers. It combines production and conversion costs with several
transport options to search for cost-efficient supply paths to a user-defined
destination.

The current workflow is split into four main stages:

1. Process raw infrastructure and techno-economic input data.
2. Create start locations and attach production and conversion costs.
3. Run the routing algorithm for all start locations.
4. Optionally create plots for the generated results.

The main algorithm can also be run for several alternative algorithm
configuration files in one batch. In that mode each scenario writes its
``location_results`` and ``algorithm_tracking`` outputs to a result folder named
exactly like the configuration file.

Start with :doc:`getting_started` for a reproducible installation and first
run. :doc:`workflow` explains the processing stages and their prerequisites,
while :doc:`parameters` documents the available configuration.

Citation
========
Soon

Big thanks to:
==============

- genthalili's SeaRoute package (https://github.com/genthalili/searoute-py/tree/main)
- NetworkX: Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

.. toctree::
   :hidden:
   :maxdepth: 3

   getting_started
   workflow
   project_overview
   results
   advanced_usage
   custom_data
   parameters

.. role:: underline
    :class: underline
