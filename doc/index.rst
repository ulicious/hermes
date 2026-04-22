.. HERMES: Hydrogen Economy Routing Model for cost-efficient Supply documentation master file, created by
   sphinx-quickstart on Tue Oct  8 11:11:51 2024.

HERMES: Hydrogen Economy Routing Model for cost-efficient Supply
================================================================

HERMES is a multi-commodity routing model for hydrogen and hydrogen-derived
energy carriers. It combines preprocessing of ports, gas pipelines, liquid
pipeline infrastructure, and location-specific conversion costs with a routing
heuristic that searches for the lowest-cost supply path to a user-defined
destination.

The current workflow is split into four main steps:

1. Process raw infrastructure and techno-economic input data.
2. Create start locations and attach production and conversion costs.
3. Run the routing algorithm for all start locations.
4. Optionally create plots for the generated results.

The documentation in this folder reflects the current repository structure,
the active YAML configuration files, and the parameter set used by the
latest code in ``_1_script_process_raw_data.py``, ``_2_create_random_locations.py``,
``_3_main.py``, and ``_4_plot_results.py``.

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
   parameters
   methodology_deepdive

.. role:: underline
    :class: underline
