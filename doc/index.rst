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

Citation
========
Soon

Big thanks to:
==============

- genthalili's SeaRoute package (https://github.com/genthalili/searoute-py/tree/main)
- NetworkX: Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

.. toctree::
   :hidden:
   :collapse: False
   :maxdepth: 3

   getting_started
   parameters
   methodology_deepdive

.. role:: underline
    :class: underline
