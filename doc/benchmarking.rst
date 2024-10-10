..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _benchmarking:

############
Benchmarking
############

The main instrument to reduce branches in the algorithm is the comparison of branches with benchmarks. In the algorithm, two types of benchmarks exist:

- Global Benchmark
- Nodal-Commodity Benchmark

Global benchmark
================

This first benchmark is a global benchmark and affects all branches. This benchmark is calculated initially based on chosen routes and commodities without the ability to improve the benchmark. From all initially benchmarks calculated, the benchmark with the lowest costs is chosen.

To calculate the initial benchmark, a combination of transport means and commodities 

All branches are compared regularly with this benchmark, and branches exceeding this benchmark are terminated. In addition, this benchmark is updated as soon as a branch reaches the final destination with the correct commodity.

Node-Commodity Benchmarks
==========================

Next to the global benchmark, node-commodity benchmarks exist. The routing algorithm explores all nodes iteratively and each branch is able to visit each node. Since branches develop parallely, it occurs that different branches visit the same node. The routing algorithm uses this circumstance to evaluate the node-commodity combination, the commodity being the current transported commodity of the branch, and set a node-commodity benchmark based on the current total costs of the branch.

The total costs of all following branches, reaching the same node and having the same commodity, will be compared to the node-commodity benchmark and terminated, if the node-commodity benchmark is lower than the total costs of the branch. This approach helps to terminate branches early.