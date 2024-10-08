..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _benchmarking:

###############################
Benchmarking
###############################

The main instrument to reduce branches in the algorithm is the comparison of branches with the benchmark. In the algorithm, two types of benchmarks exist:

Global benchmark
====

This first benchmark is a global benchmark and affects all branches. This benchmark is calculated initially based on chosen routes and commodities without the ability to improve the benchmark. From all initially benchmarks calculated, the benchmark with the lowest costs is chosen.

To calculate the initial benchmark, a combination of transport means and commodities 

All branches are compared regularly with this benchmark, and branches exceeding this benchmark are terminated. In addition, this benchmark is updated as soon as a branch reaches the final destination with the correct commodity.

