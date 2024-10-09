..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _parameter_explanation_transport:

###############################
Transport Parameter Explanation
###############################

Several parameters affect the HERMES model. Following article will describe all parameters in detail.

Parameters in algorithm_configuration.yaml
##########################################

Setting and assumptions affecting main algorithm
================================================

Affects main algorithm regarding tolerances, maximal distances of road and new pipelines, heat availability at destination etc.

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/algorithm.csv
   :delim: ;

.. math::
    a = b + c