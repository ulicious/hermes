..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _parameter_explanation_transport:

###############################
Transport Parameter Explanation
###############################

Transport assumptions and setting decide on availability of different transport means and costs for transport for each commodity. For each commodity, following structure needs to be set up:

.. code-block:: none

    commodity
        available transport means: list of transport means usable by commodity
        transport mean 1: costs
        transport mean 2: costs
        ...

All costs in EURO / MWh / 1000 km