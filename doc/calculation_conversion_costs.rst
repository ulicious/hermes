..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _calculation_conversion_costs:

###########################
Conversion Cost Calculation
###########################

Pre-calculation of Conversion Costs
-----------------------------------

In :ref:`parameter_explanation_conversion`, parameters are defined which are used to calculate conversion costs. These conversion costs are calculated prior to the routing algorithm when processing the infrastructure and creating random locations. The calculation uses the given data on feedstock costs, the capital costs and the techno-economic data of the conversion to calculate the conversion costs of 1 MWh of the initial commodity.

.. math::
    \text{Specific Conversion Costs [EURO / MWh]} = \frac{\text{Specific Investment}}{\text{Operating Hours}} \cdot (\text{Annuity Factor} + \text{Maintenance}) + \sum \text{Specific Feedstock Demand} \cdot \text{Feedstock Costs}

Interest rate and feedstock costs can be country or location-specific. Therefore, each location might have specific conversion costs of a technology.

Within the algorithm, the specific conversion costs are used to calculate the total costs after conversion. The old total costs are all previous costs including initial hydrogen production costs, previous conversion and transportation costs:

.. math::
    \text{New Total Costs} = \frac{\text{Old Total Costs} + \text{Specific Conversion Costs}}{\text{Efficiency}}

It is necessary to mention that the efficiency depends on the assumption if the heat demand can be covered by external sources (than costs are included in specific conversion costs) or will be covered by burning the initial commodity (the efficiency will be lower) (see :ref:`conversion_settings`, :ref:`locations` and :ref:`infrastructure`).
