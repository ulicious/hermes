..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _location_creation:

##########################
Starting Location Creation
##########################

Starting locations are created randomly all across the globe and complemented with their commodity production costs. All locations are created to reach the same destination, which needs to be defined in :ref:`locations`. The number of created locations can be defined there as well.

To restrict the location process, one can either define the continents of the starting location (origin_continents), or set minimal and maximal latitudes and longitudes of the area, where the locations are placed in. The restricting parameters can be found in :ref:`locations`.

Attaching Conversion Costs
==========================

Similar to attaching conversion costs to the infrastructure, conversion costs are calculated for the random locations as well to complement the production costs of other commodities since only the hydrogen production costs are given as input to the model.

Details on the calculation method is given in :ref:`calculation_conversion_costs`.