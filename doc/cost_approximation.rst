..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _cost_approximation:

##################
Cost Approximation
##################

Next to the comparison of the current total costs of a branch to global and node-commodity benchmarks, cost approximation is conducted to assess potential future total cost development of a branch. These cost approximations can be compared to the global benchmark as well and branches terminated.

Minimal costs to the destination
================================

One cost approximation is the minimal cost to the destination. This approximation is based on the direct distance between the current location of the branch and the destination. While a branch might need several iterations to reach the destination, the costs to the destination can always be approximated. The assumption here is that each transport mean can be used to transport the commodity to the destination, independent from existing infrastructure, making it the minimal costs possible and ensuring that no valid option achieves lower costs.

The minimal costs to the destination is calculated based on conversion costs of the branch commodity and the transport over the direct distance. Following pseudocode is applied:

.. code-block:: none

    minimal_costs_to_destination = Inf
    for all target_commodity in commodities convertable from branch_commodity:
        conversion_costs = conversion costs from branch_commodity to target_commodity

        for all transport_mean in possible transport means of target_commodity:
            specific_transport_costs = specific transport costs of target_commodity using transport_mean
            transport_costs = direct distance current branch location to destination * specific_transport_costs

            minimal_costs = total_costs of branch + conversion_costs + transport_costs

            if minimal_costs < minimal_costs_to_destination:
                minimal_costs_to_destination = minimal_costs

    if minimal_costs_to_destination > global_benchmark:
        terminate branch

Using Closest Infrastructure
----------------------------

The major downside of the minimal costs to destination approximation is the assumptions that the cheapest conversion and transport will be used, reducing the amount of terminated branches. To overcome this challenge, the closest infrastructure of the current branch is considered.

Outside of networks, the routing algorithm will move from one infrastructure to another (e.g. discharging shipping cargo and feeding-in into pipeline system). However, if these two infrastructures are distant (> tolerance_distance :ref:`algorithm`), then road or new pipeline transportation is necessary. These two transport means are generally more expensive than shipping or existing pipeline transport.

Using the closest node of the current branch location, additional costs can be approximated based on the road / new pipeline transport to the closest infrastructure. In this case, the type of infrastructure at the closest infrastructure is not considered to avoid complex coding exemptions. Following pseudocode is applied:

.. code-block:: none

    minimal_costs_to_destination = Inf

    # first, calculate costs from current location to closest node
    if distance_to_closest_infrastructure > tolerance_distance:  # road / new pipeline transport necessary
        costs_to_closest_infrastructure = Inf

        for all target_commodity in commodities convertable from branch_commodity:
            conversion_costs = conversion costs from branch_commodity to target_commodity

            for all transport_mean in [Road, New Pipeline]:  # only road and new pipeline possible
                specific_transport_costs = specific transport costs of target_commodity using transport_mean
                transport_costs = direct distance current branch location to destination * specific_transport_costs

                minimal_costs = total_costs of branch + conversion_costs + transport_costs

                if minimal_costs < costs_to_closest_infrastructure:
                    costs_to_closest_infrastructure = minimal_costs

        distance_to_destination = distance to destination from closest infrastructure

        branch_commodity = target_commodity  # since conversion took place, replace branch_commodity

    else: # infrastructure can be used directly
        costs_to_closest_infrastructure = 0
        distance_to_destination = distance to destination from current location

    # calculate costs to final destination --> distance was adjusted if transported to closest infrastructure
    for all target_commodity in commodities convertable from branch_commodity:
        conversion_costs = conversion costs from branch_commodity to target_commodity

        for all transport_mean in possible transport means of target_commodity:
            specific_transport_costs = specific transport costs of target_commodity using transport_mean
            transport_costs = distance_to_destination * specific_transport_costs

            minimal_costs = total_costs of branch + conversion_costs + transport_costs

            if minimal_costs < minimal_costs_to_destination:
                minimal_costs_to_destination = minimal_costs

    if minimal_costs_to_destination + costs_to_closest_infrastructure > global_benchmark:
        terminate branch

In general, the transport costs between infrastructure will have a major share on the total transport costs. Considering these will allow increased termination of branches.

Cost of Using Infrastructure
============================

Using shipping, gas and oil pipeline infrastructure are the most cost-efficient way to transport commodities. However, since not all commodities are transportable via these transport means, conversion might need to take place.

Each branch can be assessed regarding their ability to use certain transport means. Based on the current total costs and the cost to convert to a commodity that is transportable via certain transport means, it can be assessed if the branch is able to use the transport means.

.. code-block:: none

    for all transport_mean in transport_means:
        cost_transport_mean_using[transport_mean] = Inf

    for all target_commodity in commodities convertable from branch_commodity:
        conversion_costs = conversion costs from branch_commodity to target_commodity

        for all transport_mean in possible transport means of target_commodity:

            if conversion_costs < cost_transport_mean_using[transport_mean]:
                cost_transport_mean_using[transport_mean] = conversion_costs

    for all transport_mean in transport_means:
        if total_costs of branch + cost_transport_mean_using[transport_mean] > global_benchmark:
            branch cannot use transport mean
