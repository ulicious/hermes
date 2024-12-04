..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _methodology_deepdive:

####################
Methodology Deepdive
####################

.. _calculation_conversion_costs:

Conversion Cost Calculation
###########################

Pre-calculation of Conversion Costs
===================================

In :ref:`parameter_explanation_conversion`, parameters are defined which are used to calculate conversion costs. These conversion costs are calculated prior to the routing algorithm when processing the infrastructure and creating random locations. The calculation uses the given data on feedstock costs, the capital costs and the techno-economic data of the conversion to calculate the conversion costs of 1 MWh of the initial commodity.

.. math::
    \text{Specific Conversion Costs [EURO / MWh]} = \frac{\text{Specific Investment}}{\text{Operating Hours}} \cdot (\text{Annuity Factor} + \text{Maintenance}) + \sum \text{Specific Feedstock Demand} \cdot \text{Feedstock Costs}

Interest rate and feedstock costs can be country or location-specific. Therefore, each location might have specific conversion costs of a technology.

Within the algorithm, the specific conversion costs are used to calculate the total costs after conversion. The old total costs are all previous costs including initial hydrogen production costs, previous conversion and transportation costs:

.. math::
    \text{New Total Costs} = \frac{\text{Old Total Costs} + \text{Specific Conversion Costs}}{\text{Efficiency}}

It is necessary to mention that the efficiency depends on the assumption if the heat demand can be covered by external sources (than costs are included in specific conversion costs) or will be covered by burning the initial commodity (the efficiency will be lower) (see :ref:`locations`, :ref:`infrastructure` and :ref:`algorithm`).

.. _infrastructure_processing:

Infrastructure Processing
#########################

In the beginning of using HERMES, infrastructure data needs to be processed. This includes assessing the existing pipeline infrastructure, creating networks from the infrastructure data and calculate distances between infrastructure access points and conversion costs at each infrastructure access point.

Pipeline Processing
===================

Creating Networks
-----------------

Based on the raw pipeline data (QUELLE), single pipeline segments are connected to full pipeline networks. Connection takes place if:

- The two segments intersect
- The two segments are within gap_distance (see :ref:`infrastructure`)
    - Large distances are bridged by implementing new segments, connecting both existing segments
    - Short distances, mainly due to floating point errors of coordinates, are bridged by extending existing segments minimally to ensure intersection of segments

This process is repeated until no further segments can be connected.

Network Simplification
----------------------

In a later stage, access points to the pipelines will be added. the number of access points are based on the underlying pipeline structure. Unnecessary pipeline segments will increase the number of access points without adding value. Therefore, unnecessary pipeline segments are removed. These include:

- Parallel pipeline segments (same start and end; similar length; similar direction) --> These will be removed
- Short (< 5000m) pipeline segments with dead ends --> These will be removed

This approach reduces the complexity of the pipeline networks significantly without removing the overall structure of the data.

Adding Access Points
--------------------

The last step of the pipeline processing is the addition of access points. The geographical points mentioned above are not sufficient since long distances without geographical points exist, making the pipeline difficult to access.

Access points are added to each pipeline segment uniformly. Their number is based on the parameter minimal_distance_between_pipeline_connection_points (see :ref:`infrastructure`) and calculated as following:

.. math::
    \text{Number Access Points} = 2 - 1 + \Bigl \lceil \frac{\text{Length Segment}}{\text{Minimal Distance Between Access Points}} \Bigr \rceil

This number of access points ensures that the distance between access points is <= minimal_distance_between_pipeline_connection_points. All access points are distributed uniformly along the pipeline segment.

Calculating Distances
=====================

Pipeline and shipping infrastructure can be seen as graphs with nodes (pipelines: access points; shipping: ports) and edges (piplelines and shipping routes). Since this infrastructure does not change during the routing algorithm, distances can be calculated before the routing algorithm to avoid repetitive calculation of distances.

Therefore, the network data is processed as undirected graphs and the Dijkstra algorithm is used to calculate shortest distances between the nodes. The information of distances is stored on the device. These calculations are conducting using the `NetworkX <https://networkx.org/>`_ Python package.

In the case of shipping, the Python package `Sea Route <https://github.com/genthalili/searoute-py/tree/main>`_ is used to calculate the distances, using Dijkstra as well.

Attaching Conversion Costs
==========================

Another calculation taking place before the routing algorithm is the calculation of conversion costs at each pipeline access point and port. These calculations are conducted since conversion costs do not change as well and generally apply for all calculations in the routing algorithm.

Details on the calculation method is given in :ref:`calculation_conversion_costs`.

.. _location_creation:

Starting Location Creation
##########################

Starting locations are created randomly all across the globe and complemented with their commodity production costs. All locations are created to reach the same destination, which needs to be defined in :ref:`locations`. The number of created locations can be defined there as well.

To restrict the location process, one can either define the continents of the starting location (origin_continents), or set minimal and maximal latitudes and longitudes of the area, where the locations are placed in. The restricting parameters can be found in :ref:`locations`.

Attaching Conversion Costs
==========================

Similar to attaching conversion costs to the infrastructure, conversion costs are calculated for the random locations as well to complement the production costs of other commodities since only the hydrogen production costs are given as input to the model.

Details on the calculation method is given in :ref:`calculation_conversion_costs`.

.. _benchmarking:

Benchmarking
############

The main instrument to reduce branches in the algorithm is the comparison of branches with benchmarks. In the algorithm, two types of benchmarks exist:

- Global Benchmark
- Nodal-Commodity Benchmark

Global benchmark
================

This first benchmark is a global benchmark and affects all branches. This benchmark is calculated initially based on chosen routes and commodities without the ability to improve the benchmark. From all initially benchmarks calculated, the benchmark with the lowest costs is chosen.

FOLLOWS

All branches are compared regularly with this benchmark, and branches exceeding this benchmark are terminated. In addition, this benchmark is updated as soon as a branch reaches the final destination with the correct commodity.

Node-Commodity Benchmarks
==========================

Next to the global benchmark, node-commodity benchmarks exist. The routing algorithm explores all nodes iteratively and each branch is able to visit each node. Since branches develop parallely, it occurs that different branches visit the same node. The routing algorithm uses this circumstance to evaluate the node-commodity combination, the commodity being the current transported commodity of the branch, and set a node-commodity benchmark based on the current total costs of the branch.

The total costs of all following branches, reaching the same node and having the same commodity, will be compared to the node-commodity benchmark and terminated, if the node-commodity benchmark is lower than the total costs of the branch. This approach helps to terminate branches early.

.. _cost_approximation:

Cost Approximation
##################

Next to the comparison of the current total costs of a branch to global and node-commodity benchmarks, cost approximation is conducted to assess potential future total cost development of a branch. These cost approximations can be compared to the global benchmark as well and branches terminated.

Minimal costs to the destination
================================

One cost approximation is the minimal cost to the destination. This approximation is based on the direct distance between the current location of the branch and the destination. While a branch might need several iterations to reach the destination, the costs to the destination can always be approximated. The assumption here is that each transport mean can be used to transport the commodity to the destination, independent from existing infrastructure, making it the minimal costs possible and ensuring that no valid option achieves lower costs.

The minimal costs to the destination is calculated based on conversion costs of the branch commodity and the transport over the direct distance. Following pseudocode is applied:

.. code-block:: none

    for branch in branches:

        minimal_costs_to_destination = Inf
        for all target_commodity in commodities convertable from branch_commodity:
            conversion_costs = conversion costs from branch_commodity to target_commodity
            conversion_efficiency = efficiency from branch_commodity to target_commodity

            for all transport_mean in possible transport means of target_commodity:
                specific_transport_costs = specific transport costs of target_commodity using transport_mean
                transport_costs = direct distance current branch location to destination * specific_transport_costs

                minimal_costs = (total_costs of branch + conversion_costs) / conversion_efficiency + transport_costs

                if minimal_costs < minimal_costs_to_destination:
                    minimal_costs_to_destination = minimal_costs

        if minimal_costs_to_destination > global_benchmark:
            terminate branch

Using Closest Infrastructure
----------------------------

The major downside of the minimal costs to destination approximation is the assumptions that the cheapest conversion and transport will be used, reducing the amount of terminated branches. To overcome this challenge, the closest infrastructure of the current branch is considered if the following transportation is based on road or new pipeline transportation.

Outside of networks, the routing algorithm will move from one infrastructure to another (e.g. discharging shipping cargo and feeding-in into pipeline system). However, if these two infrastructures are distant (> tolerance_distance :ref:`algorithm`), then road or new pipeline transportation is necessary. These two transport means are generally more expensive than shipping or existing pipeline transport.

Using the closest infrastructure to the current branch location, additional costs can be approximated based on the road / new pipeline transport to the closest infrastructure. In this case, the type of infrastructure at the closest infrastructure is not considered to avoid complex coding exemptions. Following pseudocode is applied:

.. code-block:: none

    for branch in branches:

        minimal_costs_to_destination = Inf

        # first, calculate costs from current location to closest node
        if distance_to_closest_infrastructure > tolerance_distance:  # road / new pipeline transport necessary
            costs_to_closest_infrastructure = Inf

            for all target_commodity in commodities convertable from branch_commodity:
                conversion_costs = conversion costs from branch_commodity to target_commodity
                conversion_efficiency = efficiency from branch_commodity to target_commodity

                for all transport_mean in [Road, New Pipeline]:  # only road and new pipeline possible
                    specific_transport_costs = specific transport costs of target_commodity using transport_mean
                    transport_costs = direct distance current branch location to destination * specific_transport_costs

                    minimal_costs = (total_costs of branch + conversion_costs) / conversion_efficiency + transport_costs

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
            conversion_efficiency = efficiency from branch_commodity to target_commodity

            for all transport_mean in possible transport means of target_commodity:
                specific_transport_costs = specific transport costs of target_commodity using transport_mean
                transport_costs = distance_to_destination * specific_transport_costs

                minimal_costs = (total_costs of branch + conversion_costs) / conversion_efficiency + transport_costs

                if minimal_costs < minimal_costs_to_destination:
                    minimal_costs_to_destination = minimal_costs

        if minimal_costs_to_destination + costs_to_closest_infrastructure > global_benchmark:
            terminate branch

In general, the transport costs between infrastructure will have a major share on the total transport costs. Considering these will allow increased termination of branches.

Excluding Infrastructure
========================

Based on Conversion Costs
-------------------------

Using shipping, gas and oil pipeline infrastructure are the most cost-efficient ways to transport commodities. However, since not all commodities are transportable via these transport means, conversion might need to take place.

Each branch can be assessed regarding their ability to use certain transport means. Based on the current total costs and the cost to convert to a commodity that is transportable via certain transport means, it can be assessed if the branch is able to use the transport means.

.. code-block:: none

    for branch in branches

        for all transport_mean in transport_means:
            cost_transport_mean_using[transport_mean] = Inf
            efficiency[transport_mean] = Inf

        for all target_commodity in commodities convertable from branch_commodity:
            conversion_costs = conversion costs from branch_commodity to target_commodity
            conversion_efficiency = efficiency from branch_commodity to target_commodity

            for all transport_mean in possible transport means of target_commodity:

                if conversion_costs < cost_transport_mean_using[transport_mean]:
                    cost_transport_mean_using[transport_mean] = conversion_costs
                    efficiency[transport_mean] = conversion_efficiency

        for all transport_mean in transport_means:
            if (total_costs of branch + cost_transport_mean_using[transport_mean]) / efficiency[transport_mean] > global_benchmark:
                branch cannot use transport mean

Based on Distance
-----------------

Furthermore, infrastructures can be excluded if they are too far based on residual costs of global benchmark and the current total costs of the branch.

.. code-block:: none

    for branch in branches:
        for infrastructure in infrastructures:

            residual_costs = global_benchmark - total_costs of branch
            maximal_distance = 0

            for all target_commodity in commodities convertable from branch_commodity:
                conversion_costs = conversion costs from branch_commodity to target_commodity
                conversion_efficiency = efficiency from branch_commodity to target_commodity

                for all transport_mean in possible transport means of target_commodity:

                    specific_transport_costs = specific transport costs of target_commodity using transport_mean

                    reachable_distance = residual_costs / specific_transport_costs

                    if reachable_distance > maximal_distance:
                        maximal_distance = reachable_distance

            if maximal_distance < distance_to_infrastructure:
                exclude infrastructure for branch

.. _data:

Applied Data
############

Pipeline Infrastructure
=======================

The pipeline data was obtained from the `Global Energy Monitor <https://globalenergymonitor.org/>`_. However, only pipelines were used which are either build or are currently under construction.

Shipping Infrastructure
=======================

Shipping infrastructure is based on `World Bank <https://datacatalog.worldbank.org/search/dataset/0038118/Global---International-Ports>`_ data.

Hydrogen Production and Electricity Generation Costs
====================================================

Hydrogen production and electricity generation costs are based on a techno-economic assessment using optimization methods. The methodology and code is currently under review, and the location of the code will be published with the publication of the methodology.

Applying Custom Data
====================

Custom data can be applied as well. Two cases are possible:

- Custom data needs to be processed: Place your data in PROJECT FOLDER/raw_data/ and set the configuration use_provided_data to False (see :ref: general_configuration).
- Custom data was processed: Place your processed data into PROJECT FOLDER/processed_data/ and run the algorithm. If only parts of the data is custom, you can run the infrastructure processing and replace the respective files while keeping all other.

Important: Your custom data (processed and not processed) needs to resemble the input and processed data of the model.

To adjust the data, customizing the standard raw data is always possible to add/delete pipeline segments or ports.