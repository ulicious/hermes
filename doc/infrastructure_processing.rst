..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _infrastructure_processing:

#########################
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

In a later stage, access points to the pipelines will be added. These access points are based on the underlying pipeline structure. Unnecessary pipeline segments will increase the number of access points without adding value. Therefore, unnecessary pipeline segments are removed. These include:

- Parallel pipeline segments (same start and end; similar length; similar direction) --> These will be removed
- Short (< 5000m) pipeline segments with dead ends --> These will be removed

The created networks in the current form have no access points, only geographical points based on starting and ending points, intersections between pipelines, and locations where pipelines change directions (even slightly). To reduce the complexity further, such geographical points are consolidated if they belong to the same network and their distance is below the gap_distance. This approach removes short, redundant pipeline segments.

This approach reduces the complexity of the pipeline networks significantly without removing the overall structure of the data.

Adding Access Points
--------------------

The last step of the pipeline processing is the addition of access points. The geographical points mentioned above are not sufficient since long distances without geographical points exist, making the pipeline difficult to access.

Access points are added to each pipeline segment uniformly. Their number is based on the parameter minimal_distance_between_pipeline_connection_points (see :ref:`infrastructure`) and calculated as following:

.. math::
    \text{Number Access Points} = \ceil{2 - 1 + \frac{\text{Length Segment}}{\text{Minimal Distances Between Access Points}}}

This number of access points ensures that the distance between access points is <=minimal_distance_between_pipeline_connection_points. All access points are distributed uniformly along the pipeline segment.

Calculating Distances
=====================

2. How are distances calculated?
3. How are conversion costs attached?
