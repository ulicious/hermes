..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _data:

############
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