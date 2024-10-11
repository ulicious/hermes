..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _installation:

############
Installation
############

1. Clone Git Project
====================

Use your integrated development environment (IDE) and clone the GitHub repository via terminal:

1. Navigate to the target directory
2. Use following command in your terminal

.. code-block:: none

    git clone https://github.com/ulicious/hermes

Alternatively, if your IDE has a version control integration, you can further clone the git project without a terminal command. Mostly, this is possible when creating new projects. For detailed instructions, please see the documentation of your IDE.

2. Install requirements
=======================

To install all required packages of HERMES, use following commands in the terminal of your IDE.

Using conda
-----------

If you have conda installed, you can use the transport_model environment. First, install the environment:

.. code-block:: none

    conda env create -f doc/environment.yml

and afterwards, choose the Python .exe file in the created folder of the environment.

Using pip
---------

First, choose a python interpreter and afterwards, install all requirements with following command in the IDE terminal.

.. code-block:: none

    pip install -r doc/requirements.txt

3. Setting up folder structure
==============================

Processed data and results need to be stored. Therefore, following folder structure needs to be implemented

.. code-block:: none

    PROJECT FOLDER/
        processed_data/
        raw_data/
        results/
            location_results/
            plots/

Please indicate the path towards the PROJECT_FOLDER in :ref:`general_configuration`.