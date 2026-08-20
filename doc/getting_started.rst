.. SPDX-License-Identifier: CC-BY-4.0

.. _getting_started:

###############
Getting started
###############

HERMES currently supports Python 3.11. It uses two separate locations: the repository contains the Python code
and input templates, while a user-selected project folder contains editable
configuration, copied inputs, processed data, and results.

Commands on this page must be run from the repository root. Quote paths that
contain spaces.

.. _installation:

Install HERMES
##############

Linux and macOS (Bash)
======================

.. code-block:: bash

   git clone https://github.com/ulicious/hermes.git
   cd hermes
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

Windows (PowerShell)
====================

.. code-block:: powershell

   git clone https://github.com/ulicious/hermes.git
   Set-Location hermes
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

If PowerShell blocks activation according to the local execution policy, call
the environment interpreter directly, for example
``.\.venv\Scripts\python.exe _run_workflow.py``.

Installation normally takes approximately 10–20 minutes on a desktop computer
with a broadband connection. Network speed and binary-wheel availability for
the geographic packages can change the installation time.

Prepare a project folder
########################

Open ``_run_workflow.py`` and set ``PROJECT_FOLDER`` to a separate working
directory. An absolute path is recommended:

.. code-block:: python

   # Linux/macOS
   PROJECT_FOLDER = "/home/user/hermes_project"

   # Windows
   PROJECT_FOLDER = r"C:\Users\user\Documents\hermes_project"

For the first run, enable only:

.. code-block:: python

   RUN_SETUP_PROJECT_FOLDER = True

Then run:

.. code-block:: bash

   python _run_workflow.py

Setup creates the project structure, copies bundled inputs to ``raw_data/``,
copies four editable YAML files to the project-folder root, and stores the
absolute project path in ``1_algorithm_configuration.yaml``.

.. warning::

   Running setup again overwrites the copied YAML files and bundled raw inputs.
   Disable ``RUN_SETUP_PROJECT_FOLDER`` after initial setup.

Configure the first run
#######################

Edit the files inside the project folder:

* ``1_algorithm_configuration.yaml`` controls geography, start locations,
  routing, computational settings, and optional preprocessing behavior.
* ``2_techno_economic_data_transportation.yaml`` contains transport costs and
  losses.
* ``3_techno_economic_data_conversion.yaml`` contains commodity conversion
  assumptions.
* ``4_plotting_configuration.yaml`` selects and styles result plots.

For the demo, set ``use_minimal_example: true``. This selects the frame
35–71° N and 21° W–45° E. Start locations are additionally restricted to land
classified as Europe by Natural Earth. The setting does not overwrite other
values such as ``number_locations``. See :doc:`parameters` for the complete
parameter reference.

The measured runtime for raw-data and infrastructure processing of the minimal
example is approximately 15 minutes (14.98 minutes in the reference run).
Start-location creation takes approximately 2 minutes (1.71 minutes in the
reference run). The routing algorithm takes approximately 8 minutes
(8.38 minutes in the reference run). Optional plot-data processing takes
approximately 23 seconds (22.66 seconds in the reference run). The optional
plot-generation step takes approximately 31 seconds (30.83 seconds in the
reference run). The complete measured workflow including plot generation
therefore takes approximately 26 minutes (25.96 minutes in total).

Run HERMES
##########

Disable setup, select the required stages in ``_run_workflow.py``, and start the
runner again:

.. code-block:: bash

   python _run_workflow.py

For a first complete calculation, the normal order is:

1. process raw data;
2. create start locations;
3. run the main algorithm;
4. optionally process result data and create plots.

Stages may be enabled together, but running them separately the first time
makes failures easier to locate. The runner stops when an enabled stage fails.
The exact switches, prerequisites, and outputs are documented in
:doc:`workflow`.

Find the results
################

The most important folders are:

.. code-block:: text

   PROJECT_FOLDER/results/
   |-- location_results/       per-location routing results and status files
   |-- algorithm_tracking/     optional diagnostic tracking logs
   |-- processed_results/      tables prepared for plotting
   `-- plots/                  generated figures

Alternative scenarios have their own directory below ``results/``. See
:doc:`results` for the contents and purpose of the generated files. See
:doc:`advanced_usage` for scenario configurations, batch execution, direct
module calls, and environment-variable usage. See :doc:`project_overview` for
the complete project structure and guidance on which stages to rerun after a
configuration change.
