################
Project overview
################

HERMES separates the source-code repository from a user-selected project
folder. The repository contains executable code, templates, and documentation.
The project folder contains editable configurations, copied raw inputs,
processed data, and results. Keeping these two locations separate prevents a
model run from writing generated files into the source tree.

Repository files used by operators
##################################

.. code-block:: text

   hermes/
   |-- _run_workflow.py    select and start workflow stages
   |-- requirements.txt    Python dependencies
   |-- data/               templates and bundled inputs copied by setup
   `-- doc/                user documentation sources

Most users only edit ``_run_workflow.py`` in the repository. Configuration and
input changes should be made in the generated project folder. Direct access to
individual Python modules is described in :doc:`advanced_usage`.

Data flow
#########

.. code-block:: text

   repository data/ templates
              |
              |  0 setup
              v
   PROJECT_FOLDER/raw_data + editable YAML files
              |
              |  1 raw-data processing
              v
   PROJECT_FOLDER/processed_data
              |
              |  2 start-location creation
              v
   PROJECT_FOLDER/start_destination_combinations.csv
              |
              |  3 routing algorithm
              v
   PROJECT_FOLDER/results/location_results
              |
              |  organize result set under results/unprocessed_results/<name>
              v
   PROJECT_FOLDER/results/unprocessed_results/<name>
              |
              |  5 result processing
              v
   PROJECT_FOLDER/results/processed_results
              |
              |  6 plotting
              v
   PROJECT_FOLDER/results/plots

Generated project structure
###########################

The project folder accumulates the following structure across the workflow.
Setup creates the base directories; stage-specific files and folders appear
only after their corresponding workflow stage runs.

.. code-block:: text

   PROJECT_FOLDER/
   |-- 1_algorithm_configuration.yaml
   |-- 2_techno_economic_data_transportation.yaml
   |-- 3_techno_economic_data_conversion.yaml
   |-- 4_plotting_configuration.yaml
   |-- algorithm_configurations/
   |-- raw_data/
   |   `-- natural_earth/                 downloaded and extracted on demand
   |-- processed_data/
   |   |-- inner_infrastructure_distances/
   |   `-- mip_data/
   |-- start_destination_combinations.csv
   `-- results/
       |-- location_results/
       |-- algorithm_tracking/
       |-- export_infrastructure_branches/
       |-- unprocessed_results/
       |-- processed_results/
       `-- plots/

Alternative algorithm configurations write to
``results/<configuration filename>/``. The filename, including ``.yaml`` or
``.yml``, is intentionally used as the scenario directory name.

What to rerun after a change
############################

* Changes to raw infrastructure or preprocessing settings normally require
  steps 1 and 2 before rerunning the algorithm.
* Changes limited to infrastructure conversion assumptions can use
  ``infrastructure_update_only_conversion_costs_and_efficiency``.
* Changes limited to start-location conversion assumptions can use
  ``start_locations_update_only_conversion_costs_and_efficiency``.
* Changes only to routing settings normally require step 3, but existing
  per-location result files are treated as completed work. Use a new scenario
  configuration or deliberately move old results if every location must run
  again.
* Plot-style changes normally require only step 6. Changes to the selected
  scenarios or processed plot metrics may require steps 5 and 6.

See :doc:`workflow` for the runner switches corresponding to these stages.
