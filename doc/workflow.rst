########
Workflow
########

The central ``_run_workflow.py`` runner is the recommended way to operate
HERMES. Set ``PROJECT_FOLDER`` and enable one or more switches. Enabled stages
run once in the order listed below. All switches are disabled by default.

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Runner switch
     - Purpose
     - Main prerequisite
   * - ``RUN_SETUP_PROJECT_FOLDER``
     - Create folders, copy configuration templates, and download versioned inputs.
     - Repository checkout
   * - ``RUN_PROCESS_RAW_DATA``
     - Download supporting geography and process infrastructure and costs.
     - Completed setup; Internet access for the first download
   * - ``RUN_CREATE_START_LOCATIONS``
     - Create origins and attach production and conversion information.
     - Processed geographic and infrastructure data
   * - ``RUN_MAIN_ALGORITHM``
     - Run the heuristic routing algorithm.
     - Processed data and ``start_destination_combinations.csv``
   * - ``RUN_EXPORT_INFRASTRUCTURE``
     - Export infrastructure branches used by the algorithm.
     - Processed infrastructure and a valid algorithm configuration
   * - ``RUN_MIP_OPTIMIZATION``
     - Run optional mixed-integer validation.
     - Preprocessing with ``create_mip_data: true`` and a working solver
   * - ``RUN_PROCESS_PLOT_DATA``
     - Prepare selected result sets for plotting.
     - Per-location files organized under
       ``results/unprocessed_results/<result name>/``
   * - ``RUN_PLOT_RESULTS``
     - Create configured plots and comparisons.
     - Files in ``results/processed_results/``
   * - ``RUN_ANALYZE_ALGORITHM_TRACKING``
     - Summarize optional algorithm-tracking logs.
     - Tracking files in ``results/algorithm_tracking/``

Recommended sequence
####################

Run setup only once unless the project folder intentionally needs to be reset.
After adjusting the copied configuration, a normal first calculation uses:

.. code-block:: python

   RUN_SETUP_PROJECT_FOLDER = False
   RUN_PROCESS_RAW_DATA = True
   RUN_CREATE_START_LOCATIONS = True
   RUN_MAIN_ALGORITHM = True

These stages may run in one invocation. Separate invocations are easier to
diagnose and avoid repeating a successful expensive stage after a later stage
fails. Once raw and processed data exist, disable stages that are not needed.

Inputs and downloads
####################

Setup downloads the following standard inputs from Zenodo record ``22031725``
to ``PROJECT_FOLDER/raw_data/``:

* ``location_data.csv``
* ``country_data.csv``
* ``network_pipelines_gas.xlsx``
* ``network_pipelines_oil.xlsx``
* ``seaports.geojson``
* ``water.zip``
* ``natural_earth.zip``

These are the default filenames used by the supplied algorithm configuration.
The five main input filenames can be changed through ``country_data``,
``location_data``, ``network_pipelines_gas``, ``network_pipelines_oil``, and
``seaports``. The Zenodo record and checksums for the default downloads are
fixed by the code. Setup verifies each download and extracts
``natural_earth.zip`` into ``raw_data/natural_earth/``.
The initial setup therefore requires Internet access; raw-data processing does
not download a newer Natural Earth release independently.

Outputs by stage
################

* Raw-data processing writes reusable infrastructure and geographic data below
  ``processed_data/``.
* Start-location creation writes ``start_destination_combinations.csv``.
* The main algorithm writes one result or status file per location below
  ``results/location_results/``.
* Infrastructure export writes to ``results/export_infrastructure_branches/``.
* Tracking analysis reads optional logs from ``results/algorithm_tracking/``.
* Plot-data processing reads named collections from
  ``results/unprocessed_results/<result name>/`` and writes CSV files to
  ``results/processed_results/``.
* Plotting writes figures and comparison exports to ``results/plots/``.

Optional stages
###############

MIP validation is not required for the heuristic workflow. It requires
``create_mip_data: true`` during preprocessing and a compatible optimization
solver and licence. The Gurobi Python package is installed with the normal
``requirements.txt``; a valid Gurobi licence must be provided separately.

Plotting is also optional. Configure the result sets in
``4_plotting_configuration.yaml`` and organize their per-location final-solution
files under ``results/unprocessed_results/<result name>/``. The current workflow
does not copy algorithm outputs there automatically. Process their data and only
then run the plotting stage.

What to rerun
#############

* Raw infrastructure or preprocessing changes normally require raw-data
  processing and start-location creation before the main algorithm.
* Infrastructure conversion-only changes can use
  ``infrastructure_update_only_conversion_costs_and_efficiency``.
* Start-location conversion-only changes can use
  ``start_locations_update_only_conversion_costs_and_efficiency``.
* Routing-only changes normally require only the main algorithm. Existing
  per-location result files are treated as completed work, so use a new
  scenario or deliberately move old results when all locations must be rerun.
* Plot-style changes normally require only plotting. Changes to selected result
  sets or processed metrics may require plot-data processing first.

See :doc:`advanced_usage` for direct stage calls and multiple scenarios.
