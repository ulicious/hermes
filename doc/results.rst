#######
Results
#######

HERMES separates raw per-location algorithm output from tables prepared for
analysis and plotting.

Per-location results
####################

The default algorithm configuration writes to:

.. code-block:: text

   PROJECT_FOLDER/results/location_results/

Alternative scenario configurations write to
``results/<configuration filename>/location_results/``.

The filename starts with the integer index of the start location. A successful
route is stored as ``<location>_final_solution.csv``. This file contains the
selected route state, total and component costs, commodity and transport
history, efficiency, destination information, solving time, and a snapshot of
the corresponding start-location data.

Status files such as ``<location>_no_benchmark.csv`` or
``<location>_no_potential.csv`` indicate that no final route was written under
the active inputs and configuration. They should be retained when a run is
resumed because their location numbers are also treated as processed work.

Processed result tables
#######################

The result-processing stage reads each name listed under ``process_results``
from:

.. code-block:: text

   PROJECT_FOLDER/results/unprocessed_results/<result name>/

The current workflow does not automatically copy algorithm output into these
named collections. Copy the relevant ``*_final_solution.csv`` files from the
default or scenario-specific ``location_results/`` directory into the desired
collection before running result processing.

The stage combines those per-location solution files into:

.. code-block:: text

   PROJECT_FOLDER/results/processed_results/
       <result name>_processed_results.csv
       <result name>_destination.csv

The combined result table contains fields used by the plotting workflow,
including total, production, conversion, and transportation costs; start
commodity; coordinates; efficiency; quantity; route and commodity sequences;
solving time; and available country, continent, and geometry metadata.

The result sets to process are selected through ``process_results`` in
``4_plotting_configuration.yaml``. Plot options refer to the same result names.

Plots and comparison exports
############################

Generated figures and comparison workbooks are written to:

.. code-block:: text

   PROJECT_FOLDER/results/plots/

Only plot types enabled in ``4_plotting_configuration.yaml`` are created.
Route-based plots can additionally require processed infrastructure data.

Tracking output
###############

When algorithm tracking is enabled, JSONL logs are stored in
``results/algorithm_tracking/`` or the corresponding scenario directory. The
tracking-analysis stage produces summaries for diagnostics; these files are not
required to use the final routing results.

Units and interpretation
########################

Interpret cost, distance, quantity, and efficiency fields using the units in
the parameter reference and YAML configuration. Scenario comparisons are valid
only when their input data, destination definition, units, and result-processing
settings are compatible.
