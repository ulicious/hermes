..
  SPDX-FileCopyrightText: 2024 - Uwe Langenmayr

  SPDX-License-Identifier: CC-BY-4.0

.. _parameter_explanation_conversion:

####################
Conversion Parameter
####################

All parameter assumptions and settings are set in data/techno_economic_data_conversion.yaml. The first set of assumptions affects feedstock costs and capital costs which affect all conversions :underline:`if feedstock costs and/or capital costs are implemented as uniform` (adjustable here: :ref:`conversion_settings`)

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/conversion_cost_parameters.csv
   :width: 100
   :widths: 30, 10, 60
   :delim: ;

Next to feedstock and capital costs, conversion specific parameters are implemented. For each commodity, the target commodities, which the initial commodity can be converted into, must be specified. Furthermore, for each target commodity, the techno-economic parameters must be specified. The structure for each initial commodity looks as following:

.. code-block:: none

    initial commodity:
        target commodities: list
        target commodity 1:
            techno economic parameters target commodity 1 (see below)
        target commodity 2:
            techno economic parameters target commodity 2 (see below)

And following parameters are necessary.

.. _tea_parameters_conversion:

.. csv-table::
   :header-rows: 1
   :file: parameter_explanation/target_commodity.csv
   :width: 100
   :widths: 20, 20, 60
   :delim: ;
