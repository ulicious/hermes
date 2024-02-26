import time
import pandas as pd

from methods_potential_destinations_test import _process_options, _assess_options_in_tolerance, assess_options


def process_combined_solutions(all_options, solutions, data, configuration, benchmark, local_benchmarks, solutions_to_remove):

    # first, remove duplicate options --> only keep cheapest

    if all_options.index.tolist():  # only process further if there are options to process
        now = time.time()
        options_in_tolerance, options_outside_tolerance = _process_options(data, configuration, all_options, benchmark)

        time_split = time.time() - now

        if len(options_in_tolerance.index) > 0:
            now = time.time()
            options_in_tolerance = _assess_options_in_tolerance(data, options_in_tolerance, solutions, local_benchmarks)

            time_inside = time.time() - now
        else:
            options_in_tolerance = pd.DataFrame()
            time_inside = 0

        # todo: if we use OSRM data, we need to adjust the options outside tolerance again

        options = pd.concat([options_in_tolerance, options_outside_tolerance])

        now = time.time()
        if not options.empty:
            options = assess_options(options, data, configuration, benchmark, local_benchmarks)
        else:
            return pd.DataFrame()
        print('time split: ' + str(time_split) + ' | time inside: ' + str(time_inside) + ' | assess all options: ' + str(time.time() - now))

        return options
    else:
        return pd.DataFrame()
