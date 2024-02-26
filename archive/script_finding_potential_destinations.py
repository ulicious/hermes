import pandas as pd
import time
from shapely.geometry import Point

from methods_potential_destinations import _get_all_options, _process_options, _assess_options_outside_tolerance, \
    _assess_options_in_tolerance, calculate_distance_known_infrastructure, compare_to_local_benchmarks


def find_potential_destinations(data, configuration, solution, benchmark, local_benchmarks, new_node_count, iteration,
                                graph_data_local, solutions_to_remove):

    """
    Finding the potential destinations uses a lot of code. Therefore, this method is only the script method for better
    readability of the process

    :param data:
    :param configuration:
    :param solution:
    :param benchmark:
    :param local_benchmarks:
    :param new_node_count:
    :return:
    """

    now = time.time()
    now_total = time.time()

    all_options, new_node_count, data \
        = _get_all_options(data, configuration, solution, benchmark, new_node_count, iteration)

    print(len(all_options.index))

    # print('all options time: ' + str(time.time() - now))

    if all_options.index.tolist():  # only process further if there are options to process
        now = time.time()
        options_in_tolerance, options_outside_tolerance, local_infrastructure, local_benchmarks, solutions_to_remove \
            = _process_options(data, configuration, solution, all_options, local_benchmarks, solutions_to_remove)

        print(len(options_outside_tolerance.index))

        # print('split options: ' + str(time.time() - now))

        if len(options_in_tolerance.index) > 0:
            now = time.time()
            potential_destinations_in_tolerance, benchmark = _assess_options_in_tolerance(data, configuration, solution,
                                                                                          options_in_tolerance,
                                                                                          benchmark,
                                                                                          local_benchmarks,
                                                                                          graph_data_local)

            # print('assess inside options: ' + str(time.time() - now))
        else:
            potential_destinations_in_tolerance = pd.DataFrame()

        if len(options_outside_tolerance.index) > 0:
            now = time.time()
            options_outside_tolerance = _assess_options_outside_tolerance(data, solution, options_outside_tolerance,
                                                                          benchmark, configuration)

            print(len(options_outside_tolerance.index))

            # print('ass outside options : ' + str(time.time() - now))
        else:
            options_outside_tolerance = pd.DataFrame()

        if len(options_outside_tolerance.index) > 0:
            now = time.time()
            potential_destinations_outside_tolerance, road_transport_options_to_calculate \
                = calculate_distance_known_infrastructure(data, configuration, solution, options_outside_tolerance,
                                                          benchmark, local_benchmarks)
            # print('calculate distance time out : ' + str(time.time() - now))
        else:
            potential_destinations_outside_tolerance = pd.DataFrame()
            road_transport_options_to_calculate = pd.DataFrame()

        now = time.time()

        if len(potential_destinations_in_tolerance.index) > 0:
            if len(potential_destinations_outside_tolerance.index) > 0:
                potential_destinations = pd.concat([potential_destinations_in_tolerance,
                                                    potential_destinations_outside_tolerance])
            else:
                potential_destinations = potential_destinations_in_tolerance
        else:
            if len(potential_destinations_outside_tolerance.index) > 0:
                potential_destinations = potential_destinations_outside_tolerance
            else:
                potential_destinations = pd.DataFrame()

        # print('concat time: ' + str(time.time() - now))
        # print('total time: ' + str(time.time() - now_total))

        return potential_destinations, road_transport_options_to_calculate, new_node_count, data, benchmark, \
            local_benchmarks, solutions_to_remove
    else:
        return pd.DataFrame(), pd.DataFrame(), new_node_count, data, benchmark, local_benchmarks, solutions_to_remove



