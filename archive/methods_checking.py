from shapely.geometry import LineString, Point, MultiLineString
import pandas as pd

from _helpers import calc_distance_single_to_single

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


def check_total_costs_of_solutions(solutions, benchmark):

    new_solutions = []

    for s in solutions:
        if s.get_total_costs() <= benchmark:
            new_solutions.append(s)

    return new_solutions


def sort_solutions_by_distance_to_destination(solutions, destination):

    df_length = pd.Series(index=range(len(solutions)), dtype='float64')

    for i, s in enumerate(solutions):

        # line = LineString([s.get_current_location(), destination])
        # df_length.loc[i] = line.length
        df_length.loc[i] = s.get_total_costs()  # todo adjusted to sort by costs not distance

    df_length.sort_values(inplace=True)

    new_solutions = []
    for ind in df_length.index:
        new_solutions.append(solutions[ind])

    return new_solutions


def remove_solution_duplications(solutions, solution_names_to_remove):

    # remove solutions which are at the same location as other solutions with the same commodity but have higher costs

    solution_dict = {}
    solutions_to_remove = []

    for s in solutions:
        current_location = s.get_current_location()
        current_commodity = s.get_current_commodity_name()
        current_total_costs = s.get_total_costs()

        key = (current_location, current_commodity)

        if key in [*solution_dict.keys()]:

            if solution_dict[key]['costs'] > current_total_costs:
                solutions_to_remove.append(solution_dict[key]['solution'])
                solution_names_to_remove.append(s.get_name())
                solution_dict[key] = {'solution': s,
                                      'costs': current_total_costs}
            else:
                solutions_to_remove.append(s)
                solution_names_to_remove.append(s.get_name())
        else:
            solution_dict[key] = {'solution': s,
                                  'costs': current_total_costs}

    return list(set(solutions) - set(solutions_to_remove)), solution_names_to_remove


def remove_solutions_based_on_past_solutions(configuration, current_solutions, solutions_per_iteration):

    # todo: difference to other method?

    import itertools

    def compare_solutions(combination):
        solution_1 = combination[0]
        solution_1_location = solution_1.get_current_location()

        solution_2 = combination[1]
        solution_2_location = solution_2.get_current_location()

        distance_between_solutions = calc_distance_single_to_single(solution_1_location.y, solution_1_location.x,
                                                                    solution_2_location.y, solution_2_location.x)

        if distance_between_solutions < configuration['tolerance_distance']:
            solution_1_commodity = solution_1.get_commodity_name()
            solution_2_commodity = solution_2.get_commodity_name()

            if solution_1_commodity == solution_2_commodity:
                solution_1_total_costs = solution_1.get_total_costs()
                solution_2_total_costs = solution_2.get_total_costs()

                if solution_1_total_costs > solution_2_total_costs:
                    return solution_1
                else:
                    return None

    # Second, compare current solutions with past solutions
    for key in [*solutions_per_iteration.keys()]:

        other_solutions = solutions_per_iteration[key]

        solution_combination = list(itertools.product(current_solutions, other_solutions))
        num_cores_solution_comparing = min(120, multiprocessing.cpu_count() - 1)
        inputs_solution_comparing = tqdm(solution_combination)
        results_solution_comparing = Parallel(n_jobs=num_cores_solution_comparing)(delayed(compare_solutions)(inp)
                                                                                   for inp in
                                                                                   inputs_solution_comparing)

        for result in results_solution_comparing:
            if result is not None:
                for s in current_solutions:
                    if s.get_name() == result.get_name():
                        current_solutions.remove(s)

    # First, compare current solutions with each other
    solution_combination = list(itertools.product(current_solutions, current_solutions))
    num_cores_solution_comparing = min(120, multiprocessing.cpu_count() - 1)
    inputs_solution_comparing = tqdm(solution_combination)
    results_solution_comparing = Parallel(n_jobs=num_cores_solution_comparing)(delayed(compare_solutions)(inp)
                                                                               for inp in inputs_solution_comparing)

    for result in results_solution_comparing:
        if result is not None:
            for s in current_solutions:
                if s.get_name() == result.get_name():
                    current_solutions.remove(s)

    return current_solutions


def remove_solution_based_on_past_solutions(configuration, current_solution, solutions_per_iteration):

    import itertools

    def compare_solutions(combination):
        solution_1 = combination[0]
        solution_1_location = solution_1.get_current_location()

        solution_2 = combination[1]
        solution_2_location = solution_2.get_current_location()

        distance_between_solutions = calc_distance_single_to_single(solution_1_location.y, solution_1_location.x,
                                                                    solution_2_location.y, solution_2_location.x)

        if distance_between_solutions < configuration['tolerance_distance']:
            solution_1_commodity = solution_1.get_commodity_name()
            solution_2_commodity = solution_2.get_commodity_name()

            if solution_1_commodity == solution_2_commodity:
                solution_1_total_costs = solution_1.get_total_costs()
                solution_2_total_costs = solution_2.get_total_costs()

                if solution_1_total_costs > solution_2_total_costs:
                    return solution_1
                else:
                    return None

    # Second, compare current solutions with past solutions
    for key in [*solutions_per_iteration.keys()]:

        other_solutions = solutions_per_iteration[key]

        solution_combination = list(itertools.product([current_solution], other_solutions))
        num_cores_solution_comparing = min(120, multiprocessing.cpu_count() - 1)
        inputs_solution_comparing = tqdm(solution_combination)
        results_solution_comparing = Parallel(n_jobs=num_cores_solution_comparing)(delayed(compare_solutions)(inp)
                                                                                   for inp in
                                                                                   inputs_solution_comparing)

        keep_solution = True
        for result in results_solution_comparing:
            if result is not None:
                if current_solution.get_name() == result.get_name():
                    keep_solution = False

    if keep_solution:
        return current_solution
    else:
        return None


def remove_solutions_based_on_local_benchmark(solutions, local_benchmark):

    obsolete_solutions = []
    new_solutions = []
    for s in solutions:
        current_commodity = s.get_current_commodity_name()
        current_location = s.get_current_location()
        current_total_costs = s.get_total_costs()

        if current_total_costs <= local_benchmark[(current_location, current_commodity)]['total_costs']:
            new_solutions.append(s)
        else:
            obsolete_solutions.append(s)

    return new_solutions, obsolete_solutions


def check_solutions_after_iterative_processing(solutions, benchmark, local_benchmark):

    obsolete_solutions = []
    solution_dict = {}

    for s in solutions:
        current_location = s.get_current_location()
        current_commodity = s.get_current_commodity_name()
        current_total_costs = s.get_total_costs()

        # first check if costs are higher than benchmark
        if current_total_costs > benchmark:
            obsolete_solutions.append(s.get_name())
            continue

        # key for next to processing steps
        key = (current_location, current_commodity)

        # now check if higher costs than local benchmark
        if False:
            if key in list(local_benchmark.keys()):
                if current_total_costs > local_benchmark[(current_location, current_commodity)]['total_costs']:
                    obsolete_solutions.append(s.get_name())
                    continue
        else:
            if len(local_benchmark.index.tolist()):
                if current_total_costs > local_benchmark.at[(current_location, current_commodity), 'total_costs']:
                    obsolete_solutions.append(s.get_name())
                    continue

        # lastly, check if duplicate with lower costs exists
        if key in list(solution_dict.keys()):

            if solution_dict[key]['costs'] > current_total_costs:
                # is duplicate and lower costs
                obsolete_solutions.append(s.get_name())
                solution_dict[key] = {'solution': s,
                                      'costs': current_total_costs}
            else:
                # is duplicate and higher costs
                obsolete_solutions.append(s.get_name())
                continue
        else:
            # not (yet) duplicate
            solution_dict[key] = {'solution': s,
                                  'costs': current_total_costs}

    new_solutions = []
    for s in list(solution_dict.keys()):
        new_solutions.append(solution_dict[s]['solution'])

    return new_solutions, obsolete_solutions
