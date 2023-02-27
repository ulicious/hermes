from shapely.geometry import LineString, Point, MultiLineString
import pandas as pd
from methods_plotting import plot_line, plot_solution


def check_total_costs_of_solutions(solutions, benchmark):

    new_solutions = []

    for s in solutions:
        if s.get_total_costs() <= benchmark:
            new_solutions.append(s)

    return new_solutions


def sort_solutions_by_distance_to_destination(solutions, destination):

    df_length = pd.Series(index=range(len(solutions)), dtype='float64')

    for i, s in enumerate(solutions):

        line = LineString([s.get_current_location(), destination])
        df_length.loc[i] = line.length

    df_length.sort_values(inplace=True)

    new_solutions = []
    for ind in df_length.index:
        new_solutions.append(solutions[ind])

    return new_solutions


def get_unique_solutions(all_solutions, current_solutions, means_of_transport, commodities):

    """
    Some solutions might double as they use the same path but take two steps instead of one
    (for example, through a network). This method removes solutions which are not unique.
    :param all_solutions:
    :param current_solutions:
    :param means_of_transport:
    :param commodities:
    :return:
    """

    unique_solutions = []
    for m in means_of_transport:
        for c in commodities:
            currently_processed_solutions = []
            for s in current_solutions:
                if (c == s.get_current_commodity()) & (m == s.get_used_transport_means()[-1]):
                    currently_processed_solutions.append(s)

            # todo: Not just same path, but also same commodity used for path

            if currently_processed_solutions:

                # First, check if there are no unique solutions in the current solutions
                path_of_solutions = {}
                all_lines = []
                i = 0
                for s in currently_processed_solutions:
                    line_list = []
                    for lines in s.get_result_lines():
                        try:
                            for line in lines.geoms:
                                line_list.append(line)
                        except Exception:  # is LineString
                            line_list.append(lines)

                    if line_list:
                        merged_line = MultiLineString(line_list)
                        all_lines.append(merged_line)

                    path_of_solutions[i] = s

                    i += 1

                lines_to_remove = []
                removed_lines = []
                for i, lines_i in enumerate(all_lines):
                    for j, lines_j in enumerate(all_lines):
                        if i != j:
                            if (lines_i.equals(lines_j)) | (lines_i == lines_j):
                                if (j not in lines_to_remove) & (lines_j not in removed_lines):
                                    lines_to_remove.append(j)
                                    removed_lines.append(lines_j)

                # Check with solutions which have been used in the previous iteration
                if all_solutions:
                    all_solution_lines = []
                    for s in all_solutions:
                        line_list = []
                        for lines in s.get_result_lines():
                            try:
                                for line in lines.geoms:
                                    line_list.append(line)
                            except Exception:  # is LineString
                                line_list.append(lines)

                        if line_list:
                            merged_line = MultiLineString(line_list)
                            all_solution_lines.append(merged_line)

                    for i, lines_i in enumerate(all_solution_lines):
                        for j, lines_j in enumerate(all_lines):
                            if (lines_i.equals(lines_j)) | (lines_i == lines_j):
                                if (j not in lines_to_remove) & (lines_j not in removed_lines):
                                    lines_to_remove.append(j)
                                    removed_lines.append(lines_j)

                for i in lines_to_remove:
                    path_of_solutions.pop(i)

                for k in [*path_of_solutions.keys()]:
                    unique_solutions.append(path_of_solutions[k])

    return unique_solutions
