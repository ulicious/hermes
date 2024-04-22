import numpy as np

from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import nearest_points
from geopy.distance import geodesic as calc_distance


def close_gaps_to_network(network, line):
    network_without_segment = MultiLineString([l for l in network.geoms if l != line])
    line_segments = list(map(LineString, zip(line.coords[:-1], line.coords[1:])))

    first_coord = None
    new_line_1_length = None
    if not line_segments[0].intersects(network_without_segment):
        new_line_coords = nearest_points(line_segments[0], network_without_segment)
        length = calc_distance((new_line_coords[0].y, new_line_coords[0].x),
                               (new_line_coords[1].y, new_line_coords[1].x)).meters

        if not length > 100:
            first_coord = new_line_coords[1]
            new_line_1_length = LineString(new_line_coords).length

    last_coord = None
    new_line_2_length = None
    if not line_segments[-1].intersects(network_without_segment):
        new_line_coords = nearest_points(line_segments[-1], network_without_segment)
        length = calc_distance((new_line_coords[0].y, new_line_coords[0].x),
                               (new_line_coords[1].y, new_line_coords[1].x)).meters

        if not length > 100:
            last_coord = new_line_coords[1]
            new_line_2_length = LineString(new_line_coords).length

    # if start and end would be connected to same coordinate --> just use the one which is closer
    if first_coord is not None:
        if first_coord == last_coord:
            if new_line_1_length < new_line_2_length:
                last_coord = None
            else:
                first_coord = None

    new_line_segments = []
    if first_coord is not None:
        new_line_segments.append(first_coord)

    for segment in line_segments:
        if segment == line_segments[0]:
            new_line_segments.append(Point(segment.coords.xy[0][0],
                                           segment.coords.xy[1][0]))

        new_line_segments.append(Point(segment.coords.xy[0][1],
                                       segment.coords.xy[1][1]))

    if last_coord is not None:
        new_line_segments.append(last_coord)

    return LineString(new_line_segments)


def extend_line_in_one_direction(direction_coordinate, support_coordinate, extension_percentage):

    # Create a LineString from the two coordinates
    line = LineString([direction_coordinate, support_coordinate])

    # Calculate the direction vector of the LineString
    direction_vector = np.array([direction_coordinate.x, direction_coordinate.y]) \
        - np.array([support_coordinate.x, support_coordinate.y])

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm != 0:
        direction_vector /= norm

    # Calculate the extension length based on the keyword (percentage)
    line_length = line.length
    extension_length = line_length * extension_percentage / 100

    # Calculate the new end point
    new_end_point = Point([direction_coordinate.x + direction_vector[0] * extension_length,
                           direction_coordinate.y + direction_vector[1] * extension_length])

    return new_end_point


def extend_line_in_both_directions(coord1, coord2, extension_percentage):

    # Create a LineString from the two coordinates
    line = LineString([coord1, coord2])

    # Calculate the direction vector of the LineString
    direction_vector = np.array([coord1.x, coord1.y]) - np.array([coord2.x, coord2.y])

    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm != 0:
        direction_vector /= norm

    # Calculate the extension length based on the keyword (percentage)
    line_length = line.length
    extension_length = line_length * extension_percentage

    # Calculate the new end points in both directions
    new_end_point1 = Point([coord1.x + direction_vector[0] * extension_length,
                            coord1.y + direction_vector[1] * extension_length])
    new_end_point2 = Point([coord2.x - direction_vector[0] * extension_length,
                            coord2.y - direction_vector[1] * extension_length])

    # Create the extended LineString
    extended_linestring = LineString([new_end_point1, new_end_point2])

    return extended_linestring


def calculate_conversion_costs(specific_investment, depreciation_period, fixed_maintenance,
                               operating_hours, interest_rate,
                               electricity_costs, electricity_demand, co2_costs, co2_demand,
                               nitrogen_costs, nitrogen_demand, heat_demand=0, heat_costs=0):

    annuity_factor \
        = (interest_rate * (1 + interest_rate) ** depreciation_period) / ((1 + interest_rate) ** depreciation_period - 1)

    conversion_costs \
        = specific_investment * (annuity_factor + fixed_maintenance) / operating_hours \
        + electricity_costs * electricity_demand + co2_costs * co2_demand \
        + nitrogen_costs * nitrogen_demand + heat_demand * heat_costs

    return conversion_costs

