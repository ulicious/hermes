import copy
from copy import deepcopy
import shapely
from shapely.geometry import LineString, Point
import requests
import json
import searoute as sr
import networkx as nx
import time
import pandas as pd

from _helpers import calc_distance_single_to_single, get_country_and_continent_from_location
from methods_checking import check_total_costs_of_solutions


class Solution:

    # Iterative independent variables
    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_destination(self, destination):
        self.destination = destination

    def get_destination(self):
        return self.destination

    def set_destination_continent(self, destination_continent):
        self.destination_continent = destination_continent

    def get_destination_continent(self):
        return self.destination_continent

    def set_final_commodity(self, target_commodity):
        self.final_commodity = target_commodity

    def get_final_commodity(self):
        return self.final_commodity

    def set_total_costs(self, total_costs):
        self.total_cost = total_costs

    def get_total_costs(self):
        return self.total_cost

    def increase_total_costs(self, new_costs):
        self.total_cost += new_costs

    def set_total_transportation_costs(self, total_transportation_costs):
        self.total_transportation_costs = total_transportation_costs

    def get_total_transportation_costs(self):
        return self.total_transportation_costs

    def increase_total_transportation_costs(self, new_costs):
        self.total_transportation_costs += new_costs

    def set_total_conversion_costs(self, total_conversion_costs):
        self.total_conversion_costs = total_conversion_costs

    def get_total_conversion_costs(self):
        return self.total_conversion_costs

    def increase_total_conversion_costs(self, new_costs):
        self.total_conversion_costs += new_costs

    def set_total_production_costs(self, total_production_costs):
        self.total_production_costs = total_production_costs

    def get_set_total_production_costs(self):
        return self.total_production_costs

    def set_total_length(self, total_length):
        self.total_length = total_length

    def get_total_length(self):
        return self.total_length

    def increase_total_length(self, new_length):
        self.total_length += new_length

    # ---
    """ Iteration dependent variables """
    def prepare_for_new_iteration(self, iteration):
        self.iteration_data['location'][iteration] = self.get_current_location()
        self.iteration_data['continent'][iteration] = self.get_current_continent()
        self.iteration_data['commodity'][iteration] = self.get_current_commodity_object()
        self.iteration_data['used_node'][iteration] = None
        self.iteration_data['used_infrastructure'][iteration] = []
        self.iteration_data['used_transport_mean'][iteration] = None
        self.iteration_data['total_costs'][iteration] = 0
        self.iteration_data['transportation_costs'][iteration] = 0
        self.iteration_data['conversion_costs'][iteration] = 0
        self.iteration_data['length'][iteration] = 0
        self.iteration_data['solution'][iteration] = None
        self.iteration_data['solution_name'][iteration] = None

    def set_iteration_data(self, iteration_data):
        self.iteration_data = iteration_data

    def get_iteration_data(self):
        return self.iteration_data

    def get_iteration_data_specific_iteration(self, iteration):
        # todo: adjust as iteration is not key of iteration data
        return self.iteration_data[iteration]
    
    def set_iterations(self, iterations):
        self.iterations = iterations

    def get_iterations(self):
        return self.iterations

    def add_iteration(self, iteration):
        self.iterations.append(iteration)

    # Geographic data
    def set_locations(self, locations):
        self.iteration_data['location'] = locations

    def get_locations(self):
        return self.iteration_data['location']

    def get_location_specific_iteration(self, iteration):
        return self.iteration_data['location'][iteration]

    def get_current_location(self):
        last_key = [*self.iteration_data['location'].keys()][-1]
        return self.iteration_data['location'][last_key]

    def add_location(self, iteration, location):
        self.iteration_data['location'][iteration] = location

    def set_continents(self, continents):
        self.iteration_data['continent'] = continents

    def get_continents(self):
        return self.iteration_data['continent']

    def get_continent_specific_iteration(self, iteration):
        return self.iteration_data['continent'][iteration]

    def get_current_continent(self):
        last_key = [*self.iteration_data['continent'].keys()][-1]
        return self.iteration_data['continent'][last_key]

    def add_continent(self, iteration, continent):
        self.iteration_data['continent'][iteration] = continent

    # Commodity data
    def set_commodities(self, commodities):
        self.iteration_data['commodity'] = commodities

    def get_commodity_object_specific_iteration(self, iteration):
        return self.iteration_data['commodity'][iteration]

    def get_commodity_objects(self):
        return self.iteration_data['commodity']

    def get_current_commodity_object(self):
        last_key = [*self.iteration_data['commodity'].keys()][-1]
        return self.iteration_data['commodity'][last_key]

    def add_commodity(self, iteration, commodity):
        self.iteration_data['commodity'][iteration] = commodity
        self.iteration_data['commodity_name'][iteration] = commodity.get_name()

    def get_commodity_names(self):
        return self.iteration_data['commodity_name']

    def get_commodity_name_specific_iteration(self, iteration):
        return self.iteration_data['commodity_name'][iteration]

    def get_current_commodity_name(self):
        last_key = [*self.iteration_data['commodity_name'].keys()][-1]
        return self.iteration_data['commodity_name'][last_key]

    # Infrastructure data
    def set_used_nodes(self, used_nodes):
        self.iteration_data['used_node'] = used_nodes

    def get_used_node_specific_iteration(self, iteration):
        return self.iteration_data['used_node'][iteration]

    def get_used_nodes(self):
        return self.iteration_data['used_node']

    def add_used_node(self, iteration, used_node):
        self.iteration_data['used_node'][iteration] = used_node

    def set_used_infrastructure(self, used_infrastructure):
        self.iteration_data['used_infrastructure'] = used_infrastructure

    def get_used_infrastructure_specific_iteration(self, iteration):
        return self.iteration_data['used_infrastructure'][iteration]

    def get_used_infrastructure(self):
        return self.iteration_data['used_infrastructure']

    def add_used_infrastructure(self, iteration, infrastructure):
        self.iteration_data['used_infrastructure'][iteration] = infrastructure

    def set_used_transport_means(self, used_transport_means):
        self.iteration_data['used_transport_mean'] = used_transport_means

    def get_used_transport_means(self):
        return self.iteration_data['used_transport_mean']

    def get_used_transport_means_specific_iteration(self, iteration):
        return self.iteration_data['used_transport_mean'][iteration]

    def add_used_transport_mean(self, iteration, used_transport_mean):
        self.iteration_data['used_transport_mean'][iteration] = used_transport_mean

    # Cost data
    def set_total_costs_specific_iteration(self, iteration, total_costs):
        self.iteration_data['total_costs'][iteration] = total_costs

    def increase_total_costs_specific_iteration(self, iteration, total_costs):
        self.iteration_data['total_costs'][iteration] += total_costs

    def get_total_costs_specific_iteration(self, iteration):
        return self.iteration_data['total_costs'][iteration]

    def set_transportation_costs_specific_iteration(self, iteration, transportation_costs):
        self.iteration_data['transportation_costs'][iteration] = transportation_costs

    def get_transportation_costs_specific_iteration(self, iteration):
        return self.iteration_data['transportation_costs'][iteration]

    def increase_transportation_costs_specific_iteration(self, iteration, new_costs):
        self.iteration_data['transportation_costs'][iteration] += new_costs

    def set_conversion_costs_specific_iteration(self, iteration, conversion_costs):
        self.iteration_data['conversion_costs'][iteration] = conversion_costs

    def get_conversion_costs_specific_iteration(self, iteration):
        return self.iteration_data['conversion_costs'][iteration]

    def increase_conversion_costs_specific_iteration(self, iteration, new_costs):
        self.iteration_data['conversion_costs'][iteration] += new_costs

    def set_length_specific_iteration(self, iteration, length):
        self.iteration_data['length'][iteration] = length

    def get_length_specific_iteration(self, iteration):
        return self.iteration_data['length'][iteration]

    def increase_length_specific_iteration(self, iteration, new_length):
        self.iteration_data['length'][iteration] += new_length

    def get_potential_conversion_costs(self, conversion_costs, loss_of_product):
        return (self.get_total_costs() + conversion_costs) / loss_of_product

    # Solution data
    def set_solution(self, solution_path):
        self.iteration_data['solution'] = solution_path

    def get_solution(self):
        return self.iteration_data['solution']

    def get_solution_specific_iteration(self, iteration):
        return self.iteration_data['solution'][iteration]

    def add_solution(self, iteration, solution):
        self.iteration_data['solution'][iteration] = solution

    def set_solution_names(self, solution_names):
        self.iteration_data['solution_name'] = solution_names

    def get_solution_names(self):
        return self.iteration_data['solution_name']

    def get_solution_name_specific_iteration(self, iteration):
        return self.iteration_data['solution_name'][iteration]

    def add_solution_name(self, iteration, solution_name):
        self.iteration_data['solution_name'][iteration] = solution_name

    """ Other methods """
    def check_if_in_destination(self, destination, to_destination_tolerance):

        distance = calc_distance_single_to_single(self.get_current_location().y, self.get_current_location().x,
                                                  destination.y, destination.x)

        if distance <= to_destination_tolerance:
            return True
        else:
            return False

    def __copy__(self):

        # deepcopy mutable objects
        iteration_data = copy.deepcopy(self.iteration_data)
        iterations = copy.deepcopy(self.iterations)
        total_cost = copy.deepcopy(self.total_cost)
        total_transportation_costs = copy.deepcopy(self.total_transportation_costs)
        total_conversion_costs = copy.deepcopy(self.total_conversion_costs)
        total_length = copy.deepcopy(self.total_length)
        final_commodity = copy.deepcopy(self.final_commodity)

        return Solution(name=self.name, destination=self.destination, destination_continent=self.destination_continent,
                        final_commodity=final_commodity,
                        iteration_data=iteration_data, iterations=iterations,
                        total_cost=total_cost, total_transportation_costs=total_transportation_costs,
                        total_conversion_costs=total_conversion_costs,
                        total_production_costs=self.total_production_costs,
                        total_length=total_length)

    def __init__(self, name, destination, destination_continent, final_commodity, iteration_data, iterations,
                 total_cost=0, total_length=0, total_transportation_costs=0,
                 total_conversion_costs=0, total_production_costs=0):

        self.name = name

        if iteration_data is None:
            self.iteration_data = {}
        else:
            self.iteration_data = iteration_data

        self.iterations = iterations

        self.destination = destination
        self.destination_continent = destination_continent
        self.final_commodity = final_commodity

        self.total_cost = total_cost
        self.total_length = total_length
        self.total_transportation_costs = total_transportation_costs
        self.total_conversion_costs = total_conversion_costs
        self.total_production_costs = total_production_costs


def create_new_solution_from_conversion_result(s, scenario_count, new_commodity, iteration):

    s_new = deepcopy(s)
    s_new.set_name('S' + str(scenario_count))

    s_new.prepare_for_new_iteration(iteration)

    s_new.add_solution_name(iteration, s_new.get_name())
    s_new.add_solution(iteration, s_new)

    current_commodity = s.get_current_commodity_object()
    s_new.add_commodity(iteration, new_commodity)

    if current_commodity.get_name() != new_commodity.get_name():
        old_costs = s_new.get_total_costs()
        conversion_costs = current_commodity.get_conversion_costs_specific_commodity(new_commodity.get_name())
        conversion_efficiency = current_commodity.get_conversion_loss_of_educt_specific_commodity(new_commodity.get_name())

        # new costs are old (costs + conversion costs) / conversion_efficiency to consider losses from conversion
        # As we only increase costs, we need to subtract old costs to get difference
        new_costs = (old_costs + conversion_costs) / conversion_efficiency - old_costs

        s_new.increase_total_conversion_costs(new_costs)
        s_new.increase_conversion_costs_specific_iteration(iteration, new_costs)

        s_new.increase_total_costs(new_costs)
        s_new.increase_total_costs_specific_iteration(iteration, new_costs)

    return s_new


def create_new_solution_from_routing_result(s, scenario_count, s_commodity, mean_of_transport,
                                            target_location, distance, used_infrastructure, used_node, iteration):

    s_new = deepcopy(s)

    s_new.set_name('S' + str(scenario_count))
    s_new.add_solution_name(iteration, s_new.get_name())
    s_new.add_solution(iteration, s_new)

    s_new.prepare_for_new_iteration(iteration)
    s_new.add_location(iteration, target_location)

    country, continent = get_country_and_continent_from_location(target_location.x, target_location.y)
    s_new.add_continent(iteration, continent)

    s_new.add_used_transport_mean(iteration, mean_of_transport)

    s_new.add_used_infrastructure(iteration, used_infrastructure)
    s_new.add_used_node(iteration, used_node)

    route_costs = s_commodity.get_transportation_costs_specific_mean_of_transport(mean_of_transport) * distance / 1000
    # route_efficiency = s_commodity.get_transportation_efficiency_specific_mean_of_transport(mean_of_transport) * distance / 1000

    s_new.increase_total_length(distance)
    s_new.increase_length_specific_iteration(iteration, distance)

    s_new.increase_total_transportation_costs(route_costs)
    s_new.increase_transportation_costs_specific_iteration(iteration, route_costs)

    s_new.increase_total_costs(route_costs)
    s_new.increase_total_costs_specific_iteration(iteration, route_costs)

    return s_new


def create_solution_linestring(solution, data, pipeline_gas_geodata, colors):

    def create_road_path(start, destination):
        successful = False
        routes = None
        while not successful:

            try:
                r = requests.get(
                    f"http://router.project-osrm.org/route/v1/car/{start.x},{start.y};"
                    f"{destination.x},{destination.y}?steps=true&geometries=geojson""")
                routes = json.loads(r.content)
                successful = True

            except Exception:
                time.sleep(1)

        if routes['code'] != 'NoRoute':
            route = routes.get("routes")[0]

            point_list = []
            for point in route['geometry']['coordinates']:
                point_list.append(Point(point))

            return LineString(point_list)

    def create_network_path(start, destination):

        print(start)
        print(destination)

        start = pipeline_gas_geodata[(pipeline_gas_geodata['longitude'] == start.x) &
                                     (pipeline_gas_geodata['latitude'] == start.y)].index
        destination = pipeline_gas_geodata[(pipeline_gas_geodata['longitude'] == destination.x) &
                                           (pipeline_gas_geodata['latitude'] == destination.y)].index

        path = nx.shortest_path(graph, start, destination)

        points_list = []
        n_before = None
        for n in path:

            if n_before is not None:
                edge_index = graph_data[(graph_data['node_start'] == n_before)
                                        & (graph_data['node_end'] == n)].index

                if len(edge_index) == 0:
                    edge_index = graph_data[(graph_data['node_start'] == n)
                                            & (graph_data['node_end'] == n_before)].index

                edge = graph_data.loc[edge_index, 'line'].values[0]
                if not isinstance(edge, LineString):
                    edge = shapely.wkt.loads(edge)

                for i_x, x in enumerate(edge.coords.xy[0]):
                    x = round(x, 5)
                    y = round(edge.coords.xy[1][i_x], 5)

                    points_list.append((x, y))

            n_before = n

        return LineString(points_list)

    def create_shipping_path(start, destination):

        route = sr.searoute((start.x, start.y), (destination.x, destination.y))
        coordinates = []

        for coordinate in route.geometry['coordinates']:
            coordinates.append((coordinate[0], coordinate[1]))

        return LineString(coordinates)

    if solution.get_result_line() is not None:
        return solution

    commodities = solution.get_commodities()
    locations = solution.get_locations()
    means_of_transport = solution.get_used_transport_means()

    result_line = {}

    length = max(max(len(commodities.keys()), len(locations.keys())), len(means_of_transport.keys()))
    if length > 1:

        i_location_before = None
        for i in range(length):

            i_commodity = None
            if i in [*commodities.keys()]:
                i_commodity = commodities[i]

            i_location = None
            if i in [*locations.keys()]:
                i_location = locations[i]

            i_mean_of_transport = None
            if i in [*means_of_transport.keys()]:
                i_mean_of_transport = means_of_transport[i]

            if i_mean_of_transport is not None:
                if i_mean_of_transport == 'Road':
                    line = create_road_path(i_location_before, i_location)
                elif i_mean_of_transport in ['Railroad', 'Pipeline_Gas', 'Pipeline_Liquid']:
                    graph_id = pipeline_gas_geodata[(pipeline_gas_geodata['longitude'] == i_location.x) & \
                                                    (pipeline_gas_geodata['longitude'] == i_location.x)]['graph'].tolist()[0]

                    graph = data[i_mean_of_transport][graph_id]['Graph']
                    graph_data = data[i_mean_of_transport][graph_id]['GraphData']

                    try:
                        line = create_network_path(i_location_before, i_location)
                    except:
                        print(i)
                        print(commodities)
                        print(locations)
                        print(means_of_transport)
                else:
                    line = create_shipping_path(i_location_before, i_location)

                result_line[i] = {'line': line,
                                  'color': colors[i_mean_of_transport]}

            if i in [*locations.keys()]:
                i_location_before = locations[i]

        solution.set_result_line(result_line)

        return solution

    else:
        return None


def process_new_solution(s_new, new_solutions, final_solution,
                         benchmark, local_benchmarks, solutions_to_remove,
                         final_destination, final_commodity,
                         current_commodity, used_node,
                         configuration, solutions_reaching_end):

    # add solution to all successful solutions -> without checking benchmark to increase number of final solutions
    if s_new.check_if_in_destination(final_destination,
                                     configuration['to_final_destination_tolerance']) \
            & (current_commodity.get_name() in final_commodity):
        solutions_reaching_end.append(s_new)

    # Don't add solutions which have already higher costs than benchmark
    if s_new.get_total_costs() <= benchmark:
        # Check if solutions has arrived in destination and has right target commodity
        # If so, update benchmark and remove solutions
        
        if s_new.check_if_in_destination(final_destination,
                                         configuration['to_final_destination_tolerance']) \
                & (current_commodity.get_name() in final_commodity):

            benchmark = s_new.get_total_costs()
            final_solution = s_new

            # throw out solutions which have been added before but with the new benchmark are too expensive
            new_solutions = check_total_costs_of_solutions(new_solutions, benchmark)
        else:
            # if new solutions has not arrived at destination but has lower total costs than the benchmark,
            # it will be further processed. But first check if solution is more expensive than others based on local
            # benchmark
            total_costs = s_new.get_total_costs()
            location = s_new.get_current_location()

            if (location, current_commodity.get_name()) not in local_benchmarks.keys():
                # add solution to local benchmarks
                local_benchmarks[location, current_commodity.get_name()] = {'total_costs': total_costs,
                                                                            'solution': s_new.get_name()}
                new_solutions.append(s_new)
            else:
                if local_benchmarks[location, current_commodity.get_name()]['total_costs'] > total_costs:
                    # add old solution to solutions to remove list

                    if local_benchmarks[location, current_commodity.get_name()]['solution'] is not None:
                        solutions_to_remove.append(local_benchmarks[location, current_commodity.get_name()]['solution'])

                    # overwrite old solution with new one
                    local_benchmarks[location, current_commodity.get_name()] = {'total_costs': total_costs,
                                                                                'solution': s_new.get_name()}
                    new_solutions.append(s_new)

                elif local_benchmarks[location, current_commodity.get_name()]['total_costs'] == total_costs:
                    # as we also update the local benchmark outside of this current method, we might have a value at the
                    # local benchmark, but not associated with a solution. Therefore, add this solution now
                    if local_benchmarks[location, current_commodity.get_name()]['solution'] is None:
                        local_benchmarks[location, current_commodity.get_name()]['solution'] = s_new.get_name()

                        new_solutions.append(s_new)

    return final_solution, new_solutions, local_benchmarks, solutions_to_remove, benchmark, solutions_reaching_end


def create_solution_dataframe(solution):

    iteration_data = solution.get_iteration_data()
    iterations = [*iteration_data['commodity'].keys()]

    columns = ['location', 'commodity', 'used_transport_mean', 'length', 'used_node', 'total_costs']

    solution_df = pd.DataFrame(index=[solution.get_name()], columns=columns)

    for iteration in iterations:

        location = iteration_data['location'][iteration]
        commodity = iteration_data['commodity'][iteration]
        mean_of_transport = iteration_data['used_transport_mean'][iteration]
        used_node = iteration_data['used_node'][iteration]
        total_costs = iteration_data['total_costs'][iteration]
        length = iteration_data['length'][iteration]

        solution_df.loc[iteration, columns] = [location, commodity.get_name(), mean_of_transport, length, used_node,
                                               total_costs]

    solution_df.loc['final', 'total_costs'] = solution.get_total_costs()

    return solution_df
