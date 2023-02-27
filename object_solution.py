import copy
from copy import deepcopy

from _helpers import calc_distance


class Solution:

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_current_location(self, current_location):
        self.current_location = current_location

    def get_current_location(self):
        return self.current_location

    def get_current_commodity(self):
        """
        :return: Commodity Name
        """
        return self.current_commodity

    def set_current_commodity_object(self, current_commodity_object):
        self.current_commodity_object = current_commodity_object
        self.current_commodity = current_commodity_object.get_name()

    def get_current_commodity_object(self):
        return self.current_commodity_object

    def set_ports(self, ports):
        self.ports = ports

    def get_ports(self):
        return self.ports

    def remove_port(self, port):
        self.ports = self.ports.drop(port)

    def set_pipeline_gas_networks(self, pipeline_gas_networks):
        self.pipeline_gas_networks = pipeline_gas_networks

    def get_pipeline_gas_networks(self):
        return self.pipeline_gas_networks

    def remove_pipeline_gas_network(self, pipeline_gas_network):
        self.pipeline_gas_networks.pop(pipeline_gas_network)

    def set_pipeline_liquid_networks(self, pipeline_liquid_networks):
        self.pipeline_liquid_networks = pipeline_liquid_networks

    def get_pipeline_liquid_networks(self):
        return self.pipeline_liquid_networks

    def remove_pipeline_liquid_network(self, pipeline_liquid_network):
        self.pipeline_liquid_networks.pop(pipeline_liquid_network)

    def set_railroad_networks(self, railroad_networks):
        self.railroad_networks = railroad_networks

    def get_railroad_networks(self):
        return self.railroad_networks

    def remove_railroad_network(self, railroad_network):
        self.railroad_networks.pop(railroad_network)

    def get_destination(self):
        return self.destination

    def set_total_costs(self, total_costs):
        self.total_cost = total_costs

    def get_total_costs(self):
        return self.total_cost

    def increase_total_costs(self, increase):
        self.total_cost += increase

    def set_total_length(self, total_length):
        self.total_length = total_length

    def get_total_length(self):
        return self.total_length

    def increase_total_length(self, increase):
        self.total_length += increase

    def set_used_transport_means(self, used_transport_means):
        self.used_transport_means = used_transport_means

    def get_used_transport_means(self):
        return self.used_transport_means

    def get_last_used_transport_means(self):
        return self.used_transport_means[-1]

    def add_used_transport_mean(self, used_transport_mean):
        self.used_transport_means.append(used_transport_mean)

    def set_result_lines(self, result_lines):
        self.result_lines = result_lines

    def get_result_lines(self):
        return self.result_lines

    def add_result_line(self, line):
        self.result_lines.append(line)

    def add_previous_solution(self, solution):
        self.previous_solutions.append(solution)

    def set_final_commodity(self, target_commodity):
        self.final_commodity = target_commodity

    def get_final_commodity(self):
        return self.final_commodity

    def check_if_in_destination(self, destination, to_destination_tolerance):

        distance = calc_distance(self.current_location.y, self.current_location.x,
                                 destination.y, destination.x)

        if distance <= to_destination_tolerance:
            return True
        else:
            return False

    def __copy__(self):

        # deepcopy mutable objects
        used_transport_means = copy.deepcopy(self.used_transport_means)
        result_lines = copy.deepcopy(self.result_lines)
        previous_solutions = copy.deepcopy(self.previous_solutions)
        ports = copy.deepcopy(self.ports)
        pipeline_gas_networks = copy.deepcopy(self.pipeline_gas_networks)
        pipeline_liquid_networks = copy.deepcopy(self.pipeline_liquid_networks)
        railroad_networks = copy.deepcopy(self.railroad_networks)

        return Solution(name=self.name,
                        current_location=self.current_location, current_commodity=self.current_commodity,
                        current_commodity_object=self.current_commodity_object,
                        ports=ports,
                        pipeline_gas_networks=pipeline_gas_networks, pipeline_liquid_networks=pipeline_liquid_networks,
                        railroad_networks=railroad_networks,
                        destination=self.destination, final_commodity=self.final_commodity,
                        total_cost=self.total_cost,
                        total_length=self.total_length,
                        used_transport_means=used_transport_means, result_lines=result_lines,
                        previous_solutions=previous_solutions)

    def __init__(self, name,
                 current_location, current_commodity, current_commodity_object,
                 ports,
                 pipeline_gas_networks, pipeline_liquid_networks,
                 railroad_networks,
                 destination, final_commodity,
                 total_cost=0, total_length=0,
                 used_transport_means=None, result_lines=None, previous_solutions=None):

        self.name = name
        self.current_location = current_location
        self.current_commodity = current_commodity
        self.current_commodity_object = current_commodity_object

        self.ports = ports
        self.pipeline_gas_networks = pipeline_gas_networks
        self.pipeline_liquid_networks = pipeline_liquid_networks
        self.railroad_networks = railroad_networks

        self.destination = destination
        self.final_commodity = final_commodity

        self.total_cost = total_cost
        self.total_length = total_length

        if used_transport_means is None:
            self.used_transport_means = []
        else:
            self.used_transport_means = used_transport_means

        if result_lines is None:
            self.result_lines = []
        else:
            self.result_lines = result_lines

        if result_lines is None:
            self.previous_solutions = []
        else:
            self.previous_solutions = previous_solutions


def create_new_solution_from_conversion_result(s, new_commodity, number):

    s_new = deepcopy(s)
    s_new.set_name(s_new.get_name() + '_' + str(number))
    s_new.add_previous_solution(s)

    s_new.set_current_commodity_object(new_commodity)
    s_new.increase_total_costs(new_commodity.get_conversion_costs(new_commodity))

    return s_new


def create_new_solution_from_routing_result(s, s_commodity, mean_of_transport,
                                            target_location, distance, line, number):

    s_new = deepcopy(s)
    s_new.set_name(s_new.get_name() + '_' + str(number))

    s_new.set_current_location(target_location)

    s_new.add_used_transport_mean(mean_of_transport)
    s_new.add_result_line(line)
    s_new.add_previous_solution(s)

    route_costs = (s_commodity.get_transportation_costs_specific_mean_of_transport(mean_of_transport) / 1000) * distance
    s_new.increase_total_costs(route_costs)

    return s_new


