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

    def set_used_ports(self, used_ports):
        self.used_ports = used_ports

    def get_used_ports(self):
        return self.used_ports

    def add_used_port(self, port):
        self.used_ports = self.used_ports.append(port)

    def set_used_pipeline_gas_networks(self, used_pipeline_gas_networks):
        self.used_pipeline_gas_networks = used_pipeline_gas_networks

    def get_used_pipeline_gas_networks(self):
        return self.used_pipeline_gas_networks

    def add_used_pipeline_gas_network(self, used_pipeline_gas_network):
        self.used_pipeline_gas_networks.append(used_pipeline_gas_network)

    def set_used_pipeline_liquid_networks(self, used_pipeline_liquid_networks):
        self.used_pipeline_liquid_networks = used_pipeline_liquid_networks

    def get_used_pipeline_liquid_networks(self):
        return self.used_pipeline_liquid_networks

    def add_used_pipeline_liquid_network(self, used_pipeline_liquid_network):
        self.used_pipeline_liquid_networks.append(used_pipeline_liquid_network)

    def set_used_railroad_networks(self, used_railroad_networks):
        self.used_railroad_networks = used_railroad_networks

    def get_used_railroad_networks(self):
        return self.used_railroad_networks

    def add_used_railroad_network(self, railroad_network):
        self.used_railroad_networks.append(railroad_network)

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

        used_ports = deepcopy(self.used_ports)
        used_pipeline_gas_networks = deepcopy(self.used_pipeline_gas_networks)
        used_pipeline_liquid_networks = deepcopy(self.used_pipeline_liquid_networks)
        used_railroad_networks = deepcopy(self.used_railroad_networks)

        return Solution(name=self.name,
                        current_location=self.current_location, current_commodity=self.current_commodity,
                        current_commodity_object=self.current_commodity_object,
                        destination=self.destination, final_commodity=self.final_commodity,
                        used_ports=used_ports, used_pipeline_gas_networks=used_pipeline_gas_networks,
                        used_pipeline_liquid_networks=used_pipeline_liquid_networks,
                        used_railroad_networks=used_railroad_networks,
                        total_cost=self.total_cost,
                        total_length=self.total_length,
                        used_transport_means=used_transport_means, result_lines=result_lines,
                        previous_solutions=previous_solutions)

    def __init__(self, name,
                 current_location, current_commodity, current_commodity_object,
                 destination, final_commodity,
                 used_ports=None, used_pipeline_gas_networks=None,
                 used_pipeline_liquid_networks=None, used_railroad_networks=None,
                 total_cost=0, total_length=0,
                 used_transport_means=None, result_lines=None, previous_solutions=None):

        self.name = name
        self.current_location = current_location
        self.current_commodity = current_commodity
        self.current_commodity_object = current_commodity_object

        if used_ports is None:
            self.used_ports = []
        else:
            self.used_ports = used_ports

        if used_pipeline_gas_networks is None:
            self.used_pipeline_gas_networks = []
        else:
            self.used_pipeline_gas_networks = used_pipeline_gas_networks

        if used_pipeline_liquid_networks is None:
            self.used_pipeline_liquid_networks = []
        else:
            self.used_pipeline_liquid_networks = used_pipeline_liquid_networks

        if used_railroad_networks is None:
            self.used_railroad_networks = []
        else:
            self.used_railroad_networks = used_railroad_networks

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


