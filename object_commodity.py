import numpy as np


class Commodity:

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_production_costs(self, production_costs):
        self.production_costs = production_costs

    def get_production_costs(self):
        return self.production_costs

    def set_conversion_options(self, conversion_options):
        self.conversion_options = conversion_options

    def get_conversion_options(self):
        return self.conversion_options

    def set_conversion_options_specific_commodity(self, commodity_name, conversion_options_specific_commodity):
        self.conversion_options[commodity_name] = conversion_options_specific_commodity

    def get_conversion_options_specific_commodity(self, commodity_name):
        return self.conversion_options[commodity_name]

    def set_conversion_costs(self, conversion_costs):
        self.conversion_costs = conversion_costs

    def get_conversion_costs(self):
        return self.conversion_costs

    def set_conversion_costs_specific_commodity(self, commodity_name, conversion_costs_specific_commodity):
        self.conversion_costs[commodity_name] = conversion_costs_specific_commodity

    def get_conversion_costs_specific_commodity(self, commodity_name):
        return self.conversion_costs[commodity_name]

    def set_transportation_costs(self, transportation_costs):
        self.transportation_costs = transportation_costs

    def get_transportation_costs(self):
        return self.transportation_costs

    def set_transportation_costs_specific_mean_of_transport(self, mean_of_transport,
                                                            transportation_costs_specific_mean_of_transport):
        self.transportation_costs[mean_of_transport] = transportation_costs_specific_mean_of_transport

    def get_transportation_costs_specific_mean_of_transport(self, mean_of_transport):
        return self.transportation_costs[mean_of_transport]

    def set_transportation_options(self, transportation_options):
        self.transportation_options = transportation_options

    def get_transportation_options(self):
        return self.transportation_options

    def set_transportation_options_specific_mean_of_transport(self, mean_of_transport,
                                                            transportation_options_specific_mean_of_transport):
        self.transportation_options[mean_of_transport] = transportation_options_specific_mean_of_transport

    def get_transportation_options_specific_mean_of_transport(self, mean_of_transport):
        return self.transportation_options[mean_of_transport]

    def __init__(self, name, production_costs, conversion_options, conversion_costs, transportation_options, transportation_costs):

        self.name = name

        self.production_costs = production_costs

        self.conversion_options = conversion_options
        self.conversion_costs = conversion_costs

        self.transportation_options = transportation_options
        self.transportation_costs = transportation_costs


def create_commodity_objects(production_costs, conversion_data, transportation_data):

    commodities = []
    commodity_names = []
    commodity_names_to_commodity = {}
    means_of_transport = []
    for commodity_1 in conversion_data.index:

        conversion_options = {}
        conversion_costs = {}

        transportation_options = {}
        transportation_costs = {}

        numeric_types = [float, int, np.float64, np.int64]

        for commodity_2 in conversion_data.columns:

            if type(conversion_data.loc[commodity_1, commodity_2]) in numeric_types:
                conversion_options[commodity_2] = True
                conversion_costs[commodity_2] = float(conversion_data.loc[commodity_1, commodity_2])
            else:
                conversion_options[commodity_2] = False
                conversion_costs[commodity_2] = '-'

        for mean_of_transport in transportation_data.columns:

            if type(transportation_data.loc[commodity_1, mean_of_transport]) in numeric_types:
                transportation_options[mean_of_transport] = True
                transportation_costs[mean_of_transport] = float(transportation_data.loc[commodity_1, mean_of_transport])
            else:
                transportation_options[mean_of_transport] = False
                transportation_costs[mean_of_transport] = '-'

        commodity = Commodity(commodity_1, production_costs[commodity_1], conversion_options, conversion_costs,
                              transportation_options, transportation_costs)
        commodities.append(commodity)
        commodity_names.append(commodity_1)
        commodity_names_to_commodity[commodity_1] = commodity

    for mean_of_transport in transportation_data.columns:
        if 'New' not in mean_of_transport:
            means_of_transport.append(mean_of_transport)

    return commodities, commodity_names, commodity_names_to_commodity, means_of_transport




