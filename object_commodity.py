import numpy as np
import math


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

    def set_conversion_loss_of_educt(self, conversion_loss_of_educt):
        self.conversion_loss_of_educt = conversion_loss_of_educt

    def get_conversion_loss_of_educt(self):
        return self.conversion_loss_of_educt

    def set_conversion_loss_of_educt_specific_commodity(self, commodity_name, conversion_loss_of_educt_specific_commodity):
        self.conversion_loss_of_educt[commodity_name] = conversion_loss_of_educt_specific_commodity

    def get_conversion_loss_of_educt_specific_commodity(self, commodity_name):
        return self.conversion_loss_of_educt[commodity_name]

    def set_transportation_options(self, transportation_options):
        self.transportation_options = transportation_options

    def get_transportation_options(self):
        return self.transportation_options

    def set_transportation_options_specific_mean_of_transport(self, mean_of_transport,
                                                              transportation_options_specific_mean_of_transport):
        self.transportation_options[mean_of_transport] = transportation_options_specific_mean_of_transport

    def get_transportation_options_specific_mean_of_transport(self, mean_of_transport):
        return self.transportation_options[mean_of_transport]

    def set_transportation_costs(self, transportation_costs):
        self.transportation_costs = transportation_costs

    def get_transportation_costs(self):
        return self.transportation_costs

    def set_transportation_costs_specific_mean_of_transport(self, mean_of_transport,
                                                            transportation_costs_specific_mean_of_transport):
        self.transportation_costs[mean_of_transport] = transportation_costs_specific_mean_of_transport

    def get_transportation_costs_specific_mean_of_transport(self, mean_of_transport):
        return self.transportation_costs[mean_of_transport]

    if False:

        def set_transportation_efficiency(self, transportation_efficiency):
            """ sets all transportation efficiencies """
            self.transportation_efficiency = transportation_efficiency

        def get_transportation_efficiency(self):
            """ returns all transportation efficiencies """
            return self.transportation_efficiency

        def set_transportation_efficiency_specific_mean_of_transport(self, mean_of_transport,
                                                                     transportation_efficiency_specific_mean_of_transport):
            """ sets transportation efficiency of one specific mean of transport """
            self.transportation_efficiency[mean_of_transport] = transportation_efficiency_specific_mean_of_transport

        def get_transportation_efficiency_specific_mean_of_transport(self, mean_of_transport):
            """ returns transportation efficiency of one specific mean of transport """
            return self.transportation_efficiency[mean_of_transport]

    def __init__(self, name, production_costs, conversion_options, conversion_costs, conversion_loss_of_educt,
                 transportation_options, transportation_costs):

        """
        Creates instance of commodity object
        :param name: name of commodity
        :param production_costs: production costs of commodity
        :param conversion_options: list with conversion options
        :param conversion_costs: dictionary with conversion costs
        :param conversion_loss_of_product: dictionary with conversion efficiencies
        :param transportation_options: list with transportation options
        :param transportation_costs: dictionary with transportation costs
        """

        self.name = name

        self.production_costs = production_costs

        self.conversion_options = conversion_options
        self.conversion_costs = conversion_costs
        self.conversion_loss_of_educt = conversion_loss_of_educt

        self.transportation_options = transportation_options
        self.transportation_costs = transportation_costs


def create_commodity_objects(production_costs,
                             commodity_conversion_data, commodity_conversion_loss_of_product_data,
                             commodity_transportation_data):

    """
    This function processes and organizes data related to commodities and transportation options.

    :param production_costs: Dictionary containing production costs for commodities.
    :param commodity_conversion_data: DataFrame containing information about conversion options between commodities.
    :param commodity_conversion_loss_of_product_data: DataFrame containing information about conversion efficiencies between commodities.
    :param commodity_transportation_data: DataFrame containing information about transportation options and costs.
    :return: A tuple containing four elements:
             1. List of Commodity instances with details about each commodity.
             2. List of commodity names.
             3. Dictionary mapping commodity names to their respective Commodity instances.
             4. List of means of transport.
    """

    commodities = []
    commodity_names = []
    commodity_names_to_commodity = {}
    means_of_transport = []
    for source_commodity in commodity_conversion_data.index:

        conversion_options = {}
        conversion_costs = {}
        conversion_loss_of_product = {}

        transportation_options = {}
        transportation_costs = {}

        for target_commodity in commodity_conversion_data.columns:

            is_nan = math.isnan(commodity_conversion_data.loc[source_commodity, target_commodity])
            is_string = isinstance(commodity_conversion_data.loc[source_commodity, target_commodity], str)

            if not(is_nan or is_string):
                conversion_options[target_commodity] = True
                conversion_costs[target_commodity] = float(commodity_conversion_data.loc[source_commodity, target_commodity])
                conversion_loss_of_product[target_commodity] = float(commodity_conversion_loss_of_product_data.loc[source_commodity, target_commodity])
            else:
                conversion_options[target_commodity] = False
                conversion_costs[target_commodity] = '-'
                conversion_loss_of_product[target_commodity] = '-'

        for mean_of_transport in commodity_transportation_data.columns:

            is_nan = math.isnan(commodity_transportation_data.loc[source_commodity, mean_of_transport])
            is_string = isinstance(commodity_transportation_data.loc[source_commodity, mean_of_transport], str)

            if not(is_nan or is_string):
                transportation_options[mean_of_transport] = True
                transportation_costs[mean_of_transport] = float(commodity_transportation_data.loc[source_commodity,
                                                                                                  mean_of_transport])
            else:
                transportation_options[mean_of_transport] = False
                transportation_costs[mean_of_transport] = '-'

        if source_commodity in [*production_costs.keys()]:
            commodity = Commodity(source_commodity, production_costs[source_commodity],
                                  conversion_options, conversion_costs, conversion_loss_of_product,
                                  transportation_options, transportation_costs)

        else:
            commodity = Commodity(source_commodity, None,
                                  conversion_options, conversion_costs, conversion_loss_of_product,
                                  transportation_options, transportation_costs)

        commodities.append(commodity)
        commodity_names.append(source_commodity)
        commodity_names_to_commodity[source_commodity] = commodity

    for mean_of_transport in commodity_transportation_data.columns:
        if 'New' not in mean_of_transport:
            means_of_transport.append(mean_of_transport)

    return commodities, commodity_names, commodity_names_to_commodity, means_of_transport




