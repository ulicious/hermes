import pandas as pd
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

    def set_conversion_costs(self, conversion_costs):
        self.conversion_costs = conversion_costs

    def get_conversion_costs(self):
        return self.conversion_costs

    def get_minimal_conversion_costs(self, commodity_name):
        return self.conversion_costs[commodity_name].min()

    def set_conversion_costs_specific_commodity(self, location, commodity_name, conversion_costs):
        self.conversion_costs.loc[location, commodity_name] = conversion_costs

    def get_conversion_costs_specific_commodity(self, location, commodity_name):
        return self.conversion_costs.loc[location, commodity_name]

    def set_conversion_efficiencies(self, conversion_efficiencies):
        self.conversion_efficiencies = conversion_efficiencies

    def get_conversion_efficiencies(self):
        return self.conversion_efficiencies

    def get_minimal_conversion_efficiency(self, commodity_name):
        return self.conversion_efficiencies[commodity_name].min()

    def set_conversion_efficiency_specific_commodity(self, location, commodity_name, conversion_efficiency):
        self.conversion_efficiencies.loc[location, commodity_name] = conversion_efficiency

    def get_conversion_efficiency_specific_commodity(self, location, commodity_name):
        return self.conversion_efficiencies.loc[location, commodity_name]

    def set_transportation_options(self, transportation_options):
        self.transportation_options = transportation_options

    def get_transportation_options(self):
        return self.transportation_options

    def set_transportation_costs(self, transportation_costs):
        self.transportation_costs = transportation_costs

    def get_transportation_costs(self):
        return self.transportation_costs

    def set_transportation_costs_specific_mean_of_transport(self, mean_of_transport,
                                                            transportation_costs_specific_mean_of_transport):
        self.transportation_costs.loc[mean_of_transport] = transportation_costs_specific_mean_of_transport

    def get_transportation_costs_specific_mean_of_transport(self, mean_of_transport):
        return self.transportation_costs.loc[mean_of_transport]

    def get_transportation_options_specific_mean_of_transport(self, mean_of_transport):
        return self.transportation_options[mean_of_transport]

    def set_starting_efficiency(self, starting_efficiency):
        self.starting_efficiency = starting_efficiency

    def get_starting_efficiency(self):
        return self.starting_efficiency

    def __init__(self, name, production_costs, conversion_options, conversion_costs, conversion_efficiencies,
                 transportation_options, transportation_costs, starting_efficiency=1):

        """
        Creates instance of commodity object
        @param str name: name of commodity
        @param pandas.DataFrame production_costs: production cost of commodity at location
        @param dict conversion_options: list with conversion options
        @param pandas.DataFrame conversion_costs: DataFrame with conversion costs at location
        @param pandas.DataFrame conversion_efficiencies: DataFrame with conversion efficiencies at location
        @param dict transportation_options: list with transportation options
        @param pandas.Series transportation_costs: DataFrame with transportation costs
        """

        self.name = name

        self.production_costs = production_costs

        self.conversion_options = conversion_options
        self.conversion_costs = conversion_costs
        self.conversion_efficiencies = conversion_efficiencies

        self.transportation_options = transportation_options
        self.transportation_costs = transportation_costs

        self.starting_efficiency = starting_efficiency


def create_commodity_objects(location_data,
                             conversion_costs_and_efficiencies, techno_economic_data_conversion,
                             techno_economic_data_transportation, config_file):

    """
    This function processes and organizes data related to commodities and transportation options.

    @param pandas.DataFrame location_data: DataFrame containing information on production costs at start location
    @param pandas.DataFrame conversion_costs_and_efficiencies: DataFrame containing information on location specific conversion costs and efficiencies
    @param pandas.DataFrame techno_economic_data_conversion: DataFrame containing information about transportation options and costs.
    @param pandas.DataFrame techno_economic_data_transportation: DataFrame containing information about transportation options and costs.
    @param dict config_file: containing necessary settings and assumptions
    @return: A tuple containing two elements:
             1. List of Commodity instances with details about each commodity.
             2. List of str commodity names.
    """

    commodity_names = config_file['available_commodity']
    all_transport_means = config_file['available_transport_means']
    commodities = []

    locations_with_conversion = conversion_costs_and_efficiencies[conversion_costs_and_efficiencies['conversion_possible']]
    for source_commodity in commodity_names:

        # -- conversions
        potential_conversions = techno_economic_data_conversion[source_commodity]['potential_conversions']
        conversion_options = {}

        conversion_cost_columns = []
        conversion_cost_efficiency_columns = []

        order_commodities = []

        # get columns of costs and efficiencies
        for target_commodity in commodity_names:

            if target_commodity in potential_conversions:

                conversion_options[target_commodity] = True

                conversion_cost_columns.append(source_commodity + '_' + target_commodity + '_conversion_costs')
                conversion_cost_efficiency_columns.append(source_commodity + '_' + target_commodity + '_conversion_efficiency')

                order_commodities.append(target_commodity)

            else:

                conversion_options[target_commodity] = False

        # -- transportation
        potential_transport_means = techno_economic_data_transportation[source_commodity]['potential_transportation']
        transportation_costs = pd.Series(math.inf, index=potential_transport_means)
        transportation_options = {}

        for mean_of_transport in all_transport_means:
            if mean_of_transport in potential_transport_means:

                if 'New' in mean_of_transport:
                    # check if new pipelines can be built
                    if config_file['build_new_infrastructure']:
                        transportation_options[mean_of_transport] = True
                    else:
                        transportation_options[mean_of_transport] = False

                elif source_commodity == 'Hydrogen_Gas':
                    # if hydrogen, check if retrofitting is possible
                    if mean_of_transport == 'Pipeline_Gas':
                        if config_file['H2_ready_infrastructure']:
                            transportation_options[mean_of_transport] = True
                        else:
                            transportation_options[mean_of_transport] = False
                    else:
                        transportation_options[mean_of_transport] = True

                else:
                    transportation_options[mean_of_transport] = True

                # attach transportation costs
                transportation_costs.loc[mean_of_transport] \
                    = techno_economic_data_transportation[source_commodity][mean_of_transport] / 1000

            else:  # mean of transport is not in potential transport means
                transportation_options[mean_of_transport] = False
                transportation_costs.loc[mean_of_transport] = math.inf

        conversion_costs = locations_with_conversion[conversion_cost_columns]
        conversion_costs.columns = order_commodities

        conversion_efficiencies = locations_with_conversion[conversion_cost_efficiency_columns]
        conversion_efficiencies.columns = order_commodities

        if source_commodity == 'Hydrogen_Gas':
            starting_efficiency = 1
        else:
            if 'Hydrogen_Gas_' + source_commodity + '_conversion_efficiency' not in locations_with_conversion.columns:
                # direct conversion from hydrogen gas to target commodity to possible
                efficiency_columns = [c for c in locations_with_conversion.columns
                                      if source_commodity + '_conversion_efficiency' in c]
                input_commodity = efficiency_columns[0].split('_' + source_commodity)[0]

                h2_to_input = locations_with_conversion.loc['Start', 'Hydrogen_Gas_' + input_commodity + '_conversion_efficiency']
                input_to_target = locations_with_conversion.loc['Start', input_commodity + '_' + source_commodity + '_conversion_efficiency']

                starting_efficiency = h2_to_input * input_to_target

            else:
                starting_efficiency = locations_with_conversion.loc['Start', 'Hydrogen_Gas_' + source_commodity + '_conversion_efficiency']

        commodity = Commodity(source_commodity, location_data.loc['Start', source_commodity], conversion_options,
                              conversion_costs, conversion_efficiencies,
                              transportation_options, transportation_costs, starting_efficiency)

        commodities.append(commodity)

    return commodities, commodity_names
