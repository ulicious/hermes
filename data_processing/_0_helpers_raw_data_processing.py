

def calculate_conversion_costs(specific_investment, depreciation_period, fixed_maintenance,
                               operating_hours, interest_rate,
                               electricity_costs, electricity_demand, co2_costs, co2_demand,
                               nitrogen_costs, nitrogen_demand, heat_demand=0, heat_costs=0):

    """
    Uses annuity method, investment, operation and maintenance parameters, and costs of commodity to calculate the
    conversion costs of one commodity to another

    @param float specific_investment:
    @param float depreciation_period:
    @param float fixed_maintenance:
    @param float operating_hours:
    @param float interest_rate:
    @param float electricity_costs:
    @param float electricity_demand:
    @param float co2_costs:
    @param float co2_demand:
    @param float nitrogen_costs:
    @param float nitrogen_demand:
    @param float heat_demand:
    @param float heat_costs:
    @return: conversion costs
    """

    annuity_factor \
        = (interest_rate * (1 + interest_rate) ** depreciation_period) / ((1 + interest_rate) ** depreciation_period - 1)

    conversion_costs \
        = specific_investment * (annuity_factor + fixed_maintenance) / operating_hours \
        + electricity_costs * electricity_demand + co2_costs * co2_demand \
        + nitrogen_costs * nitrogen_demand + heat_demand * heat_costs

    return conversion_costs

