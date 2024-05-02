import numpy as np

from math import cos, sin, asin, sqrt, radians


def calc_distance_single_to_single(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    This methods calculates direct distance between two locations

    @param float latitude_1: latitude first location
    @param float longitude_1: longitude first location
    @param float latitude_2: latitude second location
    @param float longitude_2: longitude second location
    @return: single direct distance values in meter
    """

    # convert decimal degrees to radians
    longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    # haversine formula
    dlon = longitude_2 - longitude_1
    dlat = latitude_2 - latitude_1
    a = sin(dlat / 2) ** 2 + cos(latitude_1) * cos(latitude_2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 6371 * c * 1000
    return m


def calc_distance_list_to_single(latitude_list_1, longitude_list_1, latitude_2, longitude_2):
    """
    This method calculates the direct distance between a single location and an array of locations

    @param pandas.DataFrame latitude_list_1: latitudes of start locations
    @param pandas.DataFrame longitude_list_1: longitude of start locations
    @param float latitude_2: latitude of destination
    @param float longitude_2: longitude of destination
    @return: array of direct distances in meter
    """

    # convert decimal degrees to radians
    longitude_list_1 = np.radians(longitude_list_1.values.astype(float))
    latitude_list_1 = np.radians(latitude_list_1.values.astype(float))
    longitude_2, latitude_2 = map(radians, [longitude_2, latitude_2])

    # haversine formula
    dlon = longitude_2 - longitude_list_1
    dlat = latitude_2 - latitude_list_1
    a = np.sin(dlat / 2) ** 2 + np.cos(latitude_list_1) * np.cos(latitude_2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c * 1000
    return m


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
