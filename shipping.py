import pandas as pd
import searoute as sr
from shapely.geometry import LineString
from _helpers import switch_lat_lon_list
from straight_distance import get_straight_distance


# ------------------------------------------------------------- Modul zur Berechnung von Schiffsrouten und deren Kosten ---------------------------------------------------------------------


data_costs = pd.read_excel('input_data/Data Aggregation.xlsx', sheet_name='working data')

def searoute(start_coordinates, destination_coordinates):
    # Define origin and destination points --> [lon, lat]
    origin = switch_lat_lon_list(start_coordinates)
    destination = switch_lat_lon_list(destination_coordinates)
    route = sr.searoute(origin, destination)
    coordinates = []
    for coordinate in route.geometry['coordinates']:
        coordinates.append((coordinate[0], coordinate[1]))
    coordinate = switch_lat_lon_list((route.geometry.coordinates[0][0], route.geometry.coordinates[0][1]))
    new_start = switch_lat_lon_list(start_coordinates)
    coordinates.insert(0, new_start)
    line = LineString(coordinates)
    dist = round(get_straight_distance(start_coordinates, coordinate), 2)
    distance = round(float(format(route.properties['length'])), 2) + dist    # km

    return [distance, line]


# Berechnung Kosten für Schiffstransport
def shipping(sea_distance, index_shipping):
    average_driving_speed_ship = 33  # kmh, 18 Knoten
    loading_unloading_time = 2 * 24
    shipping_data = data_costs.iloc[index_shipping]
    ship_cost = shipping_data['CAPEX'] * shipping_data['annuity factor [%]'] / 100 / 365 * (sea_distance / average_driving_speed_ship * 2 / 24 + 2 * loading_unloading_time) / shipping_data['Capacity Ref.']
    ship_cost_opex = shipping_data['CAPEX'] * shipping_data['OPEX fix'] / 100 / 365 * (sea_distance / average_driving_speed_ship * 2 / 24 + 2 * loading_unloading_time) / shipping_data['Capacity Ref.']
    ship_fuel_cost = (sea_distance / average_driving_speed_ship * 2 / 24 * shipping_data['OPEX var'] * shipping_data['energy cost [€/kWhel]']) / shipping_data['Capacity Ref.']
    shipping_cost = ship_cost + ship_fuel_cost + ship_cost_opex
    return shipping_cost # €/MWh
