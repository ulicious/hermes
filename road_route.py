import openrouteservice
from openrouteservice import convert
import shapely.wkt
from shapely.geometry import LineString, Point
import geopandas as gpd
import pandas as pd
from _helpers import switch_lat_lon

#---------------------------- Modul für die Berechnung von Entfernung zu Häfen, Straßenrouten und deren Kosten --------------------------------------------------------------------------

data_costs = pd.read_excel('input_data/Data Aggregation.xlsx', sheet_name='working data')

# Berechnung Straßenroute zu nächstgelegenem Hafen
def get_closest_ports_via_road(start_coordinates, closest_ports):  # target port or pipeline
    start_coordinates = switch_lat_lon(start_coordinates)
    for p, port in enumerate(closest_ports['mit Koordinaten']):
        try:
            port_point = gpd.GeoSeries([shapely.wkt.loads(port)])
            port_longitude = port_point.apply(lambda p: p.x).to_string(index=False)
            port_latitude = port_point.apply(lambda p: p.y).to_string(index=False)

            # Route bestimmen
            coords = ((start_coordinates),(port_longitude, port_latitude))
            client = openrouteservice.Client(key='5b3ce3597851110001cf6248f769a906b1254e6485b69c1f2ef5206b') # Specify your personal API key
            routes = client.directions(coords, profile='driving-hgv')   #Route für LKW
            geometry = client.directions(coords)['routes'][0]['geometry']
            decoded = convert.decode_polyline(geometry)
            point_list = []
            for point in decoded['coordinates']:
                point_list.append(Point(point))
            line = LineString(point_list)
            distance = round(routes['routes'][0]['summary']['distance'] / 1000, 2)
            duration = round(routes['routes'][0]['summary']['duration'] / 3600, 2)
            if p < 1:
                road_routes = pd.DataFrame({'Entfernung [km]': [distance], 'Dauer [h]': [duration], 'zu Hafen': closest_ports.loc[p, 'zu Hafen'], 'Koordinaten': port, 'Route': str(line)})
            else:
                road_routes.loc[p] = distance, duration, closest_ports.loc[p, 'zu Hafen'], port, str(line)
        except Exception: #keine Straße in der Nähe
            print('Straßenroute konnte nicht gefunden werden')
            road_routes = pd.DataFrame()
            break
    if road_routes.empty:
        return road_routes
    else:
        road_routes = road_routes.sort_values(by="Entfernung [km]")
        closest_port_via_road = road_routes.loc[1, 'zu Hafen']
        port_coordinates = closest_ports[closest_ports['zu Hafen'] == closest_port_via_road]['mit Koordinaten']
        print(road_routes, '\n')
        print('kürzeste Straßenroute mit LKW:')
        print('Entfernung',road_routes.loc[1, 'Entfernung [km]'], 'km')
        print('Dauer:', road_routes.loc[1, 'Dauer [h]'], 'h')
        print('zu Hafen:',closest_port_via_road, 'mit Koordinaten', port_coordinates.to_string(index=False))
        return road_routes


# Berechnung Route und Streckenlänge
def get_road_route(start_coordinates, target_coordinates):  # target port or pipeline
    start_coordinates = switch_lat_lon(start_coordinates)
    target_coordinates = switch_lat_lon(target_coordinates)

    coords = (start_coordinates, target_coordinates)
    client = openrouteservice.Client(
        key='5b3ce3597851110001cf6248f769a906b1254e6485b69c1f2ef5206b')  # Specify your personal API key
    routes = client.directions(coords, profile='driving-hgv')
    geometry = client.directions(coords)['routes'][0]['geometry']
    decoded = convert.decode_polyline(geometry)
    point_list = []
    for point in decoded['coordinates']:
        point_list.append(Point(point))
    line = LineString(point_list)

    distance = round(routes['routes'][0]['summary']['distance'] / 1000, 2)
    duration = round(routes['routes'][0]['summary']['duration'] / 3600, 2)

    return [distance, line, duration]



# Berechnung Kosten für Straßentransport mit LKW
def truck_transport(road_distance, duration ,index_truck_transport):
    index_truck = data_costs[data_costs.process == 'Truck'].index[0]
    truck_data = data_costs.iloc[index_truck]
    truck_transport_data = data_costs.iloc[index_truck_transport]
    # Daten aus 'working data' Tabelle auslesen
    truck_cost = truck_data['depreciation cost']  # €/h
    trailer_cost = truck_transport_data['depreciation cost']            # €/h
    fuel_consumption = truck_data['OPEX var']                           # l/100km
    fuel_cost = truck_data['energy cost [€/kWhel]']                               # €/l
    loading_unloading_time = truck_transport_data['loading time']
    labor_costs = truck_transport_data['labor cost']
    # spezifische Kosten für gegebene Distanz berechnen:
    specific_fuel_cost = round((fuel_consumption / 100 * float(road_distance) * 2 * fuel_cost) / truck_transport_data['Capacity Ref.'], 4)
    specific_truck_cost = round(truck_cost * (float(duration) * 2 + loading_unloading_time) / truck_transport_data['Capacity Ref.'], 4)
    specific_trailer_cost = round(trailer_cost * (float(duration) * 2 + loading_unloading_time) / truck_transport_data['Capacity Ref.'], 4)
    specific_labor_cost = round(labor_costs * (float(duration) * 2 + loading_unloading_time) / truck_transport_data['Capacity Ref.'], 4)
    truck_transport_cost = round(specific_truck_cost + specific_trailer_cost + specific_fuel_cost + specific_labor_cost, 4)
    # print('Spritkosten:', specific_fuel_cost, 'LKW Kosten:', specific_truck_cost, 'Anhänger Kosten:', specific_trailer_cost, 'Lohnkosten:', specific_labor_cost)

    return truck_transport_cost

