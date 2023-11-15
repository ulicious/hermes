import pandas as pd
import requests
import json
import time


def get_road_distance_to_single_option(configuration, location, option, direct_distance=None):
    # todo: The tool also uses direct transportation somehow. Needs to be avoided

    successful = False
    routes = None
    while not successful:

        try:
            r = requests.get(
                f"http://router.project-osrm.org/route/v1/car/{location.x},{location.y};"
                f"{option.x},{option.y}?steps=true&geometries=geojson""")
            routes = json.loads(r.content)
            successful = True

        except Exception:
            time.sleep(1)

    if routes['code'] != 'NoRoute':
        route = routes.get("routes")[0]
        # Case, that initial starting point is not on street:
        # todo, hier gibt es anscheinend auch einen service
        #  http://project-osrm.org/docs/v5.5.1/api/?language=Python#nearest-service
        #  Wäre hilfreich für die Distanz zu Häfen, wenn kein direkter Weg hinführt
        #  (aber z.B. nur 500 m entfernt)

        return route['distance']

    else:

        return direct_distance * configuration['no_road_multiplier']


def get_road_distance_to_options(configuration, location, options, step_size=200):

    start_string = 'http://router.project-osrm.org/table/v1/driving/' + str(location.x) + ',' + str(
        location.y)
    end_string = '?sources=0&annotations=distance'

    all_options = options.index.tolist()
    all_distances = pd.DataFrame(index=options.index, columns=['road_distance'])

    while all_options:
        options_to_check = all_options[:step_size]
        destination_string = ''
        for option in options_to_check:
            destination_string += ';' + str(options.loc[option, 'destination_longitude']) + ',' + str(
                options.loc[option, 'destination_latitude'])

        request_string = start_string + destination_string + end_string

        successful = False
        while not successful:
            try:
                r = requests.get(request_string)
                routes = json.loads(r.content)
                all_distances.loc[options_to_check, 'road_distance'] = routes['distances'][0][1:]
                successful = True

            except Exception:
                time.sleep(1)

        all_options = all_options[step_size:]

    # Set distance to road multiplier * direct distance as minimal value if None
    for ind in all_distances.index:
        if all_distances.loc[ind, 'road_distance'] is None:
            all_distances.loc[ind, 'road_distance'] = options.loc[ind, 'direct_distance'] * configuration['no_road_multiplier']

    return all_distances['road_distance']


def get_road_distances_between_options(options_start, options_destination, step_size=450):

    distances = {}

    start_string = 'http://router.project-osrm.org/table/v1/driving/'
    end_string = '?sources='

    first = True
    for i, s in enumerate(options_start):

        if first:
            start_string += str(round(s.x, 4)) + ',' + str(round(s.y, 4))
            end_string += str(i)

            first = False
        else:
            start_string += ';' + str(round(s.x, 4)) + ',' + str(round(s.y, 4))
            end_string += ';' + str(i)

    end_string += '&annotations=distance'

    while options_destination:
        options_to_check = options_destination[:step_size]
        destination_string = ''
        for option in options_to_check:
            destination_string += ';' + str(round(option.x, 4)) + ',' + str(round(option.y, 4))

        request_string = start_string + destination_string + end_string

        successful = False
        while not successful:
            try:
                r = requests.get(request_string)
                routes = json.loads(r.content)

                for i, s in enumerate(routes['distances']):
                    distances[i] = s

                successful = True

            except Exception:
                print(r)
                time.sleep(5)

        options_destination = options_destination[step_size:]

    if False:
        # Set distance to road multiplier * direct distance as minimal value if None
        for ind in all_distances.index:
            if all_distances.loc[ind, 'road_distance'] is None:
                all_distances.loc[ind, 'road_distance'] = options.loc[ind, 'direct_distance'] * configuration[
                    'no_road_multiplier']

    return distances
