from shapely.geometry import LineString, Point
from pyproj import Geod
from _helpers import point_to_coordinates
from straight_distance import get_straight_distance
from pipelines import get_used_pipeline_length


# ------------------------------ Modul für die Berechnung des kürzesten Pfades durch ein gegebenes Pipelinenetzwerk (Multilinestring) bei Angabe von Start- und Endpunkt -------------------------------


def find_shortest_path(network, start_point, end_point):  # for MultiLineStrings
    print('Startpunkt:', start_point)
    print('Endpunkt:', end_point)
    point_list = []         # Liste mit allen Knotenpunkten
    all_points = []         # Liste mit allen Punkten inkl. verbundenen Punkten
    lines = []              # Liste mit allen Teilstücken / Linestrings
    used_lines = []         # Verwendete Teilstücke / Linestrings
    for r in range(len(network.geoms)):
        line1 = network.geoms[r]
        endpoints = line1.boundary
        try:
            if endpoints.geoms[0] not in point_list:
                point_list.append(endpoints.geoms[0])
            if endpoints.geoms[1] not in point_list:
                point_list.append(endpoints.geoms[1])
        except Exception:  # Falls Pipeline ein Kreis ist, können keine endpoints bestimmt werden
            line1_coords = []
            for p in line1.coords:
                line1_coords.append(p)
            del line1_coords[-1]
            try:
                line1 = LineString(line1_coords)
            except Exception:   # falls line1 nur zwei gleiche Koordinaten hat (Punkt) kann kein LineString gebildet werden
                continue
        lines.append(line1)

    # Lücken in Netzwerk schließen
    connected_lines = lines.copy()
    while len(connected_lines) > 1:
        for v in range(len(connected_lines)):
            h = 0
            try:
                line1 = connected_lines[v]
            except IndexError:
                break
            for q in range(len(connected_lines)):
                line2 = connected_lines[q]
                try:
                    if connected_lines[v] == connected_lines[q]:
                        continue
                    else:
                        if connected_lines[v].intersects(connected_lines[q]):
                            h = h + 1
                            connected_line = connected_lines[v].union(connected_lines[q])
                            connected_lines.remove(line1)
                            connected_lines.remove(line2)
                            connected_lines.append(connected_line)
                            break
                except IndexError:
                    break
            if h == 0:
                try:
                    endpoints = connected_lines[v].boundary
                    for endpt in range(len(endpoints.geoms)):
                        distances = []
                        endpoint_coord = point_to_coordinates(str(endpoints.geoms[endpt]))
                        for a in range(len(connected_lines)):
                            if connected_lines[v] == connected_lines[a]:
                                continue
                            if connected_lines[a].geom_type == 'MultiLineString':
                                for con_line in connected_lines[a].geoms:
                                    for e in range(len(con_line.coords)):
                                        pt = Point(con_line.coords[e])
                                        coord_point = point_to_coordinates(str(pt))
                                        dist_pipe = get_straight_distance(endpoint_coord, coord_point)
                                        distances.append([dist_pipe, pt, endpoints.geoms[endpt]])
                            else:
                                for e in range(len(connected_lines[a].coords)):
                                    pt = Point(connected_lines[a].coords[e])
                                    coord_point = point_to_coordinates(str(pt))
                                    dist_pipe = get_straight_distance(endpoint_coord, coord_point)
                                    distances.append([dist_pipe, pt, endpoints.geoms[endpt]])

                    distances.sort(key=lambda x: x[0])
                    distances_sorted = distances
                    start_point_new_pipe = distances_sorted[0][1]
                    closest_point_new_pipe = distances_sorted[0][2]
                    new_line = LineString([start_point_new_pipe, closest_point_new_pipe])
                    lines.append(new_line)
                    connected_lines.append(new_line)
                    if start_point_new_pipe not in point_list:
                        point_list.append(start_point_new_pipe)
                    if closest_point_new_pipe not in point_list:
                        point_list.append(closest_point_new_pipe)
                except IndexError:
                    continue

    # Bestimme Start-Linestring
    distances = []
    start_point_coords = point_to_coordinates(str(start_point))
    for line in lines:
        pt = line.interpolate(line.project(start_point))
        pt_coords = point_to_coordinates(str(pt))
        dist = get_straight_distance(start_point_coords, pt_coords)
        distances.append(dist)
    min_dist = min(distances)
    index = distances.index(min_dist)
    start_line = lines[index]
    endpoints = start_line.boundary

    distances = []
    for e in endpoints.geoms:
        e_coord = point_to_coordinates(str(e))
        dist = get_straight_distance(start_point_coords, e_coord)
        distances.append(dist)
    min_dist = min(distances)
    index = distances.index(min_dist)
    line_pt_start = endpoints.geoms[index]

    # Bestimme End-Linestring
    distances = []
    end_point_coords = point_to_coordinates(str(end_point))
    for line in lines:
        pt = line.interpolate(line.project(end_point))
        pt_coords = point_to_coordinates(str(pt))
        dist = get_straight_distance(end_point_coords, pt_coords)
        distances.append(dist)
    min_dist = min(distances)
    index = distances.index(min_dist)
    end_line = lines[index]
    endpoints = end_line.boundary

    distances = []
    for e in endpoints.geoms:
        e_coord = point_to_coordinates(str(e))
        dist = get_straight_distance(end_point_coords, e_coord)
        distances.append(dist)
    min_dist = min(distances)
    index = distances.index(min_dist)
    line_pt_end = endpoints.geoms[index]

    coords = []
    if start_line == end_line:          # falls nur ein Linestring aus Multilinestring benötigt wird
        for coord in start_line.coords:
            coords.append(Point(coord))
        pipe_start_point = start_line.interpolate(start_line.project(start_point))
        pipe_end_point = start_line.interpolate(start_line.project(end_point))
        pipe_pt_start = point_to_coordinates(str(pipe_start_point))
        pipe_pt_end = point_to_coordinates(str(pipe_end_point))
        distances_start = []
        distances_end = []
        for w in range(len(coords)):
            co1 =  point_to_coordinates(str(coords[w]))
            dist1 = get_straight_distance(co1, pipe_pt_start)
            distances_start.append(dist1)
            dist2 = get_straight_distance(co1, pipe_pt_end)
            distances_end.append(dist2)
        min_dist1 = min(distances_start)
        min_dist2 = min(distances_end)
        ind_start = distances_start.index(min_dist1)
        ind_end = distances_end.index((min_dist2))

        if line_pt_start == line_pt_end:
            index_start = 0
            index_end = 1
            coords = []
            coords.insert(index_start, pipe_start_point)
            coords.insert(index_end, pipe_end_point)
        else:
            index_start = ind_start
            coords.remove(coords[ind_start])
            coords.insert(index_start, pipe_start_point)
            index_end = ind_end
            coords.remove(coords[ind_end])
            coords.insert(index_end, pipe_end_point)

        if ind_start < ind_end:
            used_line = coords[ind_start:ind_end+1]
        elif ind_start > ind_end:
            used_line = coords[ind_end:ind_start+1]
        elif ind_start == ind_end:
            used_line = [start_point, pipe_end_point]

        if len(used_line) == 1:
            route = 0
            length = 0
        else:
            route = used_line
            line = LineString(route)
            geod = Geod(ellps="WGS84")
            length = geod.geometry_length(line) / 1000

    else:       # Start und Endpunkt liegen auf unterschiedlichen Linestrings
        # Linestrings auftrennen für Verbindungspunkte / Knotenpunkte
        repeat = True
        intersection_points = []
        while repeat:
            repeat = False
            for l in range(len(lines)):
                pipe_pts = []
                try:
                    endpoints1 = lines[l].boundary
                    if lines[l] is not lines[-1]:
                        for c in range(len(lines[l].coords)):  # einzelne Punkte in Linestrings
                            pt = Point(lines[l].coords[c])
                            pipe_pts.append(pt)

                        for k in range(l+1, len(lines)):        # vergleiche Line mit anderen Lines
                            if lines[l].intersects(lines[k]):      # Überprüfe Intersection
                                intersection_pts = lines[l].intersection(lines[k])      # Bestimme Schnittpunkte
                                lengths = []
                                if intersection_pts.geom_type == 'LineString':
                                    try:
                                        for point in intersection_pts.coords:        # falls mehrere Schnittpunkte
                                            point_pt = Point(point)
                                            length = get_used_pipeline_length(lines[l], endpoints1.geoms[0], point_pt)[0]
                                            lengths.append(length)
                                        min_length = min(lengths)
                                        index = lengths.index(min_length)
                                        intersection_pt = Point(intersection_pts.coords[index])     # nächstgelegener Schnittpunkt
                                    except Exception:
                                        continue
                                elif intersection_pts.geom_type == 'MultiPoint':
                                    intersection_pt = intersection_pts.geoms[0]
                                elif intersection_pts.geom_type == 'Point':
                                    intersection_pt = intersection_pts          # nur ein Schnittpunkt

                                if intersection_pt in intersection_points:
                                    continue
                                else:
                                    intersection_points.append(intersection_pt)     # Liste mit allen Schnittpunkten
                                index_pt = 'none'
                                intersection_point_coord = point_to_coordinates(str(intersection_pt))

                                for w in range(len(pipe_pts)):
                                    if pipe_pts[w] == intersection_pt:      # Schnittpunkt bereits in Linestring enthalten?
                                        index_pt = w

                                if index_pt == 'none':  # Punkt konnte nicht auf Line gefunden werden
                                    # Split Lines
                                    two_pipes = [lines[l], lines[k]]
                                    for j in range(len(two_pipes)):
                                        distances = []
                                        coords = []

                                        for point in range(len(two_pipes[j].coords)):
                                            pt_point = Point(two_pipes[j].coords[point])
                                            coords.append(pt_point)
                                            pt_point_coord = point_to_coordinates(str(pt_point))
                                            dist = get_straight_distance(intersection_point_coord, pt_point_coord)
                                            distances.append([dist, pt_point, point])
                                        distances.sort(key=lambda x: x[0])
                                        distances_sorted = distances
                                        index1 = distances_sorted[0][2]
                                        index2 = distances_sorted[1][2]
                                        if intersection_pt not in coords:
                                            if index2 > index1:
                                                coords.insert(index2, intersection_pt)
                                                new_line1 = LineString(coords[:index2+1])
                                                new_line2 = LineString(coords[index2:])
                                            else:
                                                coords.insert(index1, intersection_pt)
                                                new_line1 = LineString(coords[:index1 + 1])
                                                new_line2 = LineString(coords[index1:])
                                        else:
                                            index_int = coords.index(intersection_pt)
                                            if index_int > 0 and index_int < len(coords)-1:
                                                new_line1 = LineString(coords[:index_int + 1])
                                                new_line2 = LineString(coords[index_int:])
                                            else:
                                                continue
                                        if j == 0:
                                            for d in range(len(lines)):
                                                if lines[d] == two_pipes[0]:
                                                    del lines[d]
                                                    lines.insert(d, new_line1)
                                                    lines.insert(d + 1, new_line2)
                                                    repeat = True

                                        elif j == 1:
                                            for d in range(len(lines)):
                                                if lines[d] == two_pipes[1]:
                                                    del lines[d]
                                                    lines.insert(d, new_line1)
                                                    lines.insert(d + 1, new_line2)
                                                    repeat = True

                                else:
                                    two_pipes = [lines[l], lines[k]]
                                    for j in range(len(two_pipes)):
                                        pipe_pts = []
                                        for c in range(len(two_pipes[j].coords)):  # einzelne Punkte in Linestrings
                                            pt = Point(two_pipes[j].coords[c])
                                            pipe_pts.append(pt)  # Liste mit Punkten in Geometry Point Format

                                        pipe_coords = list(two_pipes[j].coords)
                                        for w in range(len(pipe_pts)):
                                            if pipe_pts[w] == intersection_pt:  # Schnittpunkt bereits in Linestring enthalten?
                                                index_pt = w

                                        if len(pipe_coords[index_pt:]) > 1 and len(pipe_coords[:index_pt + 1]) > 1:
                                            new_line2 = LineString(pipe_coords[index_pt:])
                                            new_line1 = LineString(pipe_coords[:index_pt + 1])
                                            for d in range(len(lines)):
                                                if lines[d] == two_pipes[j]:
                                                    del lines[d]
                                                    lines.insert(d, new_line1)
                                                    lines.insert(d + 1, new_line2)
                                                    repeat = True
                                                    break
                                        else:
                                            continue
                                break
                            else:
                                continue
                except IndexError:
                    continue

        for line in lines:
            endpoints = line.boundary
            try:   # update
                if endpoints.geoms[0] not in point_list:
                    point_list.append(endpoints.geoms[0])
                if endpoints.geoms[1] not in point_list:
                    point_list.append(endpoints.geoms[1])
            except Exception:   # Falls Pipeline ein Kreis ist, können keine endpoints bestimmt werden
                continue

        for line in lines:
            endpoints = line.boundary
            try:
                index1 = point_list.index((endpoints.geoms[0]))
                index2 = point_list.index((endpoints.geoms[1]))
                points = [index1, index2]
            except Exception:   # Falls Pipeline ein Kreis ist, können keine endpoints bestimmt werden
                continue
            all_points.append(points)

        # Verbindungen zusammenfügen für Knotenbildung
        repeat = True
        while repeat:
            repeat = False
            for p in range(len(all_points)):
                for e in range(p+1, len(all_points)):
                    if all_points[p] == all_points[e]:
                        continue
                    elif all_points[p][0] == all_points[e][0]:
                        pass
                        for point in all_points[e]:
                            if point not in all_points[p]:
                                all_points[p].append(point)
                        del all_points[e]
                        repeat = True
                        break
                    else: continue

        nodes = []
        # alle Punkte als Knoten anlegen
        for p in range(len(all_points)):
            nodes.append(all_points[p][0])
        for p in range(len(all_points)):
            for e in range(len(all_points)):
                if all_points[p] == all_points[e]:
                    if len(all_points) == 1:
                        for t in range(len(all_points[p])):
                            if all_points[p][t] not in nodes:
                                nodes.append(all_points[p][t])
                                all_points.append([all_points[p][t]])
                    continue
                else:
                    for s in range(1, len(all_points[p])):
                        if all_points[p][s] not in nodes:
                            nodes.append(all_points[p][s])
                            all_points.append([all_points[p][s]])
                        if all_points[p][s] == all_points[e][0]:
                            if all_points[p][0] not in all_points[e]:
                                all_points[e].append(all_points[p][0])

        all_points.sort()
        nodes.sort()

        for p in range(len(all_points)):
            for e in range(len(all_points)):
                if all_points[p] == all_points[e]:
                    continue
                else:
                    for s in range(1, len(all_points[p])):
                        if all_points[p][s] not in nodes:
                            nodes.append(all_points[p][s])
                            all_points.append([all_points[p][s]])
                        if all_points[p][s] == all_points[e][0]:
                            if all_points[p][0] not in all_points[e]:
                                all_points[e].append(all_points[p][0])

        # Duplikate entfernen
        res = []
        [res.append(x) for x in all_points if x not in res]
        all_points = res.copy()
        nodes = list(dict.fromkeys(nodes))

        def shortest_path(graph, node1, node2):
            path_list = [[node1]]
            path_index = 0
            # To keep track of previously visited nodes
            previous_nodes = {node1}
            if node1 == node2:
                return path_list[0]

            while path_index < len(path_list):
                current_path = path_list[path_index]
                last_node = current_path[-1]
                next_nodes = graph[last_node]
                # Search goal node
                if node2 in next_nodes:
                    current_path.append(node2)
                    return current_path
                # Add new paths
                for next_node in next_nodes:
                    if not next_node in previous_nodes:
                        new_path = current_path[:]
                        new_path.append(next_node)
                        path_list.append(new_path)
                        # To avoid backtracking
                        previous_nodes.add(next_node)
                # Continue to next path in list
                path_index += 1
            # No path is found
            return []

        graph = {}
        for n in nodes:
            del all_points[n][0]
            graph[n] = all_points[n]
        print()
        print('Graph:', graph)

        start_node = point_list.index(line_pt_start)
        end_node = point_list.index(line_pt_end)
        route = []
        nodes_route = shortest_path(graph, start_node, end_node)
        for r in nodes_route:
            route.append(point_list[r])
        print('Route Knoten::', nodes_route)
        if len(route) == 0:
            return [start_point, 0, start_point, end_point]

        # verwendete Pipelinestücke / Linestrings
        for pt in range(len(route)):
            if route[pt] is not route[-1]:
                for l in lines:
                    points = []
                    for c in range(len(l.coords)):
                        point = Point(l.coords[c])
                        points.append(point)
                    if route[pt] in points and route[pt + 1 ] in points:
                        if l not in used_lines:
                            used_lines.append(l)
                    else: continue
            else: break

        # Entferne lines mit selbem Start- und Endpunkt
        used_line_endpoints = []
        for ul in used_lines:
            used_line_endpoints.append(ul.boundary)
        res = []
        for ep in used_line_endpoints:
            if ep not in res:
                res.append(ep)
            else:
                index = used_line_endpoints.index(ep)
                used_lines[index] = 0

        val = 0
        while val in used_lines:
            used_lines.remove(val)

        pipe_pt_start = point_to_coordinates(str(start_point))
        pipe_pt_end = point_to_coordinates(str(end_point))

        pipe_start_point = start_line.interpolate(start_line.project(start_point))
        pipe_end_point = end_line.interpolate(end_line.project(end_point))

        distances_start = []
        distances_end = []
        for w in range(len(route)):
            co1 = point_to_coordinates(str(route[w]))
            dist1 = get_straight_distance(co1, pipe_pt_start)
            distances_start.append(dist1)
            dist2 = get_straight_distance(co1, pipe_pt_end)
            distances_end.append(dist2)
        min_dist1 = min(distances_start)
        min_dist2 = min(distances_end)
        index_start = distances_start.index(min_dist1)
        index_end = distances_end.index((min_dist2))

        if line_pt_start == line_pt_end:
            if start_point != line_pt_start:
                route.insert(0, start_point)    # ist schon Verbindungspunkt auf Pipeline
            if pipe_end_point != line_pt_end:
                route.insert(2, pipe_end_point)  # nur end Punkt nicht unbedingt auf Pipeline
        else:
            if pipe_start_point != line_pt_start:
                route.remove(line_pt_start)
                route.insert(index_start, pipe_start_point)
            if pipe_end_point != line_pt_end:
                route.remove(line_pt_end)
                route.insert(index_end, pipe_end_point)

        section_lengths = []
        total_length = 0
        every_route_point = []
        for ul in range(len(used_lines)):
            if route[ul] is not route[-1]:
                route_infos = get_used_pipeline_length(used_lines[ul], route[ul], route[ul+1])
                dist = route_infos[0]
                route_section = route_infos[2]
                for pt in route_section:
                    every_route_point.append(pt)
                total_length = total_length + dist
                section_lengths.append(dist)
            else:
                break
        print('Einzellängen:', section_lengths)
        print()
        length = total_length
        res = []
        [res.append(x) for x in every_route_point if x not in res]
        route = res.copy()

    if index_start > index_end:
        start_point = route[-1]
        end_point = route[0]
        route.reverse()

    return [route, round(length, 2), start_point, end_point]
