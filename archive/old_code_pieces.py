for n in node_order.index:
    if n in processed_nodes:
        continue

    processed_nodes.append(n)
    affected_nodes = distances[distances['level_0'] == n]['level_1'].values

    for node_to_remove in affected_nodes:

        if nodes.at[node_to_remove, 'graph'] != nodes.at[n, 'graph']:
            continue

        same_graphs = True

        # get all edges which are connected to node_to_remove
        other_edges = graphs[(graphs['node_start'] == node_to_remove) | (graphs['node_end'] == node_to_remove)]

        for o in other_edges.index:
            # replace node_to_remove with n
            if graphs.at[o, 'node_start'] == node_to_remove:
                graphs.at[o, 'node_start'] = n

                line = graphs.at[o, 'line']
                new_coords = [Point([nodes.at[n, 'longitude'], nodes.at[n, 'latitude']])]
                for coords in line.coords[1:]:
                    new_coords.append(coords)

            else:
                graphs.at[o, 'node_end'] = n

                line = graphs.at[o, 'line']
                new_coords = []
                for coords in line.coords[:-1]:
                    new_coords.append(coords)
                new_coords.append(Point([nodes.at[n, 'longitude'], nodes.at[n, 'latitude']]))

            c_before = None
            distance = 0
            for c in new_coords:
                if c_before is not None:

                    if isinstance(c, Point):
                        c_x = c.x
                        c_y = c.y
                    else:
                        c_x = c[0]
                        c_y = c[1]

                    if isinstance(c_before, Point):
                        c_before_x = c_before.x
                        c_before_y = c_before.y
                    else:
                        c_before_x = c_before[0]
                        c_before_y = c_before[1]

                    distance += calc_distance((c_y, c_x), (c_before_y, c_before_x)).meters

                c_before = c

            graphs.at[o, 'line'] = LineString(new_coords)
            graphs.at[o, 'distance'] = distance

            if (graphs.at[o, 'node_start'] == n) & (graphs.at[o, 'node_end'] == n):
                if o not in edges_to_drop:
                    edges_to_drop.append(o)

        if node_to_remove not in processed_nodes:
            processed_nodes.append(node_to_remove)