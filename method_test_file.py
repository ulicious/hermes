import osmnx as ox

means_of_transport = {}

lon = 70
lat = 50

if lat < 0:
    north = lat + 0.25
    south = lat - 0.25
else:
    north = lat - 0.25
    south = lat + 0.25

G = ox.graph_from_bbox(north, south, lon - 0.25, lon + 0.25)
G_proj = ox.project_graph(G)
nearest_edge = ox.nearest_edges(G_proj, X=lon, Y=lat, return_dist=True)

means_of_transport['Road'] = nearest_edge