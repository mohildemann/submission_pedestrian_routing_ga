### http://www.networkx.github.io
### for docu
import networkx as nx
### path to shapefile(s)
path = 'Graph/Muenster_edges.shp'
## Graph - not directed ! 
G = nx.Graph(nx.read_shp(path),strict=False, geom_attrs=True)
### START AND END ID 
### nodes are described in coordinate pairs
routes = [[(405030.89170764922, 5757094.946161799),(406135.01669398177, 5757120.137230382)],
          [(404950.52825963526, 5756361.616166964),(406135.01669398177, 5757120.137230382)],
          [(405495.13775169034, 5756342.791932008),(406135.01669398177, 5757120.137230382)]]

###  get the shortest path ; returns a list of nodes
### route to analyse
### accepts 1 to 3
route = 1
shortest_path =nx.dijkstra_path(G,routes[route-1][0],routes[route-1][1],weight='length')
### empty graph to fill with nodes and edges
H = nx.Graph()
## fill graph with nodes and edges
for i in range(len(shortest_path)):
    H.add_node(shortest_path[i])
for y in range(len(shortest_path)-1):
    H.add_edge((shortest_path[y]),(shortest_path[y+1]))
### write to output shapefiles folder 
nx.write_shp(H,'shapefiles')
