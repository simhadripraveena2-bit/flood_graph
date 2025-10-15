import networkx as nx
from sklearn.neighbors import kneighbors_graph

def build_spatial_graph(df, k=5):
    """
    Builds spatial graph where nodes = unique (lat, lon),
    edges = k-nearest neighbors by geodesic distance.
    """
    coords = df[["latitude", "longitude"]].drop_duplicates().values
    graph = kneighbors_graph(coords, k, mode='connectivity', include_self=False)
    G = nx.from_scipy_sparse_array(graph)
    mapping = {i: tuple(coords[i]) for i in range(len(coords))}
    G = nx.relabel_nodes(G, mapping)
    return G
