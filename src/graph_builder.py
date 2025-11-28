import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
import haversine

def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine.haversine((lat1, lon1), (lat2, lon2))

def build_spatio_temporal_graph(df_path, k_spatial=6, lag_days=3):
    df = pd.read_csv(df_path).copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['lat', 'lon', 'date']).reset_index(drop=True)
    
    # Create lagged features (avoid future leakage)
    for lag in range(1, lag_days+1):
        df[f'rainfall_lag_{lag}'] = df.groupby(['lat', 'lon'])['rainfall'].shift(lag)
    df = df.dropna().reset_index(drop=True)
    df['node_id'] = range(len(df))
    
    G = nx.DiGraph()
    
    # Enhanced node features: [rainfall, lat, lon, inflow, 3 lags]
    feature_cols = ['rainfall', 'lat', 'lon', 'inflow'] + [f'rainfall_lag_{i}' for i in range(1, lag_days+1)]
    scaler = StandardScaler()
    node_features = scaler.fit_transform(df[feature_cols])
    
    # Add nodes with rich features
    for i, row in df.iterrows():
        feat_dict = {col: float(node_features[i, j]) for j, col in enumerate(feature_cols)}
        feat_dict.update({'date': row['date']})
        G.add_node(int(row['node_id']), **feat_dict)
    
    # Spatial edges (k-NN with distance weights)
    for date, group in df.groupby('date'):
        pts = group[['node_id', 'lat', 'lon']].values
        for i, (nid1, lat1, lon1) in enumerate(pts):
            distances = []
            for j, (nid2, lat2, lon2) in enumerate(pts):
                if nid1 != nid2:
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    distances.append((int(nid2), dist))
            distances.sort(key=lambda x: x[1])
            for nid2, dist in distances[:k_spatial]:
                weight = 1.0 / (dist + 1e-6)
                G.add_edge(int(nid1), int(nid2), type='spatial', weight=weight)
    
    # Temporal edges (within same location)
    for _, group in df.groupby(['lat', 'lon']):
        group = group.sort_values('date')
        ids = group['node_id'].tolist()
        for a, b in zip(ids, ids[1:]):
            G.add_edge(int(a), int(b), type='temporal', weight=1.0)
    
    # Export for PyG
    nodes = list(G.nodes(data=True))
    x = np.array([[n[1][col] for col in feature_cols] for n in nodes])
    edges = np.array([[u, v] for u, v, d in G.edges(data=True)])
    edge_weights = np.array([d.get('weight', 1.0) for _, _, d in G.edges(data=True)])
    
    export_data = {
        'x': x.astype(np.float32),
        'edge_index': edges.T if edges.size > 0 else np.zeros((2, 0), dtype=np.int64),
        'edge_attr': edge_weights if len(edge_weights) > 0 else np.array([]),
        'y': df['rainfall'].values.astype(np.float32),
        'scaler_params': {
            'mean': scaler.mean_,
            'scale': scaler.scale_
        },
        'feature_cols': feature_cols
    }
    
    return export_data, df
