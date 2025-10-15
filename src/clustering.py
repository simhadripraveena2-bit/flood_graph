# src/clustering.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from haversine import haversine

def spatio_temporal_clustering(df, eps_km=50, eps_days=5, min_samples=3, auto_tune=True):
    """
    Perform spatio-temporal clustering on rainfall data.
    Combines latitude, longitude, and date information.
    """
    df = df.copy()

    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'])
    coords = df[['latitude', 'longitude']].to_numpy()
    dates = df['date'].to_numpy(dtype='datetime64[D]')  # ensure day precision

    # Compute spatial distance matrix (km)
    n = len(df)
    spatial_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(tuple(coords[i]), tuple(coords[j]))
            spatial_dist[i, j] = d
            spatial_dist[j, i] = d

    # Temporal distance (in days)
    temporal_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # difference as float days
            t = abs((dates[i] - dates[j]) / np.timedelta64(1, 'D'))
            temporal_dist[i, j] = t
            temporal_dist[j, i] = t

    # Normalize both distances
    spatial_dist /= (spatial_dist.max() + 1e-9)
    temporal_dist /= (temporal_dist.max() + 1e-9)

    # Combine (weighted)
    combined_dist = 0.7 * spatial_dist + 0.3 * temporal_dist

    # Auto-tune eps if requested
    if auto_tune:
        avg_neighbor_dist = np.mean(np.sort(combined_dist, axis=1)[:, 1:5])
        eps = avg_neighbor_dist * 1.2  # small buffer
    else:
        eps = eps_km / 6371.0  # fallback

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(combined_dist)
    df['cluster_id'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters (and {sum(labels == -1)} noise points)")
    return df, combined_dist
