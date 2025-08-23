from __future__ import annotations
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scan_utils import load_data_from_csv, build_features_data


def find_anomalies_with_dbscan_original(
    features_data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original DBSCAN implementation for comparison purposes.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_data)
    
    core_indices = dbscan.core_sample_indices_
    noise_indices = np.where(cluster_labels == -1)[0]

    sorted_noise_indices, _scores = sort_noise_points_by_distance(
        features_data, noise_indices, core_indices
    )
    return cluster_labels, sorted_noise_indices


def compute_sparse_neighborhoods_chunked(features_data: np.ndarray, eps: float, chunk_size: int) -> csr_matrix:
    """
    Compute sparse neighborhood matrix in chunks to reduce memory usage.
    
    Args:
        features_data: feature matrix (n_samples, n_features)
        eps: radius for neighborhood queries
        chunk_size: size of chunks to process at once
        
    Returns:
        Sparse distance matrix where only neighbors within eps are stored
    """
    n_samples = features_data.shape[0]
    print(f"Computing sparse neighborhoods for {n_samples} samples in chunks of {chunk_size}...")
    
    # Initialize lists to store sparse matrix components
    row_indices = []
    col_indices = []
    distances = []
    
    # Process data in chunks
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_data = features_data[start_idx:end_idx]
        
        # Create NearestNeighbors for this chunk
        nn = NearestNeighbors(radius=eps, metric='euclidean')
        nn.fit(features_data)  # Fit on full dataset
        
        # Get neighbors within radius for this chunk
        distances_chunk, indices_chunk = nn.radius_neighbors(chunk_data, return_distance=True)
        
        # Convert to sparse matrix format
        for i, (dists, idxs) in enumerate(zip(distances_chunk, indices_chunk)):
            row_idx = start_idx + i
            for dist, col_idx in zip(dists, idxs):
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                distances.append(dist)
        
        print(f"Processed chunk {start_idx//chunk_size + 1}/{(n_samples + chunk_size - 1)//chunk_size}")
    
    # Create sparse distance matrix
    distance_matrix = csr_matrix(
        (distances, (row_indices, col_indices)),
        shape=(n_samples, n_samples)
    )
    
    print(f"Created sparse distance matrix with {len(distances)} non-zero entries")
    return distance_matrix


def find_anomalies_with_dbscan(
    features_data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 15,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use DBSCAN to find anomalies using metadata features only.

    Args:
        features_data: feature matrix
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        chunk_size: size of chunks for neighborhood computation (reduces memory usage)

    Returns:
        - cluster_labels: cluster assignment for each point (-1 = noise/anomaly)
        - anomaly_indices: sorted by distance from nearest core point
    """
    print(f"Running memory-optimized DBSCAN with eps={eps}, min_samples={min_samples}, chunk_size={chunk_size}...")
    
    # Pre-compute sparse neighborhoods in chunks to reduce memory complexity
    distance_matrix = compute_sparse_neighborhoods_chunked(features_data, eps, chunk_size)
    
    # Use precomputed distance matrix with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)
    
    core_indices = dbscan.core_sample_indices_
    noise_indices = np.where(cluster_labels == -1)[0]

    sorted_noise_indices, _scores = sort_noise_points_by_distance(
        features_data, noise_indices, core_indices
    )
    return cluster_labels, sorted_noise_indices


def sort_noise_points_by_distance(features_data: np.ndarray, noise_indices: np.ndarray, core_indices: np.ndarray) -> np.ndarray:
    """
    Sort noise points by their minimum distance to any core point.
    
    Args:
        features_data: Full feature matrix (n_samples, n_features)
        noise_indices: Indices of noise points in the full dataset
        core_indices: Indices of core points in the full dataset
        
    Returns:
        Array of noise point indices sorted by minimum distance to core points
        (most anomalous/remote points first)
    """
    noise_indices = np.asarray(noise_indices, dtype=int)
    core_indices = np.asarray(core_indices, dtype=int)

    if noise_indices.size == 0 or core_indices.size == 0:
        return noise_indices, np.array([], dtype=float)

    core_points = features_data[core_indices]
    noise_points = features_data[noise_indices]

    dists = np.linalg.norm(noise_points[:, None, :] - core_points[None, :, :], axis=2)
    min_dists = dists.min(axis=1)

    order = np.argsort(min_dists)[::-1]  
    return noise_indices[order], min_dists[order]


def main():
    parser = argparse.ArgumentParser(description="Find anomalous Amazon reviews using metadata features with memory-optimized DBSCAN")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="dbscan_anomalies.csv", help="Output CSV file")
    parser.add_argument("--num-samples", type=int, default=None, help="Max number of samples to process")
    parser.add_argument("--eps", type=float, default=0.6, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples parameter")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for memory-optimized neighborhood computation")
    
    args = parser.parse_args()
    
    df = load_data_from_csv(args.csv_path, max_rows=args.num_samples)
    
    features_data = build_features_data(df)
    
    cluster_labels, anomaly_indices = find_anomalies_with_dbscan(
        features_data, 
        eps=args.eps, 
        min_samples=args.min_samples,
    )
    
    anomaly_df = df.iloc[anomaly_indices].copy()
    
    anomaly_df['original_index'] = anomaly_indices
    anomaly_df['cluster'] = cluster_labels[anomaly_indices]
    
    results_df = anomaly_df
    results_df.to_csv(args.out, index=False)
    print(f"Saved {len(anomaly_df)} anomalies to {args.out}")
    
    n_noise = np.sum(cluster_labels == -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"\nDBSCAN Results:")
    print(f"  Total points: {len(cluster_labels)}")
    print(f"  Noise points (anomalies): {n_noise}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Features used: rating, helpful_votes, verified_purchase, has_images, token_count, sentiment, rating_mismatch")

if __name__ == "__main__":
    main()
