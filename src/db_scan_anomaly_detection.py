from __future__ import annotations
import argparse
import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def load_data_from_csv(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load data directly from CSV file.
    Assumes the sample dataset format with standardized column names.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    print(f"Loaded {len(df)} reviews from CSV")
    return df


def build_features_data(df: pd.DataFrame) -> np.ndarray:
    """
    Build features from data    
    Args:
        df: DataFrame with review data (must include 'sentiment' and 'rating_mismatch' columns)
        
    Returns:
        Feature array for clustering
    """
    print("Building features from pre-calculated data...")
    
    features = {
        "rating": df['rating'].values,
        "helpful_votes": df['helpful_vote'].values,  
        "verified_purchase": df['verified_purchase'].astype(int).values,
        "has_images": df['has_images'].astype(int).values,
        "token_count": df['token_count'].values,
        "sentiment": df['sentiment'].values,  
        "rating_mismatch": df['rating_mismatch'].values
    }
    
    feature_matrix = np.column_stack([
        features["rating"],
        features["helpful_votes"], 
        features["verified_purchase"],
        features["has_images"],
        features["token_count"],
        features["sentiment"],
        features["rating_mismatch"]
    ])
    
    scaler = StandardScaler()
    features_data = scaler.fit_transform(feature_matrix)
    
    print(f"Created features of shape: {features_data.shape}")
    print(f"Features: rating, helpful_votes, verified_purchase, has_images, token_count, sentiment, rating_mismatch")
    
    return features_data


def find_anomalies_with_dbscan(
    features_data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use DBSCAN to find anomalies using metadata features only.

    Args:
        features_data: feature matrix
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        - cluster_labels: cluster assignment for each point (-1 = noise/anomaly)
        - anomaly_indices: sorted by distance from nearest core point
    """
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_data)
    
    core_indices = dbscan.core_sample_indices_
    noise_indices = np.where(cluster_labels == -1)[0]

    anomaly_indices = sort_noise_points_by_distance(features_data, noise_indices, core_indices)

    return cluster_labels, anomaly_indices


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
    if len(core_indices) == 0 or len(noise_indices) == 0:
        return np.array([])
    
    core_points = features_data[core_indices]
    noise_points = features_data[noise_indices]
    
    distances = np.zeros(len(noise_points))
    
    for i, noise_point in enumerate(noise_points):
        dists_to_cores = np.linalg.norm(noise_point - core_points, axis=1)
        distances[i] = np.min(dists_to_cores)
    
    sorted_indices = np.argsort(distances)[::-1]
    
    return noise_indices[sorted_indices]


def main():
    parser = argparse.ArgumentParser(description="Find anomalous Amazon reviews using metadata features with DBSCAN")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="dbscan_anomalies.csv", help="Output CSV file")
    parser.add_argument("--num-samples", type=int, default=20000, help="Max number of samples to process")
    parser.add_argument("--eps", type=float, default=0.6, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples parameter")
    
    args = parser.parse_args()
    
    # Load data
    df = load_data_from_csv(args.csv_path, max_rows=args.num_samples)
    
    # Build features
    features_data = build_features_data(df)
    
    # Find anomalies
    cluster_labels, anomaly_indices = find_anomalies_with_dbscan(
        features_data, 
        eps=args.eps, 
        min_samples=args.min_samples,
    )
    
    results = []
    for idx in anomaly_indices:
        results.append({
            "index": idx,
            "text": df.iloc[idx]['text'],
            "rating": df.iloc[idx]['rating'],
            "cluster": cluster_labels[idx],
            "sentiment": df.iloc[idx]['sentiment'],
            "rating_mismatch": df.iloc[idx]['rating_mismatch'],
            "verified": df.iloc[idx]['verified_purchase'],
            "helpful_votes": df.iloc[idx]['helpful_vote'],
            "has_images": df.iloc[idx]['has_images'],
            "token_count": df.iloc[idx]['token_count'],
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)
    print(f"Saved {len(results)} anomalies to {args.out}")
    
    n_noise = np.sum(cluster_labels == -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"\nDBSCAN Results:")
    print(f"  Total points: {len(cluster_labels)}")
    print(f"  Noise points (anomalies): {n_noise}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Features used: rating, helpful_votes, verified_purchase, has_images, token_count, sentiment, rating_mismatch")
    
    print(f"\nTop 5 anomalies:")
    for i, row in results_df.head().iterrows():
        cluster_type = "NOISE" if row['cluster'] == -1 else f"Cluster {row['cluster']}"
        print(f"{i+1}. Rating: {row['rating']}, {cluster_type}")
        print(f"   Sentiment: {row['sentiment']:.3f}, Mismatch: {row['rating_mismatch']:.3f}")
        print(f"   Helpful: {row['helpful_votes']}, Verified: {row['verified']}, Images: {row['has_images']}")
        print(f"   Text: {row['text']}...")
        print()


if __name__ == "__main__":
    main()
