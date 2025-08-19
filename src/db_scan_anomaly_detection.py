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


# compute_sentiment_scores function removed - sentiment scores are now pre-calculated during data creation


def build_features_data(df: pd.DataFrame) -> np.ndarray:
    """
    Build features from data    
    Args:
        df: DataFrame with review data (must include 'sentiment' and 'rating_mismatch' columns)
        
    Returns:
        Feature array for clustering
    """
    print("Building features from pre-calculated data...")
    
    # Use pre-calculated sentiment scores and rating mismatch from CSV
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
    min_samples: int = 5,
    n_anomalies: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use DBSCAN to find anomalies using metadata features only.
    
    Returns:
        - cluster_labels: cluster assignment for each point (-1 = noise/anomaly)
        - distances: distance to nearest core point for each point
        - anomaly_indices: indices of top anomalies
    """
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    
    # Run DBSCAN directly on the 7-dimensional feature space
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_data)
    
    # Calculate distances to nearest core point
    distances = np.full(len(features_data), np.inf)
    core_indices = np.where(cluster_labels != -1)[0]
    
    if len(core_indices) > 0:
        core_points = features_data[core_indices]
        for i in range(len(features_data)):
            if cluster_labels[i] == -1:  # noise point
                # Distance to nearest core point
                dists_to_cores = np.linalg.norm(features_data[i] - core_points, axis=1)
                distances[i] = np.min(dists_to_cores)
            else:
                # Distance to cluster center
                cluster_center = np.mean(features_data[cluster_labels == cluster_labels[i]], axis=0)
                distances[i] = np.linalg.norm(features_data[i] - cluster_center)
    
    # Get top anomalies (noise points first, then by distance)
    noise_indices = np.where(cluster_labels == -1)[0]
    other_indices = np.where(cluster_labels != -1)[0]
    
    # Sort noise points by distance, then other points by distance
    if len(noise_indices) > 0:
        noise_sorted = noise_indices[np.argsort(distances[noise_indices])]
    else:
        noise_sorted = np.array([])
    
    if len(other_indices) > 0:
        other_sorted = other_indices[np.argsort(distances[other_indices])]
    else:
        other_sorted = np.array([])
    
    # Combine: noise points first, then others
    anomaly_indices = np.concatenate([noise_sorted, other_sorted])[:n_anomalies]
    
    return cluster_labels, distances, anomaly_indices


def main():
    parser = argparse.ArgumentParser(description="Find anomalous Amazon reviews using metadata features with DBSCAN")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="dbscan_anomalies.csv", help="Output CSV file")
    parser.add_argument("--num-samples", type=int, default=20000, help="Max number of samples to process")
    parser.add_argument("--eps", type=float, default=0.6, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples parameter")
    parser.add_argument("--n-anomalies", type=int, default=100, help="Number of anomalies to return")
    
    args = parser.parse_args()
    
    # Load data
    df = load_data_from_csv(args.csv_path, max_rows=args.num_samples)
    
    # Build features
    features_data = build_features_data(df)
    
    # Find anomalies
    cluster_labels, distances, anomaly_indices = find_anomalies_with_dbscan(
        features_data, 
        eps=args.eps, 
        min_samples=args.min_samples,
        n_anomalies=args.n_anomalies
    )
    
    # Create results dataframe
    results = []
    for idx in anomaly_indices:
        # Use pre-calculated sentiment score from CSV
        sentiment_score = df.iloc[idx]['sentiment']
        rating_mismatch = df.iloc[idx]['rating_mismatch']
        
        results.append({
            "index": idx,
            "text": df.iloc[idx]['text'],
            "rating": df.iloc[idx]['rating'],
            "cluster": cluster_labels[idx],
            "distance": distances[idx],
            "sentiment": sentiment_score,
            "rating_mismatch": rating_mismatch,
            "verified": df.iloc[idx]['verified_purchase'],
            "helpful_votes": df.iloc[idx]['helpful_vote'],
            "has_images": df.iloc[idx]['has_images'],
            "token_count": df.iloc[idx]['token_count'],
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)
    print(f"Saved {len(results)} anomalies to {args.out}")
    
    # Print summary
    n_noise = np.sum(cluster_labels == -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"\nDBSCAN Results (Metadata + Sentiment):")
    print(f"  Total points: {len(cluster_labels)}")
    print(f"  Noise points (anomalies): {n_noise}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Features used: rating, helpful_votes, verified_purchase, has_images, token_count, sentiment, rating_mismatch")
    
    print(f"\nTop 5 anomalies:")
    for i, row in results_df.head().iterrows():
        cluster_type = "NOISE" if row['cluster'] == -1 else f"Cluster {row['cluster']}"
        print(f"{i+1}. Rating: {row['rating']}, {cluster_type}, Distance: {row['distance']:.3f}")
        print(f"   Sentiment: {row['sentiment']:.3f}, Mismatch: {row['rating_mismatch']:.3f}")
        print(f"   Helpful: {row['helpful_votes']}, Verified: {row['verified']}, Images: {row['has_images']}")
        print(f"   Text: {row['text']}...")
        print()


if __name__ == "__main__":
    main()
