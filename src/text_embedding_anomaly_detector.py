#!/usr/bin/env python3
"""
text_embedding_anomaly_detector.py

Anomaly detection using only text embeddings from Amazon reviews.
- Loads review data from CSV
- Creates text embeddings using SBERT
- Applies DBSCAN clustering
- Identifies anomalies based on distance to cluster centroids

Run:
  python text_embedding_anomaly_detector.py --csv-path data/val/electronics_val.csv --out text_anomalies.csv

Requirements:
  pip install sentence-transformers scikit-learn pandas numpy
"""

import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


def load_data_from_csv(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load data directly from CSV file.
    Assumes the sample dataset format with standardized column names.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    print(f"Loaded {len(df)} reviews from CSV")
    return df


def create_text_embeddings(
    texts: list[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Create text embeddings using SBERT.
    
    Args:
        texts: List of review texts
        model_id: SBERT model to use
        batch_size: Batch size for processing
        
    Returns:
        Embeddings array of shape (n_texts, embedding_dim)
    """
    print(f"Creating text embeddings with {model_id}...")
    model = SentenceTransformer(model_id)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"Created embeddings of shape: {embeddings.shape}")
    return embeddings


def find_anomalies_with_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    n_anomalies: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find anomalies using DBSCAN clustering.
    
    Args:
        X: Embeddings array
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        n_anomalies: Number of anomalies to return
        
    Returns:
        cluster_labels: Cluster assignment for each point (-1 = noise/anomaly)
        distances: Distance to nearest core point for each point
        anomaly_indices: Indices of top anomalies
    """
    print(f"Clustering {len(X)} points with DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    # Reduce dimensionality first
    pca = PCA(n_components=min(100, X.shape[1]))
    X_reduced = pca.fit_transform(X)
    print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_reduced)
    
    # Calculate distances to nearest core point
    distances = np.full(len(X_reduced), np.inf)
    core_indices = np.where(cluster_labels != -1)[0]
    
    if len(core_indices) > 0:
        core_points = X_reduced[core_indices]
        for i in range(len(X_reduced)):
            if cluster_labels[i] == -1:  # noise point
                # Distance to nearest core point
                dists_to_cores = np.linalg.norm(X_reduced[i] - core_points, axis=1)
                distances[i] = np.min(dists_to_cores)
            else:
                # Distance to cluster center
                cluster_center = np.mean(X_reduced[cluster_labels == cluster_labels[i]], axis=0)
                distances[i] = np.linalg.norm(X_reduced[i] - cluster_center)
    
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
    parser = argparse.ArgumentParser(description="Find anomalous Amazon reviews using text embeddings only")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="text_anomalies.csv", help="Output CSV file")
    parser.add_argument("--num-samples", type=int, default=20000, help="Max number of samples to process")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model to use")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples parameter")
    parser.add_argument("--n-anomalies", type=int, default=100, help="Number of anomalies to return")
    
    args = parser.parse_args()
    
    # Load data
    df = load_data_from_csv(args.csv_path, max_rows=args.num_samples)
    
    # Create text embeddings
    texts = df['text'].tolist()
    embeddings = create_text_embeddings(texts, model_id=args.model)
    
    # Find anomalies with DBSCAN
    cluster_labels, distances, anomaly_indices = find_anomalies_with_dbscan(
        embeddings,
        eps=args.eps,
        min_samples=args.min_samples,
        n_anomalies=args.n_anomalies
    )
    
    # Create results dataframe
    results = []
    for idx in anomaly_indices:
        results.append({
            "index": idx,
            "text": df.iloc[idx]['text'][:200],  # truncate for readability
            "rating": df.iloc[idx]['rating'],
            "cluster": cluster_labels[idx],
            "distance": distances[idx],
            "verified": df.iloc[idx]['verified_purchase'],
            "helpful_votes": df.iloc[idx]['helpful_vote'],
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out, index=False)
    print(f"Saved {len(results)} anomalies to {args.out}")
    
    # Print summary
    n_noise = np.sum(cluster_labels == -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"\nDBSCAN Results (Text Embeddings Only):")
    print(f"  Total points: {len(cluster_labels)}")
    print(f"  Noise points (anomalies): {n_noise}")
    print(f"  Clusters found: {n_clusters}")
    
    print(f"\nTop 5 anomalies:")
    for i, row in results_df.head().iterrows():
        cluster_type = "NOISE" if row['cluster'] == -1 else f"Cluster {row['cluster']}"
        print(f"{i+1}. Rating: {row['rating']}, {cluster_type}, Distance: {row['distance']:.3f}")
        print(f"   Text: {row['text']}...")
        print()


if __name__ == "__main__":
    main()
