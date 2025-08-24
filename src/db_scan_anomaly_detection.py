from __future__ import annotations
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm
from scan_utils import load_data_from_csv, build_features_data
import psutil, os
import gc


class DBScanAnomalyDetector:
    def __init__(self,features_data: np.ndarray, eps: float = 0.5, min_samples: int = 15, batch_size: int = None):
        self.eps = eps
        self.min_samples = min_samples
        self.batch_size = batch_size
        self.features_data = features_data

    def scan(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.batch_size is None:
            return self._scan(self.features_data)
        else:
            all_labels = []
            all_anomalies = []
            total_batches = (len(self.features_data) + self.batch_size - 1) // self.batch_size
            
            with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
                for i in range(0, len(self.features_data), self.batch_size):
                    batch = self.features_data[i:i+self.batch_size]
                    batch_cluster_labels, batch_anomaly_indices = self._scan(batch)
                    all_labels.append(batch_cluster_labels)
                    if len(batch_anomaly_indices) > 0:
                        all_anomalies.append(batch_anomaly_indices + i)
                    pbar.update(1)
            
            self.cluster_labels = np.concatenate(all_labels)
            self.anomaly_indices = np.concatenate(all_anomalies)

        return self.cluster_labels, self.anomaly_indices

    # def _scan(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    #     cluster_labels = dbscan.fit_predict(batch)
        
    #     core_indices = dbscan.core_sample_indices_.copy()
    #     noise_indices = np.where(cluster_labels == -1)[0]

    #     sorted_noise_indices, _ = self._sort_noise_points_by_distance(
    #         batch, noise_indices, core_indices
    #     )
        
    #     del dbscan, core_indices, noise_indices
        
    #     return cluster_labels, sorted_noise_indices

    def _scan(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        process = psutil.Process(os.getpid())
        
        print(f"Batch input size: {batch.nbytes / 1024**2:.1f}MB")
        mem_start = process.memory_info().rss / 1024**3
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        
        mem_after_create = process.memory_info().rss / 1024**3
        print(f"After DBSCAN creation: +{mem_after_create-mem_start:.1f}GB")
        
        cluster_labels = dbscan.fit_predict(batch)
        
        mem_after_fit = process.memory_info().rss / 1024**3
        print(f"After fit_predict: +{mem_after_fit-mem_after_create:.1f}GB")

        core_indices = dbscan.core_sample_indices_.copy()
        noise_indices = np.where(cluster_labels == -1)[0]

        sorted_noise_indices, _ = self._sort_noise_points_by_distance(
            batch, noise_indices, core_indices
        )

        del dbscan, core_indices, noise_indices
        gc.collect()

        return cluster_labels, sorted_noise_indices

    def _sort_noise_points_by_distance(self, batch: np.ndarray, noise_indices: np.ndarray, core_indices: np.ndarray) -> np.ndarray:
        """
        Sort noise points by their minimum distance to any core point.
        
        Args:
            batch: Batch of features data
            noise_indices: Indices of noise points in the batch
            core_indices: Indices of core points in the batch

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sorted noise indices and their minimum distances to core points
        """

        noise_indices = np.asarray(noise_indices, dtype=int)
        core_indices = np.asarray(core_indices, dtype=int)

        if noise_indices.size == 0 or core_indices.size == 0:
            return noise_indices, np.array([], dtype=float)

        core_points = batch[core_indices]
        noise_points = batch[noise_indices]

        dists = np.linalg.norm(noise_points[:, None, :] - core_points[None, :, :], axis=2)
        min_dists = dists.min(axis=1)

        order = np.argsort(min_dists)[::-1]  

        del dists, core_points, noise_points
        gc.collect()

        return noise_indices[order], min_dists[order]


def main():
    parser = argparse.ArgumentParser(description="Find anomalous Amazon reviews using metadata features with memory-optimized DBSCAN")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--out", default="dbscan_anomalies.csv", help="Output CSV file")
    parser.add_argument("--num-samples", type=int, default=None, help="Max number of samples to process")
    parser.add_argument("--eps", type=float, default=0.6, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples parameter")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for memory optimization")

    
    args = parser.parse_args()
    
    df = load_data_from_csv(args.csv_path, max_rows=args.num_samples)
    
    features_data = build_features_data(df)
    detector = DBScanAnomalyDetector(features_data, eps=args.eps, min_samples=args.min_samples, batch_size=args.batch_size)
    
    cluster_labels, anomaly_indices = detector.scan()
    
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
