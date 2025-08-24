import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scan_utils import load_data_from_csv, build_features_data, check_outliers_simple


def compute_k_distance_curve(features_data: np.ndarray, k: int) -> np.ndarray:
    """
    Return sorted k-distance values (distance to the k-th nearest neighbor for each point).
    """
    if k < 1:
        raise ValueError("k (min_samples) must be >= 1")
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="minkowski")
    nbrs.fit(features_data)
    distances, _ = nbrs.kneighbors(features_data)
    k_dist = np.sort(distances[:, -1])
    return k_dist


def suggest_eps_from_kdist(k_dist: np.ndarray) -> float:
    """
    Suggest eps using the elbow (knee) if possible; otherwise fall back to a robust percentile.
    """
    try:
        
        x = np.arange(len(k_dist))
        kl = KneeLocator(x, k_dist, curve="convex", direction="increasing")
        if kl.knee is not None:
            return float(k_dist[kl.knee])
    except Exception:
        pass
    return float(np.percentile(k_dist, 95))


def save_k_distance_plot(k_dist: np.ndarray, out_path: str, k: int):
    """
    Save the k-distance plot to out_path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_dist)
    plt.title(f"k-distance graph (sorted) for k={k}")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}-th nearest neighbor")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze_eps_parameters(features_data: np.ndarray, min_samples: int = 15, plot_path: str = "k_distance_plot.png"):
    """
    Analyze and suggest optimal eps parameter for DBSCAN.
    
    Args:
        features_data: Feature matrix for clustering
        min_samples: min_samples parameter for DBSCAN (default: 15)
        plot_path: Path to save the k-distance plot
        
    Returns:
        Suggested eps value
    """
    print(f"Computing k-distance curve with k={min_samples}...")
    k_dist = compute_k_distance_curve(features_data, min_samples)
    
    print(f"Suggesting eps using elbow method...")
    suggested_eps = suggest_eps_from_kdist(k_dist)
    
    print(f"Creating k-distance plot...")
    save_k_distance_plot(k_dist, plot_path, min_samples)
    
    print(f"\nK-distance statistics for k={min_samples}:")
    print(f"  Min distance: {k_dist[0]:.4f}")
    print(f"  Max distance: {k_dist[-1]:.4f}")
    print(f"  25th percentile: {np.percentile(k_dist, 25):.4f}")
    print(f"  50th percentile: {np.percentile(k_dist, 50):.4f}")
    print(f"  75th percentile: {np.percentile(k_dist, 75):.4f}")
    print(f"  90th percentile: {np.percentile(k_dist, 90):.4f}")
    print(f"  95th percentile: {np.percentile(k_dist, 95):.4f}")
    print(f"  99th percentile: {np.percentile(k_dist, 99):.4f}")
    
    print(f"\nSuggested eps: {suggested_eps:.4f}")
    print(f"K-distance plot saved to: {plot_path} for k={min_samples}")
    
    return suggested_eps


def main():
    parser = argparse.ArgumentParser(description="Analyze and suggest optimal eps parameter for DBSCAN")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples parameter")
    parser.add_argument("--plot", default="k_distance_plot.png", help="Output path for k-distance plot")
    
    args = parser.parse_args()
    
    
    df = load_data_from_csv(args.csv_path)
    check_outliers_simple(df)

    df_clean = df.copy()
    df_clean['helpful_vote'] = df_clean['helpful_vote'].clip(upper=df_clean['helpful_vote'].quantile(0.99))
    
    features_data = build_features_data(df_clean)
    
    suggested_eps = analyze_eps_parameters(
        features_data, 
        min_samples=args.min_samples,
        plot_path=args.plot
    )
    
    print(f"\n=== DBSCAN Parameter Selection Results ===")
    print(f"Suggested eps: {suggested_eps:.4f}")
    print(f"Min samples: {args.min_samples}")
    print(f"\nYou can now use these parameters in your DBSCAN analysis:")
    print(f"  --eps {suggested_eps:.4f} --min-samples {args.min_samples}")


if __name__ == "__main__":
    main()
