import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
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
    try:
        x = np.arange(len(k_dist))
        kl = KneeLocator(x, k_dist, curve="convex", direction="increasing")
        if kl.knee is not None:
            print(f"Knee found at index {kl.knee}, eps = {k_dist[kl.knee]:.4f}")
            return float(k_dist[kl.knee])
        else:
            print("No knee detected by KneeLocator")
    except Exception as e:
        print(f"KneeLocator failed: {e}")
    
    percentile_eps = float(np.percentile(k_dist, 95))
    print(f"Falling back to 95th percentile: {percentile_eps:.4f}")
    return percentile_eps


def save_k_distance_plot(k_dist: np.ndarray, out_path: str, k: int, category: str = None):
    """
    Save the k-distance plot to out_path.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_dist)
    
    # Create title with category if provided
    if category:
        title = f"k-distance graph (sorted) for k={k} - {category}"
    else:
        title = f"k-distance graph (sorted) for k={k}"
    
    plt.title(title)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}-th nearest neighbor")
    
    # Add vertical grid lines every 0.1 units on the y-axis for easier eps selection
    y_min, y_max = plt.ylim()
    grid_lines = np.arange(0, y_max + 0.1, 0.1)
    for line in grid_lines:
        if line <= y_max:
            plt.axhline(y=line, color='lightgray', linestyle='--', alpha=0.7, linewidth=0.5)
    
    # Add major grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def analyze_eps_parameters(features_data: np.ndarray, min_samples: int = 15, plot_path: str = "k_distance_plot.png", category: str = None):
    """
    Analyze and suggest optimal eps parameter for DBSCAN.
    
    Args:
        features_data: Feature matrix for clustering
        min_samples: min_samples parameter for DBSCAN (default: 15)
        plot_path: Path to save the k-distance plot
        category: Category name to include in plot title (optional)
        
    Returns:
        Suggested eps value
    """
    print(f"Computing k-distance curve with k={min_samples}...")
    k_dist = compute_k_distance_curve(features_data, min_samples)
    
    print(f"Suggesting eps using elbow method...")
    suggested_eps = suggest_eps_from_kdist(k_dist)
    
    print(f"Creating k-distance plot...")
    save_k_distance_plot(k_dist, plot_path, min_samples, category)
    
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


def extract_category_from_path(csv_path: str) -> str:
    """
    Extract category name from CSV file path.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Category name extracted from the file path
    """
    # Get the filename without path and extension
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Remove common suffixes like '_test', '_train', '_val'
    suffixes_to_remove = ['_test', '_train', '_val']
    category = filename
    for suffix in suffixes_to_remove:
        if category.endswith(suffix):
            category = category[:-len(suffix)]
            break
    
    # Convert underscores to spaces and capitalize words for display
    category_display = category.replace('_', ' ').title()
    
    return category_display


def main():
    parser = argparse.ArgumentParser(description="Analyze and suggest optimal eps parameter for DBSCAN")
    parser.add_argument("--csv-path", required=True, help="Path to CSV file")
    parser.add_argument("--min-samples", type=int, default=16, help="DBSCAN min_samples parameter")
    parser.add_argument("--plot", default="k_distance_plot.png", help="Output path for k-distance plot")
    
    args = parser.parse_args()
    
    # Extract category name from CSV path
    category = extract_category_from_path(args.csv_path)
    print(f"Detected category: {category}")
    
    df = load_data_from_csv(args.csv_path)
    check_outliers_simple(df)

    df_clean = df.copy()
    df_clean['helpful_vote'] = df_clean['helpful_vote'].clip(upper=df_clean['helpful_vote'].quantile(0.99))
    df_clean['reviewer_review_count'] = df_clean['reviewer_review_count'].clip(upper=df_clean['reviewer_review_count'].quantile(0.99))
    
    features_data = build_features_data(df_clean)
    
    suggested_eps = analyze_eps_parameters(
        features_data, 
        min_samples=args.min_samples,
        plot_path=args.plot,
        category=category
    )
    
    print(f"\n=== DBSCAN Parameter Selection Results ===")
    print(f"Suggested eps: {suggested_eps:.4f}")
    print(f"Min samples: {args.min_samples}")
    print(f"\nYou can now use these parameters in your DBSCAN analysis:")
    print(f"  --eps {suggested_eps:.4f} --min-samples {args.min_samples}")


if __name__ == "__main__":
    main()
