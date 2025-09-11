import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data_from_csv(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load data directly from CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df=df.dropna()
    
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    print(f"Loaded {len(df)} reviews from CSV")
    return df


def build_features_data(df: pd.DataFrame) -> np.ndarray:
    """
    Build features from data for anomaly detection.
    Args:
        df: DataFrame with review data
        
    Returns:
        Feature array for anomaly detection
    """
    print("Building features from data...")
    
    features = {
        "helpful_votes": df['helpful_vote'].values,  
        "verified_purchase": df['verified_purchase'].astype(int).values,
        "has_images": df['has_images'].astype(int).values,
        "rating_diff": df['rating_diff'].values,
        "reviewer_review_count": df['reviewer_review_count'].values,
        "rating_vs_product_avg_abs": df['rating_vs_product_avg_abs'].values
    }
    
    feature_matrix = np.column_stack([
        features["helpful_votes"], 
        features["verified_purchase"],
        features["has_images"],
        features["rating_diff"],
        features["reviewer_review_count"],
        features["rating_vs_product_avg_abs"]
    ])
    
    scaler = StandardScaler()
    features_data = scaler.fit_transform(feature_matrix)
    
    print(f"Created features of shape: {features_data.shape}")
    print(f"Features: helpful_votes, verified_purchase, has_images, rating_diff, reviewer_review_count, rating_vs_product_avg_abs")
    
    return features_data


def check_outliers_simple(df: pd.DataFrame):
    """Simple check for extreme outliers that might be causing the k-distance spike."""
    print("=== CHECKING FOR EXTREME OUTLIERS ===\n")
    
    # Check the features most likely to have extreme outliers
    features_to_check = ['helpful_vote', 'reviewer_review_count']
    
    for feature in features_to_check:
        if feature not in df.columns:
            print(f"Feature {feature} not found")
            continue
            
        data = df[feature]
        
        print(f"{feature.upper()}:")
        print(f"  Min: {data.min()}")
        print(f"  Max: {data.max()}")
        print(f"  95th percentile: {data.quantile(0.95):.2f}")
        print(f"  99th percentile: {data.quantile(0.99):.2f}")
        print(f"  99.9th percentile: {data.quantile(0.999):.2f}")
        
        ratio = data.max() / (data.quantile(0.95) + 1e-8)
        print(f"  Max/95th ratio: {ratio:.1f}x")
        
        if ratio > 10:
            print(f"   PROBLEM: Extreme outliers detected!")
            print(f"     Top 5 values: {sorted(data.nlargest(5).tolist(), reverse=True)}")
        else:
            print(f"   Outliers look reasonable")
        print()

