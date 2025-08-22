import os
import numpy as np
import pandas as pd
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
