import pandas as pd
import numpy as np
import math
import sys
import os
from typing import List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "model")
sys.path.insert(0, model_dir)

from model.model_utils import BARTRegressionPredictor


class AnomalyDetector:
    """
    Detects anomalous reviews using multiple scoring criteria.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_path: Path to the BART regression model
        """
        if model_path is None:
            model_path = os.path.join(current_dir, "model", "bart_regression_star")
        
        self.model_path = model_path
        self.predictor = None
        
    def load_model(self):
        """Load the BART regression model."""
        print("Loading BART regression model...")
        self.predictor = BARTRegressionPredictor(self.model_path)
        self.predictor.load_model()
        print("✓ Model loaded successfully!")
        
    def calculate_rating_gap(self, predicted_ratings: List[float], actual_ratings: List[float]) -> List[float]:
        """
        Calculate rating gap score.
        
        Args:
            predicted_ratings: List of predicted ratings
            actual_ratings: List of actual ratings
            
        Returns:
            List of rating gap scores
        """
        return [(pred - actual) / 4.0 for pred, actual in zip(predicted_ratings, actual_ratings)]
    
    def calculate_no_image_score(self, has_images: List[bool]) -> List[float]:
        """
        Calculate no image score.
        
        Args:
            has_images: List of boolean values indicating if review has images
            
        Returns:
            List of no image scores (1 - has_image)
        """
        return [1.0 - float(has_img) for has_img in has_images]
    
    def calculate_helpfulness_score(self, helpful_votes: List[int], max_votes: int) -> List[float]:
        """
        Calculate helpfulness score using the formula:
        helpfulness_score = 1 - log(helpful_votes + 1) / log(max_votes + 1)
        where max_votes is the maximum helpful votes in the current CSV file.
        
        Args:
            helpful_votes: List of helpful vote counts
            max_votes: Maximum helpful votes in the current dataset
            
        Returns:
            List of helpfulness scores
        """
        scores = []
        
        log_max_votes = math.log(max_votes + 1)
        
        for hv in helpful_votes:
            log_hv = math.log(hv + 1)
            if log_max_votes == 0:
                scores.append(1.0)
            else:
                score = 1.0 - (log_hv / log_max_votes)
                scores.append(max(0.0, min(1.0, score)))
        return scores
    
    def calculate_verified_score(self, verified_purchases: List[bool]) -> List[float]:
        """
        Calculate verified purchase score.
        
        Args:
            verified_purchases: List of boolean values indicating verified purchase
            
        Returns:
            List of verified scores (1 if not verified, 0 if verified)
        """
        return [0.0 if verified else 1.0 for verified in verified_purchases]
    
    def calculate_weighted_score(self, 
                                rating_gap: List[float],
                                no_image_score: List[float], 
                                helpfulness_score: List[float],
                                verified_score: List[float]) -> List[float]:
        """
        Calculate weighted anomaly score using the specified weights.
        
        Args:
            rating_gap: Rating gap scores
            no_image_score: No image scores
            helpfulness_score: Helpfulness scores
            verified_score: Verified purchase scores
            
        Returns:
            List of weighted anomaly scores
        """
        weighted_scores = []
        
        for rg, ni, hs, vs in zip(rating_gap, no_image_score, helpfulness_score, verified_score):
            weighted_score = (
                0.50 * abs(rg) + 
                0.10 * ni +
                0.25 * hs +
                0.15 * vs
            )
            weighted_scores.append(weighted_score)
        
        return weighted_scores
    
    def process_dataset(self, file: str, batch_size: int = 8) -> pd.DataFrame:
        """
        Process the entire dataset and calculate anomaly scores.
        
        Args:
            file: Path to CSV file
            batch_size: Batch size for model predictions
            
        Returns:
            DataFrame with anomaly scores added
        """
   
        print(f"Loading dataset from {file}...")
        df = pd.read_csv(file)
        print(f"✓ Loaded {len(df)} reviews")
        
        reviews_data = df['review_data'].tolist()
        actual_ratings = df['rating'].tolist()
        has_images = df['has_images'].tolist()
        helpful_votes = df['helpful_vote'].tolist()
        verified_purchases = df['verified_purchase'].tolist()
        max_votes_in_file = df['helpful_vote'].max()

        predicted_ratings = self.predictor.predict_batch(
            reviews_data, batch_size=batch_size
        )

        rating_gaps = self.calculate_rating_gap(predicted_ratings, actual_ratings)
        no_image_scores = self.calculate_no_image_score(has_images)
        helpfulness_scores = self.calculate_helpfulness_score(helpful_votes, max_votes_in_file)
        verified_scores = self.calculate_verified_score(verified_purchases)
        weighted_scores = self.calculate_weighted_score(
            rating_gaps, no_image_scores, helpfulness_scores, verified_scores
        )
        
        result_df = df.copy()
        result_df['predicted_rating'] = predicted_ratings
        result_df['rating_gap'] = rating_gaps
        result_df['no_image_score'] = no_image_scores
        result_df['helpfulness_score'] = helpfulness_scores
        result_df['verified_score'] = verified_scores
        result_df['anomaly_score'] = weighted_scores
        result_df['max_votes_in_file'] = max_votes_in_file
        
        print("✓ Anomaly scores calculated successfully!")
        result_df.to_csv(file, index=False)
        self.print_summary(result_df, max_votes_in_file)
        return result_df
    
    def print_summary(self, df: pd.DataFrame, max_votes_in_file: int):
        """Print summary statistics of the anomaly detection results."""
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        
        print(f"Total reviews processed: {len(df):,}")
        print(f"Maximum helpful votes in this file: {max_votes_in_file}")
        print(f"\nAnomaly Score Statistics:")
        print(f"  Mean: {df['anomaly_score'].mean():.4f}")
        print(f"  Std:  {df['anomaly_score'].std():.4f}")
        print(f"  Min:  {df['anomaly_score'].min():.4f}")
        print(f"  Max:  {df['anomaly_score'].max():.4f}")
        
        # Top anomalies
        print(f"\nTop 10 Most Anomalous Reviews:")
        top_anomalies = df.nlargest(10, 'anomaly_score')[
            ['anomaly_score', 'rating', 'predicted_rating', 'rating_gap', 'helpful_vote', 'title', 'category']
        ]
        
        for idx, row in top_anomalies.iterrows():
            print(f"  Score: {row['anomaly_score']:.4f} | "
                  f"Rating: {row['rating']:.1f} → {row['predicted_rating']:.2f} | "
                  f"Gap: {row['rating_gap']:.3f} | "
                  f"Helpful: {row['helpful_vote']} | "
                  f"Cat: {row['category']} | "
                  f"Title: {row['title'][:30]}...")
        
        # Score distribution
        percentiles = [50, 75, 90, 95, 99]
        print(f"\nAnomaly Score Percentiles:")
        for p in percentiles:
            value = np.percentile(df['anomaly_score'], p)
            print(f"  {p}th percentile: {value:.4f}")


def main():
    """Main function to run anomaly detection."""
    # File paths
    file = os.path.join(current_dir, "..", "data", "test", "combined_test_dataset.csv")
    
    # Initialize anomaly detector
    detector = AnomalyDetector()
    
    try:
        # Load model
        detector.load_model()
        
        # Process dataset
        result_df = detector.process_dataset(
            file=file,
            batch_size=8
        )
        
    except Exception as e:
        print(f"❌ Error during anomaly detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
