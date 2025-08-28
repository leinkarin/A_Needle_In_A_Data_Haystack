import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionEvaluator:
    """
    Comprehensive evaluation framework for DBSCAN anomaly detection results.
    """
    
    def __init__(self, results_file: str, original_data_file: str = None):
        """
        Initialize the evaluator with DBSCAN results.
        
        Args:
            results_file: Path to DBSCAN anomaly detection results CSV
            original_data_file: Path to original dataset (optional, for comparison)
        """
        self.results_file = results_file
        self.original_data_file = original_data_file
        self.anomalies_df = None
        self.original_df = None
        self.features_data = None
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load anomaly results and original data."""
        print(f"Loading anomaly results from: {self.results_file}")
        self.anomalies_df = pd.read_csv(self.results_file)
        print(f"✓ Loaded {len(self.anomalies_df)} anomalies")
        
        if self.original_data_file and os.path.exists(self.original_data_file):
            print(f"Loading original data from: {self.original_data_file}")
            self.original_df = pd.read_csv(self.original_data_file)
            print(f"✓ Loaded {len(self.original_df)} original records")
        else:
            print("⚠️ Original data file not provided or not found")
            
    def evaluate_anomaly_characteristics(self) -> Dict:
        """
        Evaluate the characteristics of detected anomalies.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("ANOMALY CHARACTERISTICS EVALUATION")
        print("="*60)
        
        results = {}
        
        # Basic statistics
        results['total_anomalies'] = len(self.anomalies_df)
        
        if self.original_df is not None:
            results['anomaly_rate'] = len(self.anomalies_df) / len(self.original_df)
            print(f"Anomaly Rate: {results['anomaly_rate']:.4f} ({results['anomaly_rate']*100:.2f}%)")
        
        # Feature analysis
        feature_columns = ['rating', 'helpful_vote', 'verified_purchase', 'has_images', 
                          'predicted_rating', 'rating_diff', 'reviewer_review_count', 
                          'rating_vs_product_avg_abs']
        
        available_features = [col for col in feature_columns if col in self.anomalies_df.columns]
        
        print(f"\nFeature Analysis for {len(available_features)} available features:")
        for feature in available_features:
            if feature in self.anomalies_df.columns:
                mean_val = self.anomalies_df[feature].mean()
                std_val = self.anomalies_df[feature].std()
                min_val = self.anomalies_df[feature].min()
                max_val = self.anomalies_df[feature].max()
                
                print(f"  {feature}:")
                print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")
                
                results[f'{feature}_mean'] = mean_val
                results[f'{feature}_std'] = std_val
                
                # Handle boolean columns differently - they don't have meaningful ranges
                if self.anomalies_df[feature].dtype == bool:
                    results[f'{feature}_range'] = None
                else:
                    results[f'{feature}_range'] = max_val - min_val
        
        # Category distribution (if available)
        if 'category' in self.anomalies_df.columns:
            print(f"\nAnomaly Distribution by Category:")
            category_counts = self.anomalies_df['category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(self.anomalies_df)) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
                results[f'category_{category}_count'] = count
                results[f'category_{category}_percentage'] = percentage
        
        # Rating distribution analysis
        if 'rating' in self.anomalies_df.columns:
            print(f"\nRating Distribution Analysis:")
            rating_counts = self.anomalies_df['rating'].value_counts().sort_index()
            print(f"  Rating distribution:")
            for rating, count in rating_counts.items():
                percentage = (count / len(self.anomalies_df)) * 100
                print(f"    {rating} stars: {count} ({percentage:.1f}%)")
                results[f'rating_{int(rating)}_count'] = count
                results[f'rating_{int(rating)}_percentage'] = percentage
        
        return results
    
    def compare_with_original_data(self) -> Dict:
        """
        Compare anomaly characteristics with original dataset.
        
        Returns:
            Dictionary with comparison metrics
        """
        if self.original_df is None:
            print("⚠️ Cannot compare with original data - file not provided")
            return {}
            
        print("\n" + "="*60)
        print("COMPARISON WITH ORIGINAL DATASET")
        print("="*60)
        
        results = {}
        
        # Feature comparisons
        feature_columns = ['rating', 'helpful_vote', 'verified_purchase', 'has_images', 
                          'predicted_rating', 'rating_diff', 'reviewer_review_count', 
                          'rating_vs_product_avg_abs']
        
        available_features = [col for col in feature_columns if col in self.anomalies_df.columns 
                            and col in self.original_df.columns]
        
        print(f"Feature Comparison Analysis:")
        for feature in available_features:
            anomaly_mean = self.anomalies_df[feature].mean()
            original_mean = self.original_df[feature].mean()
            anomaly_std = self.anomalies_df[feature].std()
            original_std = self.original_df[feature].std()
            
            mean_diff = anomaly_mean - original_mean
            std_ratio = anomaly_std / original_std if original_std > 0 else 0
            
            print(f"  {feature}:")
            print(f"    Anomaly mean: {anomaly_mean:.4f}, Original mean: {original_mean:.4f}")
            print(f"    Mean difference: {mean_diff:.4f}")
            print(f"    Anomaly std: {anomaly_std:.4f}, Original std: {original_std:.4f}")
            print(f"    Std ratio: {std_ratio:.2f}")
            
            results[f'{feature}_mean_diff'] = mean_diff
            results[f'{feature}_std_ratio'] = std_ratio
        
        # Category comparison
        if 'category' in self.anomalies_df.columns and 'category' in self.original_df.columns:
            print(f"\nCategory Distribution Comparison:")
            anomaly_cats = self.anomalies_df['category'].value_counts(normalize=True)
            original_cats = self.original_df['category'].value_counts(normalize=True)
            
            for category in set(anomaly_cats.index) | set(original_cats.index):
                anomaly_pct = anomaly_cats.get(category, 0) * 100
                original_pct = original_cats.get(category, 0) * 100
                diff = anomaly_pct - original_pct
                
                print(f"  {category}:")
                print(f"    Anomaly: {anomaly_pct:.1f}%, Original: {original_pct:.1f}%")
                print(f"    Difference: {diff:+.1f}%")
                
                results[f'category_{category}_pct_diff'] = diff
        
        return results
    
    def evaluate_clustering_quality(self) -> Dict:
        """
        Evaluate the quality of DBSCAN clustering (excluding noise points).
        
        Returns:
            Dictionary with clustering quality metrics
        """
        print("\n" + "="*60)
        print("CLUSTERING QUALITY EVALUATION")
        print("="*60)
        
        results = {}
        
        if 'cluster' not in self.anomalies_df.columns:
            print("⚠️ No cluster information found in results")
            return results
        
        # Get non-noise clusters
        cluster_labels = self.anomalies_df['cluster'].values
        non_noise_mask = cluster_labels != -1
        non_noise_labels = cluster_labels[non_noise_mask]
        
        if len(non_noise_labels) == 0:
            print("⚠️ No non-noise clusters found")
            return results
        
        # Build feature matrix for clustering evaluation
        feature_columns = ['rating', 'helpful_vote', 'verified_purchase', 'has_images', 
                          'predicted_rating', 'rating_diff', 'reviewer_review_count', 
                          'rating_vs_product_avg_abs']
        
        available_features = [col for col in feature_columns if col in self.anomalies_df.columns]
        
        if len(available_features) < 2:
            print("⚠️ Insufficient features for clustering evaluation")
            return results
        
        # Create feature matrix
        feature_data = self.anomalies_df[available_features].values[non_noise_mask]
        
        # Clustering metrics
        n_clusters = len(set(non_noise_labels))
        n_noise = np.sum(cluster_labels == -1)
        n_total = len(cluster_labels)
        
        print(f"Clustering Statistics:")
        print(f"  Total points: {n_total}")
        print(f"  Noise points: {n_noise} ({n_noise/n_total*100:.1f}%)")
        print(f"  Clustered points: {len(non_noise_labels)} ({len(non_noise_labels)/n_total*100:.1f}%)")
        print(f"  Number of clusters: {n_clusters}")
        
        results['total_points'] = n_total
        results['noise_points'] = n_noise
        results['noise_percentage'] = n_noise / n_total
        results['clustered_points'] = len(non_noise_labels)
        results['clustered_percentage'] = len(non_noise_labels) / n_total
        results['n_clusters'] = n_clusters
        
        # Cluster size distribution
        if n_clusters > 0:
            cluster_sizes = [np.sum(non_noise_labels == i) for i in range(n_clusters)]
            print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
            
            results['min_cluster_size'] = min(cluster_sizes)
            results['max_cluster_size'] = max(cluster_sizes)
            results['mean_cluster_size'] = np.mean(cluster_sizes)
            results['cluster_size_std'] = np.std(cluster_sizes)
        
        # Silhouette score (if multiple clusters)
        if n_clusters > 1 and len(non_noise_labels) > 1:
            try:
                silhouette_avg = silhouette_score(feature_data, non_noise_labels)
                print(f"  Silhouette Score: {silhouette_avg:.4f}")
                results['silhouette_score'] = silhouette_avg
            except Exception as e:
                print(f"  Silhouette Score: Could not compute ({e})")
        
        # Calinski-Harabasz score
        if n_clusters > 1:
            try:
                ch_score = calinski_harabasz_score(feature_data, non_noise_labels)
                print(f"  Calinski-Harabasz Score: {ch_score:.4f}")
                results['calinski_harabasz_score'] = ch_score
            except Exception as e:
                print(f"  Calinski-Harabasz Score: Could not compute ({e})")
        
        # Davies-Bouldin score
        if n_clusters > 1:
            try:
                db_score = davies_bouldin_score(feature_data, non_noise_labels)
                print(f"  Davies-Bouldin Score: {db_score:.4f}")
                results['davies_bouldin_score'] = db_score
            except Exception as e:
                print(f"  Davies-Bouldin Score: Could not compute ({e})")
        
        return results
    
    def analyze_anomaly_patterns(self) -> Dict:
        """
        Analyze patterns in the detected anomalies.
        
        Returns:
            Dictionary with pattern analysis results
        """
        print("\n" + "="*60)
        print("ANOMALY PATTERN ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Rating vs Predicted Rating Analysis
        if 'rating' in self.anomalies_df.columns and 'predicted_rating' in self.anomalies_df.columns:
            print("Rating vs Predicted Rating Analysis:")
            
            rating_diff = self.anomalies_df['rating'] - self.anomalies_df['predicted_rating']
            mean_diff = rating_diff.mean()
            std_diff = rating_diff.std()
            
            print(f"  Mean rating difference: {mean_diff:.4f}")
            print(f"  Std rating difference: {std_diff:.4f}")
            
            # Categorize by rating difference
            positive_diff = rating_diff > 0
            negative_diff = rating_diff < 0
            
            print(f"  Positive differences (actual > predicted): {positive_diff.sum()} ({positive_diff.mean()*100:.1f}%)")
            print(f"  Negative differences (actual < predicted): {negative_diff.sum()} ({negative_diff.mean()*100:.1f}%)")
            
            results['mean_rating_diff'] = mean_diff
            results['std_rating_diff'] = std_diff
            results['positive_diff_count'] = positive_diff.sum()
            results['negative_diff_count'] = negative_diff.sum()
            results['positive_diff_percentage'] = positive_diff.mean() * 100
            results['negative_diff_percentage'] = negative_diff.mean() * 100
        
        # Helpful votes analysis
        if 'helpful_vote' in self.anomalies_df.columns:
            print(f"\nHelpful Votes Analysis:")
            helpful_votes = self.anomalies_df['helpful_vote']
            
            zero_votes = (helpful_votes == 0).sum()
            low_votes = ((helpful_votes > 0) & (helpful_votes <= 5)).sum()
            high_votes = (helpful_votes > 5).sum()
            
            print(f"  Zero votes: {zero_votes} ({zero_votes/len(helpful_votes)*100:.1f}%)")
            print(f"  Low votes (1-5): {low_votes} ({low_votes/len(helpful_votes)*100:.1f}%)")
            print(f"  High votes (>5): {high_votes} ({high_votes/len(helpful_votes)*100:.1f}%)")
            
            results['zero_votes_count'] = zero_votes
            results['low_votes_count'] = low_votes
            results['high_votes_count'] = high_votes
            results['zero_votes_percentage'] = zero_votes / len(helpful_votes) * 100
            results['low_votes_percentage'] = low_votes / len(helpful_votes) * 100
            results['high_votes_percentage'] = high_votes / len(helpful_votes) * 100
        
        # Verified purchase analysis
        if 'verified_purchase' in self.anomalies_df.columns:
            print(f"\nVerified Purchase Analysis:")
            verified = self.anomalies_df['verified_purchase']
            
            verified_count = verified.sum()
            unverified_count = len(verified) - verified_count
            
            print(f"  Verified purchases: {verified_count} ({verified_count/len(verified)*100:.1f}%)")
            print(f"  Unverified purchases: {unverified_count} ({unverified_count/len(verified)*100:.1f}%)")
            
            results['verified_count'] = verified_count
            results['unverified_count'] = unverified_count
            results['verified_percentage'] = verified_count / len(verified) * 100
            results['unverified_percentage'] = unverified_count / len(verified) * 100
        
        # Has images analysis
        if 'has_images' in self.anomalies_df.columns:
            print(f"\nImages Analysis:")
            has_images = self.anomalies_df['has_images']
            
            with_images = has_images.sum()
            without_images = len(has_images) - with_images
            
            print(f"  Reviews with images: {with_images} ({with_images/len(has_images)*100:.1f}%)")
            print(f"  Reviews without images: {without_images} ({without_images/len(has_images)*100:.1f}%)")
            
            results['with_images_count'] = with_images
            results['without_images_count'] = without_images
            results['with_images_percentage'] = with_images / len(has_images) * 100
            results['without_images_percentage'] = without_images / len(has_images) * 100
        
        return results
    
    def create_visualizations(self, output_dir: str = "evaluation_plots"):
        """
        Create visualizations for anomaly detection results.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating visualizations in: {output_dir}")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Rating Distribution
        if 'rating' in self.anomalies_df.columns:
            plt.figure(figsize=(10, 6))
            rating_counts = self.anomalies_df['rating'].value_counts().sort_index()
            plt.bar(rating_counts.index, rating_counts.values, alpha=0.7)
            plt.title('Anomaly Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.xticks(range(1, 6))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'anomaly_rating_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Rating vs Predicted Rating Scatter
        if 'rating' in self.anomalies_df.columns and 'predicted_rating' in self.anomalies_df.columns:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.anomalies_df['predicted_rating'], self.anomalies_df['rating'], 
                       alpha=0.6, s=30)
            plt.plot([1, 5], [1, 5], 'r--', alpha=0.8, label='Perfect Prediction')
            plt.xlabel('Predicted Rating')
            plt.ylabel('Actual Rating')
            plt.title('Anomaly: Actual vs Predicted Ratings')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'anomaly_rating_vs_predicted.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Individual Feature Distributions
        feature_columns = ['helpful_vote', 'rating_diff', 'reviewer_review_count']
        available_features = [col for col in feature_columns if col in self.anomalies_df.columns]
        
        for feature in available_features:
            plt.figure(figsize=(10, 6))
            plt.hist(self.anomalies_df[feature], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Anomaly {feature.replace("_", " ").title()} Distribution')
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel('Count')
            
            # Use log scale for helpful_vote distribution
            if feature == 'helpful_vote':
                plt.yscale('log')
                plt.ylabel('Count (log scale)')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'anomaly_{feature}_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Cluster Analysis (if available)
        if 'cluster' in self.anomalies_df.columns:
            cluster_labels = self.anomalies_df['cluster'].values
            non_noise_mask = cluster_labels != -1
            
            if non_noise_mask.sum() > 0:
                # PCA for visualization
                feature_columns = ['rating', 'helpful_vote', 'verified_purchase', 'has_images', 
                                  'predicted_rating', 'rating_diff', 'reviewer_review_count', 
                                  'rating_vs_product_avg_abs']
                
                available_features = [col for col in feature_columns if col in self.anomalies_df.columns]
                
                if len(available_features) >= 2:
                    feature_data = self.anomalies_df[available_features].values
                    
                    # PCA
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(feature_data)
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Plot non-noise points
                    non_noise_data = pca_data[non_noise_mask]
                    non_noise_clusters = cluster_labels[non_noise_mask]
                    
                    scatter = plt.scatter(non_noise_data[:, 0], non_noise_data[:, 1], 
                                        c=non_noise_clusters, cmap='tab10', alpha=0.7, s=30)
                    
                    # Plot noise points
                    noise_data = pca_data[~non_noise_mask]
                    if len(noise_data) > 0:
                        plt.scatter(noise_data[:, 0], noise_data[:, 1], 
                                  c='red', marker='x', s=50, alpha=0.8, label='Noise')
                    
                    plt.title('DBSCAN Clustering Results (PCA Visualization)')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    plt.colorbar(scatter, label='Cluster')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'dbscan_clustering_visualization.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        
        print(f"✓ Saved {len(os.listdir(output_dir))} visualization files")
    
    def generate_summary_report(self, output_file: str = "anomaly_evaluation_report.txt"):
        """
        Generate a comprehensive summary report.
        
        Args:
            output_file: Path to save the report
        """
        print(f"\nGenerating summary report: {output_file}")
        
        # Run all evaluations
        anomaly_chars = self.evaluate_anomaly_characteristics()
        clustering_quality = self.evaluate_clustering_quality()
        anomaly_patterns = self.analyze_anomaly_patterns()
        
        comparison_results = {}
        if self.original_df is not None:
            comparison_results = self.compare_with_original_data()
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DBSCAN ANOMALY DETECTION EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Results File: {self.results_file}\n")
            if self.original_data_file:
                f.write(f"Original Data File: {self.original_data_file}\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Anomalies Detected: {anomaly_chars.get('total_anomalies', 'N/A')}\n")
            if 'anomaly_rate' in anomaly_chars:
                f.write(f"Anomaly Rate: {anomaly_chars['anomaly_rate']:.4f} ({anomaly_chars['anomaly_rate']*100:.2f}%)\n")
            f.write(f"Number of Clusters: {clustering_quality.get('n_clusters', 'N/A')}\n")
            f.write(f"Noise Points: {clustering_quality.get('noise_points', 'N/A')} ({clustering_quality.get('noise_percentage', 0)*100:.1f}%)\n\n")
            
            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 40 + "\n")
            
            # Rating analysis
            if 'mean_rating_diff' in anomaly_patterns:
                f.write(f"• Average rating difference: {anomaly_patterns['mean_rating_diff']:.3f}\n")
                f.write(f"• Positive rating differences: {anomaly_patterns['positive_diff_percentage']:.1f}%\n")
                f.write(f"• Negative rating differences: {anomaly_patterns['negative_diff_percentage']:.1f}%\n")
            
            # Helpful votes
            if 'zero_votes_percentage' in anomaly_patterns:
                f.write(f"• Reviews with zero helpful votes: {anomaly_patterns['zero_votes_percentage']:.1f}%\n")
            
            # Verified purchases
            if 'verified_percentage' in anomaly_patterns:
                f.write(f"• Verified purchases: {anomaly_patterns['verified_percentage']:.1f}%\n")
            
            # Clustering quality
            if 'silhouette_score' in clustering_quality:
                f.write(f"• Clustering quality (Silhouette): {clustering_quality['silhouette_score']:.3f}\n")
            
            f.write("\n")
            
            # Detailed metrics
            f.write("DETAILED METRICS\n")
            f.write("-" * 40 + "\n")
            
            # Feature statistics
            f.write("Feature Statistics:\n")
            for key, value in anomaly_chars.items():
                if key.startswith(('rating_', 'helpful_', 'verified_', 'has_images_', 'predicted_', 'reviewer_', 'rating_vs_')):
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            f.write("\n")
            
            # Clustering metrics
            f.write("Clustering Metrics:\n")
            for key, value in clustering_quality.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n")
            
            # Comparison with original data
            if comparison_results:
                f.write("Comparison with Original Data:\n")
                for key, value in comparison_results.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
        
        print(f"✓ Summary report saved to: {output_file}")
    
    def run_complete_evaluation(self, output_dir: str = "evaluation_results"):
        """
        Run complete evaluation pipeline.
        
        Args:
            output_dir: Directory to save all results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("RUNNING COMPLETE DBSCAN ANOMALY DETECTION EVALUATION")
        print("="*80)
        
        # Run all evaluations
        anomaly_chars = self.evaluate_anomaly_characteristics()
        clustering_quality = self.evaluate_clustering_quality()
        anomaly_patterns = self.analyze_anomaly_patterns()
        
        comparison_results = {}
        if self.original_df is not None:
            comparison_results = self.compare_with_original_data()
        
        # Create visualizations
        plots_dir = os.path.join(output_dir, "plots")
        self.create_visualizations(plots_dir)
        
        # Generate report
        report_file = os.path.join(output_dir, "evaluation_report.txt")
        self.generate_summary_report(report_file)
        
        # Save metrics as JSON
        import json
        all_metrics = {
            'anomaly_characteristics': anomaly_chars,
            'clustering_quality': clustering_quality,
            'anomaly_patterns': anomaly_patterns,
            'comparison_with_original': comparison_results
        }
        
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        print(f"\n✓ Complete evaluation saved to: {output_dir}")
        print(f"  - Report: {report_file}")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Plots: {plots_dir}/")
        
        return all_metrics


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate DBSCAN anomaly detection results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to DBSCAN anomaly detection results CSV'
    )
    
    parser.add_argument(
        '--original-data',
        type=str,
        default=None,
        help='Path to original dataset CSV (optional, for comparison)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        help='Create visualization plots'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AnomalyDetectionEvaluator(
        results_file=args.results_file,
        original_data_file=args.original_data
    )
    
    # Run evaluation
    metrics = evaluator.run_complete_evaluation(args.output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()