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
        
        # Feature analysis - updated to reflect new feature set focused on deception detection
        feature_columns = ['helpful_vote', 'verified_purchase', 'has_images', 
                          'rating_diff', 'reviewer_review_count', 
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
        
        # Feature comparisons - updated to reflect new feature set focused on deception detection
        feature_columns = ['helpful_vote', 'verified_purchase', 'has_images', 
                          'rating_diff', 'reviewer_review_count', 
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
        
        # Build feature matrix for clustering evaluation - updated to reflect new feature set
        feature_columns = ['helpful_vote', 'verified_purchase', 'has_images', 
                          'rating_diff', 'reviewer_review_count', 
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
        Create visualizations for anomaly detection results with comparison to normal data.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating visualizations in: {output_dir}")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Rating Distribution Comparison
        if 'rating' in self.anomalies_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Plot anomaly ratings
            anomaly_rating_counts = self.anomalies_df['rating'].value_counts().sort_index()
            plt.bar(anomaly_rating_counts.index - 0.2, anomaly_rating_counts.values, 
                   alpha=0.7, width=0.4, label='Anomalies', color='red')
            
            # Plot normal ratings if available
            if self.original_df is not None and 'rating' in self.original_df.columns:
                # Get normal data (exclude anomalies)
                anomaly_indices = set(self.anomalies_df.index)
                normal_mask = ~self.original_df.index.isin(anomaly_indices)
                normal_ratings = self.original_df.loc[normal_mask, 'rating']
                
                if len(normal_ratings) > 0:
                    normal_rating_counts = normal_ratings.value_counts().sort_index()
                    plt.bar(normal_rating_counts.index + 0.2, normal_rating_counts.values, 
                           alpha=0.7, width=0.4, label='Normal Reviews', color='blue')
            
            plt.title('Rating Distribution: Anomalies vs Normal Reviews')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.xticks(range(1, 6))
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rating_distribution_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Individual Feature Distributions - Anomalies Only
        # Save specific distributions in basic_metrics folder
        basic_metrics_dir = os.path.join(output_dir, "basic_metrics")
        os.makedirs(basic_metrics_dir, exist_ok=True)
        
        feature_columns = ['helpful_vote', 'rating_diff', 'reviewer_review_count']
        available_features = [col for col in feature_columns if col in self.anomalies_df.columns]
        
        for feature in available_features:
            plt.figure(figsize=(10, 6))
            
            # Simple histogram of anomalies only
            plt.hist(self.anomalies_df[feature], bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
            plt.title(f'Anomalies - {feature.replace("_", " ").title()} Distribution')
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel('Count')
            
            # Apply log scale for helpful_vote if needed
            if feature == 'helpful_vote':
                plt.yscale('log')
                plt.ylabel('Count (log scale)')
            
            plt.grid(True, alpha=0.3)
            
            # Add basic statistics
            feature_mean = self.anomalies_df[feature].mean()
            feature_median = self.anomalies_df[feature].median()
            feature_count = len(self.anomalies_df[feature])
            plt.text(0.02, 0.98, 
                   f'Mean: {feature_mean:.2f}\nMedian: {feature_median:.2f}\nCount: {feature_count}', 
                   transform=plt.gca().transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(basic_metrics_dir, f'{feature}_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Rating vs Rating Difference Analysis
        print(f"\nChecking conditions for rating vs rating_diff plot:")
        print(f"  'rating' in anomalies_df: {'rating' in self.anomalies_df.columns}")
        print(f"  'rating_diff' in anomalies_df: {'rating_diff' in self.anomalies_df.columns}")
        print(f"  original_df is not None: {self.original_df is not None}")
        if self.original_df is not None:
            print(f"  'rating' in original_df: {'rating' in self.original_df.columns}")
            print(f"  'rating_diff' in original_df: {'rating_diff' in self.original_df.columns}")
        
        if ('rating' in self.anomalies_df.columns and 'rating_diff' in self.anomalies_df.columns and 
            self.original_df is not None and 'rating' in self.original_df.columns and 'rating_diff' in self.original_df.columns):
            
            plt.figure(figsize=(12, 8))
            
            # Get normal data (exclude anomalies)
            anomaly_indices = set(self.anomalies_df.index)
            normal_mask = ~self.original_df.index.isin(anomaly_indices)
            normal_data = self.original_df.loc[normal_mask]
            
            # Calculate mean rating_diff for each rating value
            # For anomalies
            anomaly_rating_groups = self.anomalies_df.groupby('rating')['rating_diff'].agg(['mean', 'std', 'count']).reset_index()
            
            # For normal data
            normal_rating_groups = normal_data.groupby('rating')['rating_diff'].agg(['mean', 'std', 'count']).reset_index()
            
            # Plot mean rating_diff by rating
            plt.plot(anomaly_rating_groups['rating'], anomaly_rating_groups['mean'], 
                    'o-', color='lightcoral', linewidth=2, markersize=8, label='Anomalies')
            plt.plot(normal_rating_groups['rating'], normal_rating_groups['mean'], 
                    'o-', color='lightblue', linewidth=2, markersize=8, label='Normal Reviews')
            
            # Add error bars for standard deviation
            plt.errorbar(anomaly_rating_groups['rating'], anomaly_rating_groups['mean'], 
                        yerr=anomaly_rating_groups['std'], color='lightcoral', alpha=0.3, capsize=5)
            plt.errorbar(normal_rating_groups['rating'], normal_rating_groups['mean'], 
                        yerr=normal_rating_groups['std'], color='lightblue', alpha=0.3, capsize=5)
            
            plt.title('Mean Rating Difference by Actual Rating')
            plt.xlabel('Actual Rating (1-5 stars)')
            plt.ylabel('Mean Rating Difference (|Actual - Predicted|)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks([1, 2, 3, 4, 5])
            
            # Add sample size annotations
            for _, row in anomaly_rating_groups.iterrows():
                plt.annotate(f'n={int(row["count"])}', 
                           (row['rating'], row['mean']), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=8, color='darkred')
            
            for _, row in normal_rating_groups.iterrows():
                plt.annotate(f'n={int(row["count"])}', 
                           (row['rating'], row['mean']), 
                           textcoords="offset points", xytext=(0,-15), ha='center',
                           fontsize=8, color='darkblue')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rating_vs_rating_diff_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved rating vs rating_diff analysis plot to: {os.path.join(output_dir, 'rating_vs_rating_diff_analysis.png')}")
            plt.close()
        else:
            print("⚠️ Skipping rating vs rating_diff plot - conditions not met")
        
        # 4. Cluster Analysis (if available)
        if 'cluster' in self.anomalies_df.columns:
            cluster_labels = self.anomalies_df['cluster'].values
            non_noise_mask = cluster_labels != -1
            
            if non_noise_mask.sum() > 0:
                # PCA for visualization - updated to reflect new feature set
                feature_columns = ['helpful_vote', 'verified_purchase', 'has_images', 
                                  'rating_diff', 'reviewer_review_count', 
                                  'rating_vs_product_avg_abs']
                
                available_features = [col for col in feature_columns if col in self.anomalies_df.columns]
                
                if len(available_features) >= 2:
                    feature_data = self.anomalies_df[available_features].values
                    
                    # PCA
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(feature_data)
                    
                    plt.figure(figsize=(14, 10))
                    
                    # Plot normal data first (if available) as background
                    if self.original_df is not None:
                        # Get normal data (exclude anomalies)
                        anomaly_indices = set(self.anomalies_df.index)
                        normal_mask = ~self.original_df.index.isin(anomaly_indices)
                        normal_data = self.original_df.loc[normal_mask, available_features].values
                        
                        if len(normal_data) > 0:
                            # Sample normal data if too large for visualization
                            if len(normal_data) > 10000:
                                normal_indices = np.random.choice(len(normal_data), 10000, replace=False)
                                normal_data = normal_data[normal_indices]
                            
                            normal_pca = pca.transform(normal_data)
                            plt.scatter(normal_pca[:, 0], normal_pca[:, 1], 
                                      c='lightblue', alpha=0.3, s=20, label='Normal Reviews')
                    
                    # Plot non-noise anomaly clusters
                    non_noise_data = pca_data[non_noise_mask]
                    non_noise_clusters = cluster_labels[non_noise_mask]
                    
                    scatter = plt.scatter(non_noise_data[:, 0], non_noise_data[:, 1], 
                                        c=non_noise_clusters, cmap='tab10', alpha=0.8, s=40, 
                                        label='Anomaly Clusters')
                    
                    # Plot noise points
                    noise_data = pca_data[~non_noise_mask]
                    if len(noise_data) > 0:
                        plt.scatter(noise_data[:, 0], noise_data[:, 1], 
                                  c='red', marker='x', s=60, alpha=0.9, label='Noise (Anomalies)')
                    
                    plt.title('DBSCAN Clustering Results with Normal Data Comparison (PCA Visualization)')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                    plt.colorbar(scatter, label='Cluster')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'clustering_with_normal_data_comparison.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 4. Additional Comparison Visualizations
        if self.original_df is not None:
            self._create_comparison_plots(output_dir)
        
        print(f"✓ Saved {len(os.listdir(output_dir))} visualization files")
    
    def _create_comparison_plots(self, output_dir: str):
        """Create additional comparison plots between anomalies and normal data."""
        
        # Box plots for numerical features - updated to reflect new feature set
        # Create separate plots for each feature
        numerical_features = ['helpful_vote', 'reviewer_review_count', 'rating_diff']
        available_numerical = [col for col in numerical_features 
                             if col in self.anomalies_df.columns and col in self.original_df.columns]
        
        if available_numerical:
            # Get normal data (exclude anomalies)
            anomaly_indices = set(self.anomalies_df.index)
            normal_mask = ~self.original_df.index.isin(anomaly_indices)
            
            for feature in available_numerical:
                plt.figure(figsize=(8, 6))
                
                normal_data = self.original_df.loc[normal_mask, feature].dropna()
                anomaly_data = self.anomalies_df[feature].dropna()
                
                # Create box plot
                data_to_plot = [normal_data, anomaly_data]
                labels = ['Normal Reviews', 'Anomalies']
                
                bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color the boxes
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                plt.title(f'{feature.replace("_", " ").title()} Comparison')
                plt.ylabel(feature.replace("_", " ").title())
                plt.grid(True, alpha=0.3)
                
                # Add some statistics as text
                normal_mean = normal_data.mean()
                anomaly_mean = anomaly_data.mean()
                plt.text(0.02, 0.98, f'Normal mean: {normal_mean:.2f}\nAnomaly mean: {anomaly_mean:.2f}', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{feature}_boxplot_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Percentage comparison for categorical features
        # Create separate plots for each feature
        categorical_features = ['verified_purchase', 'has_images']
        available_categorical = [col for col in categorical_features 
                               if col in self.anomalies_df.columns and col in self.original_df.columns]
        
        if available_categorical:
            # Get normal data (exclude anomalies)
            anomaly_indices = set(self.anomalies_df.index)
            normal_mask = ~self.original_df.index.isin(anomaly_indices)
            
            for feature in available_categorical:
                plt.figure(figsize=(8, 6))
                
                normal_data = self.original_df.loc[normal_mask, feature]
                anomaly_data = self.anomalies_df[feature]
                
                # Calculate percentages
                normal_true_pct = normal_data.mean() * 100
                normal_false_pct = (1 - normal_data.mean()) * 100
                anomaly_true_pct = anomaly_data.mean() * 100
                anomaly_false_pct = (1 - anomaly_data.mean()) * 100
                
                # Create grouped bar chart
                categories = ['True', 'False']
                normal_values = [normal_true_pct, normal_false_pct]
                anomaly_values = [anomaly_true_pct, anomaly_false_pct]
                
                x = np.arange(len(categories))
                width = 0.35
                
                plt.bar(x - width/2, normal_values, width, label='Normal Reviews', color='lightblue', alpha=0.8)
                plt.bar(x + width/2, anomaly_values, width, label='Anomalies', color='lightcoral', alpha=0.8)
                
                plt.title(f'{feature.replace("_", " ").title()} Comparison')
                plt.ylabel('Percentage (%)')
                plt.xticks(x, categories)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add percentage labels on bars
                for j, (normal_val, anomaly_val) in enumerate(zip(normal_values, anomaly_values)):
                    plt.text(j - width/2, normal_val + 1, f'{normal_val:.1f}%', 
                           ha='center', va='bottom')
                    plt.text(j + width/2, anomaly_val + 1, f'{anomaly_val:.1f}%', 
                           ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{feature}_categorical_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
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
            
            # Comparison with normal data
            if comparison_results:
                f.write("\nCOMPARISON WITH NORMAL DATA:\n")
                for key, value in comparison_results.items():
                    if key.startswith('rating_') and 'mean_diff' in key:
                        f.write(f"• {key.replace('_', ' ').title()}: {value:+.3f}\n")
                    elif key.startswith('category_') and 'pct_diff' in key:
                        f.write(f"• {key.replace('_', ' ').title()}: {value:+.1f}%\n")
            
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
        '--anomaly-data-file',
        type=str,
        default='output/electronics_test_scan_anomalies_eps_0.8_min_samples_15_batch_size_100000.csv',
        help='Path to DBSCAN anomaly detection results CSV'
    )
    
    parser.add_argument(
        '--original-data',
        type=str,
        default='data/test/electronics_test.csv',
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
    
    # Display file paths being used
    print("="*80)
    print("DBSCAN ANOMALY DETECTION EVALUATION")
    print("="*80)
    print(f"Anomaly Data File: {args.anomaly_data_file}")
    print(f"Original Data File: {args.original_data}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80)
    
    # Check if files exist
    if not os.path.exists(args.anomaly_data_file):
        print(f"❌ Anomaly data file not found: {args.anomaly_data_file}")
        print("Please provide a valid path using --anomaly-data-file")
        return
    
    if args.original_data and not os.path.exists(args.original_data):
        print(f"❌ Original data file not found: {args.original_data}")
        print("Please provide a valid path using --original-data")
        return
    
    # Initialize evaluator
    evaluator = AnomalyDetectionEvaluator(
        results_file=args.anomaly_data_file,
        original_data_file=args.original_data
    )
    
    # Run evaluation
    metrics = evaluator.run_complete_evaluation(args.output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()