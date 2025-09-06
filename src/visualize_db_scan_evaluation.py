"""
Anomaly Detection Visualization Module

This module contains visualization functions for anomaly detection results,
including rating distributions, clustering visualizations, and feature analysis plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def create_anomaly_visualizations(anomalies_df: pd.DataFrame, output_dir: str = "evaluation_plots",
                                  original_df: pd.DataFrame = None):
    """
    Create comprehensive visualizations for anomaly detection results.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots
        original_df: Optional original dataset for comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCreating visualizations in: {output_dir}")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    # 1. Rating Distribution Comparison
    if 'rating' in anomalies_df.columns:
        plt.figure(figsize=(12, 8))

        anomaly_rating_counts = anomalies_df['rating'].value_counts().sort_index()
        plt.bar(anomaly_rating_counts.index - 0.2, anomaly_rating_counts.values,
                alpha=0.7, width=0.4, label='Anomalies', color='red')

        if original_df is not None and 'rating' in original_df.columns:
            # Get normal data (exclude anomalies)
            anomaly_indices = set(anomalies_df.index)
            normal_mask = ~original_df.index.isin(anomaly_indices)
            normal_ratings = original_df.loc[normal_mask, 'rating']

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
    basic_metrics_dir = os.path.join(output_dir, "basic_metrics")
    os.makedirs(basic_metrics_dir, exist_ok=True)

    feature_columns = ['helpful_vote', 'rating_diff', 'reviewer_review_count']
    available_features = [col for col in feature_columns if col in anomalies_df.columns]

    for feature in available_features:
        plt.figure(figsize=(10, 6))
        plt.hist(anomalies_df[feature], bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
        plt.title(f'Anomalies - {feature.replace("_", " ").title()} Distribution')
        plt.xlabel(feature.replace("_", " ").title())
        plt.ylabel('Count')
        if feature == 'helpful_vote':
            plt.yscale('log')
            plt.ylabel('Count (log scale)')

        plt.grid(True, alpha=0.3)

        feature_mean = anomalies_df[feature].mean()
        feature_median = anomalies_df[feature].median()
        feature_count = len(anomalies_df[feature])
        plt.text(0.02, 0.98,
                 f'Mean: {feature_mean:.2f}\nMedian: {feature_median:.2f}\nCount: {feature_count}',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(basic_metrics_dir, f'{feature}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Rating vs Rating Difference Analysis
        if original_df is not None:
            print(f"  'rating' in original_df: {'rating' in original_df.columns}")
            print(f"  'rating_diff' in original_df: {'rating_diff' in original_df.columns}")

        if ('rating' in anomalies_df.columns and 'rating_diff' in anomalies_df.columns and
                original_df is not None and 'rating' in original_df.columns and 'rating_diff' in original_df.columns):

            plt.figure(figsize=(12, 8))

            anomaly_indices = set(anomalies_df.index)
            normal_mask = ~original_df.index.isin(anomaly_indices)
            normal_data = original_df.loc[normal_mask]

            anomaly_rating_groups = anomalies_df.groupby('rating')['rating_diff'].agg(
                ['mean', 'std', 'count']).reset_index()

            normal_rating_groups = normal_data.groupby('rating')['rating_diff'].agg(
                ['mean', 'std', 'count']).reset_index()

            plt.plot(anomaly_rating_groups['rating'], anomaly_rating_groups['mean'],
                     'o-', color='lightcoral', linewidth=2, markersize=8, label='Anomalies')
            plt.plot(normal_rating_groups['rating'], normal_rating_groups['mean'],
                     'o-', color='lightblue', linewidth=2, markersize=8, label='Normal Reviews')

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

            for _, row in anomaly_rating_groups.iterrows():
                plt.annotate(f'n={int(row["count"])}',
                             (row['rating'], row['mean']),
                             textcoords="offset points", xytext=(0, 10), ha='center',
                             fontsize=8, color='darkred')

            for _, row in normal_rating_groups.iterrows():
                plt.annotate(f'n={int(row["count"])}',
                             (row['rating'], row['mean']),
                             textcoords="offset points", xytext=(0, -15), ha='center',
                             fontsize=8, color='darkblue')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rating_vs_rating_diff_analysis.png'), dpi=300, bbox_inches='tight')
            print(
                f"✓ Saved rating vs rating_diff analysis plot to: {os.path.join(output_dir, 'rating_vs_rating_diff_analysis.png')}")
            plt.close()
        else:
            print("⚠️ Skipping rating vs rating_diff plot - conditions not met")

            # 4. Cluster Analysis (if available)
            if 'cluster' in anomalies_df.columns:
                cluster_labels = anomalies_df['cluster'].values
                non_noise_mask = cluster_labels != -1

                if non_noise_mask.sum() > 0:
                    feature_columns = ['helpful_vote', 'verified_purchase', 'has_images',
                                       'rating_diff', 'reviewer_review_count',
                                       'rating_vs_product_avg_abs']

                    available_features = [col for col in feature_columns if col in anomalies_df.columns]

                    if len(available_features) >= 2:
                        feature_data = anomalies_df[available_features].values

                        pca = PCA(n_components=2)
                        pca_data = pca.fit_transform(feature_data)

                        plt.figure(figsize=(14, 10))

                        if original_df is not None:
                            anomaly_indices = set(anomalies_df.index)
                            normal_mask = ~original_df.index.isin(anomaly_indices)
                            normal_data = original_df.loc[normal_mask, available_features].values

                            if len(normal_data) > 0:
                                # Sample normal data if too large for visualization
                                if len(normal_data) > 10000:
                                    normal_indices = np.random.choice(len(normal_data), 10000, replace=False)
                                    normal_data = normal_data[normal_indices]

                                normal_pca = pca.transform(normal_data)
                                plt.scatter(normal_pca[:, 0], normal_pca[:, 1],
                                            c='lightblue', alpha=0.3, s=20, label='Normal Reviews')

                        non_noise_data = pca_data[non_noise_mask]
                        non_noise_clusters = cluster_labels[non_noise_mask]

                        scatter = plt.scatter(non_noise_data[:, 0], non_noise_data[:, 1],
                                              c=non_noise_clusters, cmap='tab10', alpha=0.8, s=40,
                                              label='Anomaly Clusters')

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
                        plt.savefig(os.path.join(output_dir, 'clustering_with_normal_data_comparison.png'), dpi=300,
                                    bbox_inches='tight')
                        plt.close()

    # 5. Temporal Pattern Analysis
    if 'timestamp' in anomalies_df.columns:
        create_temporal_pattern_visualizations(anomalies_df, output_dir)

    # 6. Comparison plots with original data (if provided)
    if original_df is not None:
        create_comparison_plots(anomalies_df, original_df, output_dir)

    print(f"✓ Saved {len(os.listdir(output_dir))} visualization files")


def create_rating_distribution_plot(anomalies_df: pd.DataFrame, output_path: str):
    """
    Create a standalone rating distribution plot.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_path: Full path to save the plot
    """
    if 'rating' not in anomalies_df.columns:
        print("Warning: 'rating' column not found in DataFrame")
        return

    plt.figure(figsize=(10, 6))
    rating_counts = anomalies_df['rating'].value_counts().sort_index()
    plt.bar(rating_counts.index, rating_counts.values, alpha=0.7)
    plt.title('Anomaly Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(range(1, 6))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved rating distribution plot to: {output_path}")


def create_clustering_visualization(anomalies_df: pd.DataFrame, output_path: str):
    """
    Create a standalone clustering visualization using PCA.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results with cluster labels
        output_path: Full path to save the plot
    """
    if 'cluster' not in anomalies_df.columns:
        print("Warning: 'cluster' column not found in DataFrame")
        return

    cluster_labels = anomalies_df['cluster'].values
    non_noise_mask = cluster_labels != -1

    if non_noise_mask.sum() == 0:
        print("Warning: No non-noise clusters found")
        return

    # PCA for visualization
    feature_columns = ['rating', 'helpful_vote', 'verified_purchase', 'has_images',
                       'predicted_rating', 'rating_diff', 'reviewer_review_count',
                       'rating_vs_product_avg_abs']

    available_features = [col for col in feature_columns if col in anomalies_df.columns]

    if len(available_features) < 2:
        print("Warning: Not enough features available for PCA visualization")
        return

    feature_data = anomalies_df[available_features].values

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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved clustering visualization to: {output_path}")


def create_temporal_pattern_visualizations(anomalies_df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive temporal pattern analysis visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results with timestamp column
        output_dir: Directory to save plots
    """
    print("Creating temporal pattern visualizations...")

    # Convert timestamp to datetime
    try:
        timestamps = pd.to_numeric(anomalies_df['timestamp'], errors='coerce')

        if timestamps.max() > 1e10:
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='s', errors='coerce')

        # Remove rows with invalid timestamps
        valid_timestamps = anomalies_df['datetime'].notna()
        if not valid_timestamps.any():
            print("⚠️ No valid timestamps found for temporal analysis")
            return

        df_temporal = anomalies_df[valid_timestamps].copy()

    except Exception as e:
        print(f"⚠️ Error processing timestamps: {e}")
        return

    create_review_bombing_analysis(df_temporal, output_dir)


def create_anomaly_timeline(df_temporal: pd.DataFrame, output_dir: str):
    """Create timeline visualization of anomaly occurrences."""

    # Monthly aggregation
    df_temporal['year_month'] = df_temporal['datetime'].dt.to_period('M')
    monthly_counts = df_temporal['year_month'].value_counts().sort_index()

    # Create timeline plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    x_labels = [str(period) for period in monthly_counts.index]
    ax1.plot(x_labels, monthly_counts.values, marker='o', linewidth=2, markersize=4)
    ax1.set_title('Anomaly Detection Timeline - Monthly Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Number of Anomalies')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add trend line
    x_numeric = range(len(monthly_counts))
    z = np.polyfit(x_numeric, monthly_counts.values, 1)
    p = np.poly1d(z)
    ax1.plot(x_labels, p(x_numeric), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.2f})')
    ax1.legend()

    # Yearly aggregation for broader view
    df_temporal['year'] = df_temporal['datetime'].dt.year
    yearly_counts = df_temporal['year'].value_counts().sort_index()

    yearly_counts.plot(kind='bar', ax=ax2, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Anomaly Detection by Year', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Anomalies')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_anomaly_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Timeline analysis: {len(monthly_counts)} months, {len(yearly_counts)} years")


def create_review_bombing_analysis(df_temporal: pd.DataFrame, output_dir: str):
    """Detect and visualize potential review bombing patterns."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Time gap analysis between consecutive anomalies
    df_sorted = df_temporal.sort_values('datetime')
    time_gaps = df_sorted['datetime'].diff().dt.total_seconds() / 3600  # Hours
    time_gaps = time_gaps.dropna()

    # Plot time gaps histogram
    ax1.hist(time_gaps[time_gaps <= 24], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax1.set_title('Time Gaps Between Consecutive Anomalies (≤24h)', fontweight='bold')
    ax1.set_xlabel('Hours Between Anomalies')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Add statistics
    rapid_reviews = (time_gaps <= 1).sum()  # Within 1 hour
    ax1.text(0.7, 0.8, f'Within 1h: {rapid_reviews}\n({rapid_reviews / len(time_gaps) * 100:.1f}%)',
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # 2. Product-based clustering (potential coordinated attacks)
    if 'asin' in df_temporal.columns:
        product_counts = df_temporal['asin'].value_counts()
        top_products = product_counts.head(20)

        top_products.plot(kind='barh', ax=ax2, alpha=0.7, color='lightgreen')
        ax2.set_title('Top 20 Products by Anomaly Count', fontweight='bold')
        ax2.set_xlabel('Number of Anomalies')
        ax2.set_ylabel('Product ASIN')
        ax2.grid(True, alpha=0.3, axis='x')

        # Highlight potential targets
        suspicious_threshold = product_counts.quantile(0.95)
        suspicious_products = product_counts[product_counts >= suspicious_threshold]
        ax2.text(0.6, 0.9, f'Suspicious products\n(≥{suspicious_threshold:.0f} anomalies): {len(suspicious_products)}',
                 transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'review_bombing_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Review bombing analysis: Time gaps and product targeting analysis completed")


def create_comparison_plots(anomalies_df, original_df, output_dir):
    """Create additional comparison plots between anomalies and normal data."""

    # Box plots for numerical features - updated to reflect new feature set
    # Create separate plots for each feature
    numerical_features = ['helpful_vote', 'reviewer_review_count', 'rating_diff']
    available_numerical = [col for col in numerical_features
                           if col in anomalies_df.columns and col in original_df.columns]

    if available_numerical:
        # Get normal data (exclude anomalies)
        anomaly_indices = set(anomalies_df.index)
        normal_mask = ~original_df.index.isin(anomaly_indices)

        for feature in available_numerical:
            plt.figure(figsize=(8, 6))

            normal_data = original_df.loc[normal_mask, feature].dropna()
            anomaly_data = anomalies_df[feature].dropna()

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
                             if col in anomalies_df.columns and col in original_df.columns]

    if available_categorical:
        # Get normal data (exclude anomalies)
        anomaly_indices = set(anomalies_df.index)
        normal_mask = ~original_df.index.isin(anomaly_indices)

        for feature in available_categorical:
            plt.figure(figsize=(8, 6))

            normal_data = original_df.loc[normal_mask, feature]
            anomaly_data = anomalies_df[feature]

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

            plt.bar(x - width / 2, normal_values, width, label='Normal Reviews', color='lightblue', alpha=0.8)
            plt.bar(x + width / 2, anomaly_values, width, label='Anomalies', color='lightcoral', alpha=0.8)

            plt.title(f'{feature.replace("_", " ").title()} Comparison')
            plt.ylabel('Percentage (%)')
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add percentage labels on bars
            for j, (normal_val, anomaly_val) in enumerate(zip(normal_values, anomaly_values)):
                plt.text(j - width / 2, normal_val + 1, f'{normal_val:.1f}%',
                         ha='center', va='bottom')
                plt.text(j + width / 2, anomaly_val + 1, f'{anomaly_val:.1f}%',
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{feature}_categorical_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
