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


def create_anomaly_visualizations(anomalies_df: pd.DataFrame, output_dir: str = "evaluation_plots", original_df: pd.DataFrame = None):
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
    
    # 1. Rating Distribution
    if 'rating' in anomalies_df.columns:
        plt.figure(figsize=(10, 6))
        rating_counts = anomalies_df['rating'].value_counts().sort_index()
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
    if 'rating' in anomalies_df.columns and 'predicted_rating' in anomalies_df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(anomalies_df['predicted_rating'], anomalies_df['rating'], 
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
    available_features = [col for col in feature_columns if col in anomalies_df.columns]
    
    for feature in available_features:
        plt.figure(figsize=(10, 6))
        plt.hist(anomalies_df[feature], bins=30, alpha=0.7, edgecolor='black')
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
    if 'cluster' in anomalies_df.columns:
        cluster_labels = anomalies_df['cluster'].values
        non_noise_mask = cluster_labels != -1
        
        if non_noise_mask.sum() > 0:
            # PCA for visualization
            feature_columns = ['rating', 'helpful_vote', 'verified_purchase', 'has_images', 
                              'predicted_rating', 'rating_diff', 'reviewer_review_count', 
                              'rating_vs_product_avg_abs']
            
            available_features = [col for col in feature_columns if col in anomalies_df.columns]
            
            if len(available_features) >= 2:
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
                plt.savefig(os.path.join(output_dir, 'dbscan_clustering_visualization.png'), dpi=300, bbox_inches='tight')
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
        # Handle both millisecond and second timestamps
        timestamps = pd.to_numeric(anomalies_df['timestamp'], errors='coerce')
        
        # Check if timestamps are in milliseconds (typical for this dataset)
        if timestamps.max() > 1e10:  # Likely milliseconds
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:  # Likely seconds
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
    
    # 1. Anomaly Timeline Distribution
    create_anomaly_timeline(df_temporal, output_dir)
    
    # 2. Review Bombing Detection
    create_review_bombing_analysis(df_temporal, output_dir)
    
    # 3. Temporal Clustering Analysis
    create_temporal_clustering_analysis(df_temporal, output_dir)
    
    # 4. Seasonal/Periodic Pattern Analysis
    create_seasonal_pattern_analysis(df_temporal, output_dir)
    
    # 5. User-Based Temporal Analysis
    create_user_temporal_analysis(df_temporal, output_dir)


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
    ax1.text(0.7, 0.8, f'Within 1h: {rapid_reviews}\n({rapid_reviews/len(time_gaps)*100:.1f}%)', 
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


def create_temporal_clustering_analysis(df_temporal: pd.DataFrame, output_dir: str):
    """Analyze temporal patterns within DBSCAN clusters."""
    
    if 'cluster' not in df_temporal.columns:
        print("  ⚠️ No cluster information available for temporal clustering analysis")
        return
    
    # Filter out noise points for cluster analysis
    clustered_data = df_temporal[df_temporal['cluster'] != -1].copy()
    
    if len(clustered_data) == 0:
        print("  ⚠️ No clustered anomalies found for temporal analysis")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cluster timeline
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(clustered_data['cluster'].unique())))
    
    for i, cluster_id in enumerate(sorted(clustered_data['cluster'].unique())):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
        cluster_data['date'] = cluster_data['datetime'].dt.date
        daily_counts = cluster_data['date'].value_counts().sort_index()
        
        ax1.plot(daily_counts.index, daily_counts.values, 
                marker='o', label=f'Cluster {cluster_id} ({len(cluster_data)} anomalies)',
                color=cluster_colors[i % len(cluster_colors)], alpha=0.7)
    
    ax1.set_title('Temporal Distribution by Cluster', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Anomalies per Day')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Cluster duration analysis
    cluster_durations = []
    cluster_sizes = []
    
    for cluster_id in sorted(clustered_data['cluster'].unique()):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
        duration = (cluster_data['datetime'].max() - cluster_data['datetime'].min()).days
        cluster_durations.append(duration)
        cluster_sizes.append(len(cluster_data))
    
    scatter = ax2.scatter(cluster_sizes, cluster_durations, alpha=0.7, s=60, c=range(len(cluster_sizes)), cmap='viridis')
    ax2.set_title('Cluster Size vs Duration', fontweight='bold')
    ax2.set_xlabel('Cluster Size (number of anomalies)')
    ax2.set_ylabel('Duration (days)')
    ax2.grid(True, alpha=0.3)
    
    # Add cluster labels
    for i, cluster_id in enumerate(sorted(clustered_data['cluster'].unique())):
        ax2.annotate(f'C{cluster_id}', (cluster_sizes[i], cluster_durations[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Monthly cluster activity heatmap
    clustered_data['year_month'] = clustered_data['datetime'].dt.to_period('M')
    cluster_monthly = clustered_data.groupby(['cluster', 'year_month']).size().unstack(fill_value=0)
    
    if len(cluster_monthly) > 0 and len(cluster_monthly.columns) > 1:
        sns.heatmap(cluster_monthly, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Anomaly Count'})
        ax3.set_title('Cluster Activity Heatmap (Monthly)', fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Cluster ID')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Cluster synchronization analysis
    # Check if clusters show coordinated timing patterns
    cluster_sync_scores = []
    cluster_pairs = []
    
    unique_clusters = sorted(clustered_data['cluster'].unique())
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            cluster1_data = clustered_data[clustered_data['cluster'] == unique_clusters[i]]
            cluster2_data = clustered_data[clustered_data['cluster'] == unique_clusters[j]]
            
            # Calculate temporal overlap
            cluster1_dates = set(cluster1_data['datetime'].dt.date)
            cluster2_dates = set(cluster2_data['datetime'].dt.date)
            
            overlap = len(cluster1_dates & cluster2_dates)
            total_unique = len(cluster1_dates | cluster2_dates)
            sync_score = overlap / total_unique if total_unique > 0 else 0
            
            cluster_sync_scores.append(sync_score)
            cluster_pairs.append(f'C{unique_clusters[i]}-C{unique_clusters[j]}')
    
    if cluster_sync_scores:
        ax4.bar(range(len(cluster_sync_scores)), cluster_sync_scores, alpha=0.7, color='lightblue')
        ax4.set_title('Cluster Temporal Synchronization', fontweight='bold')
        ax4.set_xlabel('Cluster Pairs')
        ax4.set_ylabel('Synchronization Score')
        ax4.set_xticks(range(len(cluster_pairs)))
        ax4.set_xticklabels(cluster_pairs, rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Highlight high synchronization
        high_sync_threshold = 0.3
        high_sync_pairs = [pair for i, pair in enumerate(cluster_pairs) if cluster_sync_scores[i] >= high_sync_threshold]
        if high_sync_pairs:
            ax4.axhline(y=high_sync_threshold, color='red', linestyle='--', alpha=0.8, 
                       label=f'High sync threshold ({high_sync_threshold})')
            ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_clustering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Temporal clustering analysis: {len(unique_clusters)} clusters analyzed")


def create_seasonal_pattern_analysis(df_temporal: pd.DataFrame, output_dir: str):
    """Analyze seasonal and periodic patterns in anomaly occurrences."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Day of week analysis
    df_temporal['day_of_week'] = df_temporal['datetime'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df_temporal['day_of_week'].value_counts().reindex(day_order)
    
    day_counts.plot(kind='bar', ax=ax1, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.set_title('Anomalies by Day of Week', fontweight='bold')
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Number of Anomalies')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add weekend highlighting
    weekend_total = day_counts[['Saturday', 'Sunday']].sum()
    weekday_total = day_counts[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].sum()
    weekend_pct = weekend_total / (weekend_total + weekday_total) * 100
    ax1.text(0.7, 0.9, f'Weekend: {weekend_pct:.1f}%', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Month analysis
    df_temporal['month'] = df_temporal['datetime'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df_temporal['month'].value_counts().reindex(month_order)
    
    month_counts.plot(kind='bar', ax=ax2, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('Anomalies by Month', fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Anomalies')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # Highlight seasonal patterns
    holiday_months = ['November', 'December', 'January']  # Holiday season
    holiday_total = month_counts[holiday_months].sum()
    holiday_pct = holiday_total / month_counts.sum() * 100
    ax2.text(0.7, 0.9, f'Holiday season: {holiday_pct:.1f}%', transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    
    # 3. Rating patterns over time
    if 'rating' in df_temporal.columns:
        # Monthly average rating of anomalies
        df_temporal['year_month'] = df_temporal['datetime'].dt.to_period('M')
        monthly_rating = df_temporal.groupby('year_month')['rating'].mean()
        
        monthly_rating.plot(kind='line', ax=ax3, marker='o', linewidth=2, markersize=4, color='purple')
        ax3.set_title('Average Anomaly Rating Over Time', fontweight='bold')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Average Rating')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add overall average line
        overall_avg = df_temporal['rating'].mean()
        ax3.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.8, 
                   label=f'Overall avg: {overall_avg:.2f}')
        ax3.legend()
    
    # 4. Verification status over time (if available)
    if 'verified_purchase' in df_temporal.columns:
        monthly_verification = df_temporal.groupby('year_month')['verified_purchase'].mean() * 100
        
        monthly_verification.plot(kind='line', ax=ax4, marker='s', linewidth=2, markersize=4, color='orange')
        ax4.set_title('Verified Purchase Rate Over Time', fontweight='bold')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Verification Rate (%)')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add overall average line
        overall_verification = df_temporal['verified_purchase'].mean() * 100
        ax4.axhline(y=overall_verification, color='red', linestyle='--', alpha=0.8,
                   label=f'Overall: {overall_verification:.1f}%')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_pattern_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Seasonal pattern analysis: {len(day_counts)} days, {len(month_counts)} months analyzed")


def create_temporal_anomaly_heatmap(anomalies_df: pd.DataFrame, output_path: str):
    """
    Create a standalone temporal heatmap visualization.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results with timestamp column
        output_path: Full path to save the plot
    """
    if 'timestamp' not in anomalies_df.columns:
        print("Warning: 'timestamp' column not found in DataFrame")
        return
    
    try:
        # Convert timestamp to datetime
        timestamps = pd.to_numeric(anomalies_df['timestamp'], errors='coerce')
        
        if timestamps.max() > 1e10:  # Likely milliseconds
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:  # Likely seconds
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='s', errors='coerce')
            
        # Remove rows with invalid timestamps
        valid_timestamps = anomalies_df['datetime'].notna()
        if not valid_timestamps.any():
            print("Warning: No valid timestamps found")
            return
            
        df_temporal = anomalies_df[valid_timestamps].copy()
        
        # Create day of week vs hour heatmap
        df_temporal['day_of_week'] = df_temporal['datetime'].dt.day_name()
        df_temporal['hour'] = df_temporal['datetime'].dt.hour
        
        # Create pivot table
        heatmap_data = df_temporal.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Number of Anomalies'})
        plt.title('Temporal Anomaly Heatmap - Day of Week vs Hour of Day', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved temporal heatmap to: {output_path}")
        
    except Exception as e:
        print(f"Error creating temporal heatmap: {e}")


def create_user_temporal_analysis(df_temporal: pd.DataFrame, output_dir: str):
    """
    Analyze temporal patterns for individual users with multiple reviews.
    
    Args:
        df_temporal: DataFrame containing anomaly detection results with timestamp and user_id columns
        output_dir: Directory to save plots
    """
    if 'user_id' not in df_temporal.columns:
        print("  ⚠️ No user_id column available for user temporal analysis")
        return
    
    print("Creating user-based temporal analysis...")
    
    # Filter users with more than 5 reviews
    user_counts = df_temporal['user_id'].value_counts()
    prolific_users = user_counts[user_counts > 5]
    
    if len(prolific_users) == 0:
        print("  ⚠️ No users found with more than 5 reviews")
        return
    
    print(f"  Found {len(prolific_users)} users with >5 reviews (max: {prolific_users.max()} reviews)")
    
    # Create comprehensive user analysis
    fig = plt.figure(figsize=(20, 16))
    
    # 1. User Review Count Distribution
    ax1 = plt.subplot(3, 3, 1)
    prolific_users.hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black', ax=ax1)
    ax1.set_title('Distribution of Review Counts\n(Users with >5 Reviews)', fontweight='bold')
    ax1.set_xlabel('Number of Reviews per User')
    ax1.set_ylabel('Number of Users')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(0.7, 0.8, f'Users: {len(prolific_users)}\nMean: {prolific_users.mean():.1f}\nMax: {prolific_users.max()}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Time Span Analysis for Top Users
    ax2 = plt.subplot(3, 3, 2)
    top_users = prolific_users.head(20)
    user_time_spans = []
    user_labels = []
    
    for user_id in top_users.index:
        user_reviews = df_temporal[df_temporal['user_id'] == user_id]
        time_span = (user_reviews['datetime'].max() - user_reviews['datetime'].min()).days
        user_time_spans.append(time_span)
        user_labels.append(f"{user_id[:8]}...")  # Truncate user ID for display
    
    bars = ax2.barh(range(len(user_time_spans)), user_time_spans, alpha=0.7, color='lightcoral')
    ax2.set_title('Time Span of Review Activity\n(Top 20 Most Active Users)', fontweight='bold')
    ax2.set_xlabel('Days Between First and Last Review')
    ax2.set_ylabel('User ID (truncated)')
    ax2.set_yticks(range(len(user_labels)))
    ax2.set_yticklabels(user_labels, fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Highlight suspicious short time spans
    suspicious_threshold = 7  # Reviews spanning less than a week
    for i, span in enumerate(user_time_spans):
        if span <= suspicious_threshold:
            bars[i].set_color('red')
    
    # 3. Review Frequency Analysis
    ax3 = plt.subplot(3, 3, 3)
    review_frequencies = []  # Reviews per day
    
    for user_id in top_users.index:
        user_reviews = df_temporal[df_temporal['user_id'] == user_id]
        time_span = (user_reviews['datetime'].max() - user_reviews['datetime'].min()).days + 1
        frequency = len(user_reviews) / time_span
        review_frequencies.append(frequency)
    
    ax3.scatter(user_time_spans, review_frequencies, alpha=0.7, s=60, c='purple')
    ax3.set_title('Review Frequency vs Time Span\n(Top 20 Users)', fontweight='bold')
    ax3.set_xlabel('Time Span (days)')
    ax3.set_ylabel('Reviews per Day')
    ax3.grid(True, alpha=0.3)
    
    # Highlight high-frequency users (potential bots)
    high_freq_threshold = 1.0  # More than 1 review per day on average
    for i, (span, freq) in enumerate(zip(user_time_spans, review_frequencies)):
        if freq >= high_freq_threshold:
            ax3.scatter(span, freq, color='red', s=100, alpha=0.8)
    
    ax3.axhline(y=high_freq_threshold, color='red', linestyle='--', alpha=0.8, 
                label=f'High frequency threshold ({high_freq_threshold} rev/day)')
    ax3.legend()
    
    # 4. Time Gap Patterns for Suspicious Users
    ax4 = plt.subplot(3, 3, 4)
    
    # Find users with very regular posting patterns (potential bots)
    suspicious_users = []
    for user_id in top_users.index:
        user_reviews = df_temporal[df_temporal['user_id'] == user_id].sort_values('datetime')
        if len(user_reviews) >= 8:  # Need enough data points
            time_gaps = user_reviews['datetime'].diff().dt.total_seconds() / 3600  # Hours
            time_gaps = time_gaps.dropna()
            
            # Check for very regular patterns (low standard deviation)
            if len(time_gaps) > 0 and time_gaps.std() < 24:  # Very consistent timing
                suspicious_users.append((user_id, time_gaps.mean(), time_gaps.std()))
    
    if suspicious_users:
        suspicious_users.sort(key=lambda x: x[2])  # Sort by std deviation
        top_suspicious = suspicious_users[:10]
        
        user_ids = [f"{user[0][:8]}..." for user in top_suspicious]
        std_devs = [user[2] for user in top_suspicious]
        
        bars = ax4.barh(range(len(std_devs)), std_devs, alpha=0.7, color='orange')
        ax4.set_title('Most Regular Posting Patterns\n(Potential Bot Behavior)', fontweight='bold')
        ax4.set_xlabel('Time Gap Std Deviation (hours)')
        ax4.set_ylabel('User ID (truncated)')
        ax4.set_yticks(range(len(user_ids)))
        ax4.set_yticklabels(user_ids, fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Color very regular patterns in red
        for i, std_dev in enumerate(std_devs):
            if std_dev < 12:  # Very regular (< 12 hour std dev)
                bars[i].set_color('red')
    else:
        ax4.text(0.5, 0.5, 'No suspicious regular\npatterns detected', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Most Regular Posting Patterns\n(Potential Bot Behavior)', fontweight='bold')
    
    # 5. Hourly Posting Patterns for Top Users
    ax5 = plt.subplot(3, 3, 5)
    
    # Analyze hourly patterns for top 5 most active users
    top_5_users = prolific_users.head(5)
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_5_users)))
    
    for i, user_id in enumerate(top_5_users.index):
        user_reviews = df_temporal[df_temporal['user_id'] == user_id]
        hourly_counts = user_reviews['datetime'].dt.hour.value_counts().sort_index()
        
        # Normalize to percentage
        hourly_pct = hourly_counts / hourly_counts.sum() * 100
        
        ax5.plot(hourly_pct.index, hourly_pct.values, marker='o', 
                label=f"{user_id[:8]}... ({len(user_reviews)} reviews)", 
                color=colors[i], alpha=0.7, linewidth=2)
    
    ax5.set_title('Hourly Posting Patterns\n(Top 5 Most Active Users)', fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Percentage of Reviews')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(range(0, 24, 4))
    
    # 6. Day-of-Week Patterns
    ax6 = plt.subplot(3, 3, 6)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for i, user_id in enumerate(top_5_users.index):
        user_reviews = df_temporal[df_temporal['user_id'] == user_id]
        day_counts = user_reviews['datetime'].dt.day_name().value_counts().reindex(day_order, fill_value=0)
        
        # Normalize to percentage
        day_pct = day_counts / day_counts.sum() * 100
        
        ax6.plot(range(len(day_order)), day_pct.values, marker='s', 
                label=f"{user_id[:8]}...", color=colors[i], alpha=0.7, linewidth=2)
    
    ax6.set_title('Day-of-Week Posting Patterns\n(Top 5 Most Active Users)', fontweight='bold')
    ax6.set_xlabel('Day of Week')
    ax6.set_ylabel('Percentage of Reviews')
    ax6.set_xticks(range(len(day_order)))
    ax6.set_xticklabels(day_order, rotation=45)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. Burst Detection - Users with Review Bursts
    ax7 = plt.subplot(3, 3, 7)
    
    burst_users = []
    for user_id in prolific_users.head(20).index:
        user_reviews = df_temporal[df_temporal['user_id'] == user_id].sort_values('datetime')
        
        # Look for bursts (multiple reviews within short time periods)
        time_gaps = user_reviews['datetime'].diff().dt.total_seconds() / 3600  # Hours
        time_gaps = time_gaps.dropna()
        
        # Count reviews within 1 hour of each other
        burst_reviews = (time_gaps <= 1).sum()
        burst_rate = burst_reviews / len(user_reviews) if len(user_reviews) > 0 else 0
        
        if burst_rate > 0.2:  # More than 20% of reviews in bursts
            burst_users.append((user_id, burst_rate, len(user_reviews)))
    
    if burst_users:
        burst_users.sort(key=lambda x: x[1], reverse=True)  # Sort by burst rate
        
        user_ids = [f"{user[0][:8]}..." for user in burst_users[:15]]
        burst_rates = [user[1] * 100 for user in burst_users[:15]]  # Convert to percentage
        
        bars = ax7.barh(range(len(burst_rates)), burst_rates, alpha=0.7, color='red')
        ax7.set_title('Users with Review Bursts\n(% Reviews within 1h of Another)', fontweight='bold')
        ax7.set_xlabel('Burst Rate (%)')
        ax7.set_ylabel('User ID (truncated)')
        ax7.set_yticks(range(len(user_ids)))
        ax7.set_yticklabels(user_ids, fontsize=8)
        ax7.grid(True, alpha=0.3, axis='x')
    else:
        ax7.text(0.5, 0.5, 'No significant\nburst patterns detected', 
                transform=ax7.transAxes, ha='center', va='center', fontsize=12)
        ax7.set_title('Users with Review Bursts\n(% Reviews within 1h of Another)', fontweight='bold')
    
    # 8. Rating Consistency Analysis
    ax8 = plt.subplot(3, 3, 8)
    
    if 'rating' in df_temporal.columns:
        rating_consistency = []
        
        for user_id in top_users.index:
            user_reviews = df_temporal[df_temporal['user_id'] == user_id]
            rating_std = user_reviews['rating'].std()
            rating_consistency.append((user_id, rating_std, len(user_reviews)))
        
        # Sort by consistency (low std = high consistency)
        rating_consistency.sort(key=lambda x: x[1])
        
        user_ids = [f"{user[0][:8]}..." for user in rating_consistency[:15]]
        rating_stds = [user[1] for user in rating_consistency[:15]]
        
        bars = ax8.barh(range(len(rating_stds)), rating_stds, alpha=0.7, color='lightblue')
        ax8.set_title('Rating Consistency\n(Lower = More Consistent)', fontweight='bold')
        ax8.set_xlabel('Rating Standard Deviation')
        ax8.set_ylabel('User ID (truncated)')
        ax8.set_yticks(range(len(user_ids)))
        ax8.set_yticklabels(user_ids, fontsize=8)
        ax8.grid(True, alpha=0.3, axis='x')
        
        # Highlight very consistent raters (potential fake accounts)
        for i, std in enumerate(rating_stds):
            if std < 0.5:  # Very consistent ratings
                bars[i].set_color('red')
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary statistics
    total_prolific_users = len(prolific_users)
    avg_reviews_per_user = prolific_users.mean()
    max_reviews = prolific_users.max()
    
    # Count suspicious patterns
    short_span_users = sum(1 for span in user_time_spans if span <= 7)
    high_freq_users = sum(1 for freq in review_frequencies if freq >= 1.0)
    burst_user_count = len(burst_users) if burst_users else 0
    regular_pattern_users = len(suspicious_users) if suspicious_users else 0
    
    summary_text = f"""
USER TEMPORAL ANALYSIS SUMMARY

Total Users (>5 reviews): {total_prolific_users}
Average Reviews per User: {avg_reviews_per_user:.1f}
Maximum Reviews: {max_reviews}

SUSPICIOUS PATTERNS DETECTED:
• Short Time Span (≤7 days): {short_span_users}
• High Frequency (≥1 rev/day): {high_freq_users}
• Burst Patterns: {burst_user_count}
• Regular Timing: {regular_pattern_users}

RISK INDICATORS:
• Red bars indicate suspicious behavior
• Regular patterns may suggest automation
• Burst posting could indicate coordinated attacks
• Very consistent ratings may be fake accounts
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_temporal_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ User temporal analysis: {total_prolific_users} users analyzed, {short_span_users + high_freq_users + burst_user_count + regular_pattern_users} suspicious patterns detected")


def create_comparison_plots(anomalies_df: pd.DataFrame, original_df: pd.DataFrame, output_dir: str):
    """
    Create comparison plots between anomalies and normal data.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        original_df: DataFrame containing original dataset
        output_dir: Directory to save plots
    """
    print("Creating comparison plots between anomalies and normal data...")
    
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
    
    print(f"  ✓ Comparison plots: {len(available_numerical)} numerical and {len(available_categorical)} categorical features compared")
