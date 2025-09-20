import os
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")


def create_user_anomaly_visualizations(user_analysis_data: pd.DataFrame, output_dir: str, category: str = "books"):
    """
    Create visualizations for user anomaly analysis.
    
    Args:
        user_analysis_data: DataFrame with user anomaly analysis results
        output_dir: Directory to save plots
        category: Category name
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(user_analysis_data['anomaly_count'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of Anomalous Reviews per User')
    plt.ylabel('Number of Users')
    title = f'Distribution of Anomalous Reviews per User - {category}'
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    mean_anomalies = user_analysis_data['anomaly_count'].mean()
    median_anomalies = user_analysis_data['anomaly_count'].median()
    max_anomalies = user_analysis_data['anomaly_count'].max()
    plt.text(0.7, 0.8, f'Mean: {mean_anomalies:.1f}\nMedian: {median_anomalies:.1f}\nMax: {max_anomalies}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(user_analysis_data['reviewer_review_count'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Total Reviews per User')
    plt.ylabel('Number of Users')
    title = f'Total Review Count Distribution (Users with Anomalies) - {category}'
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    mean_total = user_analysis_data['reviewer_review_count'].mean()
    median_total = user_analysis_data['reviewer_review_count'].median()
    max_total = user_analysis_data['reviewer_review_count'].max()
    plt.text(0.7, 0.8, f'Mean: {mean_total:.1f}\nMedian: {median_total:.1f}\nMax: {max_total}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_review_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(user_analysis_data['anomaly_rate'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Anomaly Rate (Anomalous / Total Reviews)')
    plt.ylabel('Number of Users')
    title = f'Distribution of Anomaly Rates - {category}'
    plt.title(title)
    plt.grid(True, alpha=0.3)

    mean_rate = user_analysis_data['anomaly_rate'].mean()
    median_rate = user_analysis_data['anomaly_rate'].median()
    plt.text(0.7, 0.8, f'Mean: {mean_rate:.3f}\nMedian: {median_rate:.3f}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_rate_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_user_analysis(anomalies_df: pd.DataFrame, output_dir: str = "evaluation_plots",
                      category: str = "books") -> Dict:
    """
    Run complete user analysis including both analysis and visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots and results
        category: Category name
        
    Returns:
        Dictionary with all user analysis results
    """

    user_plots_dir = os.path.join(output_dir, "user_analysis")
    os.makedirs(user_plots_dir, exist_ok=True)
    results = {}

    user_anomaly_counts = anomalies_df.groupby('user_id').size().reset_index(name='anomaly_count')

    user_total_counts = anomalies_df.groupby('user_id')['reviewer_review_count'].first().reset_index()

    user_analysis = pd.merge(user_anomaly_counts, user_total_counts, on='user_id', how='left')

    user_analysis['normal_reviews'] = user_analysis['reviewer_review_count'] - user_analysis['anomaly_count']
    user_analysis['anomaly_rate'] = user_analysis['anomaly_count'] / user_analysis['reviewer_review_count']

    user_analysis = user_analysis.dropna()

    if len(user_analysis) == 0:
        print("No valid user data found for analysis")
        return results

    stats = {
        'total_anomalous_users': len(user_analysis),
        'total_anomalous_reviews': int(user_analysis['anomaly_count'].sum()),
        'avg_anomalies_per_user': float(user_analysis['anomaly_count'].mean()),
        'max_anomalies_per_user': int(user_analysis['anomaly_count'].max()),
        'avg_total_reviews_per_user': float(user_analysis['reviewer_review_count'].mean()),
        'avg_normal_reviews_per_user': float(user_analysis['normal_reviews'].mean()),
        'users_with_all_anomalous': int((user_analysis['anomaly_rate'] == 1.0).sum()),
        'users_with_high_anomaly_rate': int((user_analysis['anomaly_rate'] > 0.5).sum()),
    }
    results.update(stats)
    results['user_analysis_data'] = user_analysis

    create_user_anomaly_visualizations(user_analysis, user_plots_dir, category)

    return results
