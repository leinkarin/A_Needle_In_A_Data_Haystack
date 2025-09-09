import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")


def create_user_anomaly_visualizations(user_analysis_data: pd.DataFrame, output_dir: str):
    """
    Create visualizations for user anomaly analysis.
    
    Args:
        user_analysis_data: DataFrame with user anomaly analysis results
        output_dir: Directory to save plots
    """
    print("Creating user anomaly analysis visualizations...")

    # Create user analysis subdirectory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Distribution of anomaly counts per user
    plt.figure(figsize=(10, 6))
    plt.hist(user_analysis_data['anomaly_count'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Number of Anomalous Reviews per User')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Anomalous Reviews per User')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mean_anomalies = user_analysis_data['anomaly_count'].mean()
    median_anomalies = user_analysis_data['anomaly_count'].median()
    max_anomalies = user_analysis_data['anomaly_count'].max()
    plt.text(0.7, 0.8, f'Mean: {mean_anomalies:.1f}\nMedian: {median_anomalies:.1f}\nMax: {max_anomalies}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Distribution of total reviews per user (for users with anomalies)
    plt.figure(figsize=(10, 6))
    plt.hist(user_analysis_data['reviewer_review_count'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Total Reviews per User')
    plt.ylabel('Number of Users')
    plt.title('Total Review Count Distribution (Users with Anomalies)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mean_total = user_analysis_data['reviewer_review_count'].mean()
    median_total = user_analysis_data['reviewer_review_count'].median()
    max_total = user_analysis_data['reviewer_review_count'].max()
    plt.text(0.7, 0.8, f'Mean: {mean_total:.1f}\nMedian: {median_total:.1f}\nMax: {max_total}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_review_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Anomaly rate distribution
    plt.figure(figsize=(10, 6))
    plt.hist(user_analysis_data['anomaly_rate'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Anomaly Rate (Anomalous / Total Reviews)')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Anomaly Rates')
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mean_rate = user_analysis_data['anomaly_rate'].mean()
    median_rate = user_analysis_data['anomaly_rate'].median()
    plt.text(0.7, 0.8, f'Mean: {mean_rate:.3f}\nMedian: {median_rate:.3f}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_rate_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Cumulative distribution of normal reviews
    plt.figure(figsize=(10, 6))
    normal_reviews_sorted = np.sort(user_analysis_data['normal_reviews'])
    y = np.arange(1, len(normal_reviews_sorted) + 1) / len(normal_reviews_sorted)
    plt.plot(normal_reviews_sorted, y, linewidth=2, color='green')
    plt.xlabel('Number of Normal Reviews per User')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of Normal Reviews')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    # Add percentile lines
    p50 = user_analysis_data['normal_reviews'].quantile(0.5)
    p90 = user_analysis_data['normal_reviews'].quantile(0.9)
    plt.axvline(p50, color='red', linestyle='--', alpha=0.7, label=f'50th percentile: {p50:.1f}')
    plt.axvline(p90, color='orange', linestyle='--', alpha=0.7, label=f'90th percentile: {p90:.1f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normal_reviews_cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Anomaly rate vs total reviews
    plt.figure(figsize=(10, 8))
    plt.scatter(user_analysis_data['reviewer_review_count'], user_analysis_data['anomaly_rate'],
                alpha=0.6, s=30, color='purple')
    plt.xlabel('Total Reviews per User')
    plt.ylabel('Anomaly Rate')
    plt.title('Anomaly Rate vs Total Reviews')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    # Add horizontal reference lines
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50% Anomaly Rate')
    plt.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='100% Anomaly Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_rate_vs_total_reviews.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {6} user anomaly analysis plots to: {output_dir}/")


def create_user_behavior_analysis(anomalies_df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive user behavior analysis from anomaly data.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots
    """
    print("Creating user behavior analysis...")

    if 'user_id' not in anomalies_df.columns or 'reviewer_review_count' not in anomalies_df.columns:
        print("  ⚠️ Required columns 'user_id' and 'reviewer_review_count' not found")
        return

    # Create user analysis data
    user_anomaly_counts = anomalies_df.groupby('user_id').size().reset_index(name='anomaly_count')
    user_total_counts = anomalies_df.groupby('user_id')['reviewer_review_count'].first().reset_index()
    user_analysis = pd.merge(user_anomaly_counts, user_total_counts, on='user_id', how='left')
    user_analysis['normal_reviews'] = user_analysis['reviewer_review_count'] - user_analysis['anomaly_count']
    user_analysis['anomaly_rate'] = user_analysis['anomaly_count'] / user_analysis['reviewer_review_count']
    user_analysis = user_analysis.dropna()

    if len(user_analysis) > 0:
        create_user_anomaly_visualizations(user_analysis, output_dir)
    else:
        print("  ⚠️ No valid user analysis data available")


def create_top_users_analysis(anomalies_df: pd.DataFrame, output_dir: str, top_n: int = 20):
    """
    Create analysis of top users by anomaly count.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots
        top_n: Number of top users to analyze
    """
    print(f"Creating top {top_n} users analysis...")

    if 'user_id' not in anomalies_df.columns:
        print("  ⚠️ 'user_id' column not found")
        return

    # Get top users by anomaly count
    user_anomaly_counts = anomalies_df['user_id'].value_counts().head(top_n)

    if len(user_anomaly_counts) == 0:
        print("  ⚠️ No users found with anomalies")
        return

    # Create horizontal bar chart
    plt.figure(figsize=(12, max(8, top_n * 0.4)))

    y_pos = np.arange(len(user_anomaly_counts))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(user_anomaly_counts)))

    bars = plt.barh(y_pos, user_anomaly_counts.values, color=colors, alpha=0.8)

    plt.yticks(y_pos, [f'User {uid}' for uid in user_anomaly_counts.index])
    plt.xlabel('Number of Anomalous Reviews')
    plt.title(f'Top {top_n} Users by Anomaly Count', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, user_anomaly_counts.values)):
        plt.text(bar.get_width() + max(user_anomaly_counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{value}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top_{top_n}_users_by_anomaly_count.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Top users analysis saved to: {output_dir}/")


def analyze_user_anomaly_patterns(anomalies_df: pd.DataFrame) -> Dict:
    """
    Analyze user behavior patterns in anomaly detection results.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        
    Returns:
        Dictionary with user pattern analysis results
    """
    print("\n" + "=" * 60)
    print("USER ANOMALY PATTERN ANALYSIS")
    print("=" * 60)

    results = {}

    if 'user_id' not in anomalies_df.columns:
        print("⚠️ No user_id column found in anomaly data")
        return results

    if 'reviewer_review_count' not in anomalies_df.columns:
        print("⚠️ No reviewer_review_count column found in anomaly data")
        return results

    # Count anomalous reviews per user
    user_anomaly_counts = anomalies_df.groupby('user_id').size().reset_index(name='anomaly_count')

    # Get reviewer_review_count for each user
    user_total_counts = anomalies_df.groupby('user_id')['reviewer_review_count'].first().reset_index()

    # Merge the data
    user_analysis = pd.merge(user_anomaly_counts, user_total_counts, on='user_id', how='left')

    # Calculate normal reviews and anomaly rate
    user_analysis['normal_reviews'] = user_analysis['reviewer_review_count'] - user_analysis['anomaly_count']
    user_analysis['anomaly_rate'] = user_analysis['anomaly_count'] / user_analysis['reviewer_review_count']

    # Remove users with missing data
    user_analysis = user_analysis.dropna()

    if len(user_analysis) == 0:
        print("⚠️ No valid user data found for analysis")
        return results

    # Calculate statistics
    stats = {
        'total_anomalous_users': len(user_analysis),
        'total_anomalous_reviews': int(user_analysis['anomaly_count'].sum()),
        'avg_anomalies_per_user': float(user_analysis['anomaly_count'].mean()),
        'median_anomalies_per_user': float(user_analysis['anomaly_count'].median()),
        'max_anomalies_per_user': int(user_analysis['anomaly_count'].max()),
        'avg_total_reviews_per_user': float(user_analysis['reviewer_review_count'].mean()),
        'median_total_reviews_per_user': float(user_analysis['reviewer_review_count'].median()),
        'avg_normal_reviews_per_user': float(user_analysis['normal_reviews'].mean()),
        'median_normal_reviews_per_user': float(user_analysis['normal_reviews'].median()),
        'avg_anomaly_rate': float(user_analysis['anomaly_rate'].mean()),
        'median_anomaly_rate': float(user_analysis['anomaly_rate'].median()),
        'users_with_all_anomalous': int((user_analysis['anomaly_rate'] == 1.0).sum()),
        'users_with_high_anomaly_rate': int((user_analysis['anomaly_rate'] > 0.5).sum()),
        'users_with_single_review': int((user_analysis['reviewer_review_count'] == 1).sum()),
        'users_with_multiple_reviews': int((user_analysis['reviewer_review_count'] > 1).sum())
    }

    # Print summary
    print(f"User Anomaly Analysis Summary:")
    print(f"  Users with anomalous reviews: {stats['total_anomalous_users']:,}")
    print(f"  Total anomalous reviews: {stats['total_anomalous_reviews']:,}")
    print(f"  Avg anomalies per user: {stats['avg_anomalies_per_user']:.2f}")
    print(f"  Avg total reviews per user: {stats['avg_total_reviews_per_user']:.2f}")
    print(f"  Avg normal reviews per user: {stats['avg_normal_reviews_per_user']:.2f}")
    print(f"  Avg anomaly rate: {stats['avg_anomaly_rate']:.3f} ({stats['avg_anomaly_rate'] * 100:.1f}%)")
    print(f"  Users with all reviews anomalous: {stats['users_with_all_anomalous']:,}")
    print(f"  Users with >50% anomaly rate: {stats['users_with_high_anomaly_rate']:,}")
    print(f"  Users with single review: {stats['users_with_single_review']:,}")

    # Percentile analysis
    percentiles = {
        'anomaly_count_percentiles': {
            '25th': float(user_analysis['anomaly_count'].quantile(0.25)),
            '50th': float(user_analysis['anomaly_count'].quantile(0.50)),
            '75th': float(user_analysis['anomaly_count'].quantile(0.75)),
            '90th': float(user_analysis['anomaly_count'].quantile(0.90)),
            '95th': float(user_analysis['anomaly_count'].quantile(0.95))
        },
        'total_reviews_percentiles': {
            '25th': float(user_analysis['reviewer_review_count'].quantile(0.25)),
            '50th': float(user_analysis['reviewer_review_count'].quantile(0.50)),
            '75th': float(user_analysis['reviewer_review_count'].quantile(0.75)),
            '90th': float(user_analysis['reviewer_review_count'].quantile(0.90)),
            '95th': float(user_analysis['reviewer_review_count'].quantile(0.95))
        },
        'anomaly_rate_percentiles': {
            '25th': float(user_analysis['anomaly_rate'].quantile(0.25)),
            '50th': float(user_analysis['anomaly_rate'].quantile(0.50)),
            '75th': float(user_analysis['anomaly_rate'].quantile(0.75)),
            '90th': float(user_analysis['anomaly_rate'].quantile(0.90)),
            '95th': float(user_analysis['anomaly_rate'].quantile(0.95))
        }
    }

    print(f"\nPercentile Analysis:")
    print(f"  Anomaly count - 50th: {percentiles['anomaly_count_percentiles']['50th']:.1f}, "
          f"90th: {percentiles['anomaly_count_percentiles']['90th']:.1f}, "
          f"95th: {percentiles['anomaly_count_percentiles']['95th']:.1f}")
    print(f"  Total reviews - 50th: {percentiles['total_reviews_percentiles']['50th']:.1f}, "
          f"90th: {percentiles['total_reviews_percentiles']['90th']:.1f}, "
          f"95th: {percentiles['total_reviews_percentiles']['95th']:.1f}")
    print(f"  Anomaly rate - 50th: {percentiles['anomaly_rate_percentiles']['50th']:.3f}, "
          f"90th: {percentiles['anomaly_rate_percentiles']['90th']:.3f}, "
          f"95th: {percentiles['anomaly_rate_percentiles']['95th']:.3f}")

    # Key insights
    print(f"\nKey Insights:")
    fraction_single = stats['users_with_single_review'] / stats['total_anomalous_users']
    fraction_all_anomalous = stats['users_with_all_anomalous'] / stats['total_anomalous_users']
    fraction_high_rate = stats['users_with_high_anomaly_rate'] / stats['total_anomalous_users']

    print(f"  • {fraction_single:.1%} of users have only 1 review (all anomalous)")
    print(f"  • {fraction_all_anomalous:.1%} of users have ALL reviews flagged as anomalous")
    print(f"  • {fraction_high_rate:.1%} of users have >50% anomaly rate")
    print(f"  • Average {stats['avg_normal_reviews_per_user']:.1f} normal reviews per user")
    print(f"    (These could contain undetected anomalies)")

    results.update(stats)
    results.update(percentiles)
    results['user_analysis_data'] = user_analysis

    return results


def run_user_analysis(anomalies_df: pd.DataFrame, output_dir: str = "evaluation_plots") -> Dict:
    """
    Run complete user analysis including both analysis and visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots and results
        
    Returns:
        Dictionary with all user analysis results
    """

    # Create user_analysis subdirectory
    user_plots_dir = os.path.join(output_dir, "user_analysis")
    os.makedirs(user_plots_dir, exist_ok=True)

    user_results = analyze_user_anomaly_patterns(anomalies_df)

    if 'user_id' in anomalies_df.columns and 'reviewer_review_count' in anomalies_df.columns:
        create_user_behavior_analysis(anomalies_df, user_plots_dir)
        create_top_users_analysis(anomalies_df, user_plots_dir)
    else:
        print("⚠️ Required user columns not found - skipping user visualizations")

    print("✅ User analysis complete!")
    return user_results
