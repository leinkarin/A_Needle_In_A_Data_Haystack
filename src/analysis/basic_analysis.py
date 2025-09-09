import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")


def create_rating_distribution_plot(anomalies_df: pd.DataFrame, output_dir: str):
    """
    Create a standalone rating distribution plot.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save the plot
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

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved rating distribution plot to: {os.path.join(output_dir, 'rating_distribution.png')}")


def create_basic_feature_distributions(anomalies_df: pd.DataFrame, output_dir: str):
    """
    Create individual feature distribution plots for anomalies.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots
    """

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
        plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved {len(available_features)} feature distribution plots to: {output_dir}/")


def create_rating_comparison_plot(anomalies_df: pd.DataFrame, original_df: pd.DataFrame, output_dir: str):
    """
    Create rating distribution comparison between anomalies and normal reviews.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        original_df: Optional original dataset for comparison plots
        output_dir: Directory to save plots
    """
    if 'rating' not in anomalies_df.columns:
        return

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
    plt.ylabel('Count (Log Scale)')
    plt.yscale('log')
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rating_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_rating_vs_rating_diff_analysis(anomalies_df: pd.DataFrame, original_df: pd.DataFrame, output_dir: str):
    """
    Create rating vs rating difference analysis plot.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        original_df: Original dataset for comparison
        output_dir: Directory to save plots
    """
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


def create_comparison_plots(anomalies_df: pd.DataFrame, original_df: pd.DataFrame, output_dir: str):
    """Create additional comparison plots between anomalies and normal data."""

    numerical_features = ['helpful_vote', 'reviewer_review_count', 'rating_diff']
    available_numerical = [col for col in numerical_features
                           if col in anomalies_df.columns and col in original_df.columns]

    if available_numerical:
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

    categorical_features = ['verified_purchase', 'has_images']
    available_categorical = [col for col in categorical_features
                             if col in anomalies_df.columns and col in original_df.columns]

    if available_categorical:
        anomaly_indices = set(anomalies_df.index)
        normal_mask = ~original_df.index.isin(anomaly_indices)

        for feature in available_categorical:
            plt.figure(figsize=(8, 6))

            normal_data = original_df.loc[normal_mask, feature]
            anomaly_data = anomalies_df[feature]

            normal_true_pct = normal_data.mean() * 100
            normal_false_pct = (1 - normal_data.mean()) * 100
            anomaly_true_pct = anomaly_data.mean() * 100
            anomaly_false_pct = (1 - anomaly_data.mean()) * 100

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

            for j, (normal_val, anomaly_val) in enumerate(zip(normal_values, anomaly_values)):
                plt.text(j - width / 2, normal_val + 1, f'{normal_val:.1f}%',
                         ha='center', va='bottom')
                plt.text(j + width / 2, anomaly_val + 1, f'{anomaly_val:.1f}%',
                         ha='center', va='bottom')

            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'{feature}_categorical_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()


def create_basic_metrics_visualization(anomalies_df: pd.DataFrame, original_df: pd.DataFrame, output_dir: str):
    basic_analysis_dir = os.path.join(output_dir, "basic_analysis")
    os.makedirs(basic_analysis_dir, exist_ok=True)
    create_rating_distribution_plot(anomalies_df, basic_analysis_dir)
    create_basic_feature_distributions(anomalies_df, basic_analysis_dir)
    create_rating_comparison_plot(anomalies_df, original_df, basic_analysis_dir)
    create_rating_vs_rating_diff_analysis(anomalies_df, original_df, basic_analysis_dir)
    create_comparison_plots(anomalies_df, original_df, basic_analysis_dir)


def analyze_anomaly_patterns(anomalies_df: pd.DataFrame) -> Dict:
    """
    Analyze patterns in the detected anomalies.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        
    Returns:
        Dictionary with pattern analysis results
    """
    print("\n" + "=" * 60)
    print("ANOMALY PATTERN ANALYSIS")
    print("=" * 60)

    results = {}

    # Rating vs Predicted Rating Analysis
    if 'rating' in anomalies_df.columns and 'predicted_rating' in anomalies_df.columns:
        print("Rating vs Predicted Rating Analysis:")

        rating_diff = anomalies_df['rating'] - anomalies_df['predicted_rating']
        mean_diff = rating_diff.mean()
        std_diff = rating_diff.std()

        print(f"  Mean rating difference: {mean_diff:.4f}")
        print(f"  Std rating difference: {std_diff:.4f}")

        # Categorize by rating difference
        positive_diff = rating_diff > 0
        negative_diff = rating_diff < 0

        print(
            f"  Positive differences (actual > predicted): {positive_diff.sum()} ({positive_diff.mean() * 100:.1f}%)")
        print(
            f"  Negative differences (actual < predicted): {negative_diff.sum()} ({negative_diff.mean() * 100:.1f}%)")

        results['mean_rating_diff'] = mean_diff
        results['std_rating_diff'] = std_diff
        results['positive_diff_count'] = positive_diff.sum()
        results['negative_diff_count'] = negative_diff.sum()
        results['positive_diff_percentage'] = positive_diff.mean() * 100
        results['negative_diff_percentage'] = negative_diff.mean() * 100

    # Helpful votes analysis
    if 'helpful_vote' in anomalies_df.columns:
        print(f"\nHelpful Votes Analysis:")
        helpful_votes = anomalies_df['helpful_vote']

        zero_votes = (helpful_votes == 0).sum()
        low_votes = ((helpful_votes > 0) & (helpful_votes <= 5)).sum()
        high_votes = (helpful_votes > 5).sum()

        print(f"  Zero votes: {zero_votes} ({zero_votes / len(helpful_votes) * 100:.1f}%)")
        print(f"  Low votes (1-5): {low_votes} ({low_votes / len(helpful_votes) * 100:.1f}%)")
        print(f"  High votes (>5): {high_votes} ({high_votes / len(helpful_votes) * 100:.1f}%)")

        results['zero_votes_count'] = zero_votes
        results['low_votes_count'] = low_votes
        results['high_votes_count'] = high_votes
        results['zero_votes_percentage'] = zero_votes / len(helpful_votes) * 100
        results['low_votes_percentage'] = low_votes / len(helpful_votes) * 100
        results['high_votes_percentage'] = high_votes / len(helpful_votes) * 100

    # Verified purchase analysis
    if 'verified_purchase' in anomalies_df.columns:
        print(f"\nVerified Purchase Analysis:")
        verified = anomalies_df['verified_purchase']

        verified_count = verified.sum()
        unverified_count = len(verified) - verified_count

        print(f"  Verified purchases: {verified_count} ({verified_count / len(verified) * 100:.1f}%)")
        print(f"  Unverified purchases: {unverified_count} ({unverified_count / len(verified) * 100:.1f}%)")

        results['verified_count'] = verified_count
        results['unverified_count'] = unverified_count
        results['verified_percentage'] = verified_count / len(verified) * 100
        results['unverified_percentage'] = unverified_count / len(verified) * 100

    # Has images analysis
    if 'has_images' in anomalies_df.columns:
        print(f"\nImages Analysis:")
        has_images = anomalies_df['has_images']

        with_images = has_images.sum()
        without_images = len(has_images) - with_images

        print(f"  Reviews with images: {with_images} ({with_images / len(has_images) * 100:.1f}%)")
        print(f"  Reviews without images: {without_images} ({without_images / len(has_images) * 100:.1f}%)")

        results['with_images_count'] = with_images
        results['without_images_count'] = without_images
        results['with_images_percentage'] = with_images / len(has_images) * 100
        results['without_images_percentage'] = without_images / len(has_images) * 100

    return results


def run_basic_metrics_analysis(anomalies_df: pd.DataFrame, original_df: pd.DataFrame = None,
                               output_dir: str = "evaluation_plots") -> Dict:
    """
    Run complete basic metrics analysis including both analysis and visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        original_df: Optional original dataset for comparison
        output_dir: Directory to save plots and results
        
    Returns:
        Dictionary with all basic metrics analysis results
    """
    # Run analysis
    analysis_results = analyze_anomaly_patterns(anomalies_df)

    # Create visualizations if original data is available
    if original_df is not None:
        create_basic_metrics_visualization(anomalies_df, original_df, output_dir)
    else:
        # Still create basic plots that don't require comparison
        create_rating_distribution_plot(anomalies_df, output_dir)
        create_basic_feature_distributions(anomalies_df, output_dir)

    return analysis_results
