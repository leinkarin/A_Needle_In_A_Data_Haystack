import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")


def get_available_text_columns(anomalies_df: pd.DataFrame) -> list:
    """Get available text columns from the dataframe."""
    possible_text_columns = ['text', 'review_text', 'review_data', 'summary', 'title']
    return [col for col in possible_text_columns if col in anomalies_df.columns]


def analyze_review_length_anomalies(anomalies_df: pd.DataFrame, original_df: pd.DataFrame = None) -> Dict:
    """
    Analyze review length patterns to identify unusually short or long reviews.

    Args:
        anomalies_df: DataFrame containing anomaly detection results
        original_df: Optional original dataset for comparison

    Returns:
        Dictionary with review length analysis results
    """
    print("\n" + "=" * 60)
    print("REVIEW LENGTH ANOMALY ANALYSIS")
    print("=" * 60)

    results = {}

    # Check for text-related columns
    text_columns = get_available_text_columns(anomalies_df)

    if not text_columns:
        print("⚠️ No text columns found for length analysis")
        return results

    # Analyze each text column
    for col in text_columns:
        print(f"\n{col.title()} Length Analysis:")

        # Calculate lengths
        text_data = anomalies_df[col].fillna('')
        char_lengths = text_data.str.len()
        word_lengths = text_data.str.split().str.len().fillna(0)

        # Basic statistics
        char_stats = {
            'mean': char_lengths.mean(),
            'median': char_lengths.median(),
            'std': char_lengths.std(),
            'min': char_lengths.min(),
            'max': char_lengths.max(),
            'q25': char_lengths.quantile(0.25),
            'q75': char_lengths.quantile(0.75)
        }

        word_stats = {
            'mean': word_lengths.mean(),
            'median': word_lengths.median(),
            'std': word_lengths.std(),
            'min': word_lengths.min(),
            'max': word_lengths.max(),
            'q25': word_lengths.quantile(0.25),
            'q75': word_lengths.quantile(0.75)
        }

        print(f"  Character Length Statistics:")
        print(f"    Mean: {char_stats['mean']:.1f}, Median: {char_stats['median']:.1f}")
        print(f"    Std: {char_stats['std']:.1f}, Range: [{char_stats['min']}, {char_stats['max']}]")
        print(f"    IQR: [{char_stats['q25']:.1f}, {char_stats['q75']:.1f}]")

        print(f"  Word Count Statistics:")
        print(f"    Mean: {word_stats['mean']:.1f}, Median: {word_stats['median']:.1f}")
        print(f"    Std: {word_stats['std']:.1f}, Range: [{word_stats['min']}, {word_stats['max']}]")
        print(f"    IQR: [{word_stats['q25']:.1f}, {word_stats['q75']:.1f}]")

        # Identify anomalous lengths using IQR method
        iqr_char = char_stats['q75'] - char_stats['q25']
        char_lower_bound = char_stats['q25'] - 1.5 * iqr_char
        char_upper_bound = char_stats['q75'] + 1.5 * iqr_char

        iqr_word = word_stats['q75'] - word_stats['q25']
        word_lower_bound = word_stats['q25'] - 1.5 * iqr_word
        word_upper_bound = word_stats['q75'] + 1.5 * iqr_word

        # Count anomalous lengths
        extremely_short_char = (char_lengths < max(char_lower_bound, 10)).sum()  # At least 10 chars
        extremely_long_char = (char_lengths > char_upper_bound).sum()
        extremely_short_word = (word_lengths < max(word_lower_bound, 2)).sum()  # At least 2 words
        extremely_long_word = (word_lengths > word_upper_bound).sum()

        # Empty or near-empty reviews
        empty_reviews = (char_lengths <= 5).sum()
        single_word_reviews = (word_lengths <= 1).sum()

        # Very long reviews (potential spam)
        very_long_char = (char_lengths > char_stats['mean'] + 3 * char_stats['std']).sum()
        very_long_word = (word_lengths > word_stats['mean'] + 3 * word_stats['std']).sum()

        print(f"  Length Anomaly Detection:")
        print(f"    Empty/near-empty (≤5 chars): {empty_reviews} ({empty_reviews / len(char_lengths) * 100:.1f}%)")
        print(f"    Single word reviews: {single_word_reviews} ({single_word_reviews / len(word_lengths) * 100:.1f}%)")
        print(f"    Extremely short (IQR method): {extremely_short_char} chars, {extremely_short_word} words")
        print(f"    Extremely long (IQR method): {extremely_long_char} chars, {extremely_long_word} words")
        print(f"    Very long (>3σ): {very_long_char} chars, {very_long_word} words")

        # Store results
        results[f'{col}_char_stats'] = char_stats
        results[f'{col}_word_stats'] = word_stats
        results[f'{col}_empty_reviews'] = empty_reviews
        results[f'{col}_single_word_reviews'] = single_word_reviews
        results[f'{col}_extremely_short_char'] = extremely_short_char
        results[f'{col}_extremely_long_char'] = extremely_long_char
        results[f'{col}_extremely_short_word'] = extremely_short_word
        results[f'{col}_extremely_long_word'] = extremely_long_word
        results[f'{col}_very_long_char'] = very_long_char
        results[f'{col}_very_long_word'] = very_long_word

        # Length vs rating correlation (if rating available)
        if 'rating' in anomalies_df.columns:
            char_rating_corr = char_lengths.corr(anomalies_df['rating'])
            word_rating_corr = word_lengths.corr(anomalies_df['rating'])

            print(f"  Length vs Rating Correlation:")
            print(f"    Character length: {char_rating_corr:.3f}")
            print(f"    Word count: {word_rating_corr:.3f}")

            results[f'{col}_char_rating_correlation'] = char_rating_corr
            results[f'{col}_word_rating_correlation'] = word_rating_corr

        # Length vs helpfulness correlation (if helpful_vote available)
        if 'helpful_vote' in anomalies_df.columns:
            char_help_corr = char_lengths.corr(anomalies_df['helpful_vote'])
            word_help_corr = word_lengths.corr(anomalies_df['helpful_vote'])

            print(f"  Length vs Helpfulness Correlation:")
            print(f"    Character length: {char_help_corr:.3f}")
            print(f"    Word count: {word_help_corr:.3f}")

            results[f'{col}_char_helpfulness_correlation'] = char_help_corr
            results[f'{col}_word_helpfulness_correlation'] = word_help_corr

    # Compare with original data if available
    if original_df is not None:
        print(f"\nComparison with Original Dataset:")
        for col in text_columns:
            if col in original_df.columns:
                orig_char_lengths = original_df[col].fillna('').str.len()
                orig_word_lengths = original_df[col].fillna('').str.split().str.len().fillna(0)

                # Recalculate for comparison (since we're outside the loop)
                text_data = anomalies_df[col].fillna('')
                anom_char_lengths = text_data.str.len()
                anom_word_lengths = text_data.str.split().str.len().fillna(0)

                anom_char_mean = char_lengths.mean()
                orig_char_mean = orig_char_lengths.mean()
                anom_word_mean = word_lengths.mean()
                orig_word_mean = orig_word_lengths.mean()

                char_diff_pct = ((anom_char_mean - orig_char_mean) / orig_char_mean) * 100 if orig_char_mean > 0 else 0
                word_diff_pct = ((anom_word_mean - orig_word_mean) / orig_word_mean) * 100 if orig_word_mean > 0 else 0

                print(f"  {col.title()} Length Comparison:")
                print(
                    f"    Anomaly avg chars: {anom_char_mean:.1f} vs Original: {orig_char_mean:.1f} ({char_diff_pct:+.1f}%)")
                print(
                    f"    Anomaly avg words: {anom_word_mean:.1f} vs Original: {orig_word_mean:.1f} ({word_diff_pct:+.1f}%)")

                results[f'{col}_char_length_diff_pct'] = char_diff_pct
                results[f'{col}_word_length_diff_pct'] = word_diff_pct

    return results


def create_length_comparison_visualizations(anomalies_df: pd.DataFrame, original_df: pd.DataFrame,
                                            length_analysis: dict, output_dir: str):
    """
    Create comprehensive visualizations comparing review lengths between anomalous and normal reviews.
    
    Args:
        anomalies_df: DataFrame with anomalous reviews
        original_df: DataFrame with original/normal reviews  
        length_analysis: Dictionary with length analysis results
        output_dir: Directory to save plots
    """
    print("  📊 Creating review length comparison visualizations...")

    # Get text columns
    text_cols = []
    for col in ['text', 'review_data', 'review_text']:
        if col in anomalies_df.columns and col in original_df.columns:
            text_cols.append(col)

    if not text_cols:
        print("  ⚠️ No common text columns found for length comparison")
        return

    primary_text_col = text_cols[0]

    # Calculate lengths for both datasets
    anomaly_texts = anomalies_df[primary_text_col].fillna('').astype(str)
    original_texts = original_df[primary_text_col].fillna('').astype(str)

    anomaly_char_lengths = anomaly_texts.str.len()
    anomaly_word_lengths = anomaly_texts.str.split().str.len()

    original_char_lengths = original_texts.str.len()
    original_word_lengths = original_texts.str.split().str.len()

    # 1. Character length histograms (log scale) - Separate figure
    plt.figure(figsize=(10, 6))
    bins = np.logspace(0, 4, 50)  # Log scale bins from 1 to 10,000
    plt.hist(original_char_lengths[original_char_lengths > 0], bins=bins, alpha=0.7,
             label=f'Normal Reviews (n={len(original_char_lengths):,})', color='skyblue', density=True)
    plt.hist(anomaly_char_lengths[anomaly_char_lengths > 0], bins=bins, alpha=0.7,
             label=f'Anomalous Reviews (n={len(anomaly_char_lengths):,})', color='red', density=True)
    plt.xscale('log')
    plt.xlabel('Character Count (log scale)')
    plt.ylabel('Density')
    plt.title('Character Length Distribution: Anomalous vs Normal Reviews', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'character_length_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Word length histograms (log scale) - Separate figure
    plt.figure(figsize=(10, 6))
    word_bins = np.logspace(0, 3, 50)  # Log scale bins from 1 to 1,000
    plt.hist(original_word_lengths[original_word_lengths > 0], bins=word_bins, alpha=0.7,
             label=f'Normal Reviews', color='skyblue', density=True)
    plt.hist(anomaly_word_lengths[anomaly_word_lengths > 0], bins=word_bins, alpha=0.7,
             label=f'Anomalous Reviews', color='red', density=True)
    plt.xscale('log')
    plt.xlabel('Word Count (log scale)')
    plt.ylabel('Density')
    plt.title('Word Length Distribution: Anomalous vs Normal Reviews', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_length_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Statistical summary table - Separate figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Calculate statistics
    stats_data = [
        ['Metric', 'Normal Reviews', 'Anomalous Reviews', 'Difference'],
        ['Count', f'{len(original_char_lengths):,}', f'{len(anomaly_char_lengths):,}', ''],
        ['Mean Chars', f'{original_char_lengths.mean():.1f}', f'{anomaly_char_lengths.mean():.1f}',
         f'{((anomaly_char_lengths.mean() - original_char_lengths.mean()) / original_char_lengths.mean() * 100):+.1f}%'],
        ['Median Chars', f'{original_char_lengths.median():.1f}', f'{anomaly_char_lengths.median():.1f}',
         f'{((anomaly_char_lengths.median() - original_char_lengths.median()) / original_char_lengths.median() * 100):+.1f}%'],
        ['Mean Words', f'{original_word_lengths.mean():.1f}', f'{anomaly_word_lengths.mean():.1f}',
         f'{((anomaly_word_lengths.mean() - original_word_lengths.mean()) / original_word_lengths.mean() * 100):+.1f}%'],
        ['Median Words', f'{original_word_lengths.median():.1f}', f'{anomaly_word_lengths.median():.1f}',
         f'{((anomaly_word_lengths.median() - original_word_lengths.median()) / original_word_lengths.median() * 100):+.1f}%'],
        ['Std Chars', f'{original_char_lengths.std():.1f}', f'{anomaly_char_lengths.std():.1f}', ''],
        ['Min Chars', f'{original_char_lengths.min():.0f}', f'{anomaly_char_lengths.min():.0f}', ''],
        ['Max Chars', f'{original_char_lengths.max():.0f}', f'{anomaly_char_lengths.max():.0f}', '']
    ]

    table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Color code the header
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Statistical Summary: Anomalous vs Normal Reviews', pad=20, fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_statistics_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Character length distribution saved to: length_analysis/character_length_distribution.png")
    print(f"  ✅ Word length distribution saved to: length_analysis/word_length_distribution.png")
    print(f"  ✅ Length statistics summary saved to: length_analysis/length_statistics_summary.png")


def create_detailed_length_analysis(anomalies_df: pd.DataFrame, original_df: pd.DataFrame, output_dir: str):
    """
    Create detailed length analysis with multiple visualizations.
    
    Args:
        anomalies_df: DataFrame with anomalous reviews
        original_df: DataFrame with original/normal reviews
        output_dir: Directory to save plots
    """
    print("Creating detailed length analysis visualizations...")

    # Create length analysis subdirectory
    length_plots_dir = os.path.join(output_dir, "length_analysis")
    os.makedirs(length_plots_dir, exist_ok=True)

    # Get text columns
    text_cols = []
    for col in ['text', 'review_data', 'review_text']:
        if col in anomalies_df.columns and col in original_df.columns:
            text_cols.append(col)

    if not text_cols:
        print("  ⚠️ No common text columns found for detailed length analysis")
        return

    primary_text_col = text_cols[0]

    # Calculate lengths for both datasets
    anomaly_texts = anomalies_df[primary_text_col].fillna('').astype(str)
    original_texts = original_df[primary_text_col].fillna('').astype(str)

    anomaly_char_lengths = anomaly_texts.str.len()
    anomaly_word_lengths = anomaly_texts.str.split().str.len()

    original_char_lengths = original_texts.str.len()
    original_word_lengths = original_texts.str.split().str.len()

    # 1. Box plots comparison
    create_length_boxplots(original_char_lengths, anomaly_char_lengths,
                           original_word_lengths, anomaly_word_lengths, output_dir)

    # 2. Cumulative distribution comparison
    create_length_cumulative_distributions(original_char_lengths, anomaly_char_lengths,
                                           original_word_lengths, anomaly_word_lengths, output_dir)

    # 3. Length vs rating analysis
    if 'rating' in anomalies_df.columns and 'rating' in original_df.columns:
        create_length_vs_rating_analysis(anomalies_df, original_df, primary_text_col, output_dir)

    # 4. Extreme length analysis
    create_extreme_length_analysis(original_char_lengths, anomaly_char_lengths,
                                   original_word_lengths, anomaly_word_lengths, output_dir)

    print(f"  ✓ Detailed length analysis saved to: {length_plots_dir}/")


def create_length_boxplots(original_char_lengths: pd.Series, anomaly_char_lengths: pd.Series,
                           original_word_lengths: pd.Series, anomaly_word_lengths: pd.Series,
                           output_dir: str):
    """Create box plots comparing length distributions."""

    # Character length box plots - Separate figure
    plt.figure(figsize=(10, 6))
    char_data = [original_char_lengths, anomaly_char_lengths]
    char_labels = ['Normal Reviews', 'Anomalous Reviews']

    bp1 = plt.boxplot(char_data, labels=char_labels, patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')

    plt.title('Character Length Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Character Count')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Add statistics
    plt.text(0.02, 0.98,
             f'Normal: μ={original_char_lengths.mean():.0f}, σ={original_char_lengths.std():.0f}\n'
             f'Anomaly: μ={anomaly_char_lengths.mean():.0f}, σ={anomaly_char_lengths.std():.0f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'character_length_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Word length box plots - Separate figure
    plt.figure(figsize=(10, 6))
    word_data = [original_word_lengths, anomaly_word_lengths]
    word_labels = ['Normal Reviews', 'Anomalous Reviews']

    bp2 = plt.boxplot(word_data, labels=word_labels, patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')

    plt.title('Word Length Comparison', fontweight='bold', fontsize=14)
    plt.ylabel('Word Count')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # Add statistics
    plt.text(0.02, 0.98,
             f'Normal: μ={original_word_lengths.mean():.0f}, σ={original_word_lengths.std():.0f}\n'
             f'Anomaly: μ={anomaly_word_lengths.mean():.0f}, σ={anomaly_word_lengths.std():.0f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_length_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_length_cumulative_distributions(original_char_lengths: pd.Series, anomaly_char_lengths: pd.Series,
                                           original_word_lengths: pd.Series, anomaly_word_lengths: pd.Series,
                                           output_dir: str):
    """Create cumulative distribution plots for length comparison."""

    # Character length CDF - Separate figure
    plt.figure(figsize=(10, 6))
    original_char_sorted = np.sort(original_char_lengths[original_char_lengths > 0])
    anomaly_char_sorted = np.sort(anomaly_char_lengths[anomaly_char_lengths > 0])

    y_orig_char = np.arange(1, len(original_char_sorted) + 1) / len(original_char_sorted)
    y_anom_char = np.arange(1, len(anomaly_char_sorted) + 1) / len(anomaly_char_sorted)

    plt.plot(original_char_sorted, y_orig_char, linewidth=2, color='blue', label='Normal Reviews')
    plt.plot(anomaly_char_sorted, y_anom_char, linewidth=2, color='red', label='Anomalous Reviews')

    plt.xlabel('Character Count')
    plt.ylabel('Cumulative Probability')
    plt.title('Character Length Cumulative Distribution', fontweight='bold', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add percentile lines
    for p, label in [(0.25, '25th'), (0.5, '50th'), (0.75, '75th'), (0.9, '90th')]:
        orig_val = original_char_lengths.quantile(p)
        anom_val = anomaly_char_lengths.quantile(p)
        plt.axvline(orig_val, color='blue', linestyle='--', alpha=0.5)
        plt.axvline(anom_val, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'character_length_cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Word length CDF - Separate figure
    plt.figure(figsize=(10, 6))
    original_word_sorted = np.sort(original_word_lengths[original_word_lengths > 0])
    anomaly_word_sorted = np.sort(anomaly_word_lengths[anomaly_word_lengths > 0])

    y_orig_word = np.arange(1, len(original_word_sorted) + 1) / len(original_word_sorted)
    y_anom_word = np.arange(1, len(anomaly_word_sorted) + 1) / len(anomaly_word_sorted)

    plt.plot(original_word_sorted, y_orig_word, linewidth=2, color='blue', label='Normal Reviews')
    plt.plot(anomaly_word_sorted, y_anom_word, linewidth=2, color='red', label='Anomalous Reviews')

    plt.xlabel('Word Count')
    plt.ylabel('Cumulative Probability')
    plt.title('Word Length Cumulative Distribution', fontweight='bold', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add percentile lines
    for p, label in [(0.25, '25th'), (0.5, '50th'), (0.75, '75th'), (0.9, '90th')]:
        orig_val = original_word_lengths.quantile(p)
        anom_val = anomaly_word_lengths.quantile(p)
        plt.axvline(orig_val, color='blue', linestyle='--', alpha=0.5)
        plt.axvline(anom_val, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_length_cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_length_vs_rating_analysis(anomalies_df: pd.DataFrame, original_df: pd.DataFrame,
                                     text_col: str, output_dir: str):
    """Create analysis of length vs rating patterns."""

    # Calculate lengths
    anomaly_texts = anomalies_df[text_col].fillna('').astype(str)
    original_texts = original_df[text_col].fillna('').astype(str)

    anomaly_char_lengths = anomaly_texts.str.len()
    original_char_lengths = original_texts.str.len()

    # Exclude anomalies from original data
    anomaly_indices = set(anomalies_df.index)
    normal_mask = ~original_df.index.isin(anomaly_indices)
    normal_df = original_df.loc[normal_mask].copy()
    normal_lengths = original_texts.loc[normal_mask]
    normal_char_lengths = normal_lengths.str.len()

    # 1. Length vs Rating scatter plot - Separate figure
    plt.figure(figsize=(10, 6))
    if len(normal_df) > 10000:  # Sample if too large
        sample_indices = normal_df.sample(10000).index
        sample_lengths = normal_char_lengths.loc[sample_indices]
        sample_ratings = normal_df.loc[sample_indices, 'rating']
    else:
        sample_lengths = normal_char_lengths
        sample_ratings = normal_df['rating']

    plt.scatter(sample_lengths, sample_ratings, alpha=0.3, s=10, color='blue', label='Normal Reviews')
    plt.scatter(anomaly_char_lengths, anomalies_df['rating'], alpha=0.7, s=20, color='red', label='Anomalous Reviews')

    plt.xlabel('Character Count')
    plt.ylabel('Rating')
    plt.title('Review Length vs Rating', fontweight='bold', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_vs_rating_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Average length by rating - Separate figure
    plt.figure(figsize=(10, 6))
    normal_by_rating = normal_df.groupby('rating')[text_col].apply(lambda x: x.fillna('').astype(str).str.len().mean())
    anomaly_by_rating = anomalies_df.groupby('rating')[text_col].apply(
        lambda x: x.fillna('').astype(str).str.len().mean())

    ratings = range(1, 6)
    normal_means = [normal_by_rating.get(r, 0) for r in ratings]
    anomaly_means = [anomaly_by_rating.get(r, 0) for r in ratings]

    x = np.arange(len(ratings))
    width = 0.35

    plt.bar(x - width / 2, normal_means, width, label='Normal Reviews', alpha=0.7, color='blue')
    plt.bar(x + width / 2, anomaly_means, width, label='Anomalous Reviews', alpha=0.7, color='red')

    plt.xlabel('Rating')
    plt.ylabel('Average Character Count')
    plt.title('Average Review Length by Rating', fontweight='bold', fontsize=14)
    plt.xticks(x, ratings)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (normal_val, anomaly_val) in enumerate(zip(normal_means, anomaly_means)):
        if normal_val > 0:
            plt.text(i - width / 2, normal_val + max(normal_means) * 0.01, f'{normal_val:.0f}',
                     ha='center', va='bottom', fontsize=8)
        if anomaly_val > 0:
            plt.text(i + width / 2, anomaly_val + max(anomaly_means) * 0.01, f'{anomaly_val:.0f}',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_length_by_rating.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_extreme_length_analysis(original_char_lengths: pd.Series, anomaly_char_lengths: pd.Series,
                                   original_word_lengths: pd.Series, anomaly_word_lengths: pd.Series,
                                   output_dir: str):
    """Analyze extreme length cases (very short and very long reviews)."""

    categories = ['Normal Reviews', 'Anomalous Reviews']

    # 1. Very short reviews (< 50 characters) - Separate figure
    plt.figure(figsize=(10, 6))
    short_normal = (original_char_lengths < 50).sum()
    short_anomaly = (anomaly_char_lengths < 50).sum()
    short_normal_pct = short_normal / len(original_char_lengths) * 100
    short_anomaly_pct = short_anomaly / len(anomaly_char_lengths) * 100

    short_counts = [short_normal, short_anomaly]
    short_pcts = [short_normal_pct, short_anomaly_pct]

    bars1 = plt.bar(categories, short_counts, alpha=0.7, color=['blue', 'red'])
    plt.title('Very Short Reviews (< 50 characters)', fontweight='bold', fontsize=14)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, count, pct in zip(bars1, short_counts, short_pcts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + max(short_counts) * 0.01,
                 f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'very_short_reviews_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Very long reviews (> 1000 characters) - Separate figure
    plt.figure(figsize=(10, 6))
    long_normal = (original_char_lengths > 1000).sum()
    long_anomaly = (anomaly_char_lengths > 1000).sum()
    long_normal_pct = long_normal / len(original_char_lengths) * 100
    long_anomaly_pct = long_anomaly / len(anomaly_char_lengths) * 100

    long_counts = [long_normal, long_anomaly]
    long_pcts = [long_normal_pct, long_anomaly_pct]

    bars2 = plt.bar(categories, long_counts, alpha=0.7, color=['blue', 'red'])
    plt.title('Very Long Reviews (> 1000 characters)', fontweight='bold', fontsize=14)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, count, pct in zip(bars2, long_counts, long_pcts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + max(long_counts) * 0.01,
                 f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'very_long_reviews_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Length distribution by percentiles - Separate figure
    plt.figure(figsize=(10, 6))
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    normal_percentiles = [original_char_lengths.quantile(p / 100) for p in percentiles]
    anomaly_percentiles = [anomaly_char_lengths.quantile(p / 100) for p in percentiles]

    plt.plot(percentiles, normal_percentiles, 'o-', linewidth=2, markersize=6,
             color='blue', label='Normal Reviews')
    plt.plot(percentiles, anomaly_percentiles, 's-', linewidth=2, markersize=6,
             color='red', label='Anomalous Reviews')

    plt.xlabel('Percentile')
    plt.ylabel('Character Count')
    plt.title('Length Distribution by Percentiles', fontweight='bold', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_percentiles_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Length ratio analysis - Separate figure
    plt.figure(figsize=(10, 6))
    length_bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
    length_labels = ['0-50', '50-100', '100-200', '200-500', '500-1K', '1K-2K', '2K+']

    normal_counts = pd.cut(original_char_lengths, bins=length_bins, labels=length_labels,
                           include_lowest=True).value_counts()
    anomaly_counts = pd.cut(anomaly_char_lengths, bins=length_bins, labels=length_labels,
                            include_lowest=True).value_counts()

    # Calculate ratios (anomaly/normal)
    ratios = []
    for label in length_labels:
        normal_count = normal_counts.get(label, 0)
        anomaly_count = anomaly_counts.get(label, 0)
        if normal_count > 0:
            ratio = (anomaly_count / len(anomaly_char_lengths)) / (normal_count / len(original_char_lengths))
        else:
            ratio = 0
        ratios.append(ratio)

    colors = ['green' if r < 1 else 'red' if r > 1.5 else 'orange' for r in ratios]
    bars4 = plt.bar(length_labels, ratios, alpha=0.7, color=colors)
    plt.xlabel('Character Count Range')
    plt.ylabel('Anomaly/Normal Ratio')
    plt.title('Length Distribution Ratio Analysis\n(>1 = More anomalies, <1 = Fewer anomalies)', fontweight='bold',
              fontsize=14)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal ratio')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()

    # Add value labels
    for bar, ratio in zip(bars4, ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + max(ratios) * 0.01,
                 f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_ratio_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_length_analysis(anomalies_df: pd.DataFrame, original_df: pd.DataFrame = None,
                        output_dir: str = "evaluation_plots") -> Dict:
    """
    Run complete length analysis including both analysis and visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        original_df: Optional original dataset for comparison
        output_dir: Directory to save plots and results
        
    Returns:
        Dictionary with all length analysis results
    """

    # Create length_analysis subdirectory
    length_plots_dir = os.path.join(output_dir, "length_analysis")
    os.makedirs(length_plots_dir, exist_ok=True)

    length_results = analyze_review_length_anomalies(anomalies_df, original_df)

    if length_results and original_df is not None:
        create_length_comparison_visualizations(anomalies_df, original_df, length_results, length_plots_dir)

    return length_results
