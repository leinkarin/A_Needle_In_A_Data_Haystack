import argparse
import os
import sys

import numpy as np
import pandas as pd

from post_anomaly_analysis import AnomalyTextAnalyzer


def load_and_tag(path: str, category: str, start_index: int = 0) -> pd.DataFrame:
    """
    Load a CSV file and add a category column with sequential indexing.
    
    Args:
        path: Path to the CSV file
        category: Category name to tag the data with
        start_index: Starting index for this category's data
        
    Returns:
        DataFrame with loaded data, category column, and original_index column
    """

    try:
        df = pd.read_csv(path)
        df['category'] = category
        df['original_index'] = range(start_index, start_index + len(df))
        return df
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        raise


def compute_category_z(df: pd.DataFrame, value_col: str, z_col: str, by: str = "category") -> pd.DataFrame:
    """
    Compute per-category z-scores for a given column.
    
    Args:
        df: Input DataFrame
        value_col: Column name containing values to normalize
        z_col: Column name for output z-scores
        by: Column to group by (default: "category")
        
    Returns:
        DataFrame with z-score column added
    """

    def compute_z_scores(group):
        values = group[value_col]

        valid_mask = pd.notna(values)
        if not valid_mask.any():
            print(f"No valid values for {value_col} in category {group.name}")
            group[z_col] = np.nan
            return group

        valid_values = values[valid_mask]
        mean_val = valid_values.mean()
        std_val = valid_values.std()

        if std_val == 0 or pd.isna(std_val):
            print(f"Zero or NaN std for {value_col} in category {group.name}, setting z-scores to NaN")
            group[z_col] = np.nan
        else:
            z_scores = pd.Series(np.nan, index=group.index)
            z_scores[valid_mask] = (valid_values - mean_val) / std_val
            group[z_col] = z_scores

        return group

    result = df.groupby(by, group_keys=False).apply(compute_z_scores)
    return result


def summarize_by_category(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """
    Create summary statistics by category.
    
    Args:
        df: Input DataFrame with anomaly detection results
        z_threshold: Z-score threshold used for anomaly detection
        
    Returns:
        DataFrame with summary statistics per category
    """
    summary_rows = []

    for category in df['category'].unique():
        cat_data = df[df['category'] == category]

        n_total = len(cat_data)
        n_anomalies = cat_data['is_anomaly'].sum()
        pct_anomalies = (n_anomalies / n_total * 100) if n_total > 0 else 0

        rating_diff_valid = cat_data['rating_diff'].dropna()

        summary_rows.append({
            'category': category,
            'n_total': n_total,
            'n_anomalies': int(n_anomalies),
            'pct_anomalies': round(pct_anomalies, 2),
            'mean_rating_diff': rating_diff_valid.mean() if len(rating_diff_valid) > 0 else np.nan,
            'std_rating_diff': rating_diff_valid.std() if len(rating_diff_valid) > 0 else np.nan,
            'z_threshold': z_threshold
        })

    return pd.DataFrame(summary_rows)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the input data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """

    required_cols = ['review_data', 'rating', 'predicted_rating', 'rating_diff', 'category', 'original_index']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df['rating_diff'] = pd.to_numeric(df['rating_diff'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['predicted_rating'] = pd.to_numeric(df['predicted_rating'], errors='coerce')

    df = df.dropna(subset=['rating_diff'])

    return df


def create_anomaly_indices_summary(anomalies_df: pd.DataFrame, output_path: str) -> None:
    """
    Create a summary CSV with anomaly indices by category.
    
    Args:
        anomalies_df: DataFrame containing all anomalies
        output_path: Path to save the summary CSV
    """

    categories = ['books', 'electronics', 'clothing_shoes_and_jewelry']

    category_indices = {}

    for category in categories:
        cat_anomalies = anomalies_df[anomalies_df['category'] == category]
        indices_list = cat_anomalies['original_index'].tolist() if len(cat_anomalies) > 0 else []
        category_indices[f'{category}_anomalies_indices'] = indices_list

    all_indices = anomalies_df['original_index'].tolist()
    category_indices['all_indices'] = all_indices

    max_length = max(len(indices) for indices in category_indices.values()) if category_indices else 0

    summary_dict = {}
    for col_name, indices in category_indices.items():
        padded_indices = indices + [np.nan] * (max_length - len(indices))
        summary_dict[col_name] = padded_indices

    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv(output_path, index=False)

    print(f"Saved anomaly indices summary to {output_path}")
    print(f"  Books anomalies: {len(category_indices['books_anomalies_indices'])}")
    print(f"  Electronics anomalies: {len(category_indices['electronics_anomalies_indices'])}")
    print(
        f"  Clothing, Shoes & Jewelry anomalies: {len(category_indices['clothing_shoes_and_jewelry_anomalies_indices'])}")
    print(f"  Total anomalies: {len(category_indices['all_indices'])}")


def analyze_anomalies(combined_df: pd.DataFrame, anomalies_df: pd.DataFrame):
    """
    Analyze all anomalies using the text analyzer.
    
    Args:
        combined_df: The full combined dataset
        anomalies_df: DataFrame containing detected anomalies with original_index column
    """

    print(f"🔍 Starting text analysis of {len(anomalies_df)} anomalies")

    anomaly_indices = anomalies_df['original_index'].tolist()

    analysis_df = combined_df.copy()
    if 'review_data' in analysis_df.columns:
        analysis_df['text'] = analysis_df['review_data']

    analyzer = AnomalyTextAnalyzer()
    analyzer.analyze_anomaly_text(analysis_df, anomaly_indices)

    return analyzer


def main():
    """Main function to run the anomaly detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Single-metric anomaly detection for Amazon review data using rating_diff",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/v2/test',
        help='Directory containing the CSV files'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='data/anomalies_rating_prediction',
        help='Output directory for results'
    )

    parser.add_argument(
        '--z_threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for anomaly detection'
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    file_mapping = {
        'books': 'books_test.csv',
        'electronics': 'electronics_test.csv',
        'clothing_shoes_and_jewelry': 'clothing_shoes_and_jewelry_test.csv'
    }

    dfs = []
    current_index = 0

    for category, filename in file_mapping.items():
        file_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            sys.exit(1)

        df = load_and_tag(file_path, category, start_index=current_index)

        end_index = current_index + len(df) - 1
        print(f"Loaded {category}: indices {current_index}-{end_index} ({len(df):,} rows)")

        dfs.append(df)
        current_index += len(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined_df):,} total rows")

    combined_df = validate_data(combined_df)

    print("Per-category data summary:")
    for category in combined_df['category'].unique():
        cat_data = combined_df[combined_df['category'] == category]
        print(f"  {category}: {len(cat_data):,} rows")

        rd_mean = cat_data['rating_diff'].mean()
        rd_std = cat_data['rating_diff'].std()

        print(f"    rating_diff: mean={rd_mean:.4f}, std={rd_std:.4f}")

    combined_df = compute_category_z(combined_df, 'rating_diff', 'z_rating_diff')

    combined_df['is_anomaly'] = combined_df['z_rating_diff'] > args.z_threshold

    combined_df['is_anomaly'] = combined_df['is_anomaly'].fillna(False)

    anomalies_df = combined_df[combined_df['is_anomaly'] == True].copy()

    print(f"Found {len(anomalies_df):,} anomalies out of {len(combined_df):,} total rows "
          f"({len(anomalies_df) / len(combined_df) * 100:.2f}%)")

    for category in combined_df['category'].unique():
        cat_total = len(combined_df[combined_df['category'] == category])
        cat_anomalies = len(anomalies_df[anomalies_df['category'] == category])
        pct = (cat_anomalies / cat_total * 100) if cat_total > 0 else 0
        print(f"  {category}: {cat_anomalies:,} / {cat_total:,} ({pct:.2f}%)")

    anomalies_df = anomalies_df.sort_values(
        'z_rating_diff',
        ascending=False,
        na_position='last'
    ).reset_index(drop=True)

    anomalies_df_clean = anomalies_df.drop('is_anomaly', axis=1)

    all_anomalies_file = os.path.join(args.out_dir, 'anomalies_all.csv')
    anomalies_df_clean.to_csv(all_anomalies_file, index=True, index_label='anomaly_rank')
    print(f"Saved all anomalies (sorted by z_rating_diff) to {all_anomalies_file}")

    for category in anomalies_df_clean['category'].unique():
        cat_anomalies = anomalies_df_clean[anomalies_df_clean['category'] == category]
        clean_category = category.lower()
        output_file = os.path.join(args.out_dir, f'anomalies_{clean_category}.csv')
        cat_anomalies.to_csv(output_file, index=True, index_label='anomaly_rank')

    summary_df = summarize_by_category(combined_df, args.z_threshold)
    summary_file = os.path.join(args.out_dir, 'anomalies_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    indices_summary_file = os.path.join(args.out_dir, 'anomaly_indices_summary.csv')
    create_anomaly_indices_summary(anomalies_df, indices_summary_file)

    analyze_anomalies(combined_df, anomalies_df)

if __name__ == "__main__":
    main()
