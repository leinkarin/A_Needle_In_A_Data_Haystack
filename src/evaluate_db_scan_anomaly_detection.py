import argparse
import json
import os
from typing import Dict

import pandas as pd

from analysis.basic_analysis import run_basic_metrics_analysis
from analysis.user_analysis_visualization import run_user_analysis
from analysis.length_analysis_visualization import run_length_analysis
from analysis.coordinated_attacks_visualization import run_coordinated_attacks_analysis

FEATURE_COLUMNS = [
    'helpful_vote', 'verified_purchase', 'has_images',
    'rating_diff', 'reviewer_review_count', 'rating_vs_product_avg_abs'
]


class AnomalyDetectionEvaluator:
    """
    Comprehensive evaluation framework for DBSCAN anomaly detection results.
    """

    def __init__(self, results_file: str, original_data_file: str, category: str):
        """
        Initialize the evaluator with DBSCAN results.
        
        Args:
            results_file: Path to DBSCAN anomaly detection results CSV
            original_data_file: Path to original dataset (optional, for comparison)
            category: The category of the reviews
        """
        self.results_file = results_file
        self.original_data_file = original_data_file
        self.category = category
        self.anomalies_df = None
        self.original_df = None
        self._load_data()

    def _calculate_feature_statistics(self, feature: str) -> Dict:
        """Calculate comprehensive statistics for a feature."""
        if feature not in self.anomalies_df.columns:
            return {}

        data = self.anomalies_df[feature]

        if data.dtype == bool:
            stats = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': None,
                'q25': None,
                'q75': None,
                'range': None
            }
        else:
            stats = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': data.median(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75)
            }
            stats['range'] = stats['max'] - stats['min']

        return stats

    def _load_data(self):
        """Load anomaly results and original data."""
        self.anomalies_df = pd.read_csv(self.results_file)

        if self.original_data_file and os.path.exists(self.original_data_file):
            self.original_df = pd.read_csv(self.original_data_file)

    def evaluate_anomaly_characteristics(self) -> Dict:
        """
        Evaluate the characteristics of detected anomalies.
        
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        results['total_anomalies'] = len(self.anomalies_df)

        if self.original_df is not None:
            results['anomaly_rate'] = len(self.anomalies_df) / len(self.original_df)

        available_features = [col for col in FEATURE_COLUMNS if col in self.anomalies_df.columns]

        for feature in available_features:
            stats = self._calculate_feature_statistics(feature)
            if stats:
                for stat_name, stat_value in stats.items():
                    results[f'{feature}_{stat_name}'] = stat_value

        if 'category' in self.anomalies_df.columns:
            category_counts = self.anomalies_df['category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(self.anomalies_df)) * 100
                results[f'category_{category}_count'] = count
                results[f'category_{category}_percentage'] = percentage

        if 'rating' in self.anomalies_df.columns:
            rating_counts = self.anomalies_df['rating'].value_counts().sort_index()
            for rating, count in rating_counts.items():
                percentage = (count / len(self.anomalies_df)) * 100
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
            return {}

        results = {}

        available_features = [col for col in FEATURE_COLUMNS
                              if col in self.anomalies_df.columns and col in self.original_df.columns]

        for feature in available_features:
            anomaly_mean = self.anomalies_df[feature].mean()
            original_mean = self.original_df[feature].mean()
            anomaly_std = self.anomalies_df[feature].std()
            original_std = self.original_df[feature].std()

            mean_diff = anomaly_mean - original_mean
            std_ratio = anomaly_std / original_std if original_std > 0 else 0

            results[f'{feature}_mean_diff'] = mean_diff
            results[f'{feature}_std_ratio'] = std_ratio

        if 'category' in self.anomalies_df.columns and 'category' in self.original_df.columns:
            anomaly_cats = self.anomalies_df['category'].value_counts(normalize=True)
            original_cats = self.original_df['category'].value_counts(normalize=True)

            for category in set(anomaly_cats.index) | set(original_cats.index):
                anomaly_pct = anomaly_cats.get(category, 0) * 100
                original_pct = original_cats.get(category, 0) * 100
                diff = anomaly_pct - original_pct

                results[f'category_{category}_pct_diff'] = diff

        return results

    def generate_summary_report(self, output_file: str = "anomaly_evaluation_report.txt",
                                anomaly_chars: dict = None, basic_results: dict = None,
                                user_results: dict = None, length_results: dict = None,
                                comparison_results: dict = None):
        """Generate a comprehensive summary report."""
        print(f"\nGenerating summary report: {output_file}")
        anomaly_patterns = basic_results
        user_patterns = user_results

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DBSCAN ANOMALY DETECTION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Results File: {self.results_file}\n")
            if self.original_data_file:
                f.write(f"Original Data File: {self.original_data_file}\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Anomalies Detected: {anomaly_chars.get('total_anomalies', 'N/A')}\n")
            if 'anomaly_rate' in anomaly_chars:
                f.write(
                    f"Anomaly Rate: {anomaly_chars['anomaly_rate']:.4f} ({anomaly_chars['anomaly_rate'] * 100:.2f}%)\n")

            f.write("KEY FINDINGS\n")
            f.write("-" * 40 + "\n")

            if 'mean_rating_diff' in anomaly_patterns:
                f.write(f"• Average rating difference: {anomaly_patterns['mean_rating_diff']:.3f}\n")
                f.write(f"• Positive rating differences: {anomaly_patterns['positive_diff_percentage']:.1f}%\n")
                f.write(f"• Negative rating differences: {anomaly_patterns['negative_diff_percentage']:.1f}%\n")

            if 'zero_votes_percentage' in anomaly_patterns:
                f.write(f"• Reviews with zero helpful votes: {anomaly_patterns['zero_votes_percentage']:.1f}%\n")

            if 'verified_percentage' in anomaly_patterns:
                f.write(f"• Verified purchases: {anomaly_patterns['verified_percentage']:.1f}%\n")

            if 'total_anomalous_users' in user_patterns:
                f.write(f"• Users with anomalous reviews: {user_patterns['total_anomalous_users']:,}\n")
                f.write(f"• Avg anomalies per user: {user_patterns['avg_anomalies_per_user']:.2f}\n")
                f.write(f"• Avg normal reviews per user: {user_patterns['avg_normal_reviews_per_user']:.1f}\n")
                f.write(f"• Users with all reviews anomalous: {user_patterns['users_with_all_anomalous']:,}\n")
                f.write(f"• Users with >50% anomaly rate: {user_patterns['users_with_high_anomaly_rate']:,}\n")

            if comparison_results:
                f.write("\nCOMPARISON WITH NORMAL DATA:\n")
                for key, value in comparison_results.items():
                    if key.startswith('rating_') and 'mean_diff' in key:
                        f.write(f"• {key.replace('_', ' ').title()}: {value:+.3f}\n")
                    elif key.startswith('category_') and 'pct_diff' in key:
                        f.write(f"• {key.replace('_', ' ').title()}: {value:+.1f}%\n")

            f.write("\n")

            f.write("DETAILED METRICS\n")
            f.write("-" * 40 + "\n")

            f.write("Feature Statistics:\n")
            for key, value in anomaly_chars.items():
                if key.startswith(
                        ('rating_', 'helpful_', 'verified_', 'has_images_', 'predicted_', 'reviewer_', 'rating_vs_')):
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

            f.write("\n")

            if user_patterns:
                f.write("User Pattern Metrics:\n")
                for key, value in user_patterns.items():
                    if key == 'user_analysis_data':
                        continue
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, float):
                                f.write(f"    {subkey}: {subvalue:.4f}\n")
                            else:
                                f.write(f"    {subkey}: {subvalue}\n")
                    elif isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

                f.write("\n")

            if length_results:
                f.write("Length Analysis Metrics:\n")
                for key, value in length_results.items():
                    if key.endswith('_stats') and isinstance(value, dict):
                        f.write(f"  {key.replace('_', ' ').title()}:\n")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, float):
                                f.write(f"    {subkey}: {subvalue:.2f}\n")
                            else:
                                f.write(f"    {subkey}: {subvalue}\n")
                    elif isinstance(value, (int, float)):
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.2f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    elif isinstance(value, str):
                        f.write(f"  {key}: {value}\n")

                f.write("\n")

            if comparison_results:
                f.write("Comparison with Original Data:\n")
                for key, value in comparison_results.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

    def run_complete_evaluation(self, output_dir: str = "evaluation_results"):
        """Run complete evaluation pipeline."""
        category_dir = os.path.join(output_dir, self.category)
        os.makedirs(category_dir, exist_ok=True)

        anomaly_chars = self.evaluate_anomaly_characteristics()

        plots_dir = os.path.join(category_dir, "plots")
        basic_results = run_basic_metrics_analysis(self.anomalies_df, self.original_df, plots_dir, self.category)
        user_results = run_user_analysis(self.anomalies_df, plots_dir, self.category)
        length_results = run_length_analysis(self.anomalies_df, self.original_df, plots_dir, self.category)
        coordinated_results = run_coordinated_attacks_analysis(self.anomalies_df, plots_dir, self.category)

        comparison_results = {}
        if self.original_df is not None:
            comparison_results = self.compare_with_original_data()

        report_file = os.path.join(category_dir, "evaluation_report.txt")
        self.generate_summary_report(report_file, anomaly_chars, basic_results, user_results, length_results,
                                     comparison_results)

        all_metrics = {
            'category': self.category,
            'anomaly_characteristics': anomaly_chars,
            'basic_metrics': basic_results,
            'user_analysis': user_results,
            'length_analysis': length_results,
            'coordinated_attacks': coordinated_results,
            'comparison_with_original': comparison_results
        }

        metrics_file = os.path.join(category_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)

        return all_metrics


def main():
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
        '--category',
        default='books',
        help='The category of the reviews'
    )

    args = parser.parse_args()

    if not os.path.exists(args.anomaly_data_file):
        print(f"Anomaly data file not found: {args.anomaly_data_file}")
        print("Please provide a valid path using --anomaly-data-file")
        return

    if args.original_data and not os.path.exists(args.original_data):
        print(f"Original data file not found: {args.original_data}")
        print("Please provide a valid path using --original-data")
        return

    evaluator = AnomalyDetectionEvaluator(
        results_file=args.anomaly_data_file,
        original_data_file=args.original_data,
        category=args.category
    )

    evaluator.run_complete_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
