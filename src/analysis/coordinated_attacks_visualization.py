import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

plt.style.use('default')
sns.set_palette("husl")

TEXT_COLUMNS = ['text', 'review_data', 'title']

SIMILARITY_THRESHOLDS = {
    'high': 0.75,
    'moderate': 0.5,
    'isolation': 0.3
}


def get_available_text_columns(anomalies_df: pd.DataFrame) -> List[str]:
    """Get list of available text columns in the anomalies dataframe."""
    return [col for col in TEXT_COLUMNS if col in anomalies_df.columns]


def create_tfidf_vectorizer(max_features: int = 1000, min_df: int = 1, max_df: float = 0.9) -> TfidfVectorizer:
    """Create a standardized TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df
    )


def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Preprocess texts for similarity analysis.
    
    Args:
        texts: List of text strings
        
    Returns:
        List of cleaned text strings
    """
    cleaned_texts = []

    for text in texts:
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove very common filler words that might inflate similarity
        filler_words = ['this', 'that', 'very', 'really', 'just', 'like', 'good', 'great', 'nice', 'bad']
        words = text.split()
        words = [word for word in words if word not in filler_words or len(words) > 10]
        text = ' '.join(words)

        cleaned_texts.append(text)

    return cleaned_texts


def _convert_timestamps_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp column to datetime."""
    try:
        timestamps = pd.to_numeric(df['timestamp'], errors='coerce')

        if timestamps.max() > 1e10:
            df['datetime'] = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(timestamps, unit='s', errors='coerce')

        return df
    except Exception:
        return df


def analyze_review_bombing_patterns(anomalies_df: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns to detect potential review bombing attacks.

    Args:
        anomalies_df: DataFrame containing anomaly detection results

    Returns:
        Dictionary with review bombing analysis results
    """
    print("\n" + "=" * 60)
    print("REVIEW BOMBING DETECTION ANALYSIS")
    print("=" * 60)

    results = {}

    if 'timestamp' not in anomalies_df.columns:
        print("⚠️ No timestamp column found - cannot perform temporal analysis")
        return results

    # Convert timestamps to datetime
    df_with_datetime = _convert_timestamps_to_datetime(anomalies_df.copy())

    if 'datetime' not in df_with_datetime.columns:
        print("⚠️ No timestamp column found - cannot perform temporal analysis")
        return results

    # Remove rows with invalid timestamps
    valid_timestamps = df_with_datetime['datetime'].notna()
    if not valid_timestamps.any():
        print("⚠️ No valid timestamps found for temporal analysis")
        return results

    df_temporal = df_with_datetime[valid_timestamps].copy()

    # Basic temporal statistics
    date_range = df_temporal['datetime'].max() - df_temporal['datetime'].min()
    results['temporal_span_days'] = date_range.days
    results['total_anomalies_with_timestamps'] = len(df_temporal)

    print(f"Temporal Analysis Summary:")
    print(f"  Anomalies with valid timestamps: {len(df_temporal):,}")
    print(
        f"  Date range: {df_temporal['datetime'].min().strftime('%Y-%m-%d')} to {df_temporal['datetime'].max().strftime('%Y-%m-%d')}")
    print(f"  Temporal span: {date_range.days} days")

    # Daily clustering analysis
    df_temporal['date'] = df_temporal['datetime'].dt.date
    daily_counts = df_temporal.groupby('date').size().reset_index(name='anomaly_count')

    # Statistical analysis of daily patterns
    mean_daily = daily_counts['anomaly_count'].mean()
    std_daily = daily_counts['anomaly_count'].std()
    max_daily = daily_counts['anomaly_count'].max()

    # Identify potential bombing days (>2 std above mean)
    bombing_threshold = mean_daily + 2 * std_daily
    potential_bombing_days = daily_counts[daily_counts['anomaly_count'] > bombing_threshold]

    results['mean_daily_anomalies'] = mean_daily
    results['std_daily_anomalies'] = std_daily
    results['max_daily_anomalies'] = max_daily
    results['bombing_threshold'] = bombing_threshold
    results['potential_bombing_days'] = len(potential_bombing_days)
    results['bombing_days_percentage'] = len(potential_bombing_days) / len(daily_counts) * 100

    print(f"\nDaily Pattern Analysis:")
    print(f"  Mean daily anomalies: {mean_daily:.2f}")
    print(f"  Std daily anomalies: {std_daily:.2f}")
    print(f"  Max daily anomalies: {max_daily}")
    print(
        f"  Potential bombing days (>2σ): {len(potential_bombing_days)} ({len(potential_bombing_days) / len(daily_counts) * 100:.1f}%)")

    if len(potential_bombing_days) > 0:
        print(f"  Top bombing days:")
        top_bombing = potential_bombing_days.nlargest(5, 'anomaly_count')
        for _, row in top_bombing.iterrows():
            print(f"    {row['date']}: {row['anomaly_count']} anomalies")

    # Hourly clustering analysis (for recent data)
    df_temporal['hour'] = df_temporal['datetime'].dt.hour
    hourly_counts = df_temporal.groupby('hour').size()

    # Find suspicious hourly patterns
    mean_hourly = hourly_counts.mean()
    std_hourly = hourly_counts.std()
    suspicious_hours = hourly_counts[hourly_counts > mean_hourly + 1.5 * std_hourly]

    results['suspicious_hours'] = len(suspicious_hours)
    results['peak_hour'] = hourly_counts.idxmax()
    results['peak_hour_count'] = hourly_counts.max()

    print(f"\nHourly Pattern Analysis:")
    print(f"  Peak hour: {hourly_counts.idxmax()}:00 with {hourly_counts.max()} anomalies")
    print(f"  Suspicious hours (>1.5σ): {len(suspicious_hours)}")

    # User clustering analysis (if user_id available)
    if 'user_id' in df_temporal.columns:
        user_temporal = df_temporal.groupby('user_id').agg({
            'datetime': ['count', 'min', 'max'],
            'date': 'nunique'
        }).reset_index()

        user_temporal.columns = ['user_id', 'total_anomalies', 'first_anomaly', 'last_anomaly', 'active_days']
        user_temporal['timespan_days'] = (user_temporal['last_anomaly'] - user_temporal['first_anomaly']).dt.days
        user_temporal['anomalies_per_day'] = user_temporal['total_anomalies'] / (user_temporal['timespan_days'] + 1)

        # Identify users with suspicious temporal patterns
        high_velocity_users = user_temporal[user_temporal['anomalies_per_day'] > 5]  # >5 anomalies per day
        burst_users = user_temporal[
            (user_temporal['total_anomalies'] >= 3) &
            (user_temporal['timespan_days'] <= 1)
            ]  # Multiple anomalies in ≤1 day

        results['high_velocity_users'] = len(high_velocity_users)
        results['burst_pattern_users'] = len(burst_users)
        results['max_user_anomalies_per_day'] = user_temporal['anomalies_per_day'].max()

        print(f"\nUser Temporal Pattern Analysis:")
        print(f"  High velocity users (>5 anomalies/day): {len(high_velocity_users)}")
        print(f"  Burst pattern users (≥3 anomalies in ≤1 day): {len(burst_users)}")
        print(f"  Max anomalies per day by single user: {user_temporal['anomalies_per_day'].max():.1f}")

    # Product clustering analysis (if asin available)
    if 'asin' in df_temporal.columns:
        product_temporal = df_temporal.groupby('asin').agg({
            'datetime': ['count', 'min', 'max'],
            'date': 'nunique'
        }).reset_index()

        product_temporal.columns = ['asin', 'total_anomalies', 'first_anomaly', 'last_anomaly', 'active_days']
        product_temporal['timespan_days'] = (
                product_temporal['last_anomaly'] - product_temporal['first_anomaly']).dt.days

        # Products with concentrated anomalous reviews
        targeted_products = product_temporal[
            (product_temporal['total_anomalies'] >= 5) &
            (product_temporal['timespan_days'] <= 7)
            ]  # ≥5 anomalies in ≤7 days

        results['targeted_products'] = len(targeted_products)
        results['max_product_anomalies'] = product_temporal['total_anomalies'].max()

        print(f"\nProduct Targeting Analysis:")
        print(f"  Potentially targeted products (≥5 anomalies in ≤7 days): {len(targeted_products)}")
        print(f"  Max anomalies per product: {product_temporal['total_anomalies'].max()}")

    return results


def analyze_similarity_patterns(anomalies_df: pd.DataFrame) -> Dict:
    """
    Analyze similarity patterns in anomalous reviews to detect coordinated attacks
    and validate pattern consistency.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        
    Returns:
        Dictionary with similarity analysis results
    """
    print("\n" + "=" * 60)
    print("SIMILARITY PATTERN ANALYSIS")
    print("=" * 60)

    results = {}

    # Check for text columns
    text_columns = get_available_text_columns(anomalies_df)

    if not text_columns:
        print("⚠️ No text columns found for similarity analysis")
        return results

    # Analyze similarity for each text column
    for col in text_columns:
        print(f"\n{col.title()} Similarity Analysis:")

        # Clean and prepare text data
        text_data = anomalies_df[col].fillna('').astype(str)

        # Filter out very short texts (less than 10 characters)
        valid_texts = text_data[text_data.str.len() >= 10]
        if len(valid_texts) < 2:
            print(f"  ⚠️ Insufficient valid texts for analysis ({len(valid_texts)} texts)")
            continue

        valid_indices = valid_texts.index.tolist()

        # Text preprocessing
        cleaned_texts = preprocess_texts(valid_texts.tolist())

        # TF-IDF Vectorization
        try:
            vectorizer = create_tfidf_vectorizer(max_features=1000, min_df=2, max_df=0.8)

            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

            if tfidf_matrix.shape[0] < 2:
                print(f"  ⚠️ Insufficient texts after vectorization")
                continue

        except Exception as e:
            print(f"  ⚠️ Error in vectorization: {e}")
            continue

        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Remove diagonal (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)

        # Analyze similarity distribution
        upper_triangle = np.triu(similarity_matrix, k=1)
        similarities = upper_triangle[upper_triangle > 0]

        if len(similarities) == 0:
            print(f"  ⚠️ No valid similarities found")
            continue

        # Basic similarity statistics
        mean_sim = similarities.mean()
        std_sim = similarities.std()
        max_sim = similarities.max()
        median_sim = np.median(similarities)

        print(f"  Similarity Statistics:")
        print(f"    Mean: {mean_sim:.4f}, Median: {median_sim:.4f}")
        print(f"    Std: {std_sim:.4f}, Max: {max_sim:.4f}")

        # Identify highly similar pairs (potential duplicates/coordinated)
        high_sim_threshold = max(SIMILARITY_THRESHOLDS['high'], mean_sim + 2 * std_sim)
        moderate_sim_threshold = max(SIMILARITY_THRESHOLDS['moderate'], mean_sim + std_sim)

        # Find highly similar pairs
        high_sim_pairs = []
        moderate_sim_pairs = []

        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                sim_score = similarity_matrix[i, j]
                if sim_score >= high_sim_threshold:
                    high_sim_pairs.append((valid_indices[i], valid_indices[j], sim_score))
                elif sim_score >= moderate_sim_threshold:
                    moderate_sim_pairs.append((valid_indices[i], valid_indices[j], sim_score))

        print(f"  Similarity Thresholds:")
        print(f"    High similarity (≥{high_sim_threshold:.3f}): {len(high_sim_pairs)} pairs")
        print(f"    Moderate similarity (≥{moderate_sim_threshold:.3f}): {len(moderate_sim_pairs)} pairs")

        # Analyze user patterns in similar reviews
        if 'user_id' in anomalies_df.columns and len(high_sim_pairs) > 0:
            same_user_pairs = 0
            different_user_pairs = 0

            for idx1, idx2, sim_score in high_sim_pairs:
                user1 = anomalies_df.loc[idx1, 'user_id']
                user2 = anomalies_df.loc[idx2, 'user_id']

                if user1 == user2:
                    same_user_pairs += 1
                else:
                    different_user_pairs += 1

            print(f"  User Pattern Analysis (High Similarity):")
            print(f"    Same user pairs: {same_user_pairs} ({same_user_pairs / len(high_sim_pairs) * 100:.1f}%)")
            print(
                f"    Different user pairs: {different_user_pairs} ({different_user_pairs / len(high_sim_pairs) * 100:.1f}%)")

            results[f'{col}_same_user_similar_pairs'] = same_user_pairs
            results[f'{col}_different_user_similar_pairs'] = different_user_pairs

        # Clustering analysis to find review templates/patterns
        if len(valid_texts) >= 5:
            n_clusters = min(10, len(valid_texts) // 3)
            if n_clusters >= 2:
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(tfidf_matrix)

                    # Analyze cluster sizes
                    cluster_sizes = Counter(cluster_labels)
                    large_clusters = [size for size in cluster_sizes.values() if size >= 3]

                    print(f"  Template Clustering Analysis:")
                    print(f"    Number of clusters: {n_clusters}")
                    print(f"    Large clusters (≥3 reviews): {len(large_clusters)}")
                    print(f"    Largest cluster size: {max(cluster_sizes.values())}")

                    # Find potential template patterns
                    template_clusters = []
                    for cluster_id, size in cluster_sizes.items():
                        if size >= 3:  # Potential template if ≥3 similar reviews
                            cluster_indices = [valid_indices[i] for i, label in enumerate(cluster_labels) if
                                               label == cluster_id]
                            template_clusters.append((cluster_id, size, cluster_indices))

                    results[f'{col}_template_clusters'] = len(template_clusters)
                    results[f'{col}_largest_template_size'] = max(cluster_sizes.values())

                    # Analyze users in template clusters
                    if 'user_id' in anomalies_df.columns and template_clusters:
                        template_user_analysis = []
                        for cluster_id, size, indices in template_clusters:
                            users_in_cluster = anomalies_df.loc[indices, 'user_id'].unique()
                            unique_users = len(users_in_cluster)
                            template_user_analysis.append({
                                'cluster_size': size,
                                'unique_users': unique_users,
                                'reviews_per_user': size / unique_users
                            })

                        avg_users_per_template = np.mean([t['unique_users'] for t in template_user_analysis])
                        avg_reviews_per_user_in_templates = np.mean(
                            [t['reviews_per_user'] for t in template_user_analysis])

                        print(f"    Avg unique users per template: {avg_users_per_template:.1f}")
                        print(f"    Avg reviews per user in templates: {avg_reviews_per_user_in_templates:.1f}")

                        results[f'{col}_avg_users_per_template'] = avg_users_per_template
                        results[f'{col}_avg_reviews_per_user_in_templates'] = avg_reviews_per_user_in_templates

                except Exception as e:
                    print(f"  ⚠️ Error in clustering analysis: {e}")

        # Exact duplicate detection
        exact_duplicates = text_data.duplicated().sum()
        near_duplicates = len([pair for pair in high_sim_pairs if pair[2] > 0.95])

        print(f"  Duplicate Analysis:")
        print(f"    Exact duplicates: {exact_duplicates}")
        print(f"    Near duplicates (>95% similar): {near_duplicates}")

        # Store results
        results[f'{col}_mean_similarity'] = mean_sim
        results[f'{col}_std_similarity'] = std_sim
        results[f'{col}_max_similarity'] = max_sim
        results[f'{col}_median_similarity'] = median_sim
        results[f'{col}_high_similarity_pairs'] = len(high_sim_pairs)
        results[f'{col}_moderate_similarity_pairs'] = len(moderate_sim_pairs)
        results[f'{col}_exact_duplicates'] = exact_duplicates
        results[f'{col}_near_duplicates'] = near_duplicates
        results[f'{col}_high_sim_threshold'] = high_sim_threshold
        results[f'{col}_moderate_sim_threshold'] = moderate_sim_threshold

        # Show examples of highly similar pairs
        if len(high_sim_pairs) > 0:
            print(f"  Top Similar Pairs:")
            sorted_pairs = sorted(high_sim_pairs, key=lambda x: x[2], reverse=True)[:3]
            for i, (idx1, idx2, sim_score) in enumerate(sorted_pairs):
                text1 = anomalies_df.loc[idx1, col][:100] + "..." if len(anomalies_df.loc[idx1, col]) > 100 else \
                anomalies_df.loc[idx1, col]
                text2 = anomalies_df.loc[idx2, col][:100] + "..." if len(anomalies_df.loc[idx2, col]) > 100 else \
                anomalies_df.loc[idx2, col]
                print(f"    Pair {i + 1} (similarity: {sim_score:.3f}):")
                print(f"      Text 1: {text1}")
                print(f"      Text 2: {text2}")

    return results


def create_temporal_pattern_visualizations(anomalies_df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive temporal pattern analysis visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results with timestamp column
        output_dir: Directory to save plots
    """
    print("Creating temporal pattern visualizations...")

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


def create_coordinated_attack_visualizations(anomalies_df: pd.DataFrame, bombing_patterns: dict,
                                             similarity_patterns: dict, anomaly_validation: dict,
                                             output_dir: str):
    """
    Create comprehensive visualizations for coordinated attacks and temporal patterns.
    
    Args:
        anomalies_df: DataFrame with anomaly data
        bombing_patterns: Results from review bombing analysis
        similarity_patterns: Results from similarity analysis
        anomaly_validation: Results from similarity matrix validation
        output_dir: Directory to save plots
    """
    print("Creating coordinated attack visualizations...")

    # Use the provided output directory (should already be attacks_analysis)

    # 1. Temporal Bombing Patterns Heatmap
    if 'timestamp' in anomalies_df.columns and bombing_patterns:
        create_temporal_bombing_heatmap(anomalies_df, bombing_patterns, output_dir)

    # 2. Similarity Network Visualization
    if anomaly_validation and 'coordinated_attacks' in anomaly_validation:
        create_similarity_network_plot(anomalies_df, anomaly_validation, output_dir)

    # 3. Attack Pattern Summary Dashboard
    create_attack_pattern_dashboard(bombing_patterns, similarity_patterns, anomaly_validation, output_dir)

    # 4. User Behavior Timeline
    if 'user_id' in anomalies_df.columns and 'timestamp' in anomalies_df.columns:
        create_user_behavior_timeline(anomalies_df, anomaly_validation, output_dir)

    # 5. Template Usage Analysis
    if anomaly_validation and 'review_templates' in anomaly_validation:
        create_template_usage_visualization(anomalies_df, anomaly_validation, output_dir)


def create_temporal_bombing_heatmap(anomalies_df: pd.DataFrame, bombing_patterns: dict, output_dir: str):
    """Create a heatmap showing temporal patterns of potential review bombing."""

    try:
        # Convert timestamps
        timestamps = pd.to_numeric(anomalies_df['timestamp'], errors='coerce')
        if timestamps.max() > 1e10:
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='s', errors='coerce')

        valid_df = anomalies_df[anomalies_df['datetime'].notna()].copy()

        if len(valid_df) == 0:
            print("  ⚠️ No valid timestamps for temporal heatmap")
            return

        # Create hour and day features
        valid_df['hour'] = valid_df['datetime'].dt.hour
        valid_df['day_of_week'] = valid_df['datetime'].dt.day_name()
        valid_df['date'] = valid_df['datetime'].dt.date

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Hour vs Day of Week Heatmap
        hour_dow_pivot = valid_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hour_dow_pivot = hour_dow_pivot.reindex(day_order)

        sns.heatmap(hour_dow_pivot, annot=False, cmap='Reds', ax=ax1, cbar_kws={'label': 'Anomaly Count'})
        ax1.set_title('Anomaly Distribution: Hour vs Day of Week', fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Day of Week')

        # 2. Daily Anomaly Timeline with Bombing Detection
        daily_counts = valid_df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])

        # Identify potential bombing days
        mean_daily = daily_counts['count'].mean()
        std_daily = daily_counts['count'].std()
        bombing_threshold = mean_daily + 2 * std_daily
        bombing_days = daily_counts[daily_counts['count'] > bombing_threshold]

        ax2.plot(daily_counts['date'], daily_counts['count'], 'b-', alpha=0.7, linewidth=1)
        ax2.axhline(y=bombing_threshold, color='red', linestyle='--', alpha=0.8,
                    label=f'Bombing Threshold ({bombing_threshold:.1f})')

        if len(bombing_days) > 0:
            ax2.scatter(bombing_days['date'], bombing_days['count'],
                        color='red', s=50, alpha=0.8, zorder=5, label=f'Bombing Days ({len(bombing_days)})')

        ax2.set_title('Daily Anomaly Timeline with Bombing Detection', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Anomalies per Day')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Hourly Distribution
        hourly_counts = valid_df['hour'].value_counts().sort_index()
        ax3.bar(hourly_counts.index, hourly_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Anomaly Distribution by Hour of Day', fontweight='bold')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Number of Anomalies')
        ax3.grid(True, alpha=0.3, axis='y')

        # Highlight suspicious hours
        mean_hourly = hourly_counts.mean()
        std_hourly = hourly_counts.std()
        suspicious_threshold = mean_hourly + 1.5 * std_hourly
        suspicious_hours = hourly_counts[hourly_counts > suspicious_threshold]

        if len(suspicious_hours) > 0:
            ax3.bar(suspicious_hours.index, suspicious_hours.values,
                    alpha=0.8, color='orange', edgecolor='red', linewidth=2,
                    label=f'Suspicious Hours ({len(suspicious_hours)})')
            ax3.legend()

        # 4. Weekly Pattern
        weekly_counts = valid_df['day_of_week'].value_counts().reindex(day_order)
        ax4.bar(range(len(weekly_counts)), weekly_counts.values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Anomaly Distribution by Day of Week', fontweight='bold')
        ax4.set_xlabel('Day of Week')
        ax4.set_ylabel('Number of Anomalies')
        ax4.set_xticks(range(len(day_order)))
        ax4.set_xticklabels(day_order, rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_bombing_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Temporal bombing heatmap: {len(bombing_days)} potential bombing days detected")

    except Exception as e:
        print(f"  ⚠️ Error creating temporal heatmap: {e}")


def create_similarity_network_plot(anomalies_df: pd.DataFrame, anomaly_validation: dict, output_dir: str):
    """Create network visualization of similarity patterns between anomalies."""

    try:
        import networkx as nx
        from matplotlib.patches import Patch

        coordinated_attacks = anomaly_validation.get('coordinated_attacks', {})
        spam_patterns = anomaly_validation.get('spam_patterns', {})

        if not coordinated_attacks.get('groups') and not spam_patterns.get('patterns'):
            print("  ⚠️ No coordinated attacks or spam patterns found for network visualization")
            return

        # Create network graph
        G = nx.Graph()

        # Color mapping for different types
        node_colors = {}
        node_sizes = {}

        # Add coordinated attack groups
        coord_groups = coordinated_attacks.get('groups', [])
        for i, group in enumerate(coord_groups):
            group_indices = group['indices']
            group_color = plt.cm.Set1(i % 9)  # Use different colors for different groups

            # Add nodes
            for idx in group_indices:
                if idx in anomalies_df.index:
                    user_id = anomalies_df.loc[idx, 'user_id'] if 'user_id' in anomalies_df.columns else f"user_{idx}"
                    G.add_node(idx, user_id=user_id, type='coordinated', group=i)
                    node_colors[idx] = group_color
                    node_sizes[idx] = 300

            # Add edges within group (high similarity)
            for j, idx1 in enumerate(group_indices):
                for idx2 in group_indices[j + 1:]:
                    if idx1 in anomalies_df.index and idx2 in anomalies_df.index:
                        G.add_edge(idx1, idx2, weight=group.get('avg_similarity', 0.8), type='coordinated')

        # Add spam patterns
        spam_patterns_list = spam_patterns.get('patterns', [])
        for i, pattern in enumerate(spam_patterns_list):
            pattern_indices = pattern['indices']
            spam_color = 'red'

            # Add nodes
            for idx in pattern_indices:
                if idx in anomalies_df.index and idx not in node_colors:
                    user_id = pattern['user_id']
                    G.add_node(idx, user_id=user_id, type='spam', pattern=i)
                    node_colors[idx] = spam_color
                    node_sizes[idx] = 200

            # Add edges within spam pattern
            for j, idx1 in enumerate(pattern_indices):
                for idx2 in pattern_indices[j + 1:]:
                    if (idx1 in anomalies_df.index and idx2 in anomalies_df.index and
                            not G.has_edge(idx1, idx2)):
                        G.add_edge(idx1, idx2, weight=pattern.get('avg_similarity', 0.7), type='spam')

        if len(G.nodes()) == 0:
            print("  ⚠️ No valid nodes for network visualization")
            return

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # 1. Full network layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw edges
        coord_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'coordinated']
        spam_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'spam']

        nx.draw_networkx_edges(G, pos, edgelist=coord_edges, edge_color='blue',
                               alpha=0.6, width=2, ax=ax1, label='Coordinated')
        nx.draw_networkx_edges(G, pos, edgelist=spam_edges, edge_color='red',
                               alpha=0.6, width=1, ax=ax1, label='Spam')

        # Draw nodes
        node_color_list = [node_colors.get(node, 'gray') for node in G.nodes()]
        node_size_list = [node_sizes.get(node, 100) for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_color_list,
                               node_size=node_size_list, alpha=0.8, ax=ax1)

        ax1.set_title('Anomaly Similarity Network\n(Coordinated Attacks & Spam Patterns)',
                      fontweight='bold', fontsize=14)
        ax1.axis('off')

        # Add legend
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label=f'Coordinated Groups ({len(coord_groups)})'),
            Patch(facecolor='red', alpha=0.6, label=f'Spam Patterns ({len(spam_patterns_list)})')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # 2. Network statistics
        stats_text = []
        stats_text.append(f"Network Statistics:")
        stats_text.append(f"• Total nodes: {len(G.nodes())}")
        stats_text.append(f"• Total edges: {len(G.edges())}")
        stats_text.append(f"• Connected components: {nx.number_connected_components(G)}")
        stats_text.append(f"• Coordinated attack groups: {len(coord_groups)}")
        stats_text.append(f"• Spam patterns: {len(spam_patterns_list)}")

        if len(G.nodes()) > 0:
            stats_text.append(f"• Average clustering coefficient: {nx.average_clustering(G):.3f}")
            if nx.is_connected(G):
                stats_text.append(f"• Average shortest path: {nx.average_shortest_path_length(G):.2f}")

        # Group size distribution
        if coord_groups:
            group_sizes = [len(group['indices']) for group in coord_groups]
            ax2.hist(group_sizes, bins=max(1, len(set(group_sizes))), alpha=0.7,
                     color='skyblue', edgecolor='black')
            ax2.set_title('Coordinated Attack Group Sizes', fontweight='bold')
            ax2.set_xlabel('Group Size (Number of Reviews)')
            ax2.set_ylabel('Number of Groups')
            ax2.grid(True, alpha=0.3)

            # Add statistics text
            ax2.text(0.6, 0.8, '\n'.join(stats_text), transform=ax2.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     verticalalignment='top', fontsize=10)
        else:
            ax2.text(0.5, 0.5, '\n'.join(stats_text), transform=ax2.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax2.set_title('Network Statistics', fontweight='bold')
            ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Similarity network: {len(G.nodes())} nodes, {len(G.edges())} edges")

    except ImportError:
        print("  ⚠️ NetworkX not available for network visualization")
    except Exception as e:
        print(f"  ⚠️ Error creating similarity network: {e}")


def create_attack_pattern_dashboard(bombing_patterns: dict, similarity_patterns: dict,
                                    anomaly_validation: dict, output_dir: str):
    """Create a comprehensive dashboard showing all attack patterns."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Validation Summary Pie Chart
    if anomaly_validation and 'validation_summary' in anomaly_validation:
        validation = anomaly_validation['validation_summary']

        labels = ['Validated Anomalies', 'Potential False Positives']
        sizes = [validation['validated_as_real'], validation['potential_false_positives']]
        colors = ['lightcoral', 'lightgray']
        explode = (0.1, 0)

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title(f'Anomaly Validation Results\n(Total: {validation["total_analyzed"]})',
                      fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No validation data available',
                 transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Anomaly Validation Results', fontweight='bold')

    # 2. Attack Type Distribution
    attack_types = []
    attack_counts = []

    if anomaly_validation:
        if 'coordinated_attacks' in anomaly_validation:
            coord = anomaly_validation['coordinated_attacks']
            attack_types.append('Coordinated\nAttacks')
            attack_counts.append(coord.get('total_reviews_in_groups', 0))

        if 'spam_patterns' in anomaly_validation:
            spam = anomaly_validation['spam_patterns']
            attack_types.append('Spam\nPatterns')
            attack_counts.append(spam.get('total_spam_reviews', 0))

        if 'review_templates' in anomaly_validation:
            templates = anomaly_validation['review_templates']
            attack_types.append('Template\nUsage')
            attack_counts.append(templates.get('total_template_reviews', 0))

    if attack_types:
        bars = ax2.bar(attack_types, attack_counts, alpha=0.7,
                       color=['red', 'orange', 'yellow'][:len(attack_types)],
                       edgecolor='black')
        ax2.set_title('Attack Pattern Distribution', fontweight='bold')
        ax2.set_ylabel('Number of Reviews')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, count in zip(bars, attack_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(attack_counts) * 0.01,
                     f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No attack patterns detected',
                 transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Attack Pattern Distribution', fontweight='bold')

    # 3. Temporal Bombing Statistics
    if bombing_patterns:
        bombing_stats = []
        bombing_values = []

        if 'potential_bombing_days' in bombing_patterns:
            bombing_stats.append('Bombing Days')
            bombing_values.append(bombing_patterns['potential_bombing_days'])

        if 'high_velocity_users' in bombing_patterns:
            bombing_stats.append('High Velocity\nUsers')
            bombing_values.append(bombing_patterns['high_velocity_users'])

        if 'burst_pattern_users' in bombing_patterns:
            bombing_stats.append('Burst Pattern\nUsers')
            bombing_values.append(bombing_patterns['burst_pattern_users'])

        if 'targeted_products' in bombing_patterns:
            bombing_stats.append('Targeted\nProducts')
            bombing_values.append(bombing_patterns['targeted_products'])

        if bombing_stats:
            bars = ax3.bar(bombing_stats, bombing_values, alpha=0.7,
                           color='lightblue', edgecolor='black')
            ax3.set_title('Temporal Bombing Indicators', fontweight='bold')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, value in zip(bars, bombing_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + max(bombing_values) * 0.01,
                         f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No temporal patterns detected',
                     transform=ax3.transAxes, ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, 'No temporal data available',
                 transform=ax3.transAxes, ha='center', va='center')

    ax3.set_title('Temporal Bombing Indicators', fontweight='bold')

    # 4. Confidence Distribution
    if anomaly_validation and 'confidence_analysis' in anomaly_validation:
        conf_analysis = anomaly_validation['confidence_analysis']

        confidence_levels = ['High\nConfidence', 'Medium\nConfidence', 'Low\nConfidence\n(False Positives?)']
        confidence_counts = [
            conf_analysis.get('high_confidence_count', 0),
            conf_analysis.get('medium_confidence_count', 0),
            conf_analysis.get('low_confidence_count', 0)
        ]

        colors = ['green', 'yellow', 'red']
        bars = ax4.bar(confidence_levels, confidence_counts, alpha=0.7,
                       color=colors, edgecolor='black')
        ax4.set_title('Anomaly Confidence Distribution', fontweight='bold')
        ax4.set_ylabel('Number of Anomalies')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, count in zip(bars, confidence_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + max(confidence_counts) * 0.01,
                     f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No confidence data available',
                 transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Anomaly Confidence Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_pattern_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Attack pattern dashboard created")


def create_user_behavior_timeline(anomalies_df: pd.DataFrame, anomaly_validation: dict, output_dir: str):
    """Create timeline visualization of user behavior patterns."""

    try:
        # Convert timestamps
        timestamps = pd.to_numeric(anomalies_df['timestamp'], errors='coerce')
        if timestamps.max() > 1e10:
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        else:
            anomalies_df['datetime'] = pd.to_datetime(timestamps, unit='s', errors='coerce')

        valid_df = anomalies_df[anomalies_df['datetime'].notna()].copy()

        if len(valid_df) == 0:
            print("  ⚠️ No valid timestamps for user behavior timeline")
            return

        # Get coordinated attack users and spam users
        coordinated_users = set()
        spam_users = set()

        if anomaly_validation:
            # Extract coordinated attack users
            coord_attacks = anomaly_validation.get('coordinated_attacks', {})
            for group in coord_attacks.get('groups', []):
                for idx in group['indices']:
                    if idx in valid_df.index:
                        user_id = valid_df.loc[idx, 'user_id']
                        coordinated_users.add(user_id)

            # Extract spam users
            spam_patterns = anomaly_validation.get('spam_patterns', {})
            for pattern in spam_patterns.get('patterns', []):
                spam_users.add(pattern['user_id'])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

        # 1. User Activity Timeline
        user_activity = valid_df.groupby(['user_id', valid_df['datetime'].dt.date]).size().reset_index(name='count')
        user_activity['date'] = pd.to_datetime(user_activity['datetime'])

        # Plot different user types
        normal_users = set(valid_df['user_id']) - coordinated_users - spam_users

        for user_type, users, color, label in [
            ('normal', normal_users, 'blue', 'Normal Users'),
            ('coordinated', coordinated_users, 'red', 'Coordinated Attack Users'),
            ('spam', spam_users, 'orange', 'Spam Users')
        ]:
            if users:
                user_data = user_activity[user_activity['user_id'].isin(users)]
                if len(user_data) > 0:
                    ax1.scatter(user_data['date'], user_data['user_id'],
                                s=user_data['count'] * 20, alpha=0.6, color=color, label=label)

        ax1.set_title('User Activity Timeline\n(Bubble size = reviews per day)', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('User ID')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. User Type Distribution Over Time
        valid_df['date'] = valid_df['datetime'].dt.date
        daily_user_types = []

        for date in sorted(valid_df['date'].unique()):
            day_data = valid_df[valid_df['date'] == date]
            day_users = set(day_data['user_id'])

            daily_user_types.append({
                'date': pd.to_datetime(date),
                'normal': len(day_users - coordinated_users - spam_users),
                'coordinated': len(day_users & coordinated_users),
                'spam': len(day_users & spam_users)
            })

        if daily_user_types:
            daily_df = pd.DataFrame(daily_user_types)

            ax2.plot(daily_df['date'], daily_df['normal'], 'b-', label='Normal Users', linewidth=2)
            ax2.plot(daily_df['date'], daily_df['coordinated'], 'r-', label='Coordinated Attack Users', linewidth=2)
            ax2.plot(daily_df['date'], daily_df['spam'], 'orange', label='Spam Users', linewidth=2)

            ax2.fill_between(daily_df['date'], daily_df['coordinated'], alpha=0.3, color='red')
            ax2.fill_between(daily_df['date'], daily_df['spam'], alpha=0.3, color='orange')

        ax2.set_title('Daily Active Users by Type', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Active Users')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_behavior_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ User behavior timeline: {len(coordinated_users)} coordinated, {len(spam_users)} spam users")

    except Exception as e:
        print(f"  ⚠️ Error creating user behavior timeline: {e}")


def create_template_usage_visualization(anomalies_df: pd.DataFrame, anomaly_validation: dict, output_dir: str):
    """Create visualization of review template usage patterns."""

    try:
        templates = anomaly_validation.get('review_templates', {}).get('templates', [])

        if not templates:
            print("  ⚠️ No review templates found for visualization")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Template Size Distribution
        template_sizes = [template['size'] for template in templates]
        ax1.hist(template_sizes, bins=max(1, len(set(template_sizes))), alpha=0.7,
                 color='lightgreen', edgecolor='black')
        ax1.set_title('Review Template Size Distribution', fontweight='bold')
        ax1.set_xlabel('Template Size (Number of Reviews)')
        ax1.set_ylabel('Number of Templates')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        ax1.text(0.7, 0.8,
                 f'Total templates: {len(templates)}\nMean size: {np.mean(template_sizes):.1f}\nMax size: {max(template_sizes)}',
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2. User Diversity in Templates
        user_diversities = [template.get('user_diversity', 1.0) for template in templates]
        ax2.hist(user_diversities, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_title('User Diversity in Templates\n(Lower = More Suspicious)', fontweight='bold')
        ax2.set_xlabel('User Diversity (Unique Users / Template Size)')
        ax2.set_ylabel('Number of Templates')
        ax2.grid(True, alpha=0.3)

        # Highlight suspicious templates (low diversity)
        suspicious_templates = [t for t in templates if t.get('user_diversity', 1.0) < 0.5]
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.8,
                    label=f'Suspicious threshold\n({len(suspicious_templates)} templates)')
        ax2.legend()

        # 3. Template Confidence Levels
        confidence_levels = [template.get('confidence', 'MEDIUM') for template in templates]
        confidence_counts = pd.Series(confidence_levels).value_counts()

        colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
        bar_colors = [colors.get(conf, 'gray') for conf in confidence_counts.index]

        bars = ax3.bar(confidence_counts.index, confidence_counts.values,
                       alpha=0.7, color=bar_colors, edgecolor='black')
        ax3.set_title('Template Confidence Distribution', fontweight='bold')
        ax3.set_ylabel('Number of Templates')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, count in zip(bars, confidence_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + max(confidence_counts.values) * 0.01,
                     f'{int(count)}', ha='center', va='bottom', fontweight='bold')

        # 4. Template vs Individual Analysis
        template_reviews = sum(template['size'] for template in templates)
        total_reviews = len(anomalies_df)
        individual_reviews = total_reviews - template_reviews

        labels = ['Template-based\nReviews', 'Individual\nReviews']
        sizes = [template_reviews, individual_reviews]
        colors = ['lightcoral', 'lightblue']
        explode = (0.1, 0)

        ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax4.set_title(f'Template vs Individual Reviews\n(Total: {total_reviews})', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'template_usage_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Template usage analysis: {len(templates)} templates, {template_reviews} template-based reviews")

    except Exception as e:
        print(f"  ⚠️ Error creating template usage visualization: {e}")


def run_coordinated_attacks_analysis(anomalies_df: pd.DataFrame, output_dir: str = "evaluation_plots") -> Dict:
    """
    Run complete coordinated attacks analysis including both analysis and visualizations.
    
    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots and results
        
    Returns:
        Dictionary with all coordinated attacks analysis results
    """

    # Create attacks analysis subdirectory
    attacks_plots_dir = os.path.join(output_dir, "attacks_analysis")
    os.makedirs(attacks_plots_dir, exist_ok=True)

    results = {}
    similarity_results = analyze_similarity_patterns(anomalies_df)
    results['similarity_patterns'] = similarity_results

    bombing_results = analyze_review_bombing_patterns(anomalies_df)
    results['review_bombing'] = bombing_results

    create_temporal_pattern_visualizations(anomalies_df, attacks_plots_dir)

    bombing_patterns = results.get('review_bombing', {})
    create_coordinated_attack_visualizations(
        anomalies_df,
        bombing_patterns,
        similarity_results,
        {},
        attacks_plots_dir
    )

    return results
