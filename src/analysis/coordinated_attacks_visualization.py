from typing import Dict, List
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import numpy as np
import os

plt.style.use('default')
sns.set_palette("husl")

TEXT_COLUMNS = ['text', 'review_data', 'title']

SIMILARITY_THRESHOLDS = {
    'high': 0.75,
    'moderate': 0.5,
    'isolation': 0.3
}


def _detect_user_spam_patterns(similarity_matrix: np.ndarray, valid_df: pd.DataFrame,
                               valid_indices: List[int]) -> List[Dict]:
    """Detect spam patterns within individual users."""
    spam_patterns = []

    if 'user_id' not in valid_df.columns:
        return spam_patterns

    user_groups = valid_df.groupby('user_id').groups

    for user_id, user_df_indices in user_groups.items():
        if len(user_df_indices) >= 2:
            user_sim_indices = []
            for df_idx in user_df_indices:
                try:
                    sim_idx = valid_indices.index(df_idx)
                    user_sim_indices.append(sim_idx)
                except ValueError:
                    continue

            if len(user_sim_indices) >= 2:
                similarities = []
                for i in user_sim_indices:
                    for j in user_sim_indices:
                        if i != j and similarity_matrix[i, j] > 0.9:
                            similarities.append(similarity_matrix[i, j])

                if similarities:
                    spam_patterns.append({
                        'user_id': user_id,
                        'indices': list(user_df_indices),
                        'review_count': len(user_df_indices),
                        'avg_similarity': np.mean(similarities)
                    })

    return spam_patterns


def create_review_length_distribution_plot(anomalies_df: pd.DataFrame, spam_patterns: List[Dict],
                                           output_dir: str, category: str):
    """Create a plot showing average review lengths per spam pattern user."""
    if not spam_patterns:
        return

    user_numbers = []
    avg_lengths = []
    user_ids = []
    review_counts = []

    for i, pattern in enumerate(spam_patterns):
        pattern_lengths = []

        for idx in pattern['indices']:
            if idx in anomalies_df.index and 'text' in anomalies_df.columns:
                review_text = str(anomalies_df.loc[idx, 'text'])
                review_length = len(review_text.split())
                pattern_lengths.append(review_length)

        if pattern_lengths:
            user_numbers.append(i + 1)
            avg_lengths.append(np.mean(pattern_lengths))
            user_ids.append(pattern['user_id'])
            review_counts.append(len(pattern_lengths))

    if not avg_lengths:
        print("No valid review texts found for length distribution")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    bars = ax.bar(user_numbers, avg_lengths, color='red', alpha=0.7, edgecolor='black', linewidth=1)
    for i, (user_num, avg_len, count) in enumerate(zip(user_numbers, avg_lengths, review_counts)):
        ax.text(user_num, avg_len + max(avg_lengths) * 0.01, f'{count} reviews',
                ha='center', va='bottom', fontsize=9, alpha=0.8)

    ax.set_xlabel('User Number')
    ax.set_ylabel('Average Review Length (words)')
    ax.set_title(f'Average Review Length per Spam Pattern User - {category}\n'
                 f'{len(spam_patterns)} spam patterns detected')
    ax.grid(True, alpha=0.3)

    ax.set_xticks(user_numbers)
    overall_mean = np.mean(avg_lengths)
    overall_median = np.median(avg_lengths)
    overall_std = np.std(avg_lengths)

    stats_text = f'Mean Avg Length: {overall_mean:.1f} words\nMedian Avg Length: {overall_median:.1f} words\nStd: {overall_std:.1f} words'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spam_review_length_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return {
        'user_avg_lengths': avg_lengths,
        'mean_avg_length': overall_mean,
        'median_avg_length': overall_median,
        'std_avg_length': overall_std,
        'total_patterns': len(spam_patterns)
    }


def create_spam_patterns_network_graph(anomalies_df: pd.DataFrame, output_dir: str, similarity_matrix,
                                       anomalies_indices, category: str):
    """Create improved network graph visualization for spam patterns."""
    spam_users = _detect_user_spam_patterns(
        similarity_matrix, anomalies_df, anomalies_indices
    )

    spam_users = [user for user in spam_users if user['review_count'] >= 3]

    if not spam_users:
        return None, None

    G = nx.Graph()
    node_colors = {}
    node_sizes = {}
    user_groups = {}

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(spam_users), 12)))

    for i, user in enumerate(spam_users):
        user_indices = user['indices']
        user_color = colors[i % len(colors)]
        user_groups[i] = []

        for idx in user_indices:
            if idx in anomalies_df.index:
                G.add_node(idx, user_id=user['user_id'], user=i)
                node_colors[idx] = user_color
                node_sizes[idx] = 300 + (user['review_count'] * 30)
                user_groups[i].append(idx)

        for j, idx1 in enumerate(user_indices):
            for idx2 in user_indices[j + 1:]:
                if idx1 in anomalies_df.index and idx2 in anomalies_df.index:
                    G.add_edge(idx1, idx2, weight=user.get('avg_similarity', 0.7), user=i)

    if len(G.nodes()) == 0:
        print("No valid nodes for spam users visualization")
        return None, None

    fig, ax = plt.subplots(1, 1, figsize=(14, 11))

    import math

    if len(spam_users) <= 1:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    else:
        num_users = len(spam_users)
        grid_cols = math.ceil(math.sqrt(num_users))
        grid_rows = math.ceil(num_users / grid_cols)
        cluster_spacing = 5.5

        sorted_users = sorted(enumerate(spam_users), key=lambda x: x[1]['review_count'], reverse=True)

        pos = {}
        cluster_centers = {}

        for idx, (user_idx, user) in enumerate(sorted_users):
            row = idx // grid_cols
            col = idx % grid_cols

            center_x = (col - (grid_cols - 1) / 2) * cluster_spacing
            center_y = (row - (grid_rows - 1) / 2) * cluster_spacing

            cluster_centers[user_idx] = (center_x, center_y)
        for user_idx, nodes in user_groups.items():
            if not nodes:
                continue

            center_x, center_y = cluster_centers[user_idx]

            if len(nodes) == 1:
                pos[nodes[0]] = (center_x, center_y)
            elif len(nodes) == 2:
                pos[nodes[0]] = (center_x - 0.6, center_y)
                pos[nodes[1]] = (center_x + 0.6, center_y)
            else:
                subgraph = G.subgraph(nodes)
                cluster_radius = min(1.8, cluster_spacing / 3)

                if len(subgraph.edges()) > 0:
                    sub_pos = nx.spring_layout(subgraph, k=1.5, iterations=150, seed=42)
                else:
                    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
                    sub_pos = {
                        node: (cluster_radius * 0.8 * np.cos(angle),
                               cluster_radius * 0.8 * np.sin(angle))
                        for node, angle in zip(nodes, angles)
                    }

                for node in nodes:
                    if node in sub_pos:
                        x, y = sub_pos[node]
                        x *= cluster_radius * 1.2
                        y *= cluster_radius * 1.2
                        pos[node] = (center_x + x, center_y + y)
                    else:
                        pos[node] = (center_x, center_y)

    all_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    edge_widths = [1 + w * 2 for w in all_weights]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.25, edge_color="gray", ax=ax)

    node_color_list = [node_colors.get(node, "gray") for node in G.nodes()]
    node_size_list = [node_sizes.get(node, 300) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_color_list,
                           node_size=node_size_list, alpha=0.85, ax=ax)

    for i, user in enumerate(spam_users):
        user_indices = [idx for idx in user['indices'] if idx in pos]
        if len(user_indices) >= 3:
            pts = np.array([pos[idx] for idx in user_indices])
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            center = np.mean(hull_pts, axis=0)
            expanded_pts = center + 1.3 * (hull_pts - center)
            poly = Polygon(expanded_pts, alpha=0.15, color=colors[i % len(colors)])
            ax.add_patch(poly)
        elif len(user_indices) == 2:
            pts = np.array([pos[idx] for idx in user_indices])
            center = np.mean(pts, axis=0)
            radius = np.linalg.norm(pts[0] - pts[1]) / 2 + 0.4
            circle = plt.Circle(center, radius, alpha=0.15, color=colors[i % len(colors)])
            ax.add_patch(circle)
        elif len(user_indices) == 1:
            center = pos[user_indices[0]]
            circle = plt.Circle(center, 0.4, alpha=0.15, color=colors[i % len(colors)])
            ax.add_patch(circle)

    ax.set_title(
        f"Clusters of reviews from the same user with repetitive text- {category} category\n{len(spam_users)} users detected",
        fontweight="bold", fontsize=14
    )
    ax.axis("off")

    user_info = [(user, colors[i % len(colors)], i) for i, user in enumerate(spam_users)]
    user_info.sort(key=lambda x: x[0]['review_count'])

    legend_elements = [
        Patch(facecolor=color, alpha=0.6,
              label=f"User {i + 1}: {u['review_count']} reviews (sim: {u['avg_similarity']:.2f})")
        for i, (u, color, idx) in enumerate(user_info)
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spamming_users_network.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return G, spam_users


def run_coordinated_attacks_analysis(anomalies_df: pd.DataFrame, output_dir: str, category: str) -> Dict:
    """
    Run complete coordinated attacks analysis including both analysis and visualizations.

    Args:
        anomalies_df: DataFrame containing anomaly detection results
        output_dir: Directory to save plots and results

    Returns:
        Dictionary with all coordinated attacks analysis results
    """

    attacks_plots_dir = os.path.join(output_dir, "coordinated_attacks")
    os.makedirs(attacks_plots_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    text_data = anomalies_df["text"].fillna('').astype(str)

    tfidf_matrix = vectorizer.fit_transform(text_data)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    anomalies_indices = anomalies_df.index.tolist()

    spam_graph, spam_patterns = create_spam_patterns_network_graph(
        anomalies_df, attacks_plots_dir, similarity_matrix, anomalies_indices, category
    )

    length_distribution_stats = None
    if spam_patterns:
        length_distribution_stats = create_review_length_distribution_plot(
            anomalies_df, spam_patterns, attacks_plots_dir, category
        )

    results = {
        'spam_patterns': {
            'count': len(spam_patterns) if spam_patterns else 0,
            'patterns': spam_patterns or [],
            'graph': spam_graph,
            'length_distribution': length_distribution_stats
        },
    }

    return results
