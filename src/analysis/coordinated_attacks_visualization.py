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


def _detect_coordinated_attacks(similarity_matrix: np.ndarray, valid_df: pd.DataFrame,
                                valid_indices: List[int]) -> List[Dict]:
    """Detect coordinated attacks using similarity patterns."""
    coordinated_groups = []
    high_sim_threshold = SIMILARITY_THRESHOLDS['high']

    processed_indices = set()

    for i in range(len(similarity_matrix)):
        if i in processed_indices:
            continue

        similar_indices = np.where(similarity_matrix[i] >= high_sim_threshold)[0]

        if len(similar_indices) >= 2:
            group_indices = list(set([i] + similar_indices.tolist()))
            group_df_indices = [valid_indices[idx] for idx in group_indices]

            if 'user_id' in valid_df.columns:
                users_in_group = valid_df.loc[group_df_indices, 'user_id'].unique()

                if len(users_in_group) > 1:
                    group_similarities = []
                    for gi in group_indices:
                        for gj in group_indices:
                            if gi != gj:
                                group_similarities.append(similarity_matrix[gi, gj])

                    avg_similarity = np.mean(group_similarities) if group_similarities else 0

                    coordinated_groups.append({
                        'indices': group_df_indices,
                        'size': len(group_indices),
                        'unique_users': len(users_in_group),
                        'avg_similarity': avg_similarity,
                        'confidence': 'HIGH' if avg_similarity > 0.85 else 'MEDIUM'
                    })

                    processed_indices.update(group_indices)

    return coordinated_groups


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
                        if i != j:
                            similarities.append(similarity_matrix[i, j])

                if similarities:
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)

                    if avg_similarity > 0.6 or max_similarity > 0.85:
                        spam_patterns.append({
                            'user_id': user_id,
                            'indices': list(user_df_indices),
                            'review_count': len(user_df_indices),
                            'avg_similarity': avg_similarity,
                            'max_similarity': max_similarity,
                            'confidence': 'HIGH' if avg_similarity > 0.8 else 'MEDIUM'
                        })

    return spam_patterns


def _detect_review_templates(similarity_matrix: np.ndarray, valid_df: pd.DataFrame,
                             valid_indices: List[int]) -> List[Dict]:
    """Detect review templates using clustering."""
    templates = []

    try:
        from sklearn.cluster import DBSCAN

        # Use similarity as distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # DBSCAN clustering on similarity
        clustering = DBSCAN(eps=0.25, min_samples=3, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Analyze clusters
        unique_labels = set(cluster_labels)
        for label in unique_labels:
            if label != -1:  # Not noise
                cluster_indices = np.where(cluster_labels == label)[0]
                cluster_df_indices = [valid_indices[idx] for idx in cluster_indices]

                # Calculate cluster statistics
                cluster_similarities = []
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i != j:
                            cluster_similarities.append(similarity_matrix[i, j])

                if cluster_similarities:
                    avg_similarity = np.mean(cluster_similarities)

                    # Check user diversity
                    if 'user_id' in valid_df.columns:
                        users_in_cluster = valid_df.loc[cluster_df_indices, 'user_id'].unique()
                        user_diversity = len(users_in_cluster) / len(cluster_indices)
                    else:
                        user_diversity = 1.0

                    templates.append({
                        'template_id': label,
                        'indices': cluster_df_indices,
                        'size': len(cluster_indices),
                        'avg_similarity': avg_similarity,
                        'unique_users': len(users_in_cluster) if 'user_id' in valid_df.columns else 'unknown',
                        'user_diversity': user_diversity,
                        'confidence': 'HIGH' if avg_similarity > 0.8 and user_diversity < 0.8 else 'MEDIUM'
                    })

    except Exception as e:
        print(f"  ⚠️ Error in template detection: {e}")

    return templates


def create_similarity_network_graph(anomalies_df: pd.DataFrame, output_dir: str, similarity_matrix, anomalies_indices):
    """Create network graph visualization of similarity patterns between anomalies."""

    try:
        import networkx as nx
        from matplotlib.patches import Patch

        coordinated_groups = _detect_coordinated_attacks(
            similarity_matrix, anomalies_df, anomalies_indices
        )

        coordinated_attacks = {
            'num_groups': len(coordinated_groups),
            'total_reviews_in_groups': sum(len(group['indices']) for group in coordinated_groups),
            'groups': coordinated_groups
        }

        spam_patterns = _detect_user_spam_patterns(
            similarity_matrix, anomalies_df, anomalies_indices
        )

        if not coordinated_attacks.get('groups') and not spam_patterns:
            print("  ⚠️ No coordinated attacks or spam patterns found for network visualization")
            return None, None

        # Create network graph
        G = nx.Graph()

        # Color mapping for different types
        node_colors = {}
        node_sizes = {}

        # Add coordinated attack groups
        coord_groups = coordinated_attacks.get('groups', [])
        for i, group in enumerate(coord_groups):
            group_indices = group['indices']
            group_color = plt.cm.Set1(i % 9)

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
                    if idx1 in anomalies_df.index and idx2 in anomalies_df.index and idx1 != idx2:
                        G.add_edge(idx1, idx2, weight=group.get('avg_similarity', 0.8), type='coordinated')

        for i, pattern in enumerate(spam_patterns):
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
                            idx1 != idx2 and not G.has_edge(idx1, idx2)):
                        G.add_edge(idx1, idx2, weight=pattern.get('avg_similarity', 0.7), type='spam')

        if len(G.nodes()) == 0:
            print("  ⚠️ No valid nodes for network visualization")
            return None, None

        # Create network visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw edges
        coord_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'coordinated']
        spam_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'spam']

        nx.draw_networkx_edges(G, pos, edgelist=coord_edges, edge_color='blue',
                               alpha=0.6, width=2, ax=ax, label='Coordinated')
        nx.draw_networkx_edges(G, pos, edgelist=spam_edges, edge_color='red',
                               alpha=0.6, width=1, ax=ax, label='Spam')

        # Draw nodes
        node_color_list = [node_colors.get(node, 'gray') for node in G.nodes()]
        node_size_list = [node_sizes.get(node, 100) for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_color_list,
                               node_size=node_size_list, alpha=0.8, ax=ax)

        ax.set_title('Anomaly Similarity Network\n(Coordinated Attacks & Spam Patterns)',
                     fontweight='bold', fontsize=14)
        ax.axis('off')

        # Add legend
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label=f'Coordinated Groups ({len(coord_groups)})'),
            Patch(facecolor='red', alpha=0.6, label=f'Spam Patterns ({len(spam_patterns)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_network_graph.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Similarity network graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

        return G, {'coord_groups': coord_groups, 'spam_patterns': spam_patterns}

    except ImportError:
        print("  ⚠️ NetworkX not available for network visualization")
        return None, None
    except Exception as e:
        print(f"  ⚠️ Error creating similarity network graph: {e}")
        return None, None


def run_coordinated_attacks_analysis(anomalies_df: pd.DataFrame, output_dir: str) -> Dict:
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
    cleaned_texts = preprocess_texts(text_data)

    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    anomalies_indices = anomalies_df.index.tolist()
    create_similarity_network_graph(anomalies_df, attacks_plots_dir, similarity_matrix, anomalies_indices)