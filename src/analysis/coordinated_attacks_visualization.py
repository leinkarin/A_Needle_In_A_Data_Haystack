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
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
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


def create_spam_patterns_network_graph(anomalies_df: pd.DataFrame, output_dir: str, similarity_matrix, anomalies_indices):
    """Create network graph visualization specifically for spam patterns."""
    
    try:
        import networkx as nx
        from matplotlib.patches import Patch

        spam_patterns = _detect_user_spam_patterns(
            similarity_matrix, anomalies_df, anomalies_indices
        )

        if not spam_patterns:
            print("  ⚠️ No spam patterns found for network visualization")
            return None, None

        G = nx.Graph()
        node_colors = {}
        node_sizes = {}
        
        if len(spam_patterns) <= 12:
            colors = plt.cm.tab20(np.linspace(0, 1, max(len(spam_patterns), 12)))
        else:
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
            colors3 = plt.cm.Dark2(np.linspace(0, 1, 8))
            colors = np.concatenate([colors1, colors2, colors3])[:len(spam_patterns)]

        for i, pattern in enumerate(spam_patterns):
            pattern_indices = pattern['indices']
            pattern_color = colors[i]

            for idx in pattern_indices:
                if idx in anomalies_df.index:
                    user_id = pattern['user_id']
                    G.add_node(idx, user_id=user_id, type='spam', pattern=i)
                    node_colors[idx] = pattern_color
                    size = 200 + (pattern['avg_similarity'] * 300)
                    node_sizes[idx] = size

            for j, idx1 in enumerate(pattern_indices):
                for idx2 in pattern_indices[j + 1:]:
                    if (idx1 in anomalies_df.index and idx2 in anomalies_df.index and idx1 != idx2):
                        G.add_edge(idx1, idx2, weight=pattern.get('avg_similarity', 0.7), pattern=i)

        if len(G.nodes()) == 0:
            print("  ⚠️ No valid nodes for spam patterns visualization")
            return None, None

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50)

        for i, pattern in enumerate(spam_patterns):
            pattern_indices = pattern['indices']
            pattern_color = colors[i]
            
            pattern_edges = []
            pattern_edge_weights = []
            
            for j, idx1 in enumerate(pattern_indices):
                for idx2 in pattern_indices[j + 1:]:
                    if (idx1 in anomalies_df.index and idx2 in anomalies_df.index and 
                        G.has_edge(idx1, idx2)):
                        pattern_edges.append((idx1, idx2))
                        pattern_edge_weights.append(G[idx1][idx2]['weight'])
            
            if pattern_edges:
                edge_widths = [w * 3 for w in pattern_edge_weights]
                nx.draw_networkx_edges(G, pos, edgelist=pattern_edges, 
                                     edge_color=pattern_color, alpha=0.6, 
                                     width=edge_widths, ax=ax)

        node_color_list = [node_colors.get(node, 'gray') for node in G.nodes()]
        node_size_list = [node_sizes.get(node, 200) for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_color_list,
                               node_size=node_size_list, alpha=0.8, ax=ax)

        ax.set_title(f'Spam Patterns Network\n({len(spam_patterns)} patterns detected)',
                     fontweight='bold', fontsize=14)
        ax.axis('off')

        pattern_info = [(pattern, colors[i], i) for i, pattern in enumerate(spam_patterns)]
        pattern_info.sort(key=lambda x: x[0]['review_count'])
        
        legend_elements = []
        for pattern, color, original_idx in pattern_info:
            label = f"Pattern {original_idx+1} ({pattern['review_count']} reviews)"
            legend_elements.append(Patch(facecolor=color, alpha=0.8, label=label))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spam_patterns_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Spam patterns network: {len(G.nodes())} nodes, {len(G.edges())} edges, {len(spam_patterns)} patterns")

        return G, spam_patterns

    except ImportError:
        print("  ⚠️ NetworkX not available for spam patterns visualization")
        return None, None
    except Exception as e:
        print(f"  ⚠️ Error creating spam patterns network graph: {e}")
        return None, None


def create_coordinated_attacks_network_graph(anomalies_df: pd.DataFrame, output_dir: str, similarity_matrix, anomalies_indices):
    """Create network graph visualization specifically for coordinated attacks."""
    
    try:
        import networkx as nx
        from matplotlib.patches import Patch

        coordinated_groups = _detect_coordinated_attacks(
            similarity_matrix, anomalies_df, anomalies_indices
        )

        if not coordinated_groups:
            print("No coordinated attacks found for network visualization")
            return None, None

        G = nx.Graph()
        node_colors = {}
        node_sizes = {}
        
        if len(coordinated_groups) <= 12:
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(coordinated_groups), 10)))
        else:
            colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
            colors2 = plt.cm.Set1(np.linspace(0, 1, 9))
            colors3 = plt.cm.Dark2(np.linspace(0, 1, 8))
            colors4 = plt.cm.Set3(np.linspace(0, 1, 12))
            colors = np.concatenate([colors1, colors2, colors3, colors4])[:len(coordinated_groups)]

        for i, group in enumerate(coordinated_groups):
            group_indices = group['indices']
            group_color = colors[i]

            for idx in group_indices:
                if idx in anomalies_df.index:
                    user_id = anomalies_df.loc[idx, 'user_id'] if 'user_id' in anomalies_df.columns else f"user_{idx}"
                    G.add_node(idx, user_id=user_id, type='coordinated', group=i)
                    node_colors[idx] = group_color
                    base_size = 300
                    confidence_multiplier = 1.5 if group['confidence'] == 'HIGH' else 1.0
                    node_sizes[idx] = base_size * confidence_multiplier

            for j, idx1 in enumerate(group_indices):
                for idx2 in group_indices[j + 1:]:
                    if idx1 in anomalies_df.index and idx2 in anomalies_df.index and idx1 != idx2:
                        G.add_edge(idx1, idx2, weight=group.get('avg_similarity', 0.8), group=i)

        if len(G.nodes()) == 0:
            print("No valid nodes for coordinated attacks visualization")
            return None, None

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)

        for i, group in enumerate(coordinated_groups):
            group_indices = group['indices']
            group_color = colors[i]
            
            group_edges = []
            group_edge_weights = []
            
            for j, idx1 in enumerate(group_indices):
                for idx2 in group_indices[j + 1:]:
                    if (idx1 in anomalies_df.index and idx2 in anomalies_df.index and 
                        G.has_edge(idx1, idx2)):
                        group_edges.append((idx1, idx2))
                        group_edge_weights.append(G[idx1][idx2]['weight'])
            
            if group_edges:
                edge_widths = [w * 4 for w in group_edge_weights]
                nx.draw_networkx_edges(G, pos, edgelist=group_edges, 
                                     edge_color=group_color, alpha=0.7, 
                                     width=edge_widths, ax=ax)

        node_color_list = [node_colors.get(node, 'gray') for node in G.nodes()]
        node_size_list = [node_sizes.get(node, 300) for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_color_list,
                               node_size=node_size_list, alpha=0.8, ax=ax)

        ax.set_title(f'Coordinated Attacks Network\n({len(coordinated_groups)} groups detected)',
                     fontweight='bold', fontsize=14)
        ax.axis('off')

        group_info = [(group, colors[i], i) for i, group in enumerate(coordinated_groups)]
        group_info.sort(key=lambda x: x[0]['size'])
        
        legend_elements = []
        for group, color, original_idx in group_info:
            label = f"Group {original_idx+1} ({group['size']} reviews, {group['unique_users']} users)"
            legend_elements.append(Patch(facecolor=color, alpha=0.8, label=label))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coordinated_attacks_network.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Coordinated attacks network: {len(G.nodes())} nodes, {len(G.edges())} edges, {len(coordinated_groups)} groups")

        return G, coordinated_groups

    except ImportError:
        print("NetworkX not available for coordinated attacks visualization")
        return None, None
    except Exception as e:
        print(f"Error creating coordinated attacks network graph: {e}")
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
    
    print("Creating spam patterns network visualization...")
    spam_graph, spam_patterns = create_spam_patterns_network_graph(
        anomalies_df, attacks_plots_dir, similarity_matrix, anomalies_indices
    )
    
    print("Creating coordinated attacks network visualization...")
    coord_graph, coordinated_groups = create_coordinated_attacks_network_graph(
        anomalies_df, attacks_plots_dir, similarity_matrix, anomalies_indices
    )
    
    results = {
        'spam_patterns': {
            'count': len(spam_patterns) if spam_patterns else 0,
            'patterns': spam_patterns or [],
            'graph': spam_graph
        },
        'coordinated_attacks': {
            'count': len(coordinated_groups) if coordinated_groups else 0,
            'groups': coordinated_groups or [],
            'total_reviews_in_groups': sum(len(group['indices']) for group in (coordinated_groups or [])),
            'graph': coord_graph
        }
    }
    
    return results