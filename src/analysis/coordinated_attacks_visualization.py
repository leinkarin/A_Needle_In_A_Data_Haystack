import argparse
import re
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




def create_spam_patterns_network_graph(anomalies_df: pd.DataFrame, output_dir: str, similarity_matrix,
                                       anomalies_indices, category: str):
    """Create improved network graph visualization for spam patterns."""


    spam_users = _detect_user_spam_patterns(
        similarity_matrix, anomalies_df, anomalies_indices
    )

    spam_users = [user for user in spam_users if user['review_count'] >= 3]

    if not spam_users:
        print("No spam users found for network visualization")
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
        cluster_spacing = 5.5  # Further increased spacing to accommodate larger internal spacing
        
        sorted_users = sorted(enumerate(spam_users), key=lambda x: x[1]['review_count'], reverse=True)
        
        pos = {}
        cluster_centers = {}
        
        # Assign cluster centers in a grid
        for idx, (user_idx, user) in enumerate(sorted_users):
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Center the grid
            center_x = (col - (grid_cols - 1) / 2) * cluster_spacing
            center_y = (row - (grid_rows - 1) / 2) * cluster_spacing
            
            cluster_centers[user_idx] = (center_x, center_y)
        
        # Position nodes within each cluster
        for user_idx, nodes in user_groups.items():
            if not nodes:
                continue
                
            center_x, center_y = cluster_centers[user_idx]
            
            if len(nodes) == 1:
                # Single node - place at cluster center
                pos[nodes[0]] = (center_x, center_y)
            elif len(nodes) == 2:
                # Two nodes - place them further apart
                pos[nodes[0]] = (center_x - 0.6, center_y)
                pos[nodes[1]] = (center_x + 0.6, center_y)
            else:
                # Multiple nodes - use spring layout within larger cluster bounds
                subgraph = G.subgraph(nodes)
                cluster_radius = min(1.8, cluster_spacing / 3)  # Increased radius for more space
                
                if len(subgraph.edges()) > 0:
                    # Use spring layout for connected nodes with more spacing
                    sub_pos = nx.spring_layout(subgraph, k=1.5, iterations=150, seed=42)
                else:
                    # Arrange disconnected nodes in a larger circle
                    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
                    sub_pos = {
                        node: (cluster_radius * 0.8 * np.cos(angle), 
                               cluster_radius * 0.8 * np.sin(angle))
                        for node, angle in zip(nodes, angles)
                    }
                
                # Scale and translate to cluster center with more spacing
                for node in nodes:
                    if node in sub_pos:
                        x, y = sub_pos[node]
                        # Scale to fit within larger cluster radius
                        x *= cluster_radius * 1.2  # Additional scaling for more space
                        y *= cluster_radius * 1.2
                        # Translate to cluster center
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

    # Draw cluster boundaries with improved spacing
    for i, user in enumerate(spam_users):
        user_indices = [idx for idx in user['indices'] if idx in pos]
        if len(user_indices) >= 3:
            pts = np.array([pos[idx] for idx in user_indices])
            hull = ConvexHull(pts)
            # Expand hull slightly to create better visual separation
            hull_pts = pts[hull.vertices]
            center = np.mean(hull_pts, axis=0)
            expanded_pts = center + 1.3 * (hull_pts - center)  # Slightly more expansion
            poly = Polygon(expanded_pts, alpha=0.15, color=colors[i % len(colors)])
            ax.add_patch(poly)
        elif len(user_indices) == 2:
            pts = np.array([pos[idx] for idx in user_indices])
            center = np.mean(pts, axis=0)
            radius = np.linalg.norm(pts[0] - pts[1]) / 2 + 0.4  # Larger radius for increased spacing
            circle = plt.Circle(center, radius, alpha=0.15, color=colors[i % len(colors)])
            ax.add_patch(circle)
        elif len(user_indices) == 1:
            # For single-node users, draw a small circle
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
        Patch(facecolor=color, alpha=0.6, label=f"User {i + 1}: {u['review_count']} reviews (sim: {u['avg_similarity']:.2f})")
        for i, (u, color, idx) in enumerate(user_info)
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spamming_users_network.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return G, spam_users


def create_coordinated_attacks_network_graph(anomalies_df: pd.DataFrame, output_dir: str, similarity_matrix, anomalies_indices, category: str = "books"):
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
                    # Size based on group size and similarity
                    size_multiplier = 1 + (group['size'] / 10)  # Larger groups get bigger nodes
                    similarity_multiplier = 1 + group['avg_similarity']  # Higher similarity gets bigger nodes
                    node_sizes[idx] = base_size * size_multiplier * similarity_multiplier

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

        ax.set_title(f'Coordinated Attacks Network - {category}\n({len(coordinated_groups)} groups detected)',
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
    cleaned_texts = preprocess_texts(text_data)

    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    anomalies_indices = anomalies_df.index.tolist()
    
    spam_graph, spam_patterns = create_spam_patterns_network_graph(
        anomalies_df, attacks_plots_dir, similarity_matrix, anomalies_indices, category
    )
    
    coord_graph, coordinated_groups = create_coordinated_attacks_network_graph(
        anomalies_df, attacks_plots_dir, similarity_matrix, anomalies_indices, category
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


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze coordinated attacks in DBSCAN anomaly detection results",
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
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--category',
        type=str,
        default='electronics',
        help='The category of the reviews'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COORDINATED ATTACKS ANALYSIS")
    print("=" * 80)
    print(f"Anomaly Data File: {args.anomaly_data_file}")
    print(f"Category: {args.category}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # Validate input file
    if not os.path.exists(args.anomaly_data_file):
        print(f"❌ Anomaly data file not found: {args.anomaly_data_file}")
        print("Please provide a valid path using --anomaly-data-file")
        return

    try:
        # Load anomaly data
        print(f"Loading anomaly results from: {args.anomaly_data_file}")
        anomalies_df = pd.read_csv(args.anomaly_data_file)
        print(f"✓ Loaded {len(anomalies_df)} anomalies")

        # Validate required columns
        if 'text' not in anomalies_df.columns:
            print("❌ Required column 'text' not found in anomaly data")
            print("Available columns:", list(anomalies_df.columns))
            return

        # Create output directory
        category_dir = os.path.join(args.output_dir, args.category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Run coordinated attacks analysis
        print("\nRunning coordinated attacks analysis...")
        results = run_coordinated_attacks_analysis(anomalies_df, category_dir, args.category)
        
        # Print summary results
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS SUMMARY")
        print("=" * 60)
        
        spam_count = results['spam_patterns']['count']
        coord_count = results['coordinated_attacks']['count']
        total_coord_reviews = results['coordinated_attacks']['total_reviews_in_groups']
        
        print(f"Spam Patterns Found: {spam_count}")
        print(f"Coordinated Attack Groups Found: {coord_count}")
        print(f"Total Reviews in Coordinated Groups: {total_coord_reviews}")
        
        if spam_count > 0:
            print(f"\nTop Spam Patterns:")
            for i, pattern in enumerate(results['spam_patterns']['patterns'][:3]):
                print(f"  Pattern {i+1}: {pattern['review_count']} reviews from user {pattern['user_id']}")
                print(f"    Average similarity: {pattern['avg_similarity']:.2f}")
        
        if coord_count > 0:
            print(f"\nTop Coordinated Attack Groups:")
            for i, group in enumerate(results['coordinated_attacks']['groups'][:3]):
                print(f"  Group {i+1}: {group['size']} reviews from {group['unique_users']} users")
                print(f"    Average similarity: {group['avg_similarity']:.3f}")
        
        # Save results
        plots_dir = os.path.join(category_dir, "plots", "coordinated_attacks")
        print(f"\n✓ Analysis complete! Results saved to:")
        print(f"  - Plots directory: {plots_dir}")
        
        if spam_count > 0:
            print(f"  - Spam patterns network graph: {os.path.join(plots_dir, 'spam_patterns_network.png')}")
        
        if coord_count > 0:
            print(f"  - Coordinated attacks network graph: {os.path.join(plots_dir, 'coordinated_attacks_network.png')}")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("COORDINATED ATTACKS ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()