import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

class AnomalyTextAnalyzer:
    """
    Analyze the text content of anomalies found by your DBSCAN
    """
    
    def __init__(self):
        self.vectorizer = None
        self.feature_names = None
        
    def analyze_anomaly_text(self, full_df, anomaly_indices):
        """
        Analyze what makes anomalous reviews textually different
        
        Args:
            full_df: Your full dataframe
            anomaly_indices: Indices of anomalies found by DBSCAN
            normal_indices: Sample of normal reviews for comparison (optional)
        """
        print(f"🔍 Analyzing text content of {len(anomaly_indices)} anomalies")
        print("=" * 60)
        
        # Get anomaly texts
        anomaly_texts = full_df.iloc[anomaly_indices]['text'].fillna('').tolist()
        
        # Get comparison texts (random sample of non-anomalies)
        all_indices = set(full_df.index)
        anomaly_set = set(anomaly_indices)
        normal_pool = list(all_indices - anomaly_set)
        normal_indices = np.random.choice(normal_pool, 
                                        size=min(len(anomaly_indices) * 3, len(normal_pool)), 
                                        replace=False)
    
        normal_texts = full_df.iloc[normal_indices]['text'].fillna('').tolist()
        
        print(f"📊 Comparing {len(anomaly_texts)} anomalies vs {len(normal_texts)} normal reviews")
        
        # Analyze differences
        self._compare_text_patterns(anomaly_texts, normal_texts)
        self._find_distinctive_words(anomaly_texts, normal_texts)
        self._categorize_anomalies(full_df, anomaly_indices)
        
        return self
    
    def _compare_text_patterns(self, anomaly_texts, normal_texts):
        """Compare basic text statistics"""
        
        print("\n📏 TEXT STATISTICS COMPARISON")
        print("-" * 40)
        
        def get_text_stats(texts):
            stats = {
                'avg_length': np.mean([len(text) for text in texts]),
                'avg_words': np.mean([len(text.split()) for text in texts]),
                'avg_sentences': np.mean([len(text.split('.')) for text in texts]),
                'exclamation_ratio': np.mean(['!' in text for text in texts]),
                'question_ratio': np.mean(['?' in text for text in texts]),
                'caps_ratio': np.mean([sum(c.isupper() for c in text) / max(len(text), 1) for text in texts])
            }
            return stats
        
        anomaly_stats = get_text_stats(anomaly_texts)
        normal_stats = get_text_stats(normal_texts)
        
        # Display comparison
        for metric in anomaly_stats:
            anomaly_val = anomaly_stats[metric]
            normal_val = normal_stats[metric]
            diff = ((anomaly_val - normal_val) / normal_val * 100) if normal_val > 0 else 0
            
            print(f"{metric:15s}: Anomaly {anomaly_val:8.2f} | Normal {normal_val:8.2f} | Diff {diff:+6.1f}%")
    
    def _find_distinctive_words(self, anomaly_texts, normal_texts):
        """Find words that are distinctive to anomalies vs normal reviews"""
        
        print("\n🔤 DISTINCTIVE WORDS ANALYSIS")
        print("-" * 40)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Combine texts with labels
        all_texts = anomaly_texts + normal_texts
        labels = ['anomaly'] * len(anomaly_texts) + ['normal'] * len(normal_texts)
        
        # Fit TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF for each group
        anomaly_tfidf = tfidf_matrix[:len(anomaly_texts)].mean(axis=0).A1
        normal_tfidf = tfidf_matrix[len(anomaly_texts):].mean(axis=0).A1
        
        # Find words more distinctive to anomalies
        tfidf_diff = anomaly_tfidf - normal_tfidf
        
        print("\n🚨 Words MORE common in anomalies:")
        top_anomaly_words = np.argsort(tfidf_diff)[-15:][::-1]
        for i, idx in enumerate(top_anomaly_words, 1):
            if tfidf_diff[idx] > 0:
                word = self.feature_names[idx]
                anomaly_score = anomaly_tfidf[idx]
                normal_score = normal_tfidf[idx]
                print(f"{i:2d}. '{word}' (A:{anomaly_score:.4f} vs N:{normal_score:.4f})")
        
        print("\n✅ Words MORE common in normal reviews:")
        bottom_anomaly_words = np.argsort(tfidf_diff)[:15]
        for i, idx in enumerate(bottom_anomaly_words, 1):
            if tfidf_diff[idx] < 0:
                word = self.feature_names[idx]
                anomaly_score = anomaly_tfidf[idx]
                normal_score = normal_tfidf[idx]
                print(f"{i:2d}. '{word}' (A:{anomaly_score:.4f} vs N:{normal_score:.4f})")
    
    def _categorize_anomalies(self, df, anomaly_indices):
        """Group anomalies by text patterns using clustering"""
        
        print("\n📁 ANOMALY CATEGORIZATION")
        print("-" * 40)
        
        if self.vectorizer is None:
            print("❌ Run analyze_anomaly_text first!")
            return
        
        # Get TF-IDF for anomalies only
        anomaly_texts = df.iloc[anomaly_indices]['text'].fillna('').tolist()
        anomaly_tfidf = self.vectorizer.transform(anomaly_texts)
        
        # Cluster anomalies by text similarity
        n_clusters = min(5, len(anomaly_texts) // 10)  # Reasonable number of clusters
        if n_clusters < 2:
            print("⚠️  Too few anomalies to categorize")
            return
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(anomaly_tfidf)
        
        print(f"📊 Found {n_clusters} types of anomalous text patterns:")
        
        # Analyze each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_indices = np.array(anomaly_indices)[cluster_mask]
            
            print(f"\n📂 ANOMALY TYPE {cluster_id + 1} ({cluster_mask.sum()} reviews):")
            
            # Get representative words for this cluster
            cluster_tfidf = anomaly_tfidf[cluster_mask].mean(axis=0).A1
            top_words = np.argsort(cluster_tfidf)[-5:][::-1]
            
            keywords = [self.feature_names[idx] for idx in top_words if cluster_tfidf[idx] > 0]
            print(f"   Key words: {', '.join(keywords)}")
            
            # Show example reviews
            example_indices = cluster_indices[:2]  # First 2 examples
            for idx in example_indices:
                review = df.iloc[idx]
                print(f"   Example: Rating {review['rating']}, Sentiment {review['sentiment']:.2f}")
                print(f"           '{review['text'][:150]}...'")
    
    def analyze_individual_anomaly(self, df, review_index):
        """
        Deep dive into a single anomalous review
        """
        print(f"\n🔍 INDIVIDUAL ANOMALY ANALYSIS - Review Index {review_index}")
        print("=" * 60)
        
        if self.vectorizer is None:
            print("❌ Run analyze_anomaly_text first!")
            return
        
        review = df.iloc[review_index]
        review_text = review['text']
        
        # Basic info
        print(f"Rating: {review['rating']}")
        print(f"Sentiment: {review['sentiment']:.3f}")
        print(f"Rating Mismatch: {review['rating_mismatch']:.3f}")
        print(f"Verified Purchase: {review['verified_purchase']}")
        print(f"Helpful Votes: {review['helpful_vote']}")
        
        print(f"\nFull Review Text:")
        print(f"'{review_text}'")
        
        # TF-IDF analysis
        review_tfidf = self.vectorizer.transform([review_text])
        tfidf_scores = review_tfidf.toarray()[0]
        
        # Find most distinctive words in this review
        top_word_indices = np.argsort(tfidf_scores)[-10:][::-1]
        
        print(f"\n🔤 Most distinctive words/phrases in this review:")
        for i, idx in enumerate(top_word_indices, 1):
            if tfidf_scores[idx] > 0:
                word = self.feature_names[idx]
                score = tfidf_scores[idx]
                print(f"{i:2d}. '{word}': {score:.4f}")


# STANDALONE USAGE EXAMPLE
def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_csv', type=str, required=True)
    parser.add_argument('--original_csv', type=str, required=True)
    args = parser.parse_args()
    
    try:
        # Load data
        anomaly_df = pd.read_csv(args.anomaly_csv)
        original_df = pd.read_csv(args.original_csv)
        
        # Get anomaly indices
        if 'original_index' not in anomaly_df.columns:
            raise ValueError("Anomaly CSV must contain 'original_index' column")
        anomaly_indices = anomaly_df['original_index'].tolist()
        
        # Run analysis
        analyzer = AnomalyTextAnalyzer()
        analyzer.analyze_anomaly_text(original_df, anomaly_indices)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("TF-IDF POST-HOC ANALYSIS TOOL")
    print("=" * 40)
    
    analyze()