#!/usr/bin/env python3
"""
amazon_reviews_anomaly.py

Uses the provided AmazonReviews2023Loader (kept in a separate file) to:
- stream a sample of reviews from a chosen category
- build a mid-fusion feature (SBERT text embedding + numeric/categorical metadata)
- add a semantic sentiment score and a rating–text mismatch feature
- reduce with PCA
- cluster with MiniBatch K-Means
- rank by distance-to-centroid and save top anomalies to CSV

Run:
  python amazon_reviews_anomaly.py --category Electronics --num-samples 20000 --out anomalies.csv

Requirements:
  pip install datasets sentence-transformers transformers scikit-learn pandas numpy
"""

import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amazon_reviews_loader import AmazonReviews2023Loader  # noqa: E402


# --------- Helpers to map fields across slightly different schemas ----------

TEXT_KEYS = ["text", "review_body", "content", "reviewText", "body"]
RATING_KEYS = ["rating", "star_rating", "overall", "rating_star", "stars"]
HELPFUL_KEYS = ["helpful_vote", "helpful_votes", "helpful", "vote", "helpful_vote_count"]
VERIFIED_KEYS = ["verified_purchase", "verified", "is_verified_purchase"]
TITLE_KEYS = ["title", "summary", "review_title"]
ASIN_KEYS = ["asin", "parent_asin", "item_id"]


def _first_nonnull(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def collect_rows(ds: Iterable, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Materialize a list of rows from a (possibly streaming) dataset iterator.
    Ensures we pick the required core fields, with fallback key names.
    """
    rows: List[Dict[str, Any]] = []
    for r in ds:
        text = _first_nonnull(r, TEXT_KEYS, default="")
        rating = _first_nonnull(r, RATING_KEYS, default=None)
        if not text or rating is None:
            continue
        try:
            rating_val = float(rating)
        except Exception:
            # if rating is something odd, skip the row
            continue

        row = {
            "text": str(text)[:2000],
            "rating": rating_val,
            "helpful": int(_first_nonnull(r, HELPFUL_KEYS, default=0) or 0),
            "verified": bool(_first_nonnull(r, VERIFIED_KEYS, default=False)),
            "title": _first_nonnull(r, TITLE_KEYS, default=""),
            "asin": _first_nonnull(r, ASIN_KEYS, default=None),
        }
        rows.append(row)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


# ------------------------- Embedding / Scoring ------------------------------

def embed_text_sbert(
    texts: List[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> np.ndarray:
    enc = SentenceTransformer(model_id)
    X = enc.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(X)


def polarity_scores(
    texts: List[str],
    model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device: Optional[str] = None,
    batch_size: int = 128,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    clf = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = clf(**enc).logits  # [neg, neu, pos]
        probs = logits.softmax(-1)
        score = (probs[:, 2] - probs[:, 0]).cpu().numpy()  # ∈ [-1, +1]
        out.append(score)
    return np.concatenate(out, axis=0)


# ------------------------- Feature building ---------------------------------

def build_mid_fusion(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    texts = [r["text"] for r in rows]

    print(f"Embedding {len(texts)} texts with SBERT…")
    X_text = embed_text_sbert(texts)

    print("Computing semantic polarity scores…")
    sem = polarity_scores(texts)

    # metadata frame
    df_meta = pd.DataFrame({
        "rating_scaled": (np.array([r["rating"] for r in rows]) - 3.0) / 2.0,  # map 1..5 → [-1,1]
        "helpful": [r["helpful"] for r in rows],
        "verified": [r["verified"] for r in rows],
        "sem_score": sem,
    })
    df_meta["mismatch"] = (df_meta["sem_score"] - df_meta["rating_scaled"]).abs()

    # Column-wise encoders
    num_cols = ["rating_scaled", "helpful", "sem_score", "mismatch"]
    cat_cols = ["verified"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="if_binary"), cat_cols),
    ])

    print("Fitting column transformer on metadata…")
    X_meta = pre.fit_transform(df_meta)  # verified becomes 1 col due to drop-if-binary

    # Mid fusion (concat)
    X_full = np.hstack([X_text, X_meta])  # 384 + 5 = 389 dims (for MiniLM-6)
    return X_full, df_meta, X_text


def pca_reduce(X: np.ndarray, target_dim: int = 100, random_state: int = 0) -> np.ndarray:
    n = X.shape[0]
    max_components = max(2, min(target_dim, X.shape[1] - 1, n - 1))
    print(f"Reducing from {X.shape[1]} → {max_components} dims with PCA…")
    pca = PCA(n_components=max_components, random_state=random_state)
    return pca.fit_transform(X)


def kmeans_score(
    X_red: np.ndarray, n_clusters: Optional[int] = None, batch_size: int = 10_000
) -> Tuple[np.ndarray, MiniBatchKMeans]:
    n = X_red.shape[0]
    if n_clusters is None:
        # simple heuristic: grows with data size, bounded
        n_clusters = int(np.clip(np.sqrt(n / 50), 5, 50))
    print(f"Clustering with MiniBatchKMeans (k={n_clusters})…")
    mbkm = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=min(batch_size, n), n_init="auto"
    )
    labels = mbkm.fit_predict(X_red)
    centers = mbkm.cluster_centers_[labels]
    dist = np.linalg.norm(X_red - centers, axis=1)
    return dist, mbkm


def summarize_anomalies(
    rows: List[Dict[str, Any]],
    dist: np.ndarray,
    df_meta: pd.DataFrame,
    top_quantile: float = 0.99,
    top_k_show: int = 20,
) -> pd.DataFrame:
    thr = float(np.quantile(dist, top_quantile))
    idx = np.where(dist > thr)[0]
    print(f"Anomaly threshold @ q={top_quantile:.2f}: {thr:.3f} → {len(idx)} rows flagged.")

    df = pd.DataFrame(rows)
    df["distance"] = dist
    df = pd.concat([df, df_meta.reset_index(drop=True)], axis=1)
    anomalies = df.loc[idx].sort_values("distance", ascending=False)

    # Display a few
    print("\nTop anomalies:")
    for _, r in anomalies.head(top_k_show).iterrows():
        snippet = (r["text"][:300].replace("\n", " ") + "…") if isinstance(r["text"], str) else ""
        print(
            f"[{int(r['rating'])}★] d={r['distance']:.2f}  sem={r['sem_score']:+.2f}  "
            f"mismatch={r['mismatch']:.2f}  verified={r['verified']}  helpful={r['helpful']}  asin={r.get('asin')}"
        )
        print(snippet, "\n")

    return anomalies


# ------------------------------- MAIN ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description="K-means anomaly detection on Amazon Reviews 2023")
    parser.add_argument("--category", type=str, default="Books", help="Category name (e.g., Electronics, Books)")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of reviews to stream")
    parser.add_argument("--pca-dim", type=int, default=100, help="PCA target dimensionality")
    parser.add_argument("--clusters", type=int, default=None, help="Number of K-means clusters (default: heuristic)")
    parser.add_argument("--top-quantile", type=float, default=0.99, help="Quantile for anomaly threshold (0-1)")
    parser.add_argument("--out", type=str, default="anomalies.csv", help="Where to save the anomalies CSV")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode (loads into memory)")
    args = parser.parse_args()

    loader = AmazonReviews2023Loader()

    print("=== Amazon Reviews 2023: Anomaly Detection ===")
    print(f"Category: {args.category} | Samples: {args.num_samples} | PCA: {args.pca_dim} | "
          f"Clusters: {args.clusters or 'auto'} | q*: {args.top_quantile}")

    print("\nLoading reviews…")
    reviews_data_set = loader.load_reviews(
        category=args.category,
        streaming=(not args.no_stream),
        num_samples=args.num_samples,
    )

    print("Collecting rows…")
    rows = collect_rows(reviews_data_set, max_rows=args.num_samples)

    # Build fused features
    X_full, df_meta, X_text = build_mid_fusion(rows)

    # Reduce & cluster
    X_red = pca_reduce(X_full, target_dim=args.pca_dim)
    dist, _ = kmeans_score(X_red, n_clusters=args.clusters)

    # Summarize & save
    anomalies = summarize_anomalies(rows, dist, df_meta, top_quantile=args.top_quantile, top_k_show=20)
    cols_to_save = ["asin", "rating", "helpful", "verified", "sem_score", "mismatch", "distance", "title", "text"]
    anomalies.to_csv(args.out, index=False, columns=[c for c in cols_to_save if c in anomalies.columns])
    print(f"\n✓ Saved anomalies to {args.out}")


if __name__ == "__main__":
    main()
