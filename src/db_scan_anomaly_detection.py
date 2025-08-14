from __future__ import annotations
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from amazon_reviews_loader import AmazonReviews2023Loader


TEXT_KEYS = [
    "text",
    "review_body",
    "content",
    "reviewText",
    "body",
]
RATING_KEYS = [
    "rating",
    "star_rating",
    "overall",
    "rating_star",
    "stars",
]
VERIFIED_KEYS = ["verified_purchase", "verified", "is_verified_purchase"]


def first_nonnull(row_dict: Dict[str, Any], keys: List[str], default=None):
    for key in keys:
        if key in row_dict and row_dict[key] is not None:
            return row_dict[key]
    return default

# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def collect_rows(dataset: Iterable, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Materialise review rows with canonical fields."""
    rows: List[Dict[str, Any]] = []
    for row in dataset:
        text = first_nonnull(row, TEXT_KEYS, default="")
        rating = first_nonnull(row, RATING_KEYS, default=None)
        if not text or rating is None:
            print(f"Skipping row: {row} because of missing text or rating")
        try:
            rating_val = float(rating)
        except Exception:
            print(f"Skipping row: {row} because of invalid rating")
            continue

        rows.append(
            {
                "text": str(text)[:2000],  # trim very long reviews
                "rating": rating_val,
                "verified": bool(first_nonnull(row, VERIFIED_KEYS, default=False)),
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
    return rows

def filter_reviews_by_token_limit(
    rows: List[Dict[str, Any]], 
    max_tokens: int = 4096,
    sentiment_model_id: str = "spacesedan/sentiment-analysis-longformer"
) -> List[Dict[str, Any]]:

    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_id)
    
    filtered_rows = []
    skipped_count = 0
    
    print(f"Filtering {len(rows)} reviews")
    
    for row in tqdm(rows, desc="Filtering by token count"):
        text = row["text"]
        rating = row["rating"]
        verified = row["verified"]

        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)
        
        if token_count <= max_tokens:
            filtered_rows.append(
                {
                    "text": text,
                    "rating": rating,
                    "verified": verified,
                }
            )
        else:
            skipped_count += 1
    
    print(f"Filtered dataset: {len(filtered_rows)} reviews kept, {skipped_count} reviews skipped")
    return filtered_rows

def compute_sentiment_scores(
    texts: List[str],
    batch_size: int = 128,
    model_id: str = "spacesedan/sentiment-analysis-longformer",
) -> np.ndarray:
    """Return sentiment confidence scores in [-1, 1] range."""
    clf = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id, device=-1)

    scores: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
        preds = clf(texts[i : i + batch_size])
        for p in preds:
            confidence = p["score"]
            label = p["label"].lower()
            
            if label.startswith("neg"):
                scores.append(-confidence)
            elif label.startswith("pos"):
                scores.append(confidence)
            else:
                scores.append(0.0)
    return np.asarray(scores, dtype=np.float32)


def embed_text(
    texts: List[str],
    model_id: str = "nomic-ai/nomic-embed-text-v1",
    batch_size: int = 128,
) -> np.ndarray:
    """Encode *texts* into Nomic embeddings (n, 768)."""
    model = SentenceTransformer(model_id)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# ---------------------------------------------------------------------------
#  Feature builder
# ---------------------------------------------------------------------------

def build_features(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return (X, meta_df)."""
    texts = [r["text"] for r in rows]

    # ---- heavy lifting ----------------------------------------------------
    X_text = embed_text(texts)
    sentiment = compute_sentiment_scores(texts)  # (n,)

    # ---- numeric & categorical -------------------------------------------
    rating_scaled = (np.array([r["rating"] for r in rows]) - 3.0) / 2.0  # (n,)
    mismatch = sentiment - rating_scaled  # key anomaly signal
    verified = np.array([r["verified"] for r in rows], dtype=float)[:, None]

    meta = pd.DataFrame(
        {
            "rating_scaled": rating_scaled,
            "sentiment": sentiment,
            "mismatch": mismatch,
            "verified": verified.squeeze(),
        }
    )

    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(meta[["rating_scaled", "sentiment", "mismatch"]])

    X = np.hstack([X_text, meta_scaled, verified])  # verified untouched (0/1)
    return X, meta

# ---------------------------------------------------------------------------
#  Dimensionality reduction & clustering
# ---------------------------------------------------------------------------

def pca_reduce(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    if n_components <= 0 or n_components >= X.shape[1]:
        return X
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X)


def dbscan_labels(
    X_red: np.ndarray, eps: float = 0.5, min_samples: int = 10
) -> np.ndarray:
    """Label each point; −1 ⇒ anomaly/outlier."""
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    return db.fit_predict(X_red)

# ---------------------------------------------------------------------------
#  Main entryp
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DBSCAN anomaly detection for reviews")
    ap.add_argument("--category", default="Electronics", help="Product category")
    ap.add_argument("--num-samples", type=int, default=2000, help="Sample size (0 ⇒ all)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per review")
    ap.add_argument("--pca-dim", type=int, default=50, help="PCA components (0 ⇒ skip)")
    ap.add_argument("--eps", type=float, default=0.6, help="DBSCAN eps radius")
    ap.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples")
    ap.add_argument("--out", default="dbscan_anomalies.csv", help="Output CSV file")
    args = ap.parse_args()

    loader = AmazonReviews2023Loader()

    print(
        f"=== DBSCAN anomaly detector ===\n"
        f"Category: {args.category} | N: {args.num_samples} | Max tokens: {args.max_tokens} | "
        f"PCA: {args.pca_dim} | eps: {args.eps} | min_samples: {args.min_samples}"
    )

    # -- Data ----------------------------------------------------------------
    ds = loader.load_reviews(
        category=args.category,
        streaming=True,
        split="test",
        num_samples=args.num_samples if args.num_samples > 0 else None,
    )
    
    # First collect the initial sample
    initial_rows = collect_rows(ds, max_rows=args.num_samples)
    print(f"Collected {len(initial_rows):,} initial reviews.")
    
    rows = filter_reviews_by_token_limit(initial_rows, max_tokens=args.max_tokens)
    print(f"After token filtering: {len(rows):,} reviews remaining.")

    # -- Features ------------------------------------------------------------
    X_full, meta = build_features(rows)
    X_red = pca_reduce(X_full, n_components=args.pca_dim)

    # -- Clustering ----------------------------------------------------------
    labels = dbscan_labels(X_red, eps=args.eps, min_samples=args.min_samples)
    n_out = int(np.sum(labels == -1))
    print(f"Flagged {n_out} / {len(rows)} reviews as anomalies.")

    # -- Save ----------------------------------------------------------------
    pd.DataFrame(rows)[labels == -1].to_csv(args.out, index=False)
    print(f"✓ Results written to {args.out}")


if __name__ == "__main__":
    main()
