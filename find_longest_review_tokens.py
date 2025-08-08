import argparse
import os
from typing import Iterable, Optional, List

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from transformers import AutoTokenizer

# Local utilities
try:
    from amazon_reviews_loader import AmazonReviews2023Loader
except Exception:
    AmazonReviews2023Loader = None  # type: ignore


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def iter_amazon2023_reviews(
    category: str,
    max_samples: Optional[int] = None,
) -> Iterable[str]:
    if AmazonReviews2023Loader is None:
        raise RuntimeError(
            "AmazonReviews2023Loader import failed. Ensure the script is run from the project root with PYTHONPATH including this directory."
        )
    loader = AmazonReviews2023Loader()
    dataset = loader.load_reviews(category=category, split="full", streaming=True, num_samples=max_samples)

    yielded = 0
    for example in dataset:
        text = (
            (example.get("text") if isinstance(example, dict) else None)
            or (example.get("reviewText") if isinstance(example, dict) else None)
            or (example.get("content") if isinstance(example, dict) else None)
        )
        if text:
            yield text
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                break


def compute_lengths(tokenizer, text_iter: Iterable[str]) -> List[int]:
    lengths: List[int] = []
    for text in text_iter:
        lengths.append(count_tokens(tokenizer, text))
    return lengths


def plot_histogram(
    lengths: List[int],
    category: str,
    output_path: str,
    bucket_size: Optional[int] = None,
    bins: int = 50,
) -> None:
    if not lengths:
        return

    max_len = int(max(lengths))
    mean_len = float(np.mean(lengths))

    # Fixed buckets as requested
    # [0,50), [50,500), [500,1000), [1000,1500), [1500,2000), [2500,3000), [3500,4000), [4000, +inf)
    fixed_edges = np.array([0, 50, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], dtype=int)
    arr = np.asarray(lengths)
    counts_fixed, _ = np.histogram(arr, bins=fixed_edges)
    count_4000_plus = int(np.sum(arr >= 4000))

    bar_lefts = list(fixed_edges[:-1]) + [4000]
    bar_widths = list(np.diff(fixed_edges)) + [max(max_len - 4000, 100)]
    bar_counts = list(counts_fixed) + [count_4000_plus]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.bar(bar_lefts, bar_counts, width=bar_widths, align="edge", color="#4C78A8", edgecolor="white")
    plt.axvline(mean_len, color="#2ca02c", linestyle="--", label=f"mean = {mean_len:.0f}")
    plt.axvline(max_len, color="#d62728", linestyle=":", label=f"max = {max_len}")
    plt.title(f"{category} — Tokenized review length buckets (n={len(lengths)})")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count of reviews")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-category histograms of tokenized Amazon Reviews 2023 lengths, annotated with mean and max."
    )
    parser.add_argument("--model-name", default="allenai/longformer-base-4096", help="Hugging Face model name for the tokenizer.")
    parser.add_argument("--category", default=None, help="Single Amazon 2023 category to process.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated list of categories to process (e.g., Electronics,Books). Overrides --category if both are set.")
    parser.add_argument("--list-categories", action="store_true", help="List available categories and exit.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of reviews to scan per category.")
    parser.add_argument("--bucket-size", type=int, default=None, help="Fixed-width token bucket size for the histogram (e.g., 100). If omitted, uses --bins.")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histogram when --bucket-size is not set.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "plots", "token_analyzations"),
        help="Directory to save the per-category histograms.",
    )

    args = parser.parse_args()

    if AmazonReviews2023Loader is None:
        raise SystemExit("amazon_reviews_loader could not be imported.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    available: List[str] = list(AmazonReviews2023Loader.CATEGORIES) 
    if args.list_categories:
        print("Available categories:")
        print("\n".join(available))
        return

    if args.categories:
        categories: List[str] = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.category:
        categories = [args.category]
    else:
        categories = available

    invalid = [c for c in categories if c not in available]
    if invalid:
        print(f"Unknown categories: {', '.join(invalid)}")
        print(f"Available: {', '.join(available)}")
        categories = [c for c in categories if c in available]
    if not categories:
        print("No valid categories to process. Exiting.")
        return

    print(f"Model: {args.model_name}")
    print(f"Saving histograms to: {args.output_dir}")

    for cat in categories:
        try:
            print(f"Processing category: {cat}...")
            text_iter = iter_amazon2023_reviews(category=cat, max_samples=args.max_samples)
            lengths = compute_lengths(tokenizer, text_iter)
            if not lengths:
                print(f"  No texts found; skipping {cat}.")
                continue

            filename = f"{cat.replace(' ', '_').lower()}_token_hist.png"
            output_path = os.path.join(args.output_dir, filename)
            plot_histogram(
                lengths=lengths,
                category=cat,
                output_path=output_path,
                bucket_size=args.bucket_size,
                bins=args.bins,
            )

            print(
                f"  Saved: {output_path} | n={len(lengths)} mean={int(np.mean(lengths))} max={int(np.max(lengths))}"
            )
        except Exception as e:
            print(f"  Error processing {cat}: {e}")


if __name__ == "__main__":
    main() 