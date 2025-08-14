
import matplotlib.pyplot as plt
from src.amazon_reviews_loader import AmazonReviews2023Loader
def plot_review_length_distribution(
    category: str,
    num_samples: int = 10000,
    bins: int = 50
):
    # Initialize loader
    loader = AmazonReviews2023Loader()

    # Load a finite sample (non-streaming) of reviews
    reviews = loader.load_reviews(
        category=category,
        split="full",
        streaming=False,
        num_samples=num_samples
    )

    # Extract the review text and compute word counts
    word_counts = []
    for rec in reviews:
        text = rec.get("text", "")
        # simple split on whitespace
        word_counts.append(len(text.split()))

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=bins, edgecolor='black')
    plt.xlabel("Review Length (number of words)")
    plt.ylabel("Number of Reviews")
    plt.title(f"Distribution of Review Lengths ({category}, n={len(word_counts)})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: plot for the Electronics category
    plot_review_length_distribution(category="Electronics", num_samples=10000, bins=60)
