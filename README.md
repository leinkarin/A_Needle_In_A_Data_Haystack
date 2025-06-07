# Amazon Reviews 2023 Dataset Loader

A comprehensive Python implementation to load and work with the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) from Hugging Face.

## Dataset Overview

The Amazon Reviews 2023 dataset is a large-scale collection with:
- **571.54M** user reviews
- **54.51M** users
- **48.19M** items
- **34** categories
- **Timespan**: May 1996 - Sep 2023

## Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from amazon_reviews_2023 import AmazonReviews2023Loader

# Initialize the loader
loader = AmazonReviews2023Loader()

# Load a small sample of Electronics reviews
reviews = loader.load_reviews(
    category="Electronics",
    streaming=True,
    num_samples=100
)

# Explore the data
loader.explore_sample(reviews, num_samples=2)

# Save sample data for analysis
loader.save_sample_data(reviews, "electronics_sample.json", num_samples=50)
```

### 3. Load Metadata

```python
# Load item metadata for Electronics category
metadata = loader.load_metadata(
    category="Electronics",
    streaming=True,
    num_samples=50
)
```

## Available Categories

The dataset includes 34 categories:
- All_Beauty
- Amazon_Fashion
- Appliances
- Arts_Crafts_and_Sewing
- Automotive
- Baby_Products
- Beauty_and_Personal_Care
- Books
- CDs_and_Vinyl
- Cell_Phones_and_Accessories
- Electronics
- Home_and_Kitchen
- Movies_and_TV
- Sports_and_Outdoors
- Video_Games
- ... and more

## Dataset Configurations

The dataset provides multiple configurations:

- **Raw Reviews**: `raw_review_{category}` (e.g., `raw_review_Electronics`)
- **Raw Metadata**: `raw_meta_{category}` (e.g., `raw_meta_Electronics`)
- **Processed versions**: Various filtered and processed subsets (0core, 5core variants)

## Key Features

### Review Data Fields
- `rating`: Product rating (1.0-5.0)
- `title`: Review title
- `text`: Review text content
- `images`: User-uploaded images
- `asin`: Product ID
- `parent_asin`: Parent product ID
- `user_id`: Reviewer ID
- `timestamp`: Review timestamp
- `verified_purchase`: Purchase verification
- `helpful_vote`: Helpfulness votes

### Metadata Fields
- `main_category`: Product category
- `title`: Product name
- `average_rating`: Average product rating
- `price`: Product price
- `images`: Product images
- `description`: Product description
- `features`: Product features
- `store`: Store name

## Usage Examples

### Example 1: Load Multiple Categories

```python
categories = ["Electronics", "Books", "Movies_and_TV"]

for category in categories:
    reviews = loader.load_reviews(
        category=category,
        streaming=True,
        num_samples=100
    )
    print(f"Loaded {category} reviews")
```

### Example 2: Save Data for Analysis

```python
# Load and save sample data
sample_data = loader.load_reviews(
    category="Beauty_and_Personal_Care",
    streaming=True,
    num_samples=1000
)

# Save as JSON
loader.save_sample_data(sample_data, "beauty_reviews.json", format="json")

# Save as CSV
loader.save_sample_data(sample_data, "beauty_reviews.csv", format="csv")
```

### Example 3: Get Dataset Information

```python
# Get dataset overview
info = loader.get_dataset_info()
print(f"Total reviews: {info['total_reviews']}")

# Get available configurations
configs = loader.get_available_configs()
print(f"Available review configs: {len(configs['raw_reviews'])}")
```

## Important Notes

1. **Streaming Mode**: Use `streaming=True` for large datasets to avoid memory issues
2. **Start Small**: Begin with small `num_samples` (10-100) for exploration
3. **Dataset Size**: The full dataset is extremely large - be mindful of your computational resources
4. **Authentication**: The dataset requires `trust_remote_code=True` (automatically handled)
5. **Rate Limits**: Be considerate of API rate limits when making repeated calls

## Files Included

- `amazon_reviews_2023.py`: Main loader implementation
- `example_usage.py`: Example usage demonstrations
- `requirements.txt`: Required dependencies
- `README.md`: This documentation

## Running the Examples

```bash
# Run the main demonstration
python amazon_reviews_2023.py

# Run the example usage script
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use streaming mode and limit sample sizes
2. **Network Timeouts**: Try smaller batch sizes or retry failed requests
3. **Missing Dependencies**: Ensure all packages from requirements.txt are installed

### Error Messages

- `BuilderConfig not found`: Make sure you're using the correct category names
- `Trust remote code`: The implementation automatically handles this requirement

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
