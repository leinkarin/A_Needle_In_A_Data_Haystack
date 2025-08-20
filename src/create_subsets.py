import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any
import sys
import os
import re
import argparse
from amazon_reviews_loader import AmazonReviews2023Loader
import gc
from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str) or text == "" or text == " ":
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text


def compute_sentiment_score(text: str, tokenizer, model, device) -> float:
    """
    Compute sentiment score
    
    Args:
        text: Review text to analyze
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded sentiment model
        device: Device to run inference on (CPU/GPU)
        
    Returns:
        Sentiment score between -1 (negative) and 1 (positive), or None if computation failed
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        negative_score = probabilities[0][0].item()
        positive_score = probabilities[0][1].item()
        
        # Convert to -1 to 1 scale
        sentiment_score = positive_score - negative_score
        
        return sentiment_score
        
    except Exception as e:
        print(f"Error computing sentiment: {e}")
        return None


def remove_duplicates(data_list: List[Dict]) -> List[Dict]:
    """
    Args:
        data_list: List of review dictionaries
        
    Returns:
        List with duplicates removed
    """
    if not data_list:
        return []
    
    df = pd.DataFrame(data_list)
    df_unique = df.drop_duplicates()
    
    return df_unique.to_dict('records')


def create_dataset(
    output_path: str = "data_v2",
    train_samples_per_category: int = 10000,
    val_samples_per_category: int = 10000,
    test_samples_per_category: int = 1000,
    max_tokens: int = 512,
    selected_categories: List[str] = None,
):
    """
    Create train, validation, and test datasets with specified parameters.
    
    STRICT NULL-FREE DATA GUARANTEE:
    This function ensures NO NULL VALUES in any field by:
    - Rejecting reviews with any None/null values in critical fields
    - Validating all text fields are non-empty after cleaning
    - Converting and validating data types for all fields
    - Ensuring rating is in valid range (1.0-5.0)
    - Filtering out invalid ASINs, user IDs, timestamps
    - Rejecting negative helpful votes
    
    Args:
        train_samples_per_category: Number of training samples per category
        val_samples_per_category: Number of validation samples per category
        test_samples_per_category: Number of test samples per category
        max_tokens: Maximum token length for reviews (inclusive)
        selected_categories: List of specific categories to process (None = all categories)
    Returns:
        bool: True if dataset creation successful, False otherwise
    """
    
    print("=== Amazon Reviews Dataset Creator ===")
    print(f"Train samples per category: {train_samples_per_category}")
    print(f"Validation samples per category: {val_samples_per_category}")
    print(f"Test samples per category: {test_samples_per_category}")
    print(f"Max tokens: {max_tokens}")
    
    print("\nCreating directory structure...")
    os.makedirs(f'{output_path}/train', exist_ok=True)
    os.makedirs(f'{output_path}/val', exist_ok=True)
    os.makedirs(f'{output_path}/test', exist_ok=True)
    print(f"✓ Created directories: {output_path}/train, {output_path}/val, {output_path}/test")
    
    print("\nLoading BART-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    
    print("Loading DistilBERT SST-2 sentiment model...")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_model.to(device)
    sentiment_model.eval()
    
    print(f"Using device: {device}")
    
    loader = AmazonReviews2023Loader()
    
    if selected_categories is None:
        categories_to_process = loader.CATEGORIES
    else:
        invalid_categories = [cat for cat in selected_categories if cat not in loader.CATEGORIES]
        if invalid_categories:
            print(f"Error: Invalid categories specified: {invalid_categories}")
            print(f"Available categories: {loader.CATEGORIES}")
            return False
        categories_to_process = selected_categories
    
    all_train_data = []
    all_val_data = []
    all_test_data = []
    categories_processed = 0
    
    total_samples_per_category = train_samples_per_category + val_samples_per_category + test_samples_per_category
    
    print(f"\nProcessing {len(categories_to_process)} categories: {categories_to_process}")

    for category in categories_to_process:
        print(f"\n--- Processing category: {category} ---")
        
        try:

            batch_size = int(total_samples_per_category * 2)  # Load 1.2x to account for filtering and duplicates
            dataset = loader.load_reviews(
                category=category,
                streaming=True,
                num_samples=batch_size
            )
            
            category_data = []            
            target_with_buffer = int(total_samples_per_category * 1.5)
            
            review_pbar = tqdm(total=target_with_buffer, desc=f"Processing {category} reviews",
                              unit="review")
            
            for review in dataset:
                if len(category_data) >= target_with_buffer:
                    break
                    
                try:
                    # Extract all required fields first
                    raw_title = review.get('title')
                    raw_text = review.get('text')
                    rating = review.get('rating')
                    asin = review.get('asin')
                    parent_asin = review.get('parent_asin')
                    user_id = review.get('user_id')
                    timestamp = review.get('timestamp')
                    verified_purchase = review.get('verified_purchase')
                    helpful_vote = review.get('helpful_vote')
                    images = review.get('images')
                    
                    # Check for any null/None values in critical fields
                    if (raw_title is None or raw_text is None or rating is None or 
                        asin is None or parent_asin is None or user_id is None or 
                        timestamp is None or verified_purchase is None or helpful_vote is None):
                        continue
                    
                    # Convert to strings and clean text fields
                    title = clean_text(str(raw_title))
                    text = clean_text(str(raw_text))
                    
    
                    if title == "" or text == "":
                        continue
                    
                    try:
                        rating_float = float(rating)
                        if not (1.0 <= rating_float <= 5.0):
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                    try:
                        # Ensure verified_purchase is boolean
                        verified_purchase_bool = bool(verified_purchase)
                        
                        # Ensure helpful_vote is numeric (can be 0)
                        helpful_vote_int = int(helpful_vote) if helpful_vote is not None else 0
                        if helpful_vote_int < 0:  # Negative helpful votes don't make sense
                            continue
                            
                        if not timestamp or str(timestamp).strip() == "":
                            continue
                            
                        asin_str = str(asin).strip()
                        parent_asin_str = str(parent_asin).strip()
                        user_id_str = str(user_id).strip()
                        
                        if not asin_str or not parent_asin_str or not user_id_str:
                            continue
                            
                    except (ValueError, TypeError):
                        continue

                    has_images = bool(images and len(images) > 0)

                    review_data = f"[CAT={category}] [TITLE={title}] {text}"

                    tokens = tokenizer.encode(review_data, add_special_tokens=True)
                    token_count = len(tokens)
                    
                    if token_count <= max_tokens:
                        sentiment_score = compute_sentiment_score(text, sentiment_tokenizer, sentiment_model, device)
                        
                        if sentiment_score is None:
                            continue
                        
                        
                        rating_normalized = (rating_float - 3.0) / 2.0  # Convert 1-5 to -1 to 1 scale
                        rating_mismatch = abs(sentiment_score - rating_normalized)
                        
                        category_data.append({
                            'review_data': review_data,
                            'rating': rating_float,
                            'token_count': token_count,
                            'category': category,
                            'title': title,
                            'text': text,
                            'has_images': has_images,
                            'asin': asin_str,
                            'parent_asin': parent_asin_str,
                            'user_id': user_id_str,
                            'timestamp': timestamp,
                            'verified_purchase': verified_purchase_bool,
                            'helpful_vote': helpful_vote_int,
                            'sentiment': sentiment_score,
                            'rating_mismatch': rating_mismatch
                        })
                        review_pbar.update(1)
                        review_pbar.set_postfix(collected=len(category_data), target=target_with_buffer)

                except Exception as e:
                    continue
            review_pbar.close()
                        
            print(f"Removing duplicates from {category} data...")
            original_category_count = len(category_data)
            category_data = remove_duplicates(category_data)
            print(f"Removed {original_category_count - len(category_data)} duplicates from {category}")
            
            if len(category_data) >= total_samples_per_category:
                train_data = category_data[:train_samples_per_category]
                val_data = category_data[train_samples_per_category:train_samples_per_category + val_samples_per_category]
                test_data = category_data[train_samples_per_category + val_samples_per_category:train_samples_per_category + val_samples_per_category + test_samples_per_category]
                
                category_name = category.replace(' ', '_').replace('&', 'and').lower()
                
                train_df = pd.DataFrame(train_data)
                train_filename = f"{output_path}/train/{category_name}_train.csv"
                train_df.to_csv(train_filename, index=False, encoding='utf-8')
                
                val_df = pd.DataFrame(val_data)
                val_filename = f"{output_path}/val/{category_name}_val.csv"
                val_df.to_csv(val_filename, index=False, encoding='utf-8')
                
                test_df = pd.DataFrame(test_data)
                test_filename = f"{output_path}/test/{category_name}_test.csv"
                test_df.to_csv(test_filename, index=False, encoding='utf-8')
    
                all_train_data.extend(train_data)
                all_val_data.extend(val_data)
                all_test_data.extend(test_data)
            else:
                print(f"Not enough samples after deduplication for {category} (got {len(category_data)}, needed {total_samples_per_category})")
            
            categories_processed += 1
            del category_data
            if 'train_data' in locals():
                del train_data
            if 'val_data' in locals():
                del val_data
            if 'test_data' in locals():
                del test_data
            if 'train_df' in locals():
                del train_df
            if 'val_df' in locals():
                del val_df
            if 'test_df' in locals():
                del test_df
            gc.collect()

        except Exception as e:
            print(f"Error processing category {category}: {e}")
            if 'category_data' in locals():
                del category_data
            if 'review_pbar' in locals():
                review_pbar.close()
            gc.collect()
            continue
    
    print(f"\n=== Dataset Creation Summary ===")
    print(f"Categories processed: {categories_processed}/{len(categories_to_process)}")
    print(f"Total train samples: {len(all_train_data)}")
    print(f"Total validation samples: {len(all_val_data)}")
    print(f"Total test samples: {len(all_test_data)}")
    
    if len(all_train_data) == 0 and len(all_val_data) == 0 and len(all_test_data) == 0:
        print("No data collected. Check your dataset access and try again.")
        return False
    
    print(f"\nRemoving cross-category duplicates from combined datasets...")
    
    original_train_count = len(all_train_data)
    all_train_data = remove_duplicates(all_train_data)
    print(f"Train: Removed {original_train_count - len(all_train_data)} cross-category duplicate reviews")
    
    original_val_count = len(all_val_data)
    all_val_data = remove_duplicates(all_val_data)
    print(f"Validation: Removed {original_val_count - len(all_val_data)} cross-category duplicate reviews")
    
    original_test_count = len(all_test_data)
    all_test_data = remove_duplicates(all_test_data)
    print(f"Test: Removed {original_test_count - len(all_test_data)} cross-category duplicate reviews")
        
    train_df = pd.DataFrame(all_train_data)
    train_df.to_csv(f"{output_path}/train/combined_train_dataset.csv", index=False, encoding='utf-8')
    val_df = pd.DataFrame(all_val_data)
    val_df.to_csv(f"{output_path}/val/combined_val_dataset.csv", index=False, encoding='utf-8')        
    test_df = pd.DataFrame(all_test_data)
    test_df.to_csv(f"{output_path}/test/combined_test_dataset.csv", index=False, encoding='utf-8')

    if len(all_train_data) > 0:
        print(f"\n=== Train Dataset Statistics ===")
        
        print(f"\nRating distribution:")
        rating_counts = train_df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count}")
        
        print(f"\nToken length statistics:")
        print(f"  Mean: {train_df['token_count'].mean():.1f}")
        print(f"  Median: {train_df['token_count'].median():.1f}")
        print(f"  Max: {train_df['token_count'].max()}")
        print(f"  Min: {train_df['token_count'].min()}")
    
    return True


def main():
    """Main function to create the dataset"""
    
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test datasets from Amazon reviews",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-path", 
        "-o",
        type=str, 
        default="data_v2",
        help="Output directory path for saving datasets"
    )
    
    parser.add_argument(
        "--train-samples", 
        "-t",
        type=int, 
        default=10000,
        help="Number of training samples per category"
    )
    
    parser.add_argument(
        "--val-samples", 
        "-v",
        type=int, 
        default=1000,
        help="Number of validation samples per category"
    )
    
    parser.add_argument(
        "--test-samples", 
        "-s",
        type=int, 
        default=10000,
        help="Number of test samples per category"
    )
    
    parser.add_argument(
        "--max-tokens", 
        "-m",
        type=int, 
        default=512,
        help="Maximum token length for reviews (inclusive)"
    )
    
    parser.add_argument(
        "--categories", 
        "-c",
        type=str,
        nargs='+',
        default=None,
        help="Specific categories to process (space-separated). If not specified, processes all categories. Example: --categories Electronics Books"
    )
    
    args = parser.parse_args()
    
    print(f"Creating datasets with:")
    print(f"  Output path: {args.output_path}")
    print(f"  Train: {args.train_samples} samples per category")
    print(f"  Validation: {args.val_samples} samples per category")
    print(f"  Test: {args.test_samples} samples per category")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Categories: {args.categories if args.categories else 'All available categories'}")

    
    try:
        success = create_dataset(
            output_path=args.output_path,
            train_samples_per_category=args.train_samples,
            val_samples_per_category=args.val_samples,
            test_samples_per_category=args.test_samples,
            max_tokens=args.max_tokens,
            selected_categories=args.categories,
        )
        
        if success:
            print("\n Dataset creation completed successfully!")
        else:
            print("\n Dataset creation failed.")
            
    except Exception as e:
        print(f"\n Error during dataset creation: {e}")
        raise


if __name__ == "__main__":
    main()
