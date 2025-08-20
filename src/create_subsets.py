import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import os
import re
import argparse
from amazon_reviews_loader import AmazonReviews2023Loader
import gc
from tqdm import tqdm
import glob

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
    Compute sentiment score for a single text (kept for compatibility)
    """
    scores = compute_sentiment_scores_batch([text], tokenizer, model, device)
    return scores[0] if scores and scores[0] is not None else None


def compute_sentiment_scores_batch(texts: List[str], tokenizer, model, device, batch_size: int = 32) -> List[float]:
    """
    Compute sentiment scores for multiple texts in batches (much faster)
    
    Args:
        texts: List of review texts to analyze
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded sentiment model
        device: Device to run inference on (CPU/GPU)
        batch_size: Number of texts to process at once
        
    Returns:
        Sentiment score between -1 (negative) and 1 (positive), or None if computation failed
    """
    if not texts:
        return []
    
    sentiment_scores = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Process each result in the batch
            for j in range(len(batch_texts)):
                negative_score = probabilities[j][0].item()
                positive_score = probabilities[j][1].item()
                sentiment_score = positive_score - negative_score
                sentiment_scores.append(sentiment_score)
            
    except Exception as e:
        print(f"Error computing sentiment batch: {e}")
        sentiment_scores = [None] * len(texts)
        
    return sentiment_scores


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


def combine_csv_files(input_path: str, output_filename: str = "combined.csv", use_parquet: bool = True, save_both_formats: bool = False) -> bool:
    """
    Combine all CSV files in a given directory into a single CSV file.
    
    Args:
        input_path: Path to directory containing CSV files to combine
        output_filename: Name of the output combined CSV file
        
    Returns:
        bool: True if combination successful, False otherwise
    """
    print(f"=== CSV File Combiner ===")
    print(f"Input directory: {input_path}")
    print(f"Output filename: {output_filename}")
    
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return False
    
    if not os.path.isdir(input_path):
        print(f"Error: Input path '{input_path}' is not a directory.")
        return False
    
    # Find files to combine (prefer parquet if available)
    if use_parquet:
        file_pattern = os.path.join(input_path, "*.parquet")
        files_to_combine = glob.glob(file_pattern)
        file_type = "parquet"
        if not files_to_combine:
            print("No parquet files found, falling back to CSV files...")
            file_pattern = os.path.join(input_path, "*.csv")
            files_to_combine = glob.glob(file_pattern)
            file_type = "csv"
    else:
        file_pattern = os.path.join(input_path, "*.csv")
        files_to_combine = glob.glob(file_pattern)
        file_type = "csv"
    
    if not files_to_combine:
        print(f"No {file_type} files found in directory '{input_path}'.")
        return False
    
    print(f"Found {len(files_to_combine)} {file_type} files:")
    for file_path in files_to_combine:
        print(f"  - {os.path.basename(file_path)}")
    
    combined_data = []
    total_rows = 0
    
    print(f"\nCombining {file_type} files...")
    for file_path in tqdm(files_to_combine, desc="Processing files"):
        try:
            # Read file (CSV or Parquet)
            if file_type == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, encoding='utf-8')
            
            # Add source file information
            df['source_file'] = os.path.basename(file_path)
            
            combined_data.append(df)
            total_rows += len(df)
            
            print(f"  ✓ {os.path.basename(file_path)}: {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error reading {os.path.basename(file_path)}: {e}")
            continue
    
    if not combined_data:
        print(f"No valid {file_type} files could be read.")
        return False
    
    # Combine all dataframes
    print(f"\nCombining {len(combined_data)} dataframes...")
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    print(f"Total rows before deduplication: {total_rows}")
    print(f"Total rows after concatenation: {len(combined_df)}")
    
    # Remove duplicates if any
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    duplicates_removed = original_count - len(combined_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Create output path
    output_path = os.path.join(input_path, output_filename)
    
    # Save combined CSV
    try:
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Combined CSV saved to: {output_path}")
        print(f"Final dataset size: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        # Show some basic statistics
        if 'category' in combined_df.columns:
            print(f"\nCategory distribution:")
            category_counts = combined_df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"  {category}: {count}")
        
        return True
        
    except Exception as e:
        print(f"Error saving combined CSV: {e}")
        return False


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
    
    categories_processed = 0
    
    total_samples_per_category = train_samples_per_category + val_samples_per_category + test_samples_per_category
    
    print(f"\nProcessing {len(categories_to_process)} categories: {categories_to_process}")

    for category in categories_to_process:
        print(f"\n--- Processing category: {category} ---")
        
        try:

            load_size = int(total_samples_per_category * 2)  # Load 2x to account for filtering and duplicates
            dataset = loader.load_reviews(
                category=category,
                streaming=True,
                num_samples=load_size
            )
            
            category_data = []            
            target_with_buffer = int(total_samples_per_category * 1.5)
            
            # Batch processing variables
            batch_size = 100  # Process 100 reviews at a time
            pending_reviews = []
            
            review_pbar = tqdm(total=target_with_buffer, desc=f"Processing {category} reviews",
                              unit="review")
            
            def process_review_batch(reviews_batch):
                """Process a batch of reviews for better performance"""
                if not reviews_batch:
                    return []
                
                valid_reviews = []
                texts_for_sentiment = []
                
                # First pass: validation and tokenization
                for review_info in reviews_batch:
                    review, review_data, token_count = review_info
                    
                    if token_count <= max_tokens:
                        valid_reviews.append((review, review_data, token_count))
                        texts_for_sentiment.append(review['cleaned_text'])
                
                if not valid_reviews:
                    return []
                
                # Batch sentiment computation
                sentiment_scores = compute_sentiment_scores_batch(
                    texts_for_sentiment, sentiment_tokenizer, sentiment_model, device
                )
                
                # Second pass: create final data
                batch_results = []
                for i, (review, review_data, token_count) in enumerate(valid_reviews):
                    sentiment_score = sentiment_scores[i] if i < len(sentiment_scores) else None
                    
                    if sentiment_score is None:
                        continue
                    
                    rating_normalized = (review['rating'] - 3.0) / 2.0
                    rating_mismatch = abs(sentiment_score - rating_normalized)
                    
                    batch_results.append({
                        'review_data': review_data,
                        'rating': review['rating'],
                        'token_count': token_count,
                        'category': category,
                        'title': review['title'],
                        'text': review['cleaned_text'],
                        'has_images': review['has_images'],
                        'asin': review['asin'],
                        'parent_asin': review['parent_asin'],
                        'user_id': review['user_id'],
                        'timestamp': review['timestamp'],
                        'verified_purchase': review['verified_purchase'],
                        'helpful_vote': review['helpful_vote'],
                        'sentiment': sentiment_score,
                        'rating_mismatch': rating_mismatch
                    })
                
                return batch_results
            
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
                    
                    # Store review for batch processing
                    pending_reviews.append((
                        {
                            'title': title,
                            'cleaned_text': text,
                            'rating': rating_float,
                            'has_images': has_images,
                            'asin': asin_str,
                            'parent_asin': parent_asin_str,
                            'user_id': user_id_str,
                            'timestamp': timestamp,
                            'verified_purchase': verified_purchase_bool,
                            'helpful_vote': helpful_vote_int
                        },
                        review_data,
                        token_count
                    ))
                    
                    # Process batch when it's full
                    if len(pending_reviews) >= batch_size:
                        batch_results = process_review_batch(pending_reviews)
                        category_data.extend(batch_results)
                        pending_reviews = []
                        
                        review_pbar.update(len(batch_results))
                        review_pbar.set_postfix(collected=len(category_data), target=target_with_buffer)

                except Exception as e:
                    continue
            
            # Process remaining reviews in the last batch
            if pending_reviews:
                batch_results = process_review_batch(pending_reviews)
                category_data.extend(batch_results)
                review_pbar.update(len(batch_results))
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
                train_csv = f"{output_path}/train/{category_name}_train.csv"
                train_parquet = f"{output_path}/train/{category_name}_train.parquet"
                train_df.to_csv(train_csv, index=False, encoding='utf-8')
                train_df.to_parquet(train_parquet, index=False)
                
                val_df = pd.DataFrame(val_data)
                val_csv = f"{output_path}/val/{category_name}_val.csv"
                val_parquet = f"{output_path}/val/{category_name}_val.parquet"
                val_df.to_csv(val_csv, index=False, encoding='utf-8')
                val_df.to_parquet(val_parquet, index=False)
                
                test_df = pd.DataFrame(test_data)
                test_csv = f"{output_path}/test/{category_name}_test.csv"
                test_parquet = f"{output_path}/test/{category_name}_test.parquet"
                test_df.to_csv(test_csv, index=False, encoding='utf-8')
    
                print(f"✓ Created {category} dataset files")
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
    
    if categories_processed == 0:
        print("No data collected. Check your dataset access and try again.")
        return False
    
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
    
    parser.add_argument(
        "--combine", 
        action="store_true",
        help="Combine all CSV files in a directory instead of creating new datasets"
    )
    
    parser.add_argument(
        "--combine-input", 
        type=str,
        default=None,
        help="Input directory path containing CSV files to combine (required when using --combine)"
    )
    
    parser.add_argument(
        "--combine-output", 
        type=str,
        default="combined.csv",
        help="Output filename for combined CSV file (default: combined.csv)"
    )
    
    args = parser.parse_args()
    
    # Handle combine mode
    if args.combine:
        if not args.combine_input:
            print("Error: --combine-input is required when using --combine mode")
            return
        
        print(f"Combine mode enabled:")
        print(f"  Input directory: {args.combine_input}")
        print(f"  Output filename: {args.combine_output}")
        
        try:
            success = combine_csv_files(
                input_path=args.combine_input,
                output_filename=args.combine_output
            )
            
            if success:
                print("\n✓ CSV combination completed successfully!")
            else:
                print("\n✗ CSV combination failed.")
                
        except Exception as e:
            print(f"\n✗ Error during CSV combination: {e}")
            raise
    
    else:
        # Normal dataset creation mode
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
                print("\n✓ Dataset creation completed successfully!")
            else:
                print("\n✗ Dataset creation failed.")
                
        except Exception as e:
            print(f"\n✗ Error during dataset creation: {e}")
            raise


if __name__ == "__main__":
    main()
