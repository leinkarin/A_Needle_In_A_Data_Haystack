import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict, Any
import sys
import os
import re
from amazon_reviews_loader import AmazonReviews2023Loader


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


def remove_duplicates(data_list: List[Dict]) -> List[Dict]:
    """
    Remove duplicate reviews based on text content.
    
    Args:
        data_list: List of review dictionaries
        
    Returns:
        List with duplicates removed
    """
    seen_texts = set()
    unique_data = []
    
    for item in data_list:
        text_for_comparison = re.sub(r'\s+', ' ', item['text'].lower().strip())
        
        if text_for_comparison not in seen_texts:
            seen_texts.add(text_for_comparison)
            unique_data.append(item)
    
    return unique_data


def create_dataset(
    train_samples_per_category: int = 10000,
    val_samples_per_category: int = 10000,
    test_samples_per_category: int = 1000,
    max_tokens: int = 1024,
):
    """
    Create train, validation, and test datasets with specified parameters.
    
    Args:
        train_samples_per_category: Number of training samples per category
        val_samples_per_category: Number of validation samples per category
        test_samples_per_category: Number of test samples per category
        max_tokens: Maximum token length for reviews (inclusive)
        save_individual_categories: Whether to save individual CSV files for each category
    """
    
    print("=== Amazon Reviews Dataset Creator ===")
    print(f"Train samples per category: {train_samples_per_category}")
    print(f"Validation samples per category: {val_samples_per_category}")
    print(f"Test samples per category: {test_samples_per_category}")
    print(f"Max tokens: {max_tokens}")
    
    print("\nCreating directory structure...")
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/val', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    print("✓ Created directories: data/train, data/val, data/test")
    
    print("\nLoading BART-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    
    loader = AmazonReviews2023Loader()
    
    all_train_data = []
    all_val_data = []
    all_test_data = []
    categories_processed = 0
    
    total_samples_per_category = train_samples_per_category + val_samples_per_category + test_samples_per_category
    
    print(f"\nProcessing {len(loader.CATEGORIES)} categories...")
    
    for category in loader.CATEGORIES:
        print(f"\n--- Processing category: {category} ---")
        
        try:
    
            batch_size = total_samples_per_category * 2  # Load 2x to account for filtering and duplicates
            dataset = loader.load_reviews(
                category=category,
                streaming=True,
                num_samples=batch_size
            )
            
            category_data = []            
            target_with_buffer = int(total_samples_per_category * 1.5)
            
            for review in dataset:
                if len(category_data) >= target_with_buffer:
                    break
                    
                try:
                    raw_title = str(review.get('title', ''))
                    raw_text = str(review.get('text', ''))
                    rating = review.get('rating')
                    
                    title = clean_text(raw_title)
                    text = clean_text(raw_text)

                    if title == "" or text == "" or rating is None or not (1.0 <= float(rating) <= 5.0):
                        continue

                    review_data = f"[CAT={category}] [TITLE={title}] {text}"

                    tokens = tokenizer.encode(review_data, add_special_tokens=True)
                    token_count = len(tokens)
                    
                    if token_count <= max_tokens:
                        category_data.append({
                            'review_data': review_data,
                            'rating': float(rating),
                            'token_count': token_count,
                            'category': category,
                            'title': title,
                            'text': text,
                            'images': review.get('images'),
                            'asin': review.get('asin'),
                            'parent_asin': review.get('parent_asin'),
                            'user_id': review.get('user_id'),
                            'timestamp': review.get('timestamp'),
                            'verified_purchase': review.get('verified_purchase'),
                            'helpful_vote': review.get('helpful_vote')
                        })

                except Exception as e:
                    continue
                        
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
                train_filename = f"data/train/{category_name}_train.csv"
                train_df.to_csv(train_filename, index=False, encoding='utf-8')
                
                val_df = pd.DataFrame(val_data)
                val_filename = f"data/val/{category_name}_val.csv"
                val_df.to_csv(val_filename, index=False, encoding='utf-8')
                
                test_df = pd.DataFrame(test_data)
                test_filename = f"data/test/{category_name}_test.csv"
                test_df.to_csv(test_filename, index=False, encoding='utf-8')
    
                all_train_data.extend(train_data)
                all_val_data.extend(val_data)
                all_test_data.extend(test_data)
            else:
                print(f"Not enough samples after deduplication for {category} (got {len(category_data)}, needed {total_samples_per_category})")
            
            categories_processed += 1
            
        except Exception as e:
            print(f"Error processing category {category}: {e}")
            continue
    
    print(f"\n=== Dataset Creation Summary ===")
    print(f"Categories processed: {categories_processed}/{len(loader.CATEGORIES)}")
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
    train_df.to_csv("data/train/combined_train_dataset.csv", index=False, encoding='utf-8')
    val_df = pd.DataFrame(all_val_data)
    val_df.to_csv("data/val/combined_val_dataset.csv", index=False, encoding='utf-8')        
    test_df = pd.DataFrame(all_test_data)
    test_df.to_csv("data/test/combined_test_dataset.csv", index=False, encoding='utf-8')

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

    train_samples_per_category = 10000
    val_samples_per_category = 10000
    test_samples_per_category = 1000
    max_tokens = 1024
    
    if len(sys.argv) > 1:
        try:
            train_samples_per_category = int(sys.argv[1])
        except ValueError:
            print("Error: train_samples_per_category must be an integer")
            return
    
    if len(sys.argv) > 2:
        try:
            val_samples_per_category = int(sys.argv[2])
        except ValueError:
            print("Error: val_samples_per_category must be an integer")
            return
    
    if len(sys.argv) > 3:
        try:
            test_samples_per_category = int(sys.argv[3])
        except ValueError:
            print("Error: test_samples_per_category must be an integer")
            return
    
    print(f"Creating datasets with:")
    print(f"  Train: {train_samples_per_category} samples per category")
    print(f"  Validation: {val_samples_per_category} samples per category")
    print(f"  Test: {test_samples_per_category} samples per category")
    
    try:
        success = create_dataset(
            train_samples_per_category=train_samples_per_category,
            val_samples_per_category=val_samples_per_category,
            test_samples_per_category=test_samples_per_category,
            max_tokens=max_tokens,
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
