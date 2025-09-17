"""
Lightweight script to add rating_vs_product_avg_abs and reviewer_review_count columns.

This script uses a two-step approach:
1. Build a small lookup dictionary for only the products in your CSV
2. Quickly add the columns using the lookup
"""

import pandas as pd
import argparse
import os
from amazon_reviews_loader import AmazonReviews2023Loader


def build_product_lookup(csv_path: str, category: str) -> dict:
    """
    Build a lightweight lookup dictionary for products in the CSV.
    
    Args:
        csv_path: Path to CSV file
        category: Category for metadata loading
        
    Returns:
        Dictionary mapping asin -> average_rating
    """
    df = pd.read_csv(csv_path)
    
    unique_asins = set()
    if 'asin' in df.columns:
        unique_asins.update(df['asin'].dropna().unique())
    if 'parent_asin' in df.columns:
        unique_asins.update(df['parent_asin'].dropna().unique())
    
    if len(unique_asins) == 0:
        print("No ASINs found in CSV")
        return {}
    
    loader = AmazonReviews2023Loader()
    
    try:
        metadata_dataset = loader.load_metadata(
            category=category,
            streaming=True
        )
        
        product_lookup = {}
        processed_count = 0
        found_count = 0
        
        for item in metadata_dataset:
            processed_count += 1
            
            product_asin = item.get('parent_asin')
            if product_asin and product_asin in unique_asins:
                avg_rating = item.get('average_rating')
                if avg_rating:
                    product_lookup[product_asin] = float(avg_rating)
                    found_count += 1
            
            if processed_count % 10000 == 0:
                print(f"  Processed {processed_count:,} metadata items, found {found_count} matches")
            
            if found_count == len(unique_asins):
                print(f"✓ Found all {found_count} products, stopping early")
                break
        
        
        if len(product_lookup) == 0:
        else:
            coverage = len(product_lookup) / len(unique_asins) * 100
            print(f"  Coverage: {coverage:.1f}% of CSV products found in metadata")
        
        return product_lookup
        
    except Exception as e:
        print(f"Error building lookup: {e}")
        return {}


def add_columns_fast(csv_path: str, product_lookup: dict) -> bool:
    """
    Quickly add columns using the pre-built lookup.
    
    Args:
        csv_path: Path to CSV file
        product_lookup: Dictionary mapping asin -> average_rating
        
    Returns:
        True if successful
    """
    print(f"\nStep 2: Adding columns to {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} reviews")
        
        if 'asin' not in df.columns or 'rating' not in df.columns:
            print("CSV must contain 'asin' and 'rating' columns")
            return False
        
        print("Adding rating_vs_product_avg_abs...")
        
        def get_product_avg(asin):
            return product_lookup.get(asin, None)
        
        df['product_avg_rating'] = df['asin'].map(get_product_avg)
        
        if 'parent_asin' in df.columns:
            missing_mask = df['product_avg_rating'].isna()
            if missing_mask.sum() > 0:
                df.loc[missing_mask, 'product_avg_rating'] = df.loc[missing_mask, 'parent_asin'].map(get_product_avg)
        
        valid_mask = ~df['product_avg_rating'].isna()
        df['rating_vs_product_avg_abs'] = pd.NA
        df.loc[valid_mask, 'rating_vs_product_avg_abs'] = (
            df.loc[valid_mask, 'rating'] - df.loc[valid_mask, 'product_avg_rating']
        ).abs()
        
        matched_count = valid_mask.sum()
        print(f"✓ Matched {matched_count}/{len(df)} reviews ({matched_count/len(df)*100:.1f}%)")
        
        print("Adding reviewer_review_count...")
        if 'user_id' in df.columns:
            reviewer_counts = df.groupby('user_id').size()
            df['reviewer_review_count'] = df['user_id'].map(reviewer_counts)
        else:
            df['reviewer_review_count'] = pd.NA
        
        df = df.drop(columns=['product_avg_rating'])
        
        print("rows before cleaning", len(df))
        df=df.dropna(subset=['rating_vs_product_avg_abs', 'reviewer_review_count'])
        print("rows after cleaning", len(df))
        df.to_csv(csv_path, index=False)
        
        rating_abs = df['rating_vs_product_avg_abs'].dropna()
        reviewer_counts = df['reviewer_review_count'].dropna()
        
        
        if len(rating_abs) > 0:
            print(f"  Rating deviation stats:")
            print(f"    Mean abs deviation: {rating_abs.mean():.4f}")
            print(f"    Max abs deviation: {rating_abs.max():.4f}")
            print(f"    Min abs deviation: {rating_abs.min():.4f}")
        
        if len(reviewer_counts) > 0:
            print(f"  Reviewer count stats:")
            print(f"    Mean reviews per user: {reviewer_counts.mean():.2f}")
            print(f"    Min reviews per user: {reviewer_counts.min()}")
            print(f"    Max reviews per user: {reviewer_counts.max()}")
            print(f"    Users with 1 review: {(reviewer_counts == 1).sum()}")
            print(f"    Users with >10 reviews: {(reviewer_counts > 10).sum()}")
        
        print(f"  Missing values: {df['rating_vs_product_avg_abs'].isna().sum()}")
        
        return True
        
    except Exception as e:
        print(f"Error adding columns: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Lightweight script to add columns using targeted metadata lookup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to the CSV file to process'
    )
    
    parser.add_argument(
        'category',
        type=str,
        help='Category name (e.g., Electronics, Books, Clothing_Shoes_and_Jewelry)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"File not found: {args.csv_path}")
        return
    
    print(f"Category: {args.category}")
    print("="*60)
    
    product_lookup = build_product_lookup(args.csv_path, args.category)
    
    if len(product_lookup) == 0:
        print("\nCould not build product lookup - no metadata found")
        return
    
    if add_columns_fast(args.csv_path, product_lookup):
    else:
        print(f"Failed to process {args.csv_path}")


if __name__ == "__main__":
    main()
