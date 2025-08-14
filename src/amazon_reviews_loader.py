import json
from typing import Optional, List, Dict, Any
from datasets import load_dataset, Dataset
import pandas as pd


class AmazonReviews2023Loader:
    """Loader class for Amazon Reviews 2023 dataset"""
    
    # Available categories in the dataset
    CATEGORIES = [
        "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
        "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
        "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
        "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
        "Health_and_Household", "Health_and_Personal_Care", "Home_and_Garden",
        "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
        "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
        "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Prime_Pantry",
        "Software", "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
        "Toys_and_Games", "Video_Games"
    ]
    
    def __init__(self):
        self.dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
    
    def load_reviews(self, category: Optional[str] = None, 
                    split: str = "full", 
                    streaming: bool = True,
                    num_samples: Optional[int] = None) -> Dataset:
        """
        Load Amazon reviews dataset
        
        Args:
            category: Specific category to load (e.g., "Electronics"). 
                     If None, loads all categories
            split: Dataset split to load. Options:
                   - "full": Complete dataset (default)
                   - "train": Training split (if available)
                   - "test": Test split (if available)
                   - "validation": Validation split (if available)
            streaming: Whether to use streaming mode (recommended for large datasets)
            num_samples: Number of samples to load (for testing/exploration)
            
        Returns:
            Dataset object containing the reviews
        """
        print(f"Loading Amazon Reviews 2023 dataset (split: {split})...")
        
        if category and category not in self.CATEGORIES:
            print(f"Warning: Category '{category}' not in known categories.")
            print(f"Available categories: {', '.join(self.CATEGORIES[:5])}... (and {len(self.CATEGORIES)-5} more)")
        
        try:
            # Load the dataset
            if category:
                # For reviews, use the 'raw_review_' prefix
                config_name = f"raw_review_{category}"
                dataset = load_dataset(
                    self.dataset_name, 
                    name=config_name,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True
                )
            else:
                # Load all categories - this might not be supported directly
                print("Loading all categories is not directly supported.")
                print("Please specify a category from the available options.")
                raise ValueError("Please specify a category")
            
            # Handle IterableDatasetDict vs IterableDataset
            if hasattr(dataset, 'values'):  # IterableDatasetDict
                # Get the first (and likely only) split
                dataset = next(iter(dataset.values()))
            
            # If num_samples specified, take only that many
            if num_samples and streaming:
                dataset = dataset.take(num_samples)
            elif num_samples and not streaming:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            print(f"✓ Successfully loaded dataset (split: {split})")
            if not streaming:
                print(f"  Dataset size: {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: This dataset may require authentication or special permissions.")
            raise
    
    def load_metadata(self, category: Optional[str] = None,
                     streaming: bool = True,
                     num_samples: Optional[int] = None) -> Dataset:
        """
        Load item metadata
        
        Args:
            category: Specific category metadata to load
            streaming: Whether to use streaming mode
            num_samples: Number of samples to load
            
        Returns:
            Dataset object containing item metadata
        """
        print(f"Loading item metadata...")
        
        try:
            if category:
                # For metadata, use the 'raw_meta_' prefix
                config_name = f"raw_meta_{category}"
                dataset = load_dataset(
                    self.dataset_name,
                    name=config_name,
                    streaming=streaming,
                    trust_remote_code=True
                )
            else:
                # Load all metadata - this might not be supported directly
                print("Loading all metadata is not directly supported.")
                print("Please specify a category from the available options.")
                raise ValueError("Please specify a category")
            
            # Handle IterableDatasetDict vs IterableDataset
            if hasattr(dataset, 'values'):  # IterableDatasetDict
                # Get the first (and likely only) split
                dataset = next(iter(dataset.values()))
            
            if num_samples and streaming:
                dataset = dataset.take(num_samples)
            elif num_samples and not streaming:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
                
            print(f"✓ Successfully loaded metadata")
            return dataset
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise
    
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            "name": "Amazon Reviews 2023",
            "total_reviews": "571.54M",
            "total_users": "54.51M", 
            "total_items": "48.19M",
            "categories": len(self.CATEGORIES),
            "timespan": "May 1996 - Sep 2023",
            "note": "Dataset uses 'raw_review_' prefix for reviews and 'raw_meta_' prefix for metadata",
            "features": {
                "reviews": ["rating", "title", "text", "images", "asin", "parent_asin", 
                           "user_id", "timestamp", "verified_purchase", "helpful_vote"],
                "metadata": ["main_category", "title", "average_rating", "rating_number",
                            "features", "description", "price", "images", "videos", "store",
                            "categories", "details", "parent_asin", "bought_together"]
            }
        }
    
    def get_available_configs(self) -> Dict[str, List[str]]:
        """Get available dataset configurations"""
        return {
            "raw_reviews": [f"raw_review_{cat}" for cat in self.CATEGORIES],
            "raw_metadata": [f"raw_meta_{cat}" for cat in self.CATEGORIES],
            "rating_only_0core": [f"0core_rating_only_{cat}" for cat in self.CATEGORIES],
            "rating_only_5core": [f"5core_rating_only_{cat}" for cat in self.CATEGORIES],
            "processed_splits": [
                "0core_last_out", "5core_last_out", "0core_timestamp", 
                "5core_timestamp", "0core_last_out_w_his", "5core_last_out_w_his",
                "0core_timestamp_w_his", "5core_timestamp_w_his"
            ]
        }
    
    def save_sample_data(self, dataset: Dataset, output_file: str, 
                        num_samples: int = 1000, format: str = "json") -> None:
        """
        Save sample data to file for local analysis
        
        Args:
            dataset: Dataset to sample from
            output_file: Output file path
            num_samples: Number of samples to save
            format: Output format ("json" or "csv")
        """
        print(f"Saving {num_samples} samples to {output_file}...")
        
        try:
            if hasattr(dataset, 'take'):  # Streaming dataset
                samples = list(dataset.take(num_samples))
            else:  # Regular dataset
                samples = dataset[:num_samples]
            
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
            elif format.lower() == "csv":
                df = pd.DataFrame(samples)
                df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"✓ Saved {len(samples)} samples to {output_file}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            raise


def main():
    """Example usage of the Amazon Reviews 2023 dataset loader"""
    
    # Initialize the loader
    loader = AmazonReviews2023Loader()
    
    # Print dataset information
    info = loader.get_dataset_info()
    print("=== Amazon Reviews 2023 Dataset ===")
    print(f"Total Reviews: {info['total_reviews']}")
    print(f"Total Users: {info['total_users']}")
    print(f"Total Items: {info['total_items']}")
    print(f"Categories: {info['categories']}")
    print(f"Timespan: {info['timespan']}")
    print(f"Note: {info['note']}")
    print(f"Available categories: {', '.join(loader.CATEGORIES[:10])}... (and more)")

    
    # Example 1: Load a small sample from Electronics category
    print(f"\n=== Loading Electronics Reviews Sample ===")
    try:
        electronics_reviews = loader.load_reviews(
            category="Electronics", 
            streaming=True, 
            num_samples=100
        )
        
        # Save sample data
        loader.save_sample_data(
            electronics_reviews, 
            "electronics_sample.json", 
            num_samples=50
        )
        
    except Exception as e:
        print(f"Could not load Electronics data: {e}")


if __name__ == "__main__":
    main()
