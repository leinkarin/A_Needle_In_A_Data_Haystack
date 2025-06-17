#!/usr/bin/env python3
"""
Amazon Reviews 2023 - Interesting Data Analysis

This script explores fascinating patterns in the Amazon Reviews dataset,
focusing on the relationship between review characteristics and helpfulness.

Research Questions:
1. What makes a review helpful? (length, images, verification status)
2. How has review behavior evolved from 1996 to 2023?
3. Are there category-specific patterns in review quality?
4. Do verified purchases lead to more helpful reviews?
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import load_dataset
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AmazonReviewsAnalyzer:
    """Analyzer for discovering interesting patterns in Amazon Reviews 2023"""
    
    def __init__(self):
        self.dataset_name = "McAuley-Lab/Amazon-Reviews-2023"
        self.categories = [
            "Electronics", "Books", "Movies_and_TV", "Beauty_and_Personal_Care",
            "Sports_and_Outdoors", "Home_and_Kitchen", "Video_Games"
        ]
    
    def load_sample_data(self, category: str, num_samples: int = 5000):
        """Load sample data for analysis"""
        print(f"Loading {num_samples} samples from {category}...")
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                name=f"raw_review_{category}",
                split="full",
                streaming=True,
                trust_remote_code=True
            )
            
            # Handle IterableDatasetDict
            if hasattr(dataset, 'values'):
                dataset = next(iter(dataset.values()))
            
            # Collect samples
            samples = []
            for i, sample in enumerate(dataset.take(num_samples)):
                if i >= num_samples:
                    break
                samples.append(sample)
            
            print(f"✓ Loaded {len(samples)} samples from {category}")
            return samples
            
        except Exception as e:
            print(f"Error loading {category}: {e}")
            return []
    
    def analyze_helpfulness_patterns(self, samples, category):
        """Analyze what makes reviews helpful"""
        print(f"\n=== Helpfulness Analysis for {category} ===")
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for sample in samples:
            try:
                # Extract timestamp as year
                timestamp = sample.get('timestamp', 0)
                if timestamp > 0:
                    year = datetime.fromtimestamp(timestamp / 1000).year
                else:
                    year = None
                
                # Calculate text length
                text = sample.get('text', '')
                text_length = len(text) if text else 0
                
                # Count images
                images = sample.get('images', [])
                image_count = len(images) if images else 0
                
                df_data.append({
                    'rating': sample.get('rating', 0),
                    'helpful_vote': sample.get('helpful_vote', 0),
                    'verified_purchase': sample.get('verified_purchase', False),
                    'text_length': text_length,
                    'image_count': image_count,
                    'year': year,
                    'has_images': image_count > 0,
                    'title_length': len(sample.get('title', ''))
                })
            except Exception as e:
                continue
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            print("No valid data to analyze")
            return None
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['helpful_vote', 'text_length'])
        
        print(f"Analyzing {len(df)} valid reviews...")
        
        # Analysis 1: Helpfulness vs Review Length
        self._analyze_length_helpfulness(df, category)
        
        # Analysis 2: Verified vs Unverified purchases
        self._analyze_verification_impact(df, category)
        
        # Analysis 3: Images impact on helpfulness
        self._analyze_image_impact(df, category)
        
        # Analysis 4: Rating vs Helpfulness
        self._analyze_rating_helpfulness(df, category)
        
        # Analysis 5: Temporal trends
        self._analyze_temporal_trends(df, category)
        
        return df
    
    def _analyze_length_helpfulness(self, df, category):
        """Analyze relationship between review length and helpfulness"""
        print("\n1. Review Length vs Helpfulness:")
        
        # Create length bins
        df['length_bin'] = pd.cut(df['text_length'], 
                                 bins=[0, 50, 150, 500, 1500, float('inf')],
                                 labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        
        length_stats = df.groupby('length_bin').agg({
            'helpful_vote': ['mean', 'count'],
            'rating': 'mean'
        }).round(2)
        
        print(length_stats)
        
        # Find correlation
        correlation = df['text_length'].corr(df['helpful_vote'])
        print(f"Correlation between length and helpfulness: {correlation:.3f}")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        df.boxplot(column='helpful_vote', by='length_bin', ax=plt.gca())
        plt.title(f'{category}: Helpfulness by Review Length')
        plt.xlabel('Review Length Category')
        plt.ylabel('Helpful Votes')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['text_length'], df['helpful_vote'], alpha=0.5, s=1)
        plt.xlabel('Review Text Length (characters)')
        plt.ylabel('Helpful Votes')
        plt.title(f'{category}: Length vs Helpfulness Scatter')
        
        plt.tight_layout()
        plt.savefig(f'{category.lower()}_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_verification_impact(self, df, category):
        """Analyze impact of verified purchase on helpfulness"""
        print("\n2. Verified Purchase Impact:")
        
        verification_stats = df.groupby('verified_purchase').agg({
            'helpful_vote': ['mean', 'count', 'std'],
            'rating': 'mean',
            'text_length': 'mean'
        }).round(3)
        
        print(verification_stats)
        
        # Statistical significance test
        from scipy import stats
        verified = df[df['verified_purchase'] == True]['helpful_vote']
        unverified = df[df['verified_purchase'] == False]['helpful_vote']
        
        if len(verified) > 0 and len(unverified) > 0:
            t_stat, p_value = stats.ttest_ind(verified, unverified)
            print(f"T-test p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("✓ Statistically significant difference!")
            else:
                print("No significant difference")
    
    def _analyze_image_impact(self, df, category):
        """Analyze impact of images on helpfulness"""
        print("\n3. Image Impact on Helpfulness:")
        
        image_stats = df.groupby('has_images').agg({
            'helpful_vote': ['mean', 'count'],
            'rating': 'mean',
            'text_length': 'mean'
        }).round(3)
        
        print(image_stats)
        
        # Images vs no images
        with_images = df[df['has_images'] == True]['helpful_vote'].mean()
        without_images = df[df['has_images'] == False]['helpful_vote'].mean()
        
        improvement = ((with_images - without_images) / without_images * 100) if without_images > 0 else 0
        print(f"Reviews with images are {improvement:.1f}% more helpful on average")
    
    def _analyze_rating_helpfulness(self, df, category):
        """Analyze relationship between rating and helpfulness"""
        print("\n4. Rating vs Helpfulness:")
        
        rating_stats = df.groupby('rating').agg({
            'helpful_vote': ['mean', 'count'],
            'text_length': 'mean'
        }).round(3)
        
        print(rating_stats)
        
        # Find most helpful rating category
        most_helpful_rating = rating_stats[('helpful_vote', 'mean')].idxmax()
        print(f"Most helpful rating category: {most_helpful_rating} stars")
    
    def _analyze_temporal_trends(self, df, category):
        """Analyze how review patterns changed over time"""
        print("\n5. Temporal Trends (if data available):")
        
        if 'year' in df.columns and df['year'].notna().any():
            yearly_stats = df.groupby('year').agg({
                'helpful_vote': 'mean',
                'text_length': 'mean',
                'image_count': 'mean',
                'verified_purchase': 'mean'
            }).round(3)
            
            print("Recent years trends:")
            print(yearly_stats.tail(10))
            
            # Plot trends
            if len(yearly_stats) > 5:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                yearly_stats['helpful_vote'].plot(kind='line')
                plt.title('Average Helpfulness Over Time')
                plt.ylabel('Avg Helpful Votes')
                
                plt.subplot(2, 2, 2)
                yearly_stats['text_length'].plot(kind='line')
                plt.title('Average Review Length Over Time')
                plt.ylabel('Avg Text Length')
                
                plt.subplot(2, 2, 3)
                yearly_stats['image_count'].plot(kind='line')
                plt.title('Average Images Per Review Over Time')
                plt.ylabel('Avg Image Count')
                
                plt.subplot(2, 2, 4)
                yearly_stats['verified_purchase'].plot(kind='line')
                plt.title('Verified Purchase Rate Over Time')
                plt.ylabel('Verification Rate')
                
                plt.tight_layout()
                plt.savefig(f'{category.lower()}_temporal_trends.png', dpi=300, bbox_inches='tight')
                plt.show()
        else:
            print("Limited temporal data available")
    
    def cross_category_analysis(self, category_data):
        """Compare patterns across categories"""
        print("\n=== Cross-Category Analysis ===")
        
        category_summary = {}
        
        for category, df in category_data.items():
            if df is not None and not df.empty:
                category_summary[category] = {
                    'avg_helpfulness': df['helpful_vote'].mean(),
                    'avg_length': df['text_length'].mean(),
                    'verification_rate': df['verified_purchase'].mean(),
                    'image_usage_rate': df['has_images'].mean(),
                    'avg_rating': df['rating'].mean()
                }
        
        summary_df = pd.DataFrame(category_summary).T
        print("\nCategory Comparison:")
        print(summary_df.round(3))
        
        # Find interesting patterns
        if not summary_df.empty and len(summary_df) > 0:
            most_helpful_category = summary_df['avg_helpfulness'].idxmax()
            longest_reviews_category = summary_df['avg_length'].idxmax()
            most_verified_category = summary_df['verification_rate'].idxmax()
            
            print(f"\n🏆 Most helpful reviews: {most_helpful_category}")
            print(f"📝 Longest reviews: {longest_reviews_category}")
            print(f"✅ Most verified purchases: {most_verified_category}")
        else:
            print("\nNo valid data for category comparison")
        
        return summary_df
    
    def generate_insights_report(self, category_data):
        """Generate a summary of interesting findings"""
        print("\n" + "="*60)
        print("🔍 INTERESTING FINDINGS SUMMARY")
        print("="*60)
        
        insights = []
        
        # Overall patterns
        total_reviews = sum(len(df) for df in category_data.values() if df is not None)
        print(f"📊 Analyzed {total_reviews:,} reviews across {len(category_data)} categories")
        
        # Category insights
        summary_df = self.cross_category_analysis(category_data)
        
        if not summary_df.empty and len(summary_df) > 0:
            # Find correlations
            help_length_corr = summary_df['avg_helpfulness'].corr(summary_df['avg_length'])
            help_verification_corr = summary_df['avg_helpfulness'].corr(summary_df['verification_rate'])
            
            print(f"\n📈 Cross-category correlations:")
            print(f"   Helpfulness ↔ Review Length: {help_length_corr:.3f}")
            print(f"   Helpfulness ↔ Verification Rate: {help_verification_corr:.3f}")
            
            # Specific insights
            if help_length_corr > 0.3:
                insights.append("💡 Longer reviews tend to be more helpful across categories")
            
            if help_verification_corr > 0.3:
                insights.append("💡 Categories with more verified purchases have more helpful reviews")
            
            # Category-specific insights
            top_helpful = summary_df.nlargest(3, 'avg_helpfulness')
            insights.append(f"🏆 Most helpful category: {top_helpful.index[0]} (avg {top_helpful.iloc[0]['avg_helpfulness']:.2f} helpful votes)")
        else:
            insights.append("⚠️ Limited data available for cross-category analysis")
            
        print(f"\n🎯 KEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights


def main():
    """Main analysis workflow"""
    print("🔍 Amazon Reviews 2023 - Discovering Interesting Patterns")
    print("="*60)
    
    analyzer = AmazonReviewsAnalyzer()
    
    # Categories to analyze (start with a few for quick results)
    categories_to_analyze = ["Electronics", "Books", "Beauty_and_Personal_Care"]
    
    category_data = {}
    
    # Analyze each category
    for category in categories_to_analyze:
        print(f"\n{'='*20} Analyzing {category} {'='*20}")
        
        # Load sample data
        samples = analyzer.load_sample_data(category, num_samples=2000)
        
        if samples:
            # Perform analysis
            df = analyzer.analyze_helpfulness_patterns(samples, category)
            category_data[category] = df
        else:
            category_data[category] = None
    
    # Cross-category analysis
    if category_data:
        analyzer.generate_insights_report(category_data)
    
    print(f"\n✅ Analysis complete! Check the generated PNG files for visualizations.")
    print(f"📁 Files created:")
    for category in categories_to_analyze:
        print(f"   - {category.lower()}_length_analysis.png")
        print(f"   - {category.lower()}_temporal_trends.png")


if __name__ == "__main__":
    main() 