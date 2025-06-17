#!/usr/bin/env python3
"""
Category Helpfulness Analysis

This script analyzes how the relationship between review length and helpfulness 
varies across different Amazon product categories. Some categories may favor 
longer, detailed reviews while others may prefer concise feedback.

Research Questions:
1. Which categories show stronger correlation between length and helpfulness?
2. Are there categories where shorter reviews are actually more helpful?
3. How does the optimal review length vary by product type?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from amazon_reviews_loader import AmazonReviews2023Loader

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")


class CategoryHelpfulnessAnalyzer:
    """Analyzer for category-specific helpfulness patterns"""
    
    def __init__(self):
        self.loader = AmazonReviews2023Loader()
        self.category_data = {}
        self.correlation_results = {}
    
    def load_category_data(self, categories, num_samples=5000):
        """Load review data for multiple categories"""
        print(f"Loading data for {len(categories)} categories...")
        
        for category in categories:
            print(f"📊 Loading {category}...")
            try:
                # Load reviews with helpfulness data
                dataset = self.loader.load_reviews(
                    category=category,
                    streaming=True,
                    num_samples=num_samples
                )
                
                # Extract relevant data
                reviews_data = []
                sample_count = 0
                
                for review in dataset:
                    if sample_count >= num_samples:
                        break
                    
                    # Extract fields we need
                    helpful_vote = review.get('helpful_vote', 0)
                    text = review.get('text', '') or ''
                    rating = review.get('rating', 0)
                    verified = review.get('verified_purchase', False)
                    
                    # Skip reviews without helpful votes or text
                    if helpful_vote is None or not text:
                        continue
                    
                    reviews_data.append({
                        'helpful_vote': helpful_vote,
                        'text_length': len(text),
                        'word_count': len(text.split()),
                        'rating': rating,
                        'verified_purchase': verified,
                        'category': category
                    })
                    sample_count += 1
                
                if reviews_data:
                    df = pd.DataFrame(reviews_data)
                    # Filter out outliers (reviews with 0 helpful votes and extremely long reviews)
                    df = df[df['word_count'] > 0]  # Remove empty reviews
                    df = df[df['word_count'] <= df['word_count'].quantile(0.99)]  # Remove top 1% longest
                    
                    self.category_data[category] = df
                    print(f"✅ Loaded {len(df)} valid reviews for {category}")
                else:
                    print(f"❌ No valid data found for {category}")
                    
            except Exception as e:
                print(f"❌ Error loading {category}: {e}")
                continue
    
    def calculate_correlations(self):
        """Calculate length-helpfulness correlations for each category"""
        print("\n📈 Calculating correlations...")
        
        for category, df in self.category_data.items():
            if df.empty:
                continue
            
            # Calculate correlations
            length_help_corr = df['word_count'].corr(df['helpful_vote'])
            char_help_corr = df['text_length'].corr(df['helpful_vote'])
            
            # Calculate average helpfulness by length bins
            df['length_bin'] = pd.cut(df['word_count'], 
                                    bins=[0, 25, 50, 100, 200, float('inf')],
                                    labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
            
            helpfulness_by_length = df.groupby('length_bin')['helpful_vote'].agg([
                'mean', 'count', 'std'
            ]).fillna(0)
            
            # Store results
            self.correlation_results[category] = {
                'word_length_correlation': length_help_corr,
                'char_length_correlation': char_help_corr,
                'sample_size': len(df),
                'avg_helpfulness': df['helpful_vote'].mean(),
                'avg_word_count': df['word_count'].mean(),
                'helpfulness_by_length': helpfulness_by_length,
                'data': df
            }
            
            print(f"{category:25} | Correlation: {length_help_corr:6.3f} | Samples: {len(df):5d}")
    
    def create_correlation_comparison_plot(self):
        """Create visualization comparing correlations across categories"""
        if not self.correlation_results:
            print("No correlation data available")
            return
        
        # Prepare data for plotting
        categories = list(self.correlation_results.keys())
        correlations = [self.correlation_results[cat]['word_length_correlation'] 
                       for cat in categories]
        sample_sizes = [self.correlation_results[cat]['sample_size'] 
                       for cat in categories]
        avg_helpfulness = [self.correlation_results[cat]['avg_helpfulness'] 
                          for cat in categories]
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Correlation by category (bar plot)
        ax1 = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in correlations]
        bars = ax1.barh(categories, correlations, color=colors, alpha=0.7)
        ax1.set_xlabel('Correlation (Word Count vs Helpfulness)')
        ax1.set_title('Length-Helpfulness Correlation by Category')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax1.text(corr + (0.01 if corr >= 0 else -0.01), i, 
                    f'{corr:.3f}', va='center', 
                    ha='left' if corr >= 0 else 'right', fontsize=9)
        
        # 2. Correlation vs Sample Size (scatter plot)
        ax2 = axes[0, 1]
        scatter = ax2.scatter(sample_sizes, correlations, 
                            c=avg_helpfulness, s=100, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Correlation')
        ax2.set_title('Correlation vs Sample Size')
        plt.colorbar(scatter, ax=ax2, label='Avg Helpfulness')
        
        # Add category labels to points
        for i, cat in enumerate(categories):
            ax2.annotate(cat.replace('_', '\n'), 
                        (sample_sizes[i], correlations[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # 3. Distribution of correlations
        ax3 = axes[1, 0]
        ax3.hist(correlations, bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Correlation Value')
        ax3.set_ylabel('Number of Categories')
        ax3.set_title('Distribution of Length-Helpfulness Correlations')
        ax3.axvline(x=np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.3f}')
        ax3.legend()
        
        # 4. Category comparison matrix
        ax4 = axes[1, 1]
        comparison_data = pd.DataFrame({
            'Category': categories,
            'Correlation': correlations,
            'Avg_Helpfulness': avg_helpfulness,
            'Sample_Size': sample_sizes
        }).set_index('Category')
        
        # Normalize data for heatmap
        normalized_data = comparison_data.copy()
        for col in normalized_data.columns:
            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                  (normalized_data[col].max() - normalized_data[col].min())
        
        sns.heatmap(normalized_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax4, cbar_kws={'label': 'Normalized Value'})
        ax4.set_title('Category Comparison Matrix (Normalized)')
        
        plt.tight_layout()
        plt.savefig('category_helpfulness_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_detailed_category_analysis(self):
        """Create detailed analysis for top/bottom categories"""
        if not self.correlation_results:
            return
        
        # Sort categories by correlation
        sorted_categories = sorted(self.correlation_results.items(), 
                                 key=lambda x: x[1]['word_length_correlation'], 
                                 reverse=True)
        
        # Get top 3 and bottom 3 categories
        top_categories = sorted_categories[:3]
        bottom_categories = sorted_categories[-3:]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot top categories
        for i, (category, data) in enumerate(top_categories):
            ax = axes[0, i]
            df = data['data']
            
            # Scatter plot with trend line
            ax.scatter(df['word_count'], df['helpful_vote'], alpha=0.5, s=10)
            
            # Add trend line
            z = np.polyfit(df['word_count'], df['helpful_vote'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['word_count'].min(), df['word_count'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Helpful Votes')
            ax.set_title(f'{category}\nCorr: {data["word_length_correlation"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Plot bottom categories
        for i, (category, data) in enumerate(bottom_categories):
            ax = axes[1, i]
            df = data['data']
            
            # Scatter plot with trend line
            ax.scatter(df['word_count'], df['helpful_vote'], alpha=0.5, s=10)
            
            # Add trend line
            z = np.polyfit(df['word_count'], df['helpful_vote'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['word_count'].min(), df['word_count'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Helpful Votes')
            ax.set_title(f'{category}\nCorr: {data["word_length_correlation"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Detailed Analysis: Top 3 vs Bottom 3 Categories\n(Length-Helpfulness Correlation)', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('detailed_category_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_helpfulness_patterns_plot(self):
        """Create visualization showing helpfulness patterns by length bins"""
        if not self.correlation_results:
            return
        
        # Prepare data for plotting
        length_bins = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Heatmap of helpfulness by length and category
        heatmap_data = []
        categories_with_data = []
        
        for category, data in self.correlation_results.items():
            helpfulness_by_length = data['helpfulness_by_length']
            if not helpfulness_by_length.empty:
                row = []
                for bin_name in length_bins:
                    if bin_name in helpfulness_by_length.index:
                        row.append(helpfulness_by_length.loc[bin_name, 'mean'])
                    else:
                        row.append(0)
                heatmap_data.append(row)
                categories_with_data.append(category)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, 
                                    index=categories_with_data, 
                                    columns=length_bins)
            
            sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                       ax=ax1, cbar_kws={'label': 'Average Helpful Votes'})
            ax1.set_title('Average Helpfulness by Review Length\n(Across Categories)')
            ax1.set_xlabel('Review Length Category')
            ax1.set_ylabel('Product Category')
        
        # 2. Line plot showing trends
        for category, data in list(self.correlation_results.items())[:7]:  # Limit to 7 for readability
            helpfulness_by_length = data['helpfulness_by_length']
            if not helpfulness_by_length.empty:
                y_values = []
                x_positions = []
                for i, bin_name in enumerate(length_bins):
                    if bin_name in helpfulness_by_length.index:
                        y_values.append(helpfulness_by_length.loc[bin_name, 'mean'])
                        x_positions.append(i)
                
                if len(y_values) > 1:
                    ax2.plot(x_positions, y_values, marker='o', 
                            label=category, linewidth=2, markersize=4)
        
        ax2.set_xlabel('Review Length Category')
        ax2.set_ylabel('Average Helpful Votes')
        ax2.set_title('Helpfulness Trends by Review Length')
        ax2.set_xticks(range(len(length_bins)))
        ax2.set_xticklabels(length_bins, rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('helpfulness_patterns_by_length.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        if not self.correlation_results:
            print("No data available for insights")
            return
        
        print("\n" + "="*70)
        print("🔍 CATEGORY HELPFULNESS ANALYSIS REPORT")
        print("="*70)
        
        # Sort by correlation
        sorted_results = sorted(self.correlation_results.items(), 
                              key=lambda x: x[1]['word_length_correlation'], 
                              reverse=True)
        
        print(f"\n📊 SUMMARY STATISTICS")
        print("-" * 40)
        correlations = [data['word_length_correlation'] for _, data in sorted_results]
        print(f"Total categories analyzed: {len(sorted_results)}")
        print(f"Average correlation: {np.mean(correlations):.3f}")
        print(f"Standard deviation: {np.std(correlations):.3f}")
        print(f"Range: {min(correlations):.3f} to {max(correlations):.3f}")
        
        # Categories that favor longer reviews
        positive_corr = [item for item in sorted_results if item[1]['word_length_correlation'] > 0.1]
        print(f"\n🔍 CATEGORIES FAVORING LONGER REVIEWS (correlation > 0.1)")
        print("-" * 60)
        for category, data in positive_corr:
            print(f"{category:30} | Corr: {data['word_length_correlation']:6.3f} | "
                  f"Avg Help: {data['avg_helpfulness']:5.2f}")
        
        # Categories that favor shorter reviews or show no preference
        neutral_negative = [item for item in sorted_results if item[1]['word_length_correlation'] <= 0.1]
        print(f"\n📝 CATEGORIES WITH WEAK/NEGATIVE LENGTH PREFERENCE (correlation ≤ 0.1)")
        print("-" * 70)
        for category, data in neutral_negative:
            print(f"{category:30} | Corr: {data['word_length_correlation']:6.3f} | "
                  f"Avg Help: {data['avg_helpfulness']:5.2f}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 40)
        
        if positive_corr:
            strongest = positive_corr[0]
            print(f"1. {strongest[0]} shows strongest preference for longer reviews")
            print(f"   (correlation: {strongest[1]['word_length_correlation']:.3f})")
        
        if neutral_negative:
            weakest = neutral_negative[-1]
            print(f"2. {weakest[0]} shows least preference for longer reviews")
            print(f"   (correlation: {weakest[1]['word_length_correlation']:.3f})")
        
        # Category-specific recommendations
        print(f"\n🎯 RECOMMENDATIONS BY CATEGORY TYPE")
        print("-" * 50)
        
        for category, data in sorted_results[:3]:
            corr = data['word_length_correlation']
            if corr > 0.2:
                print(f"• {category}: Detailed reviews are highly valued")
            elif corr > 0.1:
                print(f"• {category}: Longer reviews tend to be more helpful")
            else:
                print(f"• {category}: Concise reviews may be as effective as long ones")


def main():
    """Main execution function"""
    print("🔍 Category-Specific Helpfulness Analysis")
    print("=" * 50)
    
    # Categories to analyze (mix of different product types)
    categories_to_analyze = [
        "Electronics", "Books", "Beauty_and_Personal_Care", 
        "Home_and_Kitchen", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry",
        "Health_and_Household", "Toys_and_Games"
    ]
    
    print(f"Analyzing {len(categories_to_analyze)} categories:")
    for cat in categories_to_analyze:
        print(f"  • {cat}")
    
    # Initialize analyzer
    analyzer = CategoryHelpfulnessAnalyzer()
    
    # Load data
    analyzer.load_category_data(categories_to_analyze, num_samples=3000)
    
    if not analyzer.category_data:
        print("❌ No data loaded successfully")
        return
    
    # Calculate correlations
    analyzer.calculate_correlations()
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    analyzer.create_correlation_comparison_plot()
    analyzer.create_detailed_category_analysis()
    analyzer.create_helpfulness_patterns_plot()
    
    # Generate insights
    analyzer.generate_insights_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Generated files:")
    print(f"  • category_helpfulness_correlations.png")
    print(f"  • detailed_category_analysis.png") 
    print(f"  • helpfulness_patterns_by_length.png")


if __name__ == "__main__":
    main() 