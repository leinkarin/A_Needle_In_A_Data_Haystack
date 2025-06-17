#!/usr/bin/env python3
"""
Books Category: Length vs Helpfulness Analysis

This script specifically analyzes whether longer reviews in the Books category 
receive more helpful votes. It provides detailed statistical analysis and 
visualizations to answer this specific question.

Research Question:
Do longer book reviews get more helpful votes than shorter ones?
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
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")


class BooksReviewLengthAnalyzer:
    """Specialized analyzer for Books category length-helpfulness relationship"""
    
    def __init__(self):
        self.loader = AmazonReviews2023Loader()
        self.books_data = None
        self.analysis_results = {}
    
    def load_books_data(self, num_samples=None):
        """Load Books category review data - defaults to entire dataset"""
        if num_samples is None:
            print(f"📚 Loading ALL available Books reviews (entire dataset)...")
        else:
            print(f"📚 Loading {num_samples:,} Books reviews...")
        
        try:
            # Load Books reviews
            dataset = self.loader.load_reviews(
                category="Books",
                streaming=True,
                num_samples=num_samples
            )
            
            # Extract relevant data
            reviews_data = []
            count = 0
            
            for review in dataset:
                # If num_samples is None, we load everything, otherwise stop at limit
                if num_samples is not None and count >= num_samples:
                    break
                
                text = review.get('text', '') or ''
                helpful_vote = review.get('helpful_vote', 0)
                
                # Skip reviews without text or helpful vote data
                if not text or helpful_vote is None:
                    continue
                
                reviews_data.append({
                    'helpful_vote': helpful_vote,
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'rating': review.get('rating', 0),
                    'verified_purchase': review.get('verified_purchase', False),
                    'title_length': len(review.get('title', '') or ''),
                    'has_images': len(review.get('images', [])) > 0 if review.get('images') else False
                })
                
                count += 1
                
                # Progress indicator for large datasets
                if count % 10000 == 0:
                    print(f"   Processed {count:,} reviews...")
            
            if reviews_data:
                self.books_data = pd.DataFrame(reviews_data)
                
                # Clean data: remove outliers and invalid entries
                initial_count = len(self.books_data)
                self.books_data = self.books_data[self.books_data['word_count'] > 0]
                self.books_data = self.books_data[self.books_data['word_count'] <= self.books_data['word_count'].quantile(0.99)]
                self.books_data = self.books_data[self.books_data['helpful_vote'] >= 0]
                
                cleaned_count = len(self.books_data)
                print(f"✅ Successfully loaded {cleaned_count:,} valid book reviews")
                print(f"   (Removed {initial_count - cleaned_count:,} invalid/outlier reviews)")
                return True
            else:
                print("❌ No valid data found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading Books data: {e}")
            return False
    
    def analyze_length_helpfulness_relationship(self):
        """Analyze the relationship between review length and helpfulness"""
        if self.books_data is None:
            print("No data loaded. Please run load_books_data() first.")
            return
        
        print("\n📊 ANALYZING LENGTH-HELPFULNESS RELATIONSHIP")
        print("=" * 60)
        
        # Basic correlation analysis
        word_corr = self.books_data['word_count'].corr(self.books_data['helpful_vote'])
        char_corr = self.books_data['text_length'].corr(self.books_data['helpful_vote'])
        
        print(f"📈 Correlation Coefficients:")
        print(f"   Word Count ↔ Helpful Votes: {word_corr:.4f}")
        print(f"   Character Count ↔ Helpful Votes: {char_corr:.4f}")
        
        # Statistical significance test
        from scipy.stats import pearsonr
        corr_stat, p_value = pearsonr(self.books_data['word_count'], self.books_data['helpful_vote'])
        print(f"   Statistical significance (p-value): {p_value:.2e}")
        
        if p_value < 0.001:
            significance = "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "Very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "Significant (p < 0.05)"
        else:
            significance = "Not significant (p ≥ 0.05)"
        
        print(f"   Significance level: {significance}")
        
        # Length categories analysis
        self.books_data['length_category'] = pd.cut(
            self.books_data['word_count'],
            bins=[0, 25, 50, 100, 200, 500, float('inf')],
            labels=['Very Short\n(≤25)', 'Short\n(26-50)', 'Medium\n(51-100)', 
                   'Long\n(101-200)', 'Very Long\n(201-500)', 'Extremely Long\n(500+)']
        )
        
        length_stats = self.books_data.groupby('length_category')['helpful_vote'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(3)
        
        print(f"\n📊 Helpfulness by Review Length:")
        print(length_stats)
        
        # Calculate effect size (practical significance)
        very_short = self.books_data[self.books_data['word_count'] <= 25]['helpful_vote']
        very_long = self.books_data[self.books_data['word_count'] >= 200]['helpful_vote']
        
        if len(very_short) > 0 and len(very_long) > 0:
            effect_size = (very_long.mean() - very_short.mean()) / np.sqrt(
                ((len(very_long)-1) * very_long.var() + (len(very_short)-1) * very_short.var()) / 
                (len(very_long) + len(very_short) - 2)
            )
            print(f"\n📏 Effect Size (Cohen's d): {effect_size:.3f}")
            
            if abs(effect_size) >= 0.8:
                effect_interpretation = "Large effect"
            elif abs(effect_size) >= 0.5:
                effect_interpretation = "Medium effect"
            elif abs(effect_size) >= 0.2:
                effect_interpretation = "Small effect"
            else:
                effect_interpretation = "Negligible effect"
            
            print(f"   Interpretation: {effect_interpretation}")
            
            improvement = ((very_long.mean() - very_short.mean()) / very_short.mean() * 100)
            print(f"   Practical impact: Very long reviews get {improvement:.1f}% more helpful votes on average")
        
        self.analysis_results = {
            'correlation': word_corr,
            'p_value': p_value,
            'significance': significance,
            'length_stats': length_stats,
            'effect_size': effect_size if 'effect_size' in locals() else None
        }
    
    def create_comprehensive_visualizations(self):
        """Create detailed visualizations of the length-helpfulness relationship"""
        if self.books_data is None:
            print("No data to visualize")
            return
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Main scatter plot with trend line
        plt.subplot(3, 3, 1)
        # Sample for better visualization if dataset is very large
        plot_sample = self.books_data.sample(min(20000, len(self.books_data)))
        plt.scatter(plot_sample['word_count'], plot_sample['helpful_vote'], 
                   alpha=0.4, s=8, color='darkblue')
        
        # Add trend line
        z = np.polyfit(self.books_data['word_count'], self.books_data['helpful_vote'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.books_data['word_count'].min(), 
                             self.books_data['word_count'].max(), 100)
        plt.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=3)
        
        plt.xlabel('Word Count')
        plt.ylabel('Helpful Votes')
        plt.title(f'Books: Word Count vs Helpful Votes\nCorrelation: {self.analysis_results["correlation"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 2. Scatter plot with color-coded helpful votes (darker = more helpful)
        plt.subplot(3, 3, 2)
        # Sample data for better visualization performance
        scatter_sample = self.books_data.sample(min(15000, len(self.books_data)))
        
        # Create scatter plot with color mapping (darker = more helpful votes)
        scatter = plt.scatter(scatter_sample['word_count'], scatter_sample['helpful_vote'],
                            c=scatter_sample['helpful_vote'], 
                            s=20, alpha=0.6, cmap='Greys', 
                            vmin=0, vmax=scatter_sample['helpful_vote'].quantile(0.95))
        
        plt.xlabel('Word Count')
        plt.ylabel('Helpful Votes')
        plt.title('Word Count vs Helpful Votes\n(Darker = More Helpful Votes)')
        plt.colorbar(scatter, label='Helpful Votes')
        plt.grid(True, alpha=0.3)
        
        # 3. Average helpfulness by length category
        plt.subplot(3, 3, 3)
        length_means = self.books_data.groupby('length_category')['helpful_vote'].mean()
        bars = plt.bar(range(len(length_means)), length_means.values, 
                      color='steelblue', alpha=0.7)
        plt.xlabel('Length Category')
        plt.ylabel('Average Helpful Votes')
        plt.title('Average Helpfulness by Length')
        plt.xticks(range(len(length_means)), length_means.index, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. Violin plot for distribution shape
        plt.subplot(3, 3, 4)
        # Sample data for better visualization
        sample_data = self.books_data.sample(min(10000, len(self.books_data)))
        
        length_categories = sample_data['length_category'].cat.categories
        violin_data = [sample_data[sample_data['length_category'] == cat]['helpful_vote'].values 
                      for cat in length_categories]
        
        parts = plt.violinplot(violin_data, positions=range(len(length_categories)))
        plt.xlabel('Length Category')
        plt.ylabel('Helpful Votes')
        plt.title('Helpfulness Distribution Shape')
        plt.xticks(range(len(length_categories)), length_categories, rotation=45)
        
        # 5. Cumulative helpfulness by percentiles
        plt.subplot(3, 3, 5)
        # Create percentile bins
        percentiles = np.arange(0, 101, 10)
        word_count_percentiles = [self.books_data['word_count'].quantile(p/100) for p in percentiles]
        helpfulness_by_percentile = []
        
        for i in range(len(percentiles)-1):
            mask = ((self.books_data['word_count'] >= word_count_percentiles[i]) & 
                   (self.books_data['word_count'] < word_count_percentiles[i+1]))
            avg_help = self.books_data[mask]['helpful_vote'].mean()
            helpfulness_by_percentile.append(avg_help)
        
        plt.plot(percentiles[:-1], helpfulness_by_percentile, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Word Count Percentile')
        plt.ylabel('Average Helpful Votes')
        plt.title('Helpfulness by Length Percentile')
        plt.grid(True, alpha=0.3)
        
        # 6. Length distribution histogram
        plt.subplot(3, 3, 6)
        plt.hist(self.books_data['word_count'], bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Number of Reviews')
        plt.title('Distribution of Review Lengths')
        plt.axvline(self.books_data['word_count'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.books_data["word_count"].mean():.0f}')
        plt.axvline(self.books_data['word_count'].median(), color='orange', linestyle='--', 
                   label=f'Median: {self.books_data["word_count"].median():.0f}')
        plt.legend()
        
        # 7. Helpfulness distribution
        plt.subplot(3, 3, 7)
        helpful_votes = self.books_data['helpful_vote']
        plt.hist(helpful_votes[helpful_votes <= helpful_votes.quantile(0.95)], 
                bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Helpful Votes')
        plt.ylabel('Number of Reviews')
        plt.title('Distribution of Helpful Votes (95th percentile)')
        
        # 8. Rating vs Length vs Helpfulness (3D relationship)
        plt.subplot(3, 3, 8)
        # Create scatter plot colored by rating - sample for performance
        scatter_sample = self.books_data.sample(min(15000, len(self.books_data)))
        scatter = plt.scatter(scatter_sample['word_count'], scatter_sample['helpful_vote'],
                            c=scatter_sample['rating'], alpha=0.5, s=10, cmap='RdYlBu_r')
        plt.xlabel('Word Count')
        plt.ylabel('Helpful Votes')
        plt.title('Length vs Helpfulness (colored by Rating)')
        plt.colorbar(scatter, label='Rating')
        
        # 9. Statistical summary table
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Create summary statistics text
        summary_text = f"""
        📚 BOOKS REVIEW ANALYSIS SUMMARY
        
        Sample Size: {len(self.books_data):,} reviews
        
        Correlation Analysis:
        • Word Count ↔ Helpful Votes: {self.analysis_results['correlation']:.4f}
        • Statistical Significance: {self.analysis_results['significance']}
        
        Length Statistics:
        • Mean Length: {self.books_data['word_count'].mean():.0f} words
        • Median Length: {self.books_data['word_count'].median():.0f} words
        • Standard Deviation: {self.books_data['word_count'].std():.0f} words
        
        Helpfulness Statistics:
        • Mean Helpful Votes: {self.books_data['helpful_vote'].mean():.2f}
        • Median Helpful Votes: {self.books_data['helpful_vote'].median():.2f}
        • % Reviews with >0 votes: {(self.books_data['helpful_vote'] > 0).mean():.1%}
        
        Key Finding:
        {"✅ Longer reviews DO get more helpful votes" if self.analysis_results['correlation'] > 0.1 else 
         "❌ Longer reviews do NOT get significantly more helpful votes" if self.analysis_results['correlation'] < 0.05 else
         "⚠️ Weak relationship between length and helpfulness"}
        """
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('books_length_helpfulness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """Generate comprehensive findings report"""
        if not self.analysis_results:
            print("No analysis results available")
            return
        
        print("\n" + "="*80)
        print("📚 BOOKS CATEGORY: LENGTH vs HELPFULNESS ANALYSIS REPORT")
        print("="*80)
        
        correlation = self.analysis_results['correlation']
        
        print(f"\n🎯 MAIN QUESTION: Do longer book reviews get more helpful votes?")
        print("-" * 60)
        
        if correlation > 0.2:
            answer = "✅ YES - There is a STRONG positive relationship"
            interpretation = "Longer book reviews consistently receive more helpful votes"
        elif correlation > 0.1:
            answer = "✅ YES - There is a MODERATE positive relationship"
            interpretation = "Longer book reviews tend to receive more helpful votes"
        elif correlation > 0.05:
            answer = "⚠️ WEAK - There is a small positive relationship"
            interpretation = "Longer book reviews receive slightly more helpful votes"
        elif correlation > -0.05:
            answer = "❌ NO - There is no meaningful relationship"
            interpretation = "Review length does not significantly affect helpfulness"
        else:
            answer = "❌ NO - Longer reviews actually get fewer helpful votes"
            interpretation = "Shorter book reviews tend to be more helpful"
        
        print(f"\n{answer}")
        print(f"Correlation coefficient: {correlation:.4f}")
        print(f"Interpretation: {interpretation}")
        print(f"Statistical significance: {self.analysis_results['significance']}")
        
        # Practical implications
        print(f"\n💡 PRACTICAL IMPLICATIONS:")
        print("-" * 40)
        
        if correlation > 0.1:
            print("📝 For book reviewers:")
            print("   • Writing longer, more detailed reviews is likely to get more helpful votes")
            print("   • Aim for comprehensive coverage of plot, characters, writing style")
            print("   • Include specific examples and detailed opinions")
            
            print("\n📖 For book readers:")
            print("   • Longer reviews tend to be more helpful for making purchase decisions")
            print("   • Look for detailed reviews when evaluating books")
            print("   • Value comprehensive reviews over brief opinions")
        else:
            print("📝 For book reviewers:")
            print("   • Focus on quality over quantity - length doesn't guarantee helpfulness")
            print("   • Concise, well-structured reviews can be just as valuable")
            print("   • Emphasize key insights rather than extensive detail")
            
            print("\n📖 For book readers:")
            print("   • Both short and long reviews can be equally helpful")
            print("   • Focus on review content quality rather than length")
            print("   • Look for specific, actionable insights regardless of review length")
        
        # Category-specific insights
        print(f"\n📚 BOOKS CATEGORY INSIGHTS:")
        print("-" * 40)
        
        avg_length = self.books_data['word_count'].mean()
        avg_helpfulness = self.books_data['helpful_vote'].mean()
        
        print(f"• Average book review length: {avg_length:.0f} words")
        print(f"• Average helpful votes per review: {avg_helpfulness:.2f}")
        print(f"• Percentage of reviews with helpful votes: {(self.books_data['helpful_vote'] > 0).mean():.1%}")
        
        # Most helpful length range
        length_stats = self.analysis_results['length_stats']
        most_helpful_category = length_stats['mean'].idxmax()
        print(f"• Most helpful length category: {most_helpful_category}")
        print(f"  (Average {length_stats.loc[most_helpful_category, 'mean']:.2f} helpful votes)")
        
        print(f"\n🎯 RECOMMENDATION:")
        print("-" * 40)
        
        if correlation > 0.15:
            print("For maximum helpfulness in book reviews, aim for detailed, comprehensive reviews")
            print(f"covering multiple aspects of the book. The data shows a clear benefit to longer reviews.")
        elif correlation > 0.05:
            print("While longer reviews tend to be slightly more helpful, focus on providing")
            print("valuable insights regardless of length. Quality matters more than quantity.")
        else:
            print("Review length has minimal impact on helpfulness for books. Focus on providing")
            print("clear, actionable insights that help other readers make informed decisions.")


def main():
    """Main execution function"""
    print("📚 Books Category: Length vs Helpfulness Analysis")
    print("=" * 60)
    print("Research Question: Do longer book reviews get more helpful votes?")
    
    # Initialize analyzer
    analyzer = BooksReviewLengthAnalyzer()
    
    # Get sample size from user (default is now the entire dataset)
    print(f"\nAnalysis Options:")
    print(f"1. Analyze ENTIRE dataset (recommended for complete analysis)")
    print(f"2. Analyze specific number of samples (faster for testing)")
    
    try:
        choice = input(f"Enter choice (1-2, default 1): ").strip()
        
        if choice == "2":
            sample_input = input(f"Enter number of book reviews to analyze: ").strip()
            num_samples = int(sample_input) if sample_input else None
            if num_samples is not None:
                num_samples = min(num_samples, 100000)  # Cap for performance
        else:
            num_samples = None  # Use entire dataset
            
    except ValueError:
        print("Invalid input. Using entire dataset.")
        num_samples = None
    
    # Load data
    if not analyzer.load_books_data(num_samples):
        print("❌ Failed to load data. Exiting.")
        return
    
    # Perform analysis
    print("\n🔍 Analyzing relationship between review length and helpfulness...")
    analyzer.analyze_length_helpfulness_relationship()
    
    # Create visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    analyzer.create_comprehensive_visualizations()
    
    # Generate report
    analyzer.generate_detailed_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Generated file: books_length_helpfulness_analysis.png")


if __name__ == "__main__":
    main() 