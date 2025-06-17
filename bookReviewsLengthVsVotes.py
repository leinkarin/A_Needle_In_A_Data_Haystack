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
        """Load Books category review data"""
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
            for review in dataset:
                text = review.get('text', '') or ''
                helpful_vote = review.get('helpful_vote', 0)
                
                # Skip reviews without text or helpful vote data
                if not text:
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
            
            if reviews_data:
                self.books_data = pd.DataFrame(reviews_data)
                
                # Clean data: remove outliers and invalid entries
                self.books_data = self.books_data[self.books_data['word_count'] > 0]
                self.books_data = self.books_data[self.books_data['word_count'] <= self.books_data['word_count'].quantile(0.99)]
                self.books_data = self.books_data[self.books_data['helpful_vote'] >= 0]
                
                print(f"✅ Successfully loaded {len(self.books_data):,} valid book reviews")
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
            return
        
        print("\n📊 ANALYZING LENGTH-HELPFULNESS RELATIONSHIP")
        print("=" * 60)
        
        # Basic correlation analysis
        word_corr = self.books_data['word_count'].corr(self.books_data['helpful_vote'])
        char_corr = self.books_data['text_length'].corr(self.books_data['helpful_vote'])
        
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
        
        # 1. Box plot by length categories
        plt.subplot(2, 2, 1)
        box_data = [group['helpful_vote'].values for name, group in 
                   self.books_data.groupby('length_category')]
        box_labels = [name for name, group in self.books_data.groupby('length_category')]
        
        plt.boxplot(box_data, labels=box_labels)
        plt.ylabel('Helpful Votes')
        plt.title('Helpfulness Distribution by Length Category')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 2. Bar chart of average helpfulness
        plt.subplot(2, 2, 2)
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
        
        # 3. Scatter plot: Length Category vs Average Helpfulness (colored by individual helpful votes)
        plt.subplot(2, 2, 3)
        
        # Sample data for better visualization performance
        scatter_sample = self.books_data.sample(min(15000, len(self.books_data)))
        
        # Create mapping from length category to numeric values for x-axis
        length_categories = scatter_sample['length_category'].cat.categories
        category_mapping = {cat: i for i, cat in enumerate(length_categories)}
        
        # Convert categorical to numeric properly
        scatter_sample = scatter_sample.copy()
        scatter_sample['length_category_numeric'] = scatter_sample['length_category'].map(category_mapping).astype(float)
        
        # Create scatter plot with jitter for better visibility
        x_jitter = scatter_sample['length_category_numeric'].values + np.random.normal(0, 0.1, len(scatter_sample))
        
        scatter = plt.scatter(x_jitter, scatter_sample['helpful_vote'],
                            c=scatter_sample['helpful_vote'], 
                            s=20, alpha=0.6, cmap='Greys',
                            vmin=0, vmax=scatter_sample['helpful_vote'].quantile(0.95))
        
        # Calculate and overlay average helpfulness for each category
        avg_helpfulness = self.books_data.groupby('length_category')['helpful_vote'].mean()
        category_positions = range(len(length_categories))
        plt.plot(category_positions, avg_helpfulness.values, 'ro-', 
                linewidth=3, markersize=8, label='Average Helpfulness', alpha=0.8)
        
        plt.xlabel('Length Category')
        plt.ylabel('Helpful Votes')
        plt.title('Length Category vs Helpfulness\n(Darker = More Helpful Votes)')
        plt.xticks(category_positions, length_categories, rotation=45)
        plt.colorbar(scatter, label='Helpful Votes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
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

    def plot_average_helpfulness_by_length(self):
        """Create a line plot showing average helpful votes by review length groups"""
        if self.books_data is None:
            print("No data to visualize")
            return
        
        # Ensure length categories are created
        if 'length_category' not in self.books_data.columns:
            self.books_data['length_category'] = pd.cut(
                self.books_data['word_count'],
                bins=[0, 25, 50, 100, 200, 500, float('inf')],
                labels=['Very Short\n(≤25)', 'Short\n(26-50)', 'Medium\n(51-100)', 
                       'Long\n(101-200)', 'Very Long\n(201-500)', 'Extremely Long\n(500+)']
            )
        
        # Calculate average helpful votes for each length category
        avg_helpfulness = self.books_data.groupby('length_category')['helpful_vote'].agg([
            'mean', 'count', 'std'
        ]).reset_index()
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Create x-axis positions
        x_positions = range(len(avg_helpfulness))
        
        # Bar chart with average helpful votes
        bars = plt.bar(x_positions, avg_helpfulness['mean'], 
                      alpha=0.7, color='steelblue', edgecolor='darkblue',
                      label='Average Helpful Votes')
        
        # Line connecting the averages
        plt.plot(x_positions, avg_helpfulness['mean'], 
                'ro-', linewidth=3, markersize=8, 
                color='red', label='Trend Line')
        
        # Add value labels on bars
        for i, (bar, mean_val, count) in enumerate(zip(bars, avg_helpfulness['mean'], avg_helpfulness['count'])):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize the plot
        plt.xlabel('Review Length Categories', fontsize=13, fontweight='bold')
        plt.ylabel('Average Helpful Votes', fontsize=13, fontweight='bold')
        plt.title('Average Helpful Votes by Review Length Groups\n(Books Category)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis labels
        plt.xticks(x_positions, avg_helpfulness['length_category'], rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        plt.legend(loc='upper left')

        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('books_avg_helpfulness_by_length.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Average Helpfulness by Length Category:")
        print("-" * 50)
        for _, row in avg_helpfulness.iterrows():
            print(f"   {row['length_category']}: {row['mean']:.3f} votes (n={row['count']:,})")
        
        return avg_helpfulness

    def plot(self):
        # Filter data to helpful votes <= 100 for better visualization
        plot_data = self.books_data[self.books_data['helpful_vote'] <= 100].copy()
        
        print(f"📊 Visualization info:")
        print(f"   Showing {len(plot_data):,} of {len(self.books_data):,} reviews ({len(plot_data)/len(self.books_data):.1%})")
        
        # Create binned helpful vote categories
        bins = [0, 5, 10, 20, 50, 100]
        labels = ['0-5', '6-10', '11-20', '21-50', '51-100']
        plot_data['helpful_vote_group'] = pd.cut(
            plot_data['helpful_vote'], bins=bins, labels=labels, right=True
        )

        sns.set(style="whitegrid", palette="pastel")

        plt.figure(figsize=(14, 8))

        sns.scatterplot(
            data=plot_data,
            x="word_count",
            y="helpful_vote",
            hue="helpful_vote_group",  
            s=50,  # Fixed size to avoid size legend
            palette="viridis",
            alpha=0.6,
            edgecolor=None
        )

        # Calculate and add average helpful votes per length category
        # Create length categories for filtered data
        plot_data['length_category'] = pd.cut(
            plot_data['word_count'],
            bins=[0, 25, 50, 100, 200, 500, float('inf')],
            labels=['Very Short\n(≤25)', 'Short\n(26-50)', 'Medium\n(51-100)', 
                   'Long\n(101-200)', 'Very Long\n(201-500)', 'Extremely Long\n(500+)']
        )
        
        # Calculate average helpful votes and median word count for each category
        category_stats = plot_data.groupby('length_category').agg({
            'helpful_vote': 'mean',
            'word_count': 'median'
        }).reset_index()
        
        # Titles and labels
        plt.title("Book Reviews: Length vs Helpfulness Analysis)", fontsize=16, fontweight='bold')
        plt.xlabel("Review Length (Word Count)", fontsize=13)
        plt.ylabel("Helpful Votes", fontsize=13)        

        # Fix legend positioning and content
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, title="Helpful Vote Ranges", 
                  bbox_to_anchor=(1.05, 1), loc='upper left')

        averages_text = "Category Averages:\n"
        for idx, row in category_stats.iterrows():
            category_name = str(row['length_category']).replace('\n', ' ')
            averages_text += f"{category_name}: {row['helpful_vote']:.2f}\n"
        
        correlation = plot_data['word_count'].corr(plot_data['helpful_vote'])
        averages_text += f"\nCorrelation: {correlation:.3f}"
        averages_text += f"\nData shown: {len(plot_data)/len(self.books_data):.1%}"
        
        plt.text(1.05, 0.4, averages_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.show()
        


def main():
    """Main execution function"""
    print("📚 Books Category: Length vs Helpfulness Analysis")
    print("=" * 60)
    print("Question: Do longer book reviews get more helpful votes?")
    
    # Initialize analyzer
    analyzer = BooksReviewLengthAnalyzer()
    
    # Get sample size from user
    try:
        sample_input = input(f"\nEnter number of book reviews to analyze: ").strip()
        num_samples = int(sample_input) if sample_input else None
    except ValueError:
        print("Invalid input. Using default of 10,000 samples")
        num_samples = 10000
    
    # Load data
    if not analyzer.load_books_data(num_samples):
        print("❌ Failed to load data. Exiting.")
        return
    
    # Perform analysis
    print("\n🔍 Analyzing relationship between review length and helpfulness...")
    analyzer.analyze_length_helpfulness_relationship()
    
    # Create visualizations
    # print(f"\n📊 Creating comprehensive visualizations...")
    # analyzer.create_comprehensive_visualizations()
    
    # Generate report
    analyzer.generate_detailed_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Generated file: books_length_helpfulness_analysis.png")

    # Create additional visualization
    print(f"\n📊 Creating average helpfulness by length groups visualization...")
    analyzer.plot_average_helpfulness_by_length()
    
    analyzer.plot()

   
if __name__ == "__main__":
    main()
