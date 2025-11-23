import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

print("🎯 TASK 3: DATA STORYTELLING - CLEAN VISUALIZATION VERSION")
print("=" * 70)

def load_and_clean_data(file_path='books.xlsx'):
    """Load and clean data for storytelling"""
    print("📊 Loading and preparing data...")
    
    df = pd.read_excel(file_path)
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    df_clean = df_clean.dropna(subset=['Price', 'Rating_from_5', 'Category'])
    df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce')
    df_clean['Rating'] = pd.to_numeric(df_clean['Rating_from_5'], errors='coerce')
    df_clean = df_clean[(df_clean['Price'] > 0) & (df_clean['Price'] < 1000)]
    df_clean = df_clean[(df_clean['Rating'] >= 1) & (df_clean['Rating'] <= 5)]
    df_clean = df_clean[~df_clean['Category'].isin(['Default', 'Add a comment'])]
    
    print(f"✅ Data ready: {len(df_clean)} books, {df_clean['Category'].nunique()} categories")
    return df_clean

df = load_and_clean_data()

print("\n📊 Creating Clean Visualizations...")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['Price'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Book Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Number of Books')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
rating_counts = df['Rating'].value_counts().sort_index()
plt.bar(rating_counts.index, rating_counts.values, color='lightcoral', alpha=0.7)
plt.title('Customer Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Books')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
top_cats = df['Category'].value_counts().head(6)
plt.barh(range(len(top_cats)), top_cats.values, color='lightgreen', alpha=0.7)
plt.yticks(range(len(top_cats)), top_cats.index)
plt.title('Top 6 Categories by Book Count')
plt.xlabel('Number of Books')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(df['Rating'], df['Price'], alpha=0.6, s=50, color='blue')
plt.xlabel('Rating')
plt.ylabel('Price ($)')
plt.title('Price vs Rating Relationship')
plt.grid(True, alpha=0.3)

correlation, p_value = stats.pearsonr(df['Rating'], df['Price'])
plt.text(1, df['Price'].max() * 0.9, f'Correlation: {correlation:.3f}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.subplot(1, 2, 2)
avg_price_by_rating = df.groupby('Rating')['Price'].mean()
plt.plot(avg_price_by_rating.index, avg_price_by_rating.values, 
         marker='o', linewidth=2, markersize=8, color='green')
plt.xlabel('Rating')
plt.ylabel('Average Price ($)')
plt.title('Average Price by Rating Level')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
top_8_categories = df['Category'].value_counts().head(8).index
df_top_cats = df[df['Category'].isin(top_8_categories)]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
avg_price_by_cat = df_top_cats.groupby('Category')['Price'].mean().sort_values(ascending=True)
plt.barh(range(len(avg_price_by_cat)), avg_price_by_cat.values, color='orange', alpha=0.7)
plt.yticks(range(len(avg_price_by_cat)), avg_price_by_cat.index)
plt.xlabel('Average Price ($)')
plt.title('Average Price by Category')
plt.subplot(1, 3, 2)
avg_rating_by_cat = df_top_cats.groupby('Category')['Rating'].mean().sort_values(ascending=True)
plt.barh(range(len(avg_rating_by_cat)), avg_rating_by_cat.values, color='purple', alpha=0.7)
plt.yticks(range(len(avg_rating_by_cat)), avg_rating_by_cat.index)
plt.xlabel('Average Rating')
plt.title('Average Rating by Category')
plt.subplot(1, 3, 3)
cat_counts = df_top_cats['Category'].value_counts().sort_values(ascending=True)
plt.barh(range(len(cat_counts)), cat_counts.values, color='teal', alpha=0.7)
plt.yticks(range(len(cat_counts)), cat_counts.index)
plt.xlabel('Number of Books')
plt.title('Book Count by Category')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))

hidden_gems = df[(df['Rating'] >= 4.5) & (df['Price'] < df['Price'].median())]
premium_excellence = df[(df['Rating'] >= 4.5) & (df['Price'] > df['Price'].quantile(0.75))]
budget_options = df[(df['Price'] < df['Price'].quantile(0.25))]

plt.subplot(1, 3, 1)
segments = ['Hidden Gems', 'Premium Excellence', 'Budget Options']
counts = [len(hidden_gems), len(premium_excellence), len(budget_options)]
colors = ['gold', 'red', 'lightblue']

bars = plt.bar(segments, counts, color=colors, alpha=0.7)
plt.title('Strategic Book Segments')
plt.ylabel('Number of Books')
plt.xticks(rotation=45)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{count}', ha='center', va='bottom')

plt.subplot(1, 3, 2)
price_segments = ['Budget', 'Mid-Range', 'Premium']
price_ranges = [
    df[df['Price'] < df['Price'].quantile(0.33)]['Price'].mean(),
    df[(df['Price'] >= df['Price'].quantile(0.33)) & 
       (df['Price'] <= df['Price'].quantile(0.66))]['Price'].mean(),
    df[df['Price'] > df['Price'].quantile(0.66)]['Price'].mean()
]

plt.bar(price_segments, price_ranges, color=['lightgreen', 'orange', 'red'], alpha=0.7)
plt.title('Average Price by Segment')
plt.ylabel('Average Price ($)')

plt.subplot(1, 3, 3)
high_rated_pct = (len(df[df['Rating'] >= 4]) / len(df)) * 100
five_star_pct = (len(df[df['Rating'] == 5]) / len(df)) * 100

performance_metrics = ['4+ Stars', '5 Stars']
performance_values = [high_rated_pct, five_star_pct]

plt.bar(performance_metrics, performance_values, color=['lightblue', 'gold'], alpha=0.7)
plt.title('Quality Performance')
plt.ylabel('Percentage of Books (%)')

for i, v in enumerate(performance_values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
print("\n" + "=" * 70)
print("DATA STORY: Strategic Insights for Bookstore Success")
print("=" * 70)

def tell_clean_data_story(df):
        total_books = len(df)
    avg_price = df['Price'].mean()
    avg_rating = df['Rating'].mean()
    
    hidden_gems = df[(df['Rating'] >= 4.5) & (df['Price'] < df['Price'].median())]
    premium_books = df[(df['Rating'] >= 4.5) & (df['Price'] > df['Price'].quantile(0.75))]
    
    category_stats = df.groupby('Category').agg({
        'Price': ['count', 'mean'],
        'Rating': 'mean'
    }).round(2)
    category_stats.columns = ['Book_Count', 'Avg_Price', 'Avg_Rating']
    
    print(f"""
 EXECUTIVE SUMMARY

Our analysis of {total_books:,} books reveals a marketplace full of strategic opportunities:

 MARKET OVERVIEW:
• Total Books Analyzed: {total_books:,}
• Average Price: ${avg_price:.2f}
• Average Rating: {avg_rating:.2f}/5
• Categories: {df['Category'].nunique()}

 KEY DISCOVERIES:
1. We found {len(hidden_gems)} "Hidden Gems" - high-quality books at affordable prices
2. Identified {len(premium_books)} premium books with exceptional ratings
3. Price and rating show minimal correlation (r={stats.pearsonr(df['Rating'], df['Price'])[0]:.3f})
""")

    top_categories = category_stats.nlargest(3, 'Book_Count')
    expensive_categories = category_stats.nlargest(3, 'Avg_Price')
    high_rated_categories = category_stats.nlargest(3, 'Avg_Rating')
    
    print(f"""
 CATEGORY INTELLIGENCE:

MOST POPULAR CATEGORIES (by volume):
""")
    for i, (cat, data) in enumerate(top_categories.iterrows(), 1):
        print(f"   {i}. {cat}: {data['Book_Count']} books")

    print(f"""
 PREMIUM CATEGORIES (highest prices):
""")
    for i, (cat, data) in enumerate(expensive_categories.iterrows(), 1):
        print(f"   {i}. {cat}: ${data['Avg_Price']} average")

    print(f"""
 HIGHEST RATED CATEGORIES:
""")
    for i, (cat, data) in enumerate(high_rated_categories.iterrows(), 1):
        print(f"   {i}. {cat}: {data['Avg_Rating']}/5 rating")

tell_clean_data_story(df)

print("\n" + "=" * 70)
print(" STRATEGIC RECOMMENDATIONS")
print("=" * 70)

print("""
 IMMEDIATE ACTIONS:

1.  COLLECTION STRATEGY
   • Feature {len(hidden_gems)} "Hidden Gems" in promotional campaigns
   • Create a "Premium Excellence" section for high-end books
   • Balance inventory across price segments

2.  PRICING STRATEGY  
   • Implement three-tier pricing: Budget, Mid-Range, Premium
   • Use premium books for margin optimization
   • Use budget gems for customer acquisition

3.  MARKETING STRATEGY
   • Target three customer segments with tailored messaging
   • Highlight value propositions for each price tier
   • Create "Staff Picks" from hidden gems

4.  PERFORMANCE TRACKING
   • Monitor category performance monthly
   • Track price-to-rating ratios
   • Identify emerging high-performers

 KEY INSIGHT: Quality isn't tied to price - focus on curating excellent 
   content across all price points to maximize customer satisfaction and revenue.
""")

print("\n" + "=" * 70)
print(" EXECUTIVE DASHBOARD SUMMARY")
print("=" * 70)

plt.figure(figsize=(14, 10))

plt.suptitle('Bookstore Intelligence: Strategic Dashboard', fontsize=16, fontweight='bold', y=0.98)

plt.subplot(2, 3, 1)
metrics = ['Total Books', 'Avg Price', 'Avg Rating', 'Categories']
values = [len(df), df['Price'].mean(), df['Rating'].mean(), df['Category'].nunique()]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

bars = plt.bar(metrics, values, color=colors, alpha=0.8)
plt.title('Key Performance Indicators', fontweight='bold')
plt.xticks(rotation=45)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.05, 
             f'{value:.0f}' if value == int(value) else f'{value:.1f}', 
             ha='center', va='bottom', fontweight='bold')

plt.subplot(2, 3, 2)
rating_pct = [(df['Rating'] == i).sum() / len(df) * 100 for i in range(1, 6)]
colors = ['#ff6b6b', '#ffa36b', '#ffd56b', '#a3de83', '#2e8b57']
plt.pie(rating_pct, labels=[f'{i} Star' for i in range(1, 6)], autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Customer Rating Distribution', fontweight='bold')

plt.subplot(2, 3, 3)
opportunities = ['Hidden Gems', 'Premium Books', 'Total High-Rated']
counts = [
    len(df[(df['Rating'] >= 4.5) & (df['Price'] < df['Price'].median())]),
    len(df[(df['Rating'] >= 4.5) & (df['Price'] > df['Price'].quantile(0.75))]),
    len(df[df['Rating'] >= 4.5])
]
plt.bar(opportunities, counts, color=['gold', 'red', 'lightblue'], alpha=0.8)
plt.title('Strategic Opportunities', fontweight='bold')
plt.ylabel('Number of Books')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
plt.scatter(df['Rating'], df['Price'], alpha=0.5, s=30, color='blue')
plt.xlabel('Rating')
plt.ylabel('Price ($)')
plt.title('Price vs Rating Relationship', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
top_cats = df['Category'].value_counts().head(6)
plt.barh(range(len(top_cats)), top_cats.values, color='lightgreen', alpha=0.8)
plt.yticks(range(len(top_cats)), top_cats.index)
plt.xlabel('Number of Books')
plt.title('Top Categories by Volume', fontweight='bold')

plt.subplot(2, 3, 6)
plt.axis('off')
summary_text = f"""
STRATEGIC SUMMARY:

 Total Books: {len(df):,}
 Avg Price: ${df['Price'].mean():.2f}
 Avg Rating: {df['Rating'].mean():.2f}/5
 Categories: {df['Category'].nunique()}

KEY OPPORTUNITIES:
• {len(hidden_gems)} Hidden Gems
• Premium positioning available
• Balanced quality across prices

RECOMMENDATION:
Focus on curation excellence
across all price segments.
"""
plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
         fontsize=10, va='top', linespacing=1.5,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("STORYTELLING TASK COMPLETED SUCCESSFULLY!")
print("=" * 70)
