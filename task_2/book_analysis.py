# COMPLETE BOOK ANALYSIS CODE FOR EXCEL FILES
# Run this entire code in one Python file

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. SETUP AND DATA LOADING
# =============================================================================

print("📚 STARTING BOOK DATASET ANALYSIS")
print("=" * 60)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_books_from_excel(file_path, sheet_name=0):
    """Load books data from Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("✅ Excel file loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

def clean_books_data(df):
    """Clean and prepare the books data for analysis"""
    df_clean = df.copy()
    
    print("\n🧹 DATA CLEANING PROCESS")
    print("Original data shape:", df_clean.shape)
    
    # Clean column names
    df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    # Handle missing values
    df_clean = df_clean.dropna(subset=['Price', 'Rating_from_5', 'Category'])
    
    # Clean price and rating data
    df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce')
    df_clean['Rating'] = pd.to_numeric(df_clean['Rating_from_5'], errors='coerce')
    
    # Remove invalid data
    df_clean = df_clean[(df_clean['Price'] > 0) & (df_clean['Price'] < 1000)]
    df_clean = df_clean[(df_clean['Rating'] >= 1) & (df_clean['Rating'] <= 5)]
    
    # Handle problematic categories
    df_clean = df_clean[~df_clean['Category'].isin(['Default', 'Add a comment'])]
    
    print(f"✅ Data after cleaning: {df_clean.shape}")
    return df_clean

# Load and clean data
df = load_books_from_excel('books.xlsx')  # Change filename if needed
if df is not None:
    df_clean = clean_books_data(df)
else:
    print("❌ Failed to load data. Please check your file path.")
    exit()

# Display basic info
print("\n📊 CLEANED DATA OVERVIEW:")
print(f"Total books: {len(df_clean):,}")
print(f"Average price: ${df_clean['Price'].mean():.2f}")
print(f"Average rating: {df_clean['Rating'].mean():.2f}/5")
print(f"Number of categories: {df_clean['Category'].nunique()}")

# =============================================================================
# 2. HYPOTHESIS 1: BOOK PRICES VARY BY CATEGORY
# =============================================================================

print("\n" + "=" * 60)
print("HYPOTHESIS 1: Book prices vary significantly by category")
print("=" * 60)

# Get top categories for visualization
top_categories = df_clean['Category'].value_counts().head(15).index
df_top_categories = df_clean[df_clean['Category'].isin(top_categories)]

# Visualization
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df_top_categories, x='Category', y='Price')
plt.xticks(rotation=45, ha='right')
plt.title('Price Distribution by Category (Top 15)')

plt.subplot(1, 2, 2)
category_avg_price = df_top_categories.groupby('Category')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=category_avg_price.values, y=category_avg_price.index)
plt.title('Average Price by Category (Top 15)')
plt.xlabel('Average Price ($)')

plt.tight_layout()
plt.show()

# Statistical test
if len(df_top_categories['Category'].unique()) > 1:
    categories = [df_top_categories[df_top_categories['Category'] == cat]['Price'].values 
                  for cat in df_top_categories['Category'].unique()]
    f_stat, p_value = stats.f_oneway(*categories)
    
    print(f"📈 ANOVA Test Results:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    if p_value < 0.05:
        print("✅ CONCLUSION: Significant price differences between categories - Hypothesis SUPPORTED")
    else:
        print("❌ CONCLUSION: No significant price differences - Hypothesis NOT SUPPORTED")

# =============================================================================
# 3. HYPOTHESIS 2: HIGHER-RATED BOOKS ARE MORE EXPENSIVE
# =============================================================================

print("\n" + "=" * 60)
print("HYPOTHESIS 2: Higher-rated books are more expensive")
print("=" * 60)

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=df_clean, x='Rating', y='Price', alpha=0.6, s=60)
plt.title('Price vs Rating')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
sns.boxplot(data=df_clean, x='Rating', y='Price')
plt.title('Price Distribution by Rating')

plt.subplot(1, 3, 3)
rating_avg_price = df_clean.groupby('Rating')['Price'].mean()
plt.plot(rating_avg_price.index, rating_avg_price.values, marker='o', linewidth=2, markersize=8)
plt.title('Average Price by Rating')
plt.xlabel('Rating')
plt.ylabel('Average Price ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical analysis
correlation_coef, p_value = stats.pearsonr(df_clean['Rating'], df_clean['Price'])
print(f"📈 Correlation Analysis:")
print(f"Correlation coefficient: {correlation_coef:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    if correlation_coef > 0:
        print("✅ CONCLUSION: Positive correlation - Higher-rated books tend to be more expensive")
    else:
        print("❌ CONCLUSION: Negative correlation - Higher-rated books are actually cheaper")
else:
    print("❌ CONCLUSION: No significant correlation - Hypothesis NOT SUPPORTED")

# =============================================================================
# 4. HYPOTHESIS 3: RATING DISTRIBUTION VARIES BY CATEGORY
# =============================================================================

print("\n" + "=" * 60)
print("HYPOTHESIS 3: Rating distribution varies across categories")
print("=" * 60)

# Get top categories for visualization
top_8_categories = df_clean['Category'].value_counts().head(8).index
df_top_8 = df_clean[df_clean['Category'].isin(top_8_categories)]

# Visualization
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
rating_by_category = pd.crosstab(df_top_8['Category'], df_top_8['Rating'])
rating_by_category.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Rating Distribution by Category (Top 8)')
plt.xlabel('Category')
plt.ylabel('Number of Books')
plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
category_avg_rating = df_top_8.groupby('Category')['Rating'].mean().sort_values(ascending=False)
sns.barplot(x=category_avg_rating.values, y=category_avg_rating.index)
plt.title('Average Rating by Category (Top 8)')
plt.xlabel('Average Rating')

plt.subplot(2, 2, 3)
rating_pivot = df_top_8.pivot_table(index='Category', columns='Rating', values='Price', aggfunc='count', fill_value=0)
sns.heatmap(rating_pivot, annot=True, fmt='g', cmap='YlOrRd')
plt.title('Number of Books by Category and Rating')

plt.subplot(2, 2, 4)
sns.violinplot(data=df_top_8, x='Category', y='Rating')
plt.xticks(rotation=45)
plt.title('Rating Distribution by Category (Violin Plot)')

plt.tight_layout()
plt.show()

# Statistical test
if len(df_top_8['Category'].unique()) > 1:
    categories_rating = [df_top_8[df_top_8['Category'] == cat]['Rating'].values 
                        for cat in df_top_8['Category'].unique()]
    f_stat, p_value = stats.f_oneway(*categories_rating)
    
    print(f"📈 ANOVA Test for Rating Differences:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    if p_value < 0.05:
        print("✅ CONCLUSION: Significant rating differences between categories - Hypothesis SUPPORTED")
    else:
        print("❌ CONCLUSION: No significant rating differences - Hypothesis NOT SUPPORTED")

# =============================================================================
# 5. COMPREHENSIVE DASHBOARD
# =============================================================================

print("\n" + "=" * 60)
print("GENERATING COMPREHENSIVE DASHBOARD")
print("=" * 60)

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 16))

# 1. Price distribution histogram
plt.subplot(3, 3, 1)
plt.hist(df_clean['Price'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
plt.title('Overall Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 2. Rating distribution
plt.subplot(3, 3, 2)
rating_counts = df_clean['Rating'].value_counts().sort_index()
plt.bar(rating_counts.index, rating_counts.values, alpha=0.7, edgecolor='black', color='lightcoral')
plt.title('Overall Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Books')
plt.grid(True, alpha=0.3)

# 3. Top categories by book count
plt.subplot(3, 3, 3)
top_categories = df_clean['Category'].value_counts().head(8)
plt.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%', 
        startangle=90, colors=sns.color_palette("Set3"))
plt.title('Top 8 Categories by Book Count')

# 4. Price vs Rating with category colors
plt.subplot(3, 3, 4)
top_6_categories = df_clean['Category'].value_counts().head(6).index
df_top_6 = df_clean[df_clean['Category'].isin(top_6_categories)]
scatter = plt.scatter(df_top_6['Rating'], df_top_6['Price'], 
                     c=pd.factorize(df_top_6['Category'])[0], alpha=0.7, cmap='tab10', s=50)
plt.colorbar(scatter, label='Category')
plt.xlabel('Rating')
plt.ylabel('Price ($)')
plt.title('Price vs Rating (Top 6 Categories)')
plt.grid(True, alpha=0.3)

# 5. Category price comparison
plt.subplot(3, 3, 5)
category_stats = df_top_8.groupby('Category').agg({'Price': ['mean', 'std'], 'Rating': 'mean'}).round(2)
category_stats.columns = ['Avg_Price', 'Price_Std', 'Avg_Rating']
category_stats = category_stats.sort_values('Avg_Price', ascending=False)

y_pos = np.arange(len(category_stats))
plt.barh(y_pos, category_stats['Avg_Price'], xerr=category_stats['Price_Std'], 
         alpha=0.7, capsize=5, color='lightgreen')
plt.yticks(y_pos, category_stats.index)
plt.xlabel('Average Price ($)')
plt.title('Category Price Comparison\n(with standard deviation)')

# 6. Correlation heatmap
plt.subplot(3, 3, 6)
numeric_df = df_clean[['Price', 'Rating']]
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix: Price vs Rating')

# 7. Price quartiles by rating
plt.subplot(3, 3, 7)
price_by_rating = [df_clean[df_clean['Rating'] == rating]['Price'] for rating in sorted(df_clean['Rating'].unique())]
plt.boxplot(price_by_rating, labels=sorted(df_clean['Rating'].unique()), patch_artist=True)
plt.xlabel('Rating')
plt.ylabel('Price ($)')
plt.title('Price Distribution by Rating')
plt.grid(True, alpha=0.3)

# 8. Category count vs average price
plt.subplot(3, 3, 8)
category_summary = df_clean.groupby('Category').agg({'Title': 'count', 'Price': 'mean'}).reset_index()
plt.scatter(category_summary['Title'], category_summary['Price'], alpha=0.7, s=60)
plt.xlabel('Number of Books in Category')
plt.ylabel('Average Price ($)')
plt.title('Category Size vs Average Price')
plt.grid(True, alpha=0.3)

# Add text labels for outliers
for i, row in category_summary.iterrows():
    if row['Title'] > category_summary['Title'].quantile(0.8) or row['Price'] > category_summary['Price'].quantile(0.8):
        plt.annotate(row['Category'], (row['Title'], row['Price']), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8, alpha=0.8)

plt.tight_layout()
plt.show()

# =============================================================================
# 6. FINAL STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL STATISTICAL SUMMARY")
print("=" * 60)

# Basic statistics
print("\n📊 BASIC STATISTICS:")
print(f"Total books analyzed: {len(df_clean):,}")
print(f"Average price: ${df_clean['Price'].mean():.2f}")
print(f"Average rating: {df_clean['Rating'].mean():.2f}/5")
print(f"Price range: ${df_clean['Price'].min():.2f} - ${df_clean['Price'].max():.2f}")
print(f"Most common rating: {df_clean['Rating'].mode().iloc[0]}/5")

# Category analysis
print("\n📚 CATEGORY ANALYSIS:")
print(f"Number of categories: {df_clean['Category'].nunique()}")
print("Top 5 categories by book count:")
for i, (cat, count) in enumerate(df_clean['Category'].value_counts().head().items(), 1):
    print(f"  {i}. {cat}: {count} books")

# Price analysis by rating groups
print("\n💰 PRICE ANALYSIS BY RATING GROUPS:")
low_rated = df_clean[df_clean['Rating'] <= 3]
high_rated = df_clean[df_clean['Rating'] >= 4]

print(f"Low-rated books (≤3): {len(low_rated):,} books, avg price: ${low_rated['Price'].mean():.2f}")
print(f"High-rated books (≥4): {len(high_rated):,} books, avg price: ${high_rated['Price'].mean():.2f}")

# T-test for price difference between high and low rated books
if len(low_rated) > 1 and len(high_rated) > 1:
    t_stat, p_value = stats.ttest_ind(low_rated['Price'], high_rated['Price'])
    print(f"\n📈 T-test for price difference (low vs high rated):")
    print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
    if p_value < 0.05:
        price_diff = high_rated['Price'].mean() - low_rated['Price'].mean()
        print(f"✅ Significant difference! High-rated books are ${price_diff:.2f} {'more' if price_diff > 0 else 'less'} expensive")
    else:
        print("❌ No significant price difference between rating groups")

print("\n" + "=" * 60)
print("🎉 ANALYSIS COMPLETE! All hypotheses tested and visualizations generated.")
print("=" * 60)