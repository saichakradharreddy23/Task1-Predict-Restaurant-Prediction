"""
eda.py
======
Phase 2: Exploratory Data Analysis (EDA) for Restaurant Rating Prediction.
Generates 12 comprehensive, publication-quality visualizations using Matplotlib & Seaborn
and saves them to 'outputs/eda/'.
"""

import os
import sys
import tempfile

# Prevent matplotlib directory warning on Windows
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure UTF-8 printing
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data_loader import load_data, COUNTRY_MAP

# Set global aesthetics
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'eda')

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_eda_visualizations(df: pd.DataFrame = None):
    """Generates and saves all 12 required EDA figures."""
    if df is None:
        df = load_data()
        
    ensure_output_dir()
    df_clean = df.copy()
    
    # Map country names for human-readable charts
    df_clean['Country Name'] = df_clean['Country Code'].map(COUNTRY_MAP).fillna('Other')
    
    # Extract primary cuisine
    df_clean['Primary Cuisine'] = df_clean['Cuisines'].fillna('Unknown').apply(
        lambda x: [c.strip() for c in str(x).split(',')][0]
    )

    print(f"Generating 12 EDA visualizations to '{OUTPUT_DIR}'...")

    # -------------------------------------------------------------
    # 1. Distribution of Aggregate Restaurant Ratings
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df_clean['Aggregate rating'], bins=30, kde=True, color='#2b5c8f', ax=ax, edgecolor='white')
    ax.axvline(df_clean['Aggregate rating'].mean(), color='#e63946', linestyle='--', linewidth=2, label=f"Mean: {df_clean['Aggregate rating'].mean():.2f}")
    ax.axvline(df_clean['Aggregate rating'].median(), color='#2a9d8f', linestyle='-', linewidth=2, label=f"Median: {df_clean['Aggregate rating'].median():.2f}")
    ax.set_title("Distribution of Aggregate Restaurant Ratings", pad=12, fontweight='bold')
    ax.set_xlabel("Aggregate Rating (0.0 to 5.0)")
    ax.set_ylabel("Number of Restaurants")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_rating_distribution.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Histogram of Numerical Features
    # -------------------------------------------------------------
    num_cols = ['Average Cost for two', 'Price range', 'Votes', 'Aggregate rating']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = ['#457b9d', '#e76f51', '#2a9d8f', '#e63946']
    
    for i, col in enumerate(num_cols):
        if col == 'Average Cost for two':
            # Use log scale or 95th percentile clipping for readable visualization
            clip_val = df_clean[col].quantile(0.95)
            sns.histplot(df_clean[df_clean[col] <= clip_val][col], bins=30, kde=True, ax=axes[i], color=colors[i], edgecolor='white')
            axes[i].set_title(f"Histogram: {col} (<= 95th percentile: {clip_val:,.0f})", fontweight='bold')
        elif col == 'Votes':
            clip_val = df_clean[col].quantile(0.95)
            sns.histplot(df_clean[df_clean[col] <= clip_val][col], bins=30, kde=True, ax=axes[i], color=colors[i], edgecolor='white')
            axes[i].set_title(f"Histogram: {col} (<= 95th percentile: {clip_val:,.0f})", fontweight='bold')
        else:
            sns.histplot(df_clean[col], bins=25 if col == 'Aggregate rating' else 4, kde=(col == 'Aggregate rating'), ax=axes[i], color=colors[i], edgecolor='white')
            axes[i].set_title(f"Histogram: {col}", fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_numerical_histograms.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Correlation Heatmap
    # -------------------------------------------------------------
    corr_cols = ['Country Code', 'Average Cost for two', 'Price range', 'Votes', 'Aggregate rating']
    corr_matrix = df_clean[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', vmin=-0.2, vmax=1.0, square=True,
                linewidths=1.5, cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
    ax.set_title("Correlation Heatmap: Numerical Features & Rating", pad=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Average Restaurant Rating by City (Top 15 Cities by Count)
    # -------------------------------------------------------------
    top_cities = df_clean['City'].value_counts().head(15).index
    city_avg = df_clean[df_clean['City'].isin(top_cities)].groupby('City')['Aggregate rating'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=city_avg.values, y=city_avg.index, hue=city_avg.index, palette='crest', legend=False, ax=ax)
    ax.set_title("Average Restaurant Rating Across Top 15 Cities (by volume)", pad=12, fontweight='bold')
    ax.set_xlabel("Mean Aggregate Rating")
    ax.set_ylabel("City")
    for i, v in enumerate(city_avg.values):
        ax.text(v + 0.05, i, f"{v:.2f}", va='center', fontsize=9, fontweight='semibold')
    ax.set_xlim(0, 5.0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_avg_rating_by_city.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Average Restaurant Rating by Country
    # -------------------------------------------------------------
    country_stats = df_clean.groupby('Country Name').agg(
        avg_rating=('Aggregate rating', 'mean'),
        count=('Aggregate rating', 'count')
    ).sort_values(by='avg_rating')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=country_stats['avg_rating'].values, y=country_stats.index, hue=country_stats.index, palette='flare', legend=False, ax=ax)
    ax.set_title("Average Restaurant Rating by Country", pad=12, fontweight='bold')
    ax.set_xlabel("Mean Aggregate Rating")
    ax.set_ylabel("Country")
    for i, (cntry, row) in enumerate(country_stats.iterrows()):
        ax.text(row['avg_rating'] + 0.05, i, f"{row['avg_rating']:.2f} (n={row['count']})", va='center', fontsize=8)
    ax.set_xlim(0, 5.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_avg_rating_by_country.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Rating Distribution by Price Range
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x='Price range', y='Aggregate rating', data=df_clean, hue='Price range', palette='Set2', legend=False, ax=ax, showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    ax.set_title("Rating Distribution by Price Range (1: Budget to 4: Luxury)", pad=12, fontweight='bold')
    ax.set_xlabel("Price Range Tier")
    ax.set_ylabel("Aggregate Rating")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '06_rating_by_price_range.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 7. Votes versus Aggregate Rating
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Votes', y='Aggregate rating', data=df_clean, alpha=0.4, color='#1d3557', ax=ax)
    sns.regplot(x='Votes', y='Aggregate rating', data=df_clean, scatter=False, color='#e63946', ax=ax)
    ax.set_title("Restaurant Votes vs. Aggregate Rating", pad=12, fontweight='bold')
    ax.set_xlabel("Total Votes Received")
    ax.set_ylabel("Aggregate Rating")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '07_votes_vs_rating.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 8. Average Cost for Two versus Aggregate Rating
    # -------------------------------------------------------------
    # Filter extreme currency outliers (e.g. Indonesian Rupiah with millions) for clean visualization
    cost_95th = df_clean['Average Cost for two'].quantile(0.95)
    filtered_cost = df_clean[df_clean['Average Cost for two'] <= cost_95th]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Average Cost for two', y='Aggregate rating', data=filtered_cost, alpha=0.3, color='#2a9d8f', ax=ax)
    sns.regplot(x='Average Cost for two', y='Aggregate rating', data=filtered_cost, scatter=False, color='#e76f51', ax=ax)
    ax.set_title(f"Average Cost for Two vs. Aggregate Rating (<= 95th percentile: {cost_95th:,.0f})", pad=12, fontweight='bold')
    ax.set_xlabel("Average Cost for Two (Local Currency)")
    ax.set_ylabel("Aggregate Rating")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '08_cost_vs_rating.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 9. Online Delivery Availability versus Ratings
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=df_clean, hue='Has Online delivery', palette=['#e76f51', '#2a9d8f'], legend=False, ax=ax, showmeans=True)
    ax.set_title("Impact of Online Delivery Availability on Aggregate Rating", pad=12, fontweight='bold')
    ax.set_xlabel("Has Online Delivery?")
    ax.set_ylabel("Aggregate Rating")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '09_online_delivery_vs_rating.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 10. Table Booking Availability versus Ratings
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Has Table booking', y='Aggregate rating', data=df_clean, hue='Has Table booking', palette=['#457b9d', '#e63946'], legend=False, ax=ax, showmeans=True)
    ax.set_title("Impact of Table Booking Availability on Aggregate Rating", pad=12, fontweight='bold')
    ax.set_xlabel("Has Table Booking?")
    ax.set_ylabel("Aggregate Rating")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '10_table_booking_vs_rating.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 11. Top Cuisines Based on Average Ratings (min 20 restaurants)
    # -------------------------------------------------------------
    cuisine_counts = df_clean['Primary Cuisine'].value_counts()
    frequent_cuisines = cuisine_counts[cuisine_counts >= 20].index
    top_cuisines_rating = (
        df_clean[df_clean['Primary Cuisine'].isin(frequent_cuisines)]
        .groupby('Primary Cuisine')['Aggregate rating']
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_cuisines_rating.values, y=top_cuisines_rating.index, hue=top_cuisines_rating.index, palette='viridis', legend=False, ax=ax)
    ax.set_title("Top 15 Primary Cuisines by Average Rating (Minimum 20 Restaurants)", pad=12, fontweight='bold')
    ax.set_xlabel("Mean Aggregate Rating")
    ax.set_ylabel("Cuisine")
    for i, v in enumerate(top_cuisines_rating.values):
        ax.text(v + 0.04, i, f"{v:.2f}", va='center', fontsize=9, fontweight='semibold')
    ax.set_xlim(0, 5.0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '11_top_cuisines_by_rating.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 12. Outlier Analysis Using Boxplots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y='Average Cost for two', data=df_clean, ax=axes[0], color='#2a9d8f')
    axes[0].set_yscale('log')
    axes[0].set_title("Outlier Analysis: Average Cost for Two (Log Scale)", fontweight='bold')
    axes[0].set_ylabel("Average Cost for Two (Log Scale)")

    sns.boxplot(y='Votes', data=df_clean, ax=axes[1], color='#e76f51')
    axes[1].set_yscale('log')
    axes[1].set_title("Outlier Analysis: Votes (Log Scale)", fontweight='bold')
    axes[1].set_ylabel("Total Votes (Log Scale)")
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '12_outlier_boxplots.png'), dpi=300)
    plt.close()

    print("All 12 EDA visualizations generated and saved successfully!")

if __name__ == '__main__':
    generate_eda_visualizations()
