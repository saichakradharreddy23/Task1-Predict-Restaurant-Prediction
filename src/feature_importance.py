"""
feature_importance.py
=====================
Phase 6: Feature Importance & Model Interpretation for Restaurant Rating Prediction.
Extracts:
1. Tree-based feature importances (Gradient Boosting & Random Forest)
2. Permutation Importance on the held-out test set
3. Linear Regression standardized coefficients

Generates ranking tables, CSV exports, and 3 diagnostic visualization charts:
- outputs/feature_importance/01_feature_importance_bar_chart.png
- outputs/feature_importance/02_permutation_importance_bar_chart.png
- outputs/feature_importance/03_model_feature_comparison.png
"""

import os
import sys
import json
import joblib
import tempfile

# Prevent matplotlib directory warning on Windows
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_train_test_data

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'feature_importance')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def clean_feature_name(raw_name: str) -> str:
    """Converts pipeline internal names into clean, readable labels."""
    name = raw_name
    if name.startswith('num__'):
        name = name[5:]
    elif name.startswith('cat__'):
        name = name[5:]
        parts = name.split('_', 1)
        if len(parts) == 2:
            name = f"{parts[0]}: {parts[1]}"
    return name

def analyze_feature_importance():
    """Calculates and visualizes feature importances and interpretations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data, pipeline, and trained models...")
    _, _, y_train, y_test, X_train_proc, X_test_proc, pipeline, raw_feat_names = get_train_test_data(
        test_size=0.2, random_state=42, save_pipeline=False
    )

    clean_names = [clean_feature_name(n) for n in raw_feat_names]

    # Load models
    best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.pkl'))

    # 1. Gradient Boosting MDI Importance (Best Model)
    gb_importances = best_model.feature_importances_
    
    # 2. Random Forest MDI Importance
    rf_importances = rf_model.feature_importances_

    # 3. Permutation Importance on Test Set (Gradient Boosting)
    print("Computing Permutation Importance on test set (5 repeats)...")
    perm_result = permutation_importance(
        best_model, X_test_proc, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring='r2'
    )
    perm_mean = perm_result.importances_mean

    # 4. Linear Regression standardized coefficients
    lr_coefs = lr_model.coef_

    # Assemble comprehensive DataFrame
    importance_df = pd.DataFrame({
        'Feature_Raw': raw_feat_names,
        'Feature': clean_names,
        'Importance_GB': gb_importances,
        'Importance_RF': rf_importances,
        'Permutation_Importance_Mean': perm_mean,
        'LR_Coefficient': lr_coefs
    })

    # Sort by Gradient Boosting importance
    importance_df = importance_df.sort_values(by='Importance_GB', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = importance_df.index + 1

    # Save ranking CSV
    ranking_csv = os.path.join(OUTPUT_DIR, 'feature_ranking.csv')
    importance_df.to_csv(ranking_csv, index=False)
    print(f"Full feature ranking saved to '{ranking_csv}'.")

    # Top 10 Features
    top10 = importance_df.head(10).copy()
    print("\n--- TOP 10 MOST INFLUENTIAL FEATURES ---")
    for idx, row in top10.iterrows():
        print(f"Rank {row['Rank']:>2}: {row['Feature']:<30} | GB Imp: {row['Importance_GB']:.4f} | Perm Imp: {row['Permutation_Importance_Mean']:.4f}")

    # -------------------------------------------------------------
    # 1. Feature Importance Horizontal Bar Chart (Gradient Boosting)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    top10_sorted = top10.sort_values(by='Importance_GB', ascending=True)
    sns.barplot(x='Importance_GB', y='Feature', data=top10_sorted, hue='Feature', palette='Blues_d', legend=False, ax=ax)
    ax.set_title("Top 10 Most Influential Features (Gradient Boosting MDI)", pad=12, fontweight='bold')
    ax.set_xlabel("Relative Feature Importance (MDI)")
    ax.set_ylabel("Feature")
    for i, v in enumerate(top10_sorted['Importance_GB']):
        ax.text(v + 0.005, i, f"{v * 100:.2f}%", va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(top10_sorted['Importance_GB']) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_feature_importance_bar_chart.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Permutation Importance Horizontal Bar Chart
    # -------------------------------------------------------------
    top10_perm = importance_df.sort_values(by='Permutation_Importance_Mean', ascending=False).head(10)
    top10_perm_sorted = top10_perm.sort_values(by='Permutation_Importance_Mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Permutation_Importance_Mean', y='Feature', data=top10_perm_sorted, hue='Feature', palette='Reds_d', legend=False, ax=ax)
    ax.set_title("Top 10 Features by Permutation Importance (Test Set R² Drop)", pad=12, fontweight='bold')
    ax.set_xlabel("Mean Decrease in R² Score when Shuffled")
    ax.set_ylabel("Feature")
    for i, v in enumerate(top10_perm_sorted['Permutation_Importance_Mean']):
        ax.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(top10_perm_sorted['Permutation_Importance_Mean']) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_permutation_importance_bar_chart.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Model Comparison: Gradient Boosting vs Random Forest
    # -------------------------------------------------------------
    comparison_top10 = importance_df.head(10)[['Feature', 'Importance_GB', 'Importance_RF']]
    comparison_melted = pd.melt(
        comparison_top10,
        id_vars=['Feature'],
        value_vars=['Importance_GB', 'Importance_RF'],
        var_name='Model',
        value_name='Importance'
    )
    comparison_melted['Model'] = comparison_melted['Model'].map({
        'Importance_GB': 'Gradient Boosting',
        'Importance_RF': 'Random Forest'
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', hue='Model', data=comparison_melted, palette=['#1d3557', '#e76f51'], ax=ax)
    ax.set_title("Feature Importance Comparison: Gradient Boosting vs. Random Forest", pad=12, fontweight='bold')
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_ylabel("Feature")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_model_feature_comparison.png'), dpi=300)
    plt.close()

    # Interpretations JSON payload
    interpretations = {
        'top_10_features': top10[['Rank', 'Feature', 'Importance_GB', 'Permutation_Importance_Mean']].to_dict(orient='records'),
        'key_insights': [
            {
                'feature': 'Votes',
                'role': 'Dominant positive driver. High volume of customer reviews strongly correlates with established customer engagement and validated ratings.',
                'interpretation': 'Restaurants with zero or near-zero votes account for unrated 0.0 entries. High-vote restaurants typically converge around higher aggregate ratings.'
            },
            {
                'feature': 'Price range & Average Cost for two',
                'role': 'Affordability and quality signal.',
                'interpretation': 'Higher price range tiers (Price range 3 and 4) maintain higher median customer ratings, reflecting enhanced ambiance, premium ingredients, and professional service.'
            },
            {
                'feature': 'Online delivery & Table booking',
                'role': 'Service accessibility conveniences.',
                'interpretation': 'Venues offering table reservations and online ordering demonstrate active digital adoption and customer-centric operational standards.'
            },
            {
                'feature': 'Location (Country Code & City)',
                'role': 'Geographical market context.',
                'interpretation': 'Regional rating cultures, platform penetration, and local dining competition significantly benchmark baseline expected scores.'
            }
        ]
    }

    json_path = os.path.join(OUTPUT_DIR, 'feature_importance_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(interpretations, f, indent=4)
    print(f"Feature importance summary JSON saved to '{json_path}'.")

    return importance_df

if __name__ == '__main__':
    analyze_feature_importance()
