"""
evaluate.py
===========
Phase 5: Model Evaluation module for Restaurant Rating Prediction.
Evaluates all trained regression models across MAE, MSE, RMSE, and R2.
Generates evaluation comparison tables and 6 publication-grade diagnostic plots:
1. outputs/evaluation/01_model_comparison.png
2. outputs/evaluation/02_actual_vs_predicted.png
3. outputs/evaluation/03_residual_distribution.png
4. outputs/evaluation/04_residuals_vs_predicted.png
5. outputs/evaluation/05_train_vs_test_performance.png
6. outputs/evaluation/06_predicted_rating_distribution.png
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
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_train_test_data

# Set global plot styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'evaluation')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def evaluate_all_models():
    """Evaluates all saved models, generates comparison tables and diagnostic plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data and preprocessing pipeline...")
    _, _, y_train, y_test, X_train_proc, X_test_proc, pipeline, feat_names = get_train_test_data(
        test_size=0.2, random_state=42, save_pipeline=False
    )

    model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
    model_files = ['linear_regression.pkl', 'decision_tree.pkl', 'random_forest.pkl', 'gradient_boosting.pkl']

    records = []
    predictions_test = {}
    predictions_train = {}
    loaded_models = {}

    for name, filename in zip(model_names, model_files):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{filename}' not found in {MODELS_DIR}. Run train.py first.")
        
        model = joblib.load(path)
        loaded_models[name] = model

        # Test metrics
        y_test_pred = model.predict(X_test_proc)
        predictions_test[name] = y_test_pred

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        # Train metrics
        y_train_pred = model.predict(X_train_proc)
        predictions_train[name] = y_train_pred

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        records.append({
            'Model': name,
            'Train MAE': round(train_mae, 4),
            'Test MAE': round(test_mae, 4),
            'Train MSE': round(train_mse, 4),
            'Test MSE': round(test_mse, 4),
            'Train RMSE': round(train_rmse, 4),
            'Test RMSE': round(test_rmse, 4),
            'Train R2': round(train_r2, 4),
            'Test R2': round(test_r2, 4)
        })

    summary_df = pd.DataFrame(records)
    summary_csv = os.path.join(OUTPUT_DIR, 'metrics_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nEvaluation summary saved to '{summary_csv}':\n")
    print(summary_df.to_string(index=False))

    # Identify best model
    best_idx = summary_df['Test R2'].idxmax()
    best_model_name = summary_df.loc[best_idx, 'Model']
    best_preds = predictions_test[best_model_name]
    residuals = y_test - best_preds

    # -------------------------------------------------------------
    # 1. Model Performance Comparison Chart
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = [('Test R2', 'R-squared (Higher is Better)', '#2a9d8f'),
                       ('Test RMSE', 'RMSE (Lower is Better)', '#e76f51'),
                       ('Test MAE', 'MAE (Lower is Better)', '#457b9d')]

    for i, (m_col, label, color) in enumerate(metrics_to_plot):
        sns.barplot(x='Model', y=m_col, data=summary_df, hue='Model', palette='Blues_r' if 'R2' in m_col else 'Reds_r', legend=False, ax=axes[i])
        axes[i].set_title(label, fontweight='bold', pad=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(m_col)
        axes[i].tick_params(axis='x', rotation=15)
        for p in axes[i].patches:
            val = p.get_height()
            axes[i].annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                             textcoords='offset points', fontweight='bold')
                             
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_model_comparison.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Actual vs. Predicted Ratings (Best Model)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(x=y_test, y=best_preds, alpha=0.35, color='#1d3557', s=35, ax=ax, label='Predictions')
    # Perfect diagonal reference
    min_val, max_val = min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#e63946', linestyle='--', linewidth=2.5, label='Ideal Perfect Fit (y = x)')
    ax.set_title(f"Actual vs. Predicted Ratings ({best_model_name})", pad=12, fontweight='bold')
    ax.set_xlabel("Actual Aggregate Rating")
    ax.set_ylabel("Predicted Aggregate Rating")
    ax.text(0.05, 0.90, f"R² = {summary_df.loc[best_idx, 'Test R2']:.4f}\nRMSE = {summary_df.loc[best_idx, 'Test RMSE']:.4f}\nMAE = {summary_df.loc[best_idx, 'Test MAE']:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_actual_vs_predicted.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Residual Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=40, kde=True, color='#457b9d', ax=ax, edgecolor='white')
    ax.axvline(0, color='#e63946', linestyle='--', linewidth=2, label="Zero Error Line")
    ax.axvline(residuals.mean(), color='#2a9d8f', linestyle=':', linewidth=2, label=f"Mean Residual: {residuals.mean():.3f}")
    ax.set_title(f"Residual Distribution ({best_model_name})", pad=12, fontweight='bold')
    ax.set_xlabel("Residual (Actual Rating - Predicted Rating)")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_residual_distribution.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Residuals vs. Predicted Values (Homoscedasticity check)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=best_preds, y=residuals, alpha=0.35, color='#2a9d8f', s=35, ax=ax)
    ax.axhline(0, color='#e63946', linestyle='--', linewidth=2, label="Zero Residual")
    ax.set_title(f"Residuals vs. Predicted Values ({best_model_name})", pad=12, fontweight='bold')
    ax.set_xlabel("Predicted Aggregate Rating")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_residuals_vs_predicted.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Training vs. Testing Performance (Overfitting Inspection)
    # -------------------------------------------------------------
    perf_df = pd.melt(
        summary_df,
        id_vars=['Model'],
        value_vars=['Train R2', 'Test R2'],
        var_name='Split',
        value_name='R2 Score'
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Model', y='R2 Score', hue='Split', data=perf_df, palette=['#457b9d', '#e76f51'], ax=ax)
    ax.set_title("Training vs. Testing R² Score (Generalization Analysis)", pad=12, fontweight='bold')
    ax.set_ylabel("R² Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_train_vs_test_performance.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Predicted vs. Actual Rating Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(y_test, label='Actual Test Ratings', color='#1d3557', fill=True, alpha=0.25, linewidth=2.5, ax=ax)
    sns.kdeplot(best_preds, label=f'Predicted Ratings ({best_model_name})', color='#e63946', fill=True, alpha=0.25, linewidth=2.5, ax=ax)
    ax.set_title("Density Comparison: Actual vs. Predicted Rating Distributions", pad=12, fontweight='bold')
    ax.set_xlabel("Aggregate Rating")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '06_predicted_rating_distribution.png'), dpi=300)
    plt.close()

    print("\nAll 6 model evaluation visualizations generated and saved to 'outputs/evaluation/' successfully!")

if __name__ == '__main__':
    evaluate_all_models()
