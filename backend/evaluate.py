"""
backend/evaluate.py
===================
Phase 5: Evaluation module for Random Forest Regressor.
Calculates MSE, RMSE, MAE, R2 on the held-out test set.
Generates 5 required visual diagnostic charts inside outputs/evaluation/.
"""

import os
import sys
import json
import joblib
import tempfile

os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_train_test_data

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'evaluation')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def evaluate_random_forest():
    """Evaluates the Random Forest model and generates diagnostic plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data and Random Forest model...")
    _, _, y_train, y_test, X_tr_p, X_te_p, pipeline, feat_names = get_train_test_data(
        test_size=0.2, random_state=42, save_pipeline=False
    )

    model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
    rf_model = joblib.load(model_path)

    # Predictions
    y_test_pred = rf_model.predict(X_te_p)
    y_train_pred = rf_model.predict(X_tr_p)

    # Calculate actual test metrics
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Calculate train metrics for overfitting check
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    residuals = y_test - y_test_pred
    errors = np.abs(residuals)

    print("\n--- RANDOM FOREST TEST METRICS ---")
    print(f"Mean Squared Error (MSE)       : {test_mse:.4f}")
    print(f"Root Mean Squared Error (RMSE) : {test_rmse:.4f}")
    print(f"Mean Absolute Error (MAE)      : {test_mae:.4f}")
    print(f"R-squared (R²) Score           : {test_r2:.4f}")

    # -------------------------------------------------------------
    # 1. Actual vs Predicted Restaurant Ratings
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.35, color='#c2410c', s=35, ax=ax, label='Predictions')
    min_v, max_v = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], color='#1c1917', linestyle='--', linewidth=2, label='Ideal Perfect Fit (y = x)')
    ax.set_title("Actual vs. Predicted Restaurant Ratings (Random Forest)", pad=12, fontweight='bold')
    ax.set_xlabel("Actual Aggregate Rating")
    ax.set_ylabel("Predicted Aggregate Rating")
    ax.text(0.05, 0.90, f"R² = {test_r2:.4f}\nRMSE = {test_rmse:.4f}\nMAE = {test_mae:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_actual_vs_predicted.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Residual Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=40, kde=True, color='#ea580c', ax=ax, edgecolor='white')
    ax.axvline(0, color='#1c1917', linestyle='--', linewidth=2, label="Zero Error Line")
    ax.axvline(residuals.mean(), color='#0284c7', linestyle=':', linewidth=2, label=f"Mean Residual: {residuals.mean():.4f}")
    ax.set_title("Residual Distribution (Actual Rating - Predicted Rating)", pad=12, fontweight='bold')
    ax.set_xlabel("Residual Error")
    ax.set_ylabel("Count")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_residual_distribution.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Residuals vs Predicted Ratings (Homoscedasticity)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.35, color='#0284c7', s=35, ax=ax)
    ax.axhline(0, color='#c2410c', linestyle='--', linewidth=2, label="Zero Residual")
    ax.set_title("Residuals vs. Predicted Ratings", pad=12, fontweight='bold')
    ax.set_xlabel("Predicted Aggregate Rating")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_residuals_vs_predicted.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Prediction Error Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(errors, bins=35, kde=True, color='#d97706', ax=ax, edgecolor='white')
    ax.axvline(errors.mean(), color='#c2410c', linestyle='--', linewidth=2, label=f"Mean Absolute Error: {errors.mean():.4f}")
    ax.axvline(np.median(errors), color='#16a34a', linestyle=':', linewidth=2, label=f"Median Absolute Error: {np.median(errors):.4f}")
    ax.set_title("Absolute Prediction Error Distribution (|Actual - Predicted|)", pad=12, fontweight='bold')
    ax.set_xlabel("Absolute Error (|Rating Delta|)")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_prediction_error_distribution.png'), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Model Evaluation Metrics (Train vs Test Overfitting Check)
    # -------------------------------------------------------------
    metrics_comparison = pd.DataFrame([
        {'Metric': 'R-squared (R²)', 'Train': train_r2, 'Test': test_r2},
        {'Metric': 'RMSE', 'Train': train_rmse, 'Test': test_rmse},
        {'Metric': 'MAE', 'Train': train_mae, 'Test': test_mae},
        {'Metric': 'MSE', 'Train': train_mse, 'Test': test_mse}
    ])
    melted = pd.melt(metrics_comparison, id_vars=['Metric'], value_vars=['Train', 'Test'], var_name='Dataset', value_name='Value')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Metric', y='Value', hue='Dataset', data=melted, palette=['#1c1917', '#c2410c'], ax=ax)
    ax.set_title("Random Forest: Training vs. Testing Metrics (Generalization Analysis)", pad=12, fontweight='bold')
    ax.set_ylabel("Score / Error Value")
    ax.set_xlabel("")
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points', fontweight='semibold')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_train_vs_test_metrics.png'), dpi=300)
    plt.close()

    print("All 5 evaluation plots generated and saved to 'outputs/evaluation/' successfully!")

if __name__ == '__main__':
    evaluate_random_forest()
