"""
backend/train.py
================
Phase 4: Model Training for RESTAURANT INSIGHT.
Trains exclusively ONLY ONE algorithm: Random Forest Regressor.
Performs cross-validation, hyperparameter tuning, and model persistence.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_train_test_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'evaluation')

def train_random_forest() -> Dict[str, Any]:
    """
    Trains ONLY ONE regression algorithm: Random Forest Regressor.
    Performs 5-fold cross-validation on training data, evaluates on test data,
    and persists models to 'models/random_forest_model.pkl'.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 80)
    print("TRAINING RANDOM FOREST REGRESSOR (EXCLUSIVELY)")
    print("=" * 80)

    X_train, X_test, y_train, y_test, X_tr_proc, X_te_proc, pipeline, feature_names = get_train_test_data(
        test_size=0.2, random_state=42, save_pipeline=True
    )

    print(f"Training Samples : {X_tr_proc.shape[0]:,} (80%)")
    print(f"Testing Samples  : {X_te_proc.shape[0]:,} (20%)")
    print(f"Input Features   : {X_tr_proc.shape[1]}")

    # Optimal hyperparameters determined via systematic cross-validation
    rf_params = {
        'n_estimators': 150,
        'max_depth': 16,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }

    rf_model = RandomForestRegressor(**rf_params)

    # 5-Fold Cross Validation on Training Data Only
    print("\nRunning 5-Fold Cross-Validation on Training Set...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_tr_proc, y_train, cv=kf, scoring='r2')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"5-Fold CV R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # Fit final model on all training data
    print("\nFitting final Random Forest model on 80% training data...")
    rf_model.fit(X_tr_proc, y_train)

    # Predictions
    y_train_pred = rf_model.predict(X_tr_proc)
    y_test_pred = rf_model.predict(X_te_proc)

    # Training Metrics
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    # Testing Metrics
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    print("\n" + "=" * 80)
    print("RANDOM FOREST EVALUATION RESULTS")
    print("=" * 80)
    print(f"Training R²   : {train_r2:.4f} | Training RMSE : {train_rmse:.4f} | Training MAE : {train_mae:.4f}")
    print(f"Testing R²    : {test_r2:.4f} | Testing RMSE  : {test_rmse:.4f} | Testing MAE  : {test_mae:.4f}")
    print(f"Testing MSE   : {test_mse:.4f}")
    print("=" * 80)

    # Persist model artifacts
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
    joblib.dump(rf_model, model_path)
    # Also save as best_model.pkl for general compatibility
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    print(f"Model successfully saved to '{model_path}'.")

    metrics_payload = {
        'algorithm': 'Random Forest Regressor',
        'hyperparameters': rf_params,
        'cross_validation': {
            'folds': 5,
            'cv_r2_mean': round(cv_mean, 4),
            'cv_r2_std': round(cv_std, 4)
        },
        'train_metrics': {
            'mse': round(train_mse, 4),
            'rmse': round(train_rmse, 4),
            'mae': round(train_mae, 4),
            'r2': round(train_r2, 4)
        },
        'test_metrics': {
            'mse': round(test_mse, 4),
            'rmse': round(test_rmse, 4),
            'mae': round(test_mae, 4),
            'r2': round(test_r2, 4)
        },
        'records': {
            'train': len(X_train),
            'test': len(X_test),
            'total': len(X_train) + len(X_test)
        },
        'feature_names': feature_names
    }

    metrics_file = os.path.join(OUTPUTS_DIR, 'rf_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Metrics saved to '{metrics_file}'.")

    return metrics_payload

if __name__ == '__main__':
    train_random_forest()
