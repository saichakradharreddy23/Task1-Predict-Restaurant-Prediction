"""
train.py
========
Phase 4: Machine Learning Model Development for Restaurant Rating Prediction.
Trains, tunes, cross-validates, and compares 4 regression algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Selects and persists the best performing model based on validation metrics,
and saves individual model artifacts to 'models/'.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from preprocessing import get_train_test_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'evaluation')

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard regression evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4)
    }

def train_and_tune_models() -> Dict[str, Any]:
    """
    Trains and tunes all 4 regression algorithms, evaluates them via 5-fold CV,
    evaluates on test data, and saves the models and comparison metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading preprocessed dataset...")
    X_train, X_test, y_train, y_test, X_tr_proc, X_te_proc, pipeline, feature_names = get_train_test_data(
        test_size=0.2, random_state=42, save_pipeline=True
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Linear Regression
    print("\n[1/4] Training Linear Regression...")
    lr = LinearRegression()
    lr_cv = cross_val_score(lr, X_tr_proc, y_train, cv=kf, scoring='r2')
    lr.fit(X_tr_proc, y_train)

    # 2. Decision Tree Regressor (Tuned to prevent overfitting)
    print("\n[2/4] Tuning Decision Tree Regressor...")
    dt_base = DecisionTreeRegressor(random_state=42)
    dt_param_grid = {
        'max_depth': [6, 8, 10, 12],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 4, 8]
    }
    dt_grid = GridSearchCV(dt_base, dt_param_grid, cv=kf, scoring='r2', n_jobs=-1)
    dt_grid.fit(X_tr_proc, y_train)
    dt_best = dt_grid.best_estimator_
    dt_cv = cross_val_score(dt_best, X_tr_proc, y_train, cv=kf, scoring='r2')
    print(f"  Best Decision Tree Params: {dt_grid.best_params_}")

    # 3. Random Forest Regressor (Tuned)
    print("\n[3/4] Tuning Random Forest Regressor...")
    rf_base = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [12, 16, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=kf, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_tr_proc, y_train)
    rf_best = rf_grid.best_estimator_
    rf_cv = cross_val_score(rf_best, X_tr_proc, y_train, cv=kf, scoring='r2')
    print(f"  Best Random Forest Params: {rf_grid.best_params_}")

    # 4. Gradient Boosting Regressor (Tuned)
    print("\n[4/4] Tuning Gradient Boosting Regressor...")
    gb_base = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5]
    }
    gb_grid = GridSearchCV(gb_base, gb_param_grid, cv=kf, scoring='r2', n_jobs=-1)
    gb_grid.fit(X_tr_proc, y_train)
    gb_best = gb_grid.best_estimator_
    gb_cv = cross_val_score(gb_best, X_tr_proc, y_train, cv=kf, scoring='r2')
    print(f"  Best Gradient Boosting Params: {gb_grid.best_params_}")

    trained_models = {
        'Linear Regression': (lr, lr_cv, {}),
        'Decision Tree': (dt_best, dt_cv, dt_grid.best_params_),
        'Random Forest': (rf_best, rf_cv, rf_grid.best_params_),
        'Gradient Boosting': (gb_best, gb_cv, gb_grid.best_params_)
    }

    results = {}
    best_model_name = None
    best_cv_score = -float('inf')
    best_model_obj = None

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE EVALUATION & COMPARISON")
    print("=" * 80)

    for name, (model, cv_scores, best_params) in trained_models.items():
        # Predictions
        train_preds = model.predict(X_tr_proc)
        test_preds = model.predict(X_te_proc)

        train_metrics = compute_metrics(y_train, train_preds)
        test_metrics = compute_metrics(y_test, test_preds)

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Check for best model based on validation / test R2 and RMSE
        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_model_name = name
            best_model_obj = model

        # Save individual model
        slug = name.lower().replace(' ', '_')
        model_filename = os.path.join(MODELS_DIR, f"{slug}.pkl")
        joblib.dump(model, model_filename)

        results[name] = {
            'CV_R2_Mean': round(cv_mean, 4),
            'CV_R2_Std': round(cv_std, 4),
            'Train_Metrics': train_metrics,
            'Test_Metrics': test_metrics,
            'Best_Hyperparameters': best_params,
            'Model_File': f"{slug}.pkl"
        }

        print(f"\nModel: {name}")
        print(f"  5-Fold CV R2 Score : {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"  Train -> R2: {train_metrics['R2']:.4f} | RMSE: {train_metrics['RMSE']:.4f} | MAE: {train_metrics['MAE']:.4f}")
        print(f"  Test  -> R2: {test_metrics['R2']:.4f} | RMSE: {test_metrics['RMSE']:.4f} | MAE: {test_metrics['MAE']:.4f}")

    # Persist Best Model
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model_obj, best_model_path)
    print(f"\n>>> Best Performing Model: '{best_model_name}' (CV R2: {best_cv_score:.4f})")
    print(f">>> Saved as '{best_model_path}'")

    summary_payload = {
        'best_model': best_model_name,
        'best_cv_r2': round(best_cv_score, 4),
        'models': results,
        'feature_names': feature_names
    }

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUTS_DIR, 'model_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=4)
    print(f"Detailed metrics summary saved to '{metrics_path}'.")

    return summary_payload

if __name__ == '__main__':
    train_and_tune_models()
