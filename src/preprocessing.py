"""
preprocessing.py
================
Phase 3: Data Preprocessing Pipeline for Restaurant Rating Prediction.
Constructs leak-free Scikit-learn Pipeline and ColumnTransformer,
performs train-test splitting (80/20), handles missing values,
applies scaling, and encodes categorical variables.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Ensure module import path
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data
from feature_engineering import RestaurantFeatureEngineer

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Explicitly defined feature groups after feature engineering
NUMERICAL_FEATURES = [
    'Average Cost for two',
    'Price range',
    'Votes',
    'Cuisine Count',
    'Has Table booking',
    'Has Online delivery'
]

CATEGORICAL_FEATURES = [
    'Country Code',
    'City',
    'Locality',
    'Primary Cuisine'
]

def build_preprocessing_pipeline() -> Pipeline:
    """
    Builds the complete end-to-end preprocessing pipeline.
    Combines custom feature engineering and Scikit-learn ColumnTransformer.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('feature_engineer', RestaurantFeatureEngineer()),
        ('column_transformer', col_transformer)
    ])

    return full_pipeline

def get_train_test_data(
    test_size: float = 0.2,
    random_state: int = 42,
    save_pipeline: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray, Pipeline, List[str]]:
    """
    Loads raw dataset, performs 80/20 train-test split, fits preprocessing
    pipeline strictly on X_train to prevent data leakage, and transforms
    both train and test sets.
    """
    df = load_data()
    
    # Target and predictor separation
    target_col = 'Aggregate rating'
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # 80/20 reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fit pipeline ONLY on training data
    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Extract readable feature names
    col_trans = pipeline.named_steps['column_transformer']
    feature_names = list(col_trans.get_feature_names_out())
    
    if save_pipeline:
        os.makedirs(MODELS_DIR, exist_ok=True)
        pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        print(f"Preprocessing pipeline saved successfully to '{pipeline_path}'.")
        
    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, pipeline, feature_names

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_train_proc, X_test_proc, pipeline, feat_names = get_train_test_data()
    print("=" * 80)
    print("PHASE 3: PREPROCESSING & TRAIN-TEST SPLIT SUMMARY")
    print("=" * 80)
    print(f"Raw Training Records   : {len(X_train):,} (80%)")
    print(f"Raw Testing Records    : {len(X_test):,} (20%)")
    print(f"Processed Train Matrix : {X_train_proc.shape}")
    print(f"Processed Test Matrix  : {X_test_proc.shape}")
    print(f"Total Encoded Features : {len(feat_names)}")
    print(f"Feature Names (First 10): {feat_names[:10]}")
    print("=" * 80)
