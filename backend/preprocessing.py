"""
backend/preprocessing.py
========================
Data Preprocessing Pipeline for RESTAURANT INSIGHT.
Leak-free Scikit-learn Pipeline & ColumnTransformer,
80/20 Train-Test split with random_state=42.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Target and feature groupings
TARGET_COLUMN = 'Aggregate rating'

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

class RestaurantFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer for restaurant prediction:
    1. Extracts 'Primary Cuisine' and 'Cuisine Count'.
    2. Maps binary flags 'Has Table booking' and 'Has Online delivery' to 1/0.
    3. Groups rare categories in City, Locality, and Primary Cuisine into 'Other'.
    4. Removes identifiers and target leakage columns.
    """
    def __init__(self, top_cities_n: int = 20, top_localities_n: int = 30, top_cuisines_n: int = 25):
        self.top_cities_n = top_cities_n
        self.top_localities_n = top_localities_n
        self.top_cuisines_n = top_cuisines_n
        self.frequent_cities_: List[str] = []
        self.frequent_localities_: List[str] = []
        self.frequent_cuisines_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        if 'City' in X_df.columns:
            self.frequent_cities_ = (
                X_df['City'].dropna().value_counts().head(self.top_cities_n).index.tolist()
            )
        if 'Locality' in X_df.columns:
            self.frequent_localities_ = (
                X_df['Locality'].dropna().value_counts().head(self.top_localities_n).index.tolist()
            )
        if 'Cuisines' in X_df.columns:
            primary = X_df['Cuisines'].fillna('Unknown').apply(
                lambda s: [c.strip() for c in str(s).split(',')][0]
            )
            self.frequent_cuisines_ = (
                primary.value_counts().head(self.top_cuisines_n).index.tolist()
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        cuisines_col = (
            X_df['Cuisines'].fillna('Unknown')
            if 'Cuisines' in X_df.columns
            else pd.Series('Unknown', index=X_df.index)
        )

        X_df['Primary Cuisine'] = cuisines_col.apply(
            lambda s: [c.strip() for c in str(s).split(',')][0] if s != 'Unknown' else 'Unknown'
        )
        X_df['Cuisine Count'] = cuisines_col.apply(
            lambda s: len([c.strip() for c in str(s).split(',')]) if s != 'Unknown' else 1
        )

        if self.frequent_cities_ and 'City' in X_df.columns:
            X_df['City'] = X_df['City'].apply(lambda x: x if x in self.frequent_cities_ else 'Other')

        if self.frequent_localities_ and 'Locality' in X_df.columns:
            X_df['Locality'] = X_df['Locality'].apply(lambda x: x if x in self.frequent_localities_ else 'Other')

        if self.frequent_cuisines_:
            X_df['Primary Cuisine'] = X_df['Primary Cuisine'].apply(
                lambda x: x if x in self.frequent_cuisines_ else 'Other'
            )

        if 'Has Table booking' in X_df.columns:
            X_df['Has Table booking'] = (
                X_df['Has Table booking'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
            )

        if 'Has Online delivery' in X_df.columns:
            X_df['Has Online delivery'] = (
                X_df['Has Online delivery'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
            )

        cols_to_drop = [
            'Restaurant ID', 'Restaurant Name', 'Address', 'Locality Verbose',
            'Rating color', 'Rating text', 'Switch to order menu', 'Is delivering now',
            'Currency', 'Cuisines'
        ]
        existing_drop = [c for c in cols_to_drop if c in X_df.columns]
        X_df = X_df.drop(columns=existing_drop)
        return X_df

def build_preprocessing_pipeline() -> Pipeline:
    """Builds the complete end-to-end preprocessing pipeline."""
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
    Splits dataset into 80% train and 20% test sets (random_state=42),
    fits preprocessing strictly on X_train to prevent data leakage,
    transforms X_train and X_test, and saves the pipeline.
    """
    df = load_data()
    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)

    col_trans = pipeline.named_steps['column_transformer']
    feature_names = list(col_trans.get_feature_names_out())

    if save_pipeline:
        os.makedirs(MODELS_DIR, exist_ok=True)
        pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        print(f"Preprocessing pipeline saved to '{pipeline_path}'.")

    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, pipeline, feature_names

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_tr_p, X_te_p, pipe, feats = get_train_test_data()
    print("Preprocessing & Split Completed:")
    print(f"Training Records: {len(X_train):,} (80%)")
    print(f"Testing Records: {len(X_test):,} (20%)")
    print(f"Transformed Features: {X_tr_p.shape[1]}")
