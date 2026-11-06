"""
feature_engineering.py
======================
Phase 3: Feature Engineering module for Restaurant Rating Prediction.
Extracts informative derived features, handles high-cardinality categories,
and cleans input features without data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional

class RestaurantFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for restaurant feature engineering:
    1. Extracts 'Primary Cuisine' and 'Cuisine Count' from comma-separated 'Cuisines'.
    2. Encodes binary flags ('Has Table booking', 'Has Online delivery') to 1/0.
    3. Groups infrequent categories in 'City', 'Locality', and 'Primary Cuisine'
       into an 'Other' bucket based on training set frequencies.
    4. Drops leakage and irrelevant columns (Restaurant ID, Restaurant Name, Address,
       Locality Verbose, Switch to order menu, Rating color, Rating text).
    """
    
    def __init__(
        self,
        top_cities_n: int = 20,
        top_localities_n: int = 30,
        top_cuisines_n: int = 25
    ):
        self.top_cities_n = top_cities_n
        self.top_localities_n = top_localities_n
        self.top_cuisines_n = top_cuisines_n
        
        # Learned vocabulary for frequent categories (to prevent leakage)
        self.frequent_cities_: List[str] = []
        self.frequent_localities_: List[str] = []
        self.frequent_cuisines_: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        
        # 1. City top categories
        if 'City' in X_df.columns:
            self.frequent_cities_ = (
                X_df['City'].dropna().value_counts().head(self.top_cities_n).index.tolist()
            )
            
        # 2. Locality top categories
        if 'Locality' in X_df.columns:
            self.frequent_localities_ = (
                X_df['Locality'].dropna().value_counts().head(self.top_localities_n).index.tolist()
            )
            
        # 3. Cuisine top categories
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
        
        # Missing Cuisines handling
        cuisines_col = X_df['Cuisines'].fillna('Unknown') if 'Cuisines' in X_df.columns else pd.Series('Unknown', index=X_df.index)
        
        # Primary cuisine extraction
        X_df['Primary Cuisine'] = cuisines_col.apply(
            lambda s: [c.strip() for c in str(s).split(',')][0] if s != 'Unknown' else 'Unknown'
        )
        
        # Cuisine Count feature (number of cuisines offered)
        X_df['Cuisine Count'] = cuisines_col.apply(
            lambda s: len([c.strip() for c in str(s).split(',')]) if s != 'Unknown' else 1
        )
        
        # High cardinality mapping: map infrequent to 'Other'
        if self.frequent_cities_ and 'City' in X_df.columns:
            X_df['City'] = X_df['City'].apply(lambda x: x if x in self.frequent_cities_ else 'Other')
            
        if self.frequent_localities_ and 'Locality' in X_df.columns:
            X_df['Locality'] = X_df['Locality'].apply(lambda x: x if x in self.frequent_localities_ else 'Other')
            
        if self.frequent_cuisines_:
            X_df['Primary Cuisine'] = X_df['Primary Cuisine'].apply(
                lambda x: x if x in self.frequent_cuisines_ else 'Other'
            )
            
        # Binary flags conversion
        if 'Has Table booking' in X_df.columns:
            X_df['Has Table booking'] = X_df['Has Table booking'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
            
        if 'Has Online delivery' in X_df.columns:
            X_df['Has Online delivery'] = X_df['Has Online delivery'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
            
        # Drop columns not intended for modeling or causing leakage
        cols_to_drop = [
            'Restaurant ID', 'Restaurant Name', 'Address', 'Locality Verbose',
            'Rating color', 'Rating text', 'Switch to order menu', 'Is delivering now',
            'Currency', 'Cuisines'
        ]
        existing_drop = [c for c in cols_to_drop if c in X_df.columns]
        X_df = X_df.drop(columns=existing_drop)
        
        return X_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper to apply feature engineering directly."""
    fe = RestaurantFeatureEngineer()
    return fe.fit_transform(df)

if __name__ == '__main__':
    from data_loader import load_data
    df = load_data()
    fe = RestaurantFeatureEngineer()
    df_engineered = fe.fit_transform(df)
    print("Engineered DataFrame Shape:", df_engineered.shape)
    print("Engineered Columns:", list(df_engineered.columns))
    print(df_engineered.head(3))
