"""
backend/data_loader.py
======================
Dataset loading, structural validation, and inspection module for
RESTAURANT INSIGHT – AI POWERED RESTAURANT RATING PREDICTION.
"""

import os
import sys
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np

# Force UTF-8 standard output for terminal printing on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Dataset search paths
POSSIBLE_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Dataset .csv'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset .csv'),
    'data/Dataset .csv',
    'Dataset .csv'
]

# Standard Country Code mapping
COUNTRY_MAP = {
    1: 'India',
    14: 'Australia',
    30: 'Brazil',
    37: 'Canada',
    94: 'Indonesia',
    148: 'New Zealand',
    162: 'Philippines',
    166: 'Qatar',
    184: 'Singapore',
    189: 'South Africa',
    191: 'Sri Lanka',
    208: 'Turkey',
    214: 'United Arab Emirates',
    215: 'United Kingdom',
    216: 'United States'
}

COUNTRY_NAME_TO_CODE = {v.lower(): k for k, v in COUNTRY_MAP.items()}

def locate_dataset(filepath: Optional[str] = None) -> str:
    """Finds the dataset path among supported default locations."""
    if filepath and os.path.exists(filepath):
        return filepath
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate 'Dataset .csv'. Ensure it is placed in 'data/' or the project root.")

def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads the restaurant dataset using Pandas with automatic encoding fallback."""
    path = locate_dataset(filepath)
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')
    return df

def identify_feature_types(df: pd.DataFrame, target_col: str = 'Aggregate rating') -> Tuple[List[str], List[str]]:
    """
    Automatically distinguishes numerical and categorical feature columns,
    excluding target leakage and identifier columns.
    """
    excluded_cols = {
        target_col,
        'Restaurant ID',
        'Restaurant Name',
        'Rating color',
        'Rating text',
        'Address',
        'Locality Verbose',
        'Switch to order menu',
        'Is delivering now'
    }
    
    numerical_features = []
    categorical_features = []
    
    for col in df.columns:
        if col in excluded_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_features.append(col)
        else:
            categorical_features.append(col)
            
    return numerical_features, categorical_features

def inspect_dataset(df: pd.DataFrame, target_col: str = 'Aggregate rating') -> Dict[str, Any]:
    """Performs full structural inspection of the dataset."""
    total_records, total_columns = df.shape
    columns = list(df.columns)
    dtypes = df.dtypes.astype(str).to_dict()
    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_records = int(df.duplicated().sum())
    
    categorical_unique_counts = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_unique_counts[col] = int(df[col].nunique())
        
    target_series = df[target_col] if target_col in df.columns else None
    target_stats = {}
    if target_series is not None:
        target_stats = {
            'count': int(target_series.count()),
            'mean': round(float(target_series.mean()), 2),
            'std': round(float(target_series.std()), 2),
            'min': float(target_series.min()),
            '25%': float(target_series.quantile(0.25)),
            '50%': float(target_series.median()),
            '75%': float(target_series.quantile(0.75)),
            'max': float(target_series.max()),
            'zero_rating_count': int((target_series == 0.0).sum()),
            'positive_rating_count': int((target_series > 0.0).sum())
        }
        
    num_features, cat_features = identify_feature_types(df, target_col)
    
    return {
        'total_records': total_records,
        'total_columns': total_columns,
        'columns': columns,
        'dtypes': dtypes,
        'missing_values': missing_values,
        'total_missing': total_missing,
        'duplicate_records': duplicate_records,
        'categorical_unique_counts': categorical_unique_counts,
        'target_stats': target_stats,
        'numerical_features': num_features,
        'categorical_features': cat_features
    }

if __name__ == '__main__':
    df = load_data()
    info = inspect_dataset(df)
    print("Dataset Loaded Successfully:")
    print(f"Shape: {info['total_records']} rows, {info['total_columns']} columns")
    print(f"Duplicates: {info['duplicate_records']}")
    print(f"Missing Values: {info['missing_values']}")
    print(f"Target Stats: {info['target_stats']}")
    print(f"Numerical Features: {info['numerical_features']}")
    print(f"Categorical Features: {info['categorical_features']}")
