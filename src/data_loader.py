"""
data_loader.py
==============
Phase 1: Project Inspection and Dataset Loading for Restaurant Rating Prediction.
Loads 'data/Dataset .csv', performs comprehensive structural validation,
and outputs dataset metadata and statistics.
"""

import os
import sys
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

# Force UTF-8 standard output for terminal printing on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Default relative paths to look for the dataset
POSSIBLE_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Dataset .csv'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset .csv'),
    'data/Dataset .csv',
    'Dataset .csv'
]

# Standard Country Code mapping for Zomato dataset
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

def locate_dataset(filepath: str = None) -> str:
    """Finds the dataset path among supported default locations."""
    if filepath and os.path.exists(filepath):
        return filepath
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate 'Dataset .csv'. Ensure it is placed in 'data/' or the root project folder.")

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Loads the restaurant dataset using Pandas.
    Handles encoding fallback automatically.
    """
    path = locate_dataset(filepath)
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')
    return df

def identify_feature_types(df: pd.DataFrame, target_col: str = 'Aggregate rating') -> Tuple[List[str], List[str]]:
    """
    Automatically identifies numerical and categorical feature columns,
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
        'Switch to order menu'  # Constant column with single value
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
    """
    Performs full structural inspection of the dataset.
    """
    total_records, total_columns = df.shape
    columns = list(df.columns)
    dtypes = df.dtypes.astype(str).to_dict()
    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_records = int(df.duplicated().sum())
    
    # Categorical cardinality
    categorical_unique_counts = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_unique_counts[col] = int(df[col].nunique())
        
    # Statistical summary
    statistical_summary = df.describe().to_dict()
    
    # Target variable inspection
    target_series = df[target_col] if target_col in df.columns else None
    target_stats = {}
    if target_series is not None:
        target_stats = {
            'count': int(target_series.count()),
            'mean': float(target_series.mean()),
            'std': float(target_series.std()),
            'min': float(target_series.min()),
            '25%': float(target_series.quantile(0.25)),
            '50% (median)': float(target_series.median()),
            '75%': float(target_series.quantile(0.75)),
            'max': float(target_series.max()),
            'missing_count': int(target_series.isnull().sum()),
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
        'statistical_summary': statistical_summary,
        'target_stats': target_stats,
        'numerical_features': num_features,
        'categorical_features': cat_features
    }

def print_inspection_report(df: pd.DataFrame, inspection: Dict[str, Any]):
    """Prints a structured Phase 1 summary report."""
    print("=" * 80)
    print("PHASE 1: RESTAURANT RATING PREDICTION - DATASET INSPECTION REPORT")
    print("=" * 80)
    print(f"Total Records  : {inspection['total_records']:,}")
    print(f"Total Columns  : {inspection['total_columns']}")
    print(f"Duplicates     : {inspection['duplicate_records']}")
    print(f"Total Missing  : {inspection['total_missing']}")
    print("\n--- Missing Values by Column ---")
    for col, count in inspection['missing_values'].items():
        if count > 0:
            pct = (count / inspection['total_records']) * 100
            print(f"  • {col:<22}: {count} missing ({pct:.2f}%)")
    if inspection['total_missing'] == 0:
        print("  • No missing values detected in any column.")
        
    print("\n--- Target Variable Analysis: 'Aggregate rating' ---")
    ts = inspection['target_stats']
    print(f"  • Mean Rating       : {ts.get('mean', 0):.2f}")
    print(f"  • Median Rating     : {ts.get('50% (median)', 0):.2f}")
    print(f"  • Std Deviation     : {ts.get('std', 0):.2f}")
    print(f"  • Min Rating        : {ts.get('min', 0):.1f}")
    print(f"  • Max Rating        : {ts.get('max', 0):.1f}")
    print(f"  • Zero (Unrated)    : {ts.get('zero_rating_count', 0):,} ({ts.get('zero_rating_count', 0)/inspection['total_records']*100:.1f}%)")
    print(f"  • Positive Rated    : {ts.get('positive_rating_count', 0):,} ({ts.get('positive_rating_count', 0)/inspection['total_records']*100:.1f}%)")
    
    print("\n--- Automatically Identified Features ---")
    print(f"  • Numerical Features   : {inspection['numerical_features']}")
    print(f"  • Categorical Features : {inspection['categorical_features']}")
    
    print("\n--- Categorical Unique Values Cardinality ---")
    for col, count in inspection['categorical_unique_counts'].items():
        print(f"  • {col:<22}: {count:>5} unique values")
        
    print("\n--- First 5 Records ---")
    print(df.head(5)[['Restaurant Name', 'City', 'Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']])
    print("\n--- Last 5 Records ---")
    print(df.tail(5)[['Restaurant Name', 'City', 'Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']])
    print("=" * 80)

if __name__ == '__main__':
    df = load_data()
    inspection = inspect_dataset(df)
    print_inspection_report(df, inspection)
