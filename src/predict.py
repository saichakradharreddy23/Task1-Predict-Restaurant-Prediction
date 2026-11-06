"""
predict.py
==========
Prediction service module for Restaurant Rating Prediction.
Loads saved preprocessing pipeline and trained regression models to predict
restaurant aggregate rating given operational and location features.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from data_loader import COUNTRY_MAP

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Reverse country mapping: Name -> Code
COUNTRY_NAME_TO_CODE = {v.lower(): k for k, v in COUNTRY_MAP.items()}

# Rating band classification
def get_rating_category(rating: float) -> Tuple[str, str]:
    """Returns qualitative rating text and badge color."""
    if rating >= 4.5:
        return 'Excellent', '#1f7a1f'       # Dark Green
    elif rating >= 4.0:
        return 'Very Good', '#38b000'       # Green
    elif rating >= 3.5:
        return 'Good', '#ffb703'            # Yellow / Amber
    elif rating >= 2.5:
        return 'Average', '#fb8500'         # Orange
    elif rating > 0.0:
        return 'Poor', '#d90429'            # Red
    else:
        return 'Not Rated', '#6c757d'       # Muted Gray

class RestaurantRatingPredictor:
    """Production predictor instance with model caching."""

    def __init__(self, model_name: str = 'best_model'):
        self.model_name = model_name
        self.pipeline = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}. Run preprocessing.py first.")
        self.pipeline = joblib.load(pipeline_path)

        filename = f"{self.model_name}.pkl" if not self.model_name.endswith('.pkl') else self.model_name
        model_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{filename}' not found at {model_path}. Run train.py first.")
        self.model = joblib.load(model_path)

    def validate_inputs(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        """Validates and standardizes input dictionary into a clean DataFrame."""
        # Normalize keys
        clean_input = {}
        for k, v in raw_input.items():
            clean_input[k.strip()] = v

        # Country code
        country_code = clean_input.get('Country Code', None)
        if country_code is None and 'Country' in clean_input:
            c_name = str(clean_input['Country']).strip().lower()
            country_code = COUNTRY_NAME_TO_CODE.get(c_name, 1)
        try:
            country_code = int(country_code) if country_code is not None else 1
        except (ValueError, TypeError):
            country_code = 1

        # Average cost for two
        try:
            avg_cost = max(0.0, float(clean_input.get('Average Cost for two', 500)))
        except (ValueError, TypeError):
            avg_cost = 500.0

        # Price range (1 to 4)
        try:
            price_range = int(clean_input.get('Price range', 2))
            price_range = max(1, min(4, price_range))
        except (ValueError, TypeError):
            price_range = 2

        # Votes
        try:
            votes = max(0, int(clean_input.get('Votes', 50)))
        except (ValueError, TypeError):
            votes = 50

        # Booking & Delivery flags
        def parse_bool_str(val) -> str:
            if isinstance(val, bool):
                return 'Yes' if val else 'No'
            s = str(val).strip().lower()
            return 'Yes' if s in ['yes', '1', 'true', 'y'] else 'No'

        table_booking = parse_bool_str(clean_input.get('Has Table booking', 'No'))
        online_delivery = parse_bool_str(clean_input.get('Has Online delivery', 'No'))

        city = str(clean_input.get('City', 'New Delhi')).strip()
        locality = str(clean_input.get('Locality', 'Other')).strip()
        cuisines = str(clean_input.get('Cuisines', 'North Indian')).strip()

        df_row = pd.DataFrame([{
            'Country Code': country_code,
            'City': city,
            'Locality': locality,
            'Cuisines': cuisines,
            'Average Cost for two': avg_cost,
            'Price range': price_range,
            'Has Table booking': table_booking,
            'Has Online delivery': online_delivery,
            'Votes': votes
        }])

        return df_row

    def predict(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Runs end-to-end inference for a single restaurant."""
        input_df = self.validate_inputs(raw_input)
        
        # Transform via fitted preprocessing pipeline
        X_proc = self.pipeline.transform(input_df)
        
        # Predict
        raw_pred = float(self.model.predict(X_proc)[0])
        # Clamping to valid aggregate rating scale [0.0, 5.0]
        clamped_rating = round(max(0.0, min(5.0, raw_pred)), 2)
        
        rating_text, rating_color = get_rating_category(clamped_rating)

        return {
            'predicted_rating': clamped_rating,
            'rating_text': rating_text,
            'rating_color': rating_color,
            'model_used': self.model_name,
            'input_summary': input_df.to_dict(orient='records')[0],
            'disclaimer': (
                "Notice: This aggregate rating is an empirical machine-learning estimate "
                "derived from historical restaurant patterns and does not constitute a guaranteed "
                "actual consumer review or official platform rating."
            )
        }

def predict_restaurant_rating(features: Dict[str, Any], model_name: str = 'best_model') -> Dict[str, Any]:
    """Convenience functional wrapper for inference."""
    predictor = RestaurantRatingPredictor(model_name=model_name)
    return predictor.predict(features)

if __name__ == '__main__':
    print("=" * 80)
    print("TESTING RESTAURANT RATING PREDICTION ENGINE")
    print("=" * 80)
    
    test_cases = [
        {
            'name': "Premium Fine Dining (High Votes, High Price Tier)",
            'features': {
                'Country Code': 1,
                'City': 'New Delhi',
                'Locality': 'Connaught Place',
                'Cuisines': 'North Indian, Mughlai, Continental',
                'Average Cost for two': 2500,
                'Price range': 4,
                'Has Table booking': 'Yes',
                'Has Online delivery': 'Yes',
                'Votes': 850
            }
        },
        {
            'name': "Budget Street Food Stall (Low Votes, Budget Tier)",
            'features': {
                'Country Code': 1,
                'City': 'New Delhi',
                'Locality': 'Shahdara',
                'Cuisines': 'Street Food, Fast Food',
                'Average Cost for two': 150,
                'Price range': 1,
                'Has Table booking': 'No',
                'Has Online delivery': 'No',
                'Votes': 8
            }
        },
        {
            'name': "Unseen International Restaurant (Handling Unknown Values)",
            'features': {
                'Country Code': 999,
                'City': 'Unknown Metropolis',
                'Locality': 'Exotic Street 42',
                'Cuisines': 'Fusion, Molecular Gastronomy',
                'Average Cost for two': 1200,
                'Price range': 3,
                'Has Table booking': 'Yes',
                'Has Online delivery': 'No',
                'Votes': 140
            }
        }
    ]

    predictor = RestaurantRatingPredictor()
    for tc in test_cases:
        res = predictor.predict(tc['features'])
        print(f"\nScenario: {tc['name']}")
        print(f"  Predicted Rating : {res['predicted_rating']} / 5.0 ({res['rating_text']})")
        print(f"  Model Employed   : {res['model_used']}")
        print(f"  Input Features   : {res['input_summary']['Cuisines']} in {res['input_summary']['City']} (Cost: {res['input_summary']['Average Cost for two']}, Votes: {res['input_summary']['Votes']})")
    print("=" * 80)
