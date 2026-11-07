"""
backend/predict.py
==================
Inference engine for RESTAURANT INSIGHT.
Loads trained Random Forest model and preprocessing pipeline to predict
restaurant aggregate rating given operational features.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from data_loader import COUNTRY_MAP, COUNTRY_NAME_TO_CODE

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def get_rating_category(rating: float) -> Tuple[str, str, str]:
    """Returns qualitative rating label, badge color, and hex accent."""
    if rating >= 4.5:
        return 'Excellent', 'bg-emerald-600 text-white', '#059669'
    elif rating >= 4.0:
        return 'Very Good', 'bg-emerald-500 text-white', '#10b981'
    elif rating >= 3.5:
        return 'Good', 'bg-amber-500 text-white', '#f59e0b'
    elif rating >= 2.5:
        return 'Average', 'bg-orange-500 text-white', '#f97316'
    elif rating > 0.0:
        return 'Poor', 'bg-rose-600 text-white', '#e11d48'
    else:
        return 'Not Rated', 'bg-stone-500 text-white', '#78716c'

class RestaurantRatingPredictor:
    """Predictor instance with cached model and pipeline."""

    def __init__(self):
        self.pipeline = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        pipeline_path = os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl')
        model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline not found at {pipeline_path}. Run preprocessing.py first.")
        if not os.path.exists(model_path):
            # Fallback to best_model if present
            model_path = os.path.join(MODELS_DIR, 'best_model.pkl')

        self.pipeline = joblib.load(pipeline_path)
        self.model = joblib.load(model_path)

        # Cross-version compatibility patch for SimpleImputer (_fill_dtype / _fit_dtype)
        try:
            ct = getattr(self.pipeline, 'named_steps', {}).get('column_transformer', None)
            if ct and hasattr(ct, 'transformers_'):
                for _, trans, _ in ct.transformers_:
                    if hasattr(trans, 'steps'):
                        for _, sub in trans.steps:
                            if hasattr(sub, 'statistics_'):
                                stat_dtype = getattr(sub.statistics_, 'dtype', None)
                                if not hasattr(sub, '_fill_dtype'):
                                    sub._fill_dtype = getattr(sub, '_fit_dtype', stat_dtype)
                                if not hasattr(sub, '_fit_dtype'):
                                    sub._fit_dtype = getattr(sub, '_fill_dtype', stat_dtype)
        except Exception:
            pass

    def validate_inputs(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        """Validates and standardizes input dictionary into a clean DataFrame."""
        clean = {k.strip(): v for k, v in raw_input.items()}

        # Country
        country_code = clean.get('Country Code', None)
        if country_code is None and 'Country' in clean:
            c_name = str(clean['Country']).strip().lower()
            country_code = COUNTRY_NAME_TO_CODE.get(c_name, 1)
        try:
            country_code = int(country_code) if country_code is not None else 1
        except (ValueError, TypeError):
            country_code = 1

        # Average cost for two
        try:
            avg_cost = max(0.0, float(clean.get('Average Cost for two', 500)))
        except (ValueError, TypeError):
            avg_cost = 500.0

        # Price range (1 to 4)
        try:
            price_range = int(clean.get('Price range', 2))
            price_range = max(1, min(4, price_range))
        except (ValueError, TypeError):
            price_range = 2

        # Votes
        try:
            votes = max(0, int(clean.get('Votes', 50)))
        except (ValueError, TypeError):
            votes = 50

        # Boolean flags
        def parse_bool(val) -> str:
            if isinstance(val, bool):
                return 'Yes' if val else 'No'
            s = str(val).strip().lower()
            return 'Yes' if s in ['yes', '1', 'true', 'y'] else 'No'

        table_booking = parse_bool(clean.get('Has Table booking', 'No'))
        online_delivery = parse_bool(clean.get('Has Online delivery', 'No'))

        city = str(clean.get('City', 'New Delhi')).strip()
        locality = str(clean.get('Locality', 'Other')).strip()
        cuisines = str(clean.get('Cuisines', 'North Indian')).strip()

        return pd.DataFrame([{
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

    def predict(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference and returns rating, category, and strategic advice."""
        input_df = self.validate_inputs(raw_input)
        X_proc = self.pipeline.transform(input_df)

        raw_pred = float(self.model.predict(X_proc)[0])
        clamped_rating = round(max(0.0, min(5.0, raw_pred)), 2)
        rating_label, badge_class, color_hex = get_rating_category(clamped_rating)

        # Operational insights
        row = input_df.iloc[0]
        advice = []
        if row['Votes'] < 30:
            advice.append("Review Velocity: Encouraging verified table feedback or digital receipt prompts can boost rating credibility.")
        if row['Has Online delivery'] == 'No':
            advice.append("Digital Access: Activating online delivery can increase customer reach and frequency of customer reviews.")
        if row['Has Table booking'] == 'No' and row['Price range'] >= 3:
            advice.append("Table Booking: Providing table reservations for mid-to-high tier venues substantially enhances diner satisfaction.")
        if not advice:
            advice.append("Strong operational positioning: The venue exhibits well-balanced service conveniences and menu depth.")

        return {
            'predicted_rating': clamped_rating,
            'rating_label': rating_label,
            'badge_class': badge_class,
            'color_hex': color_hex,
            'algorithm_used': 'Random Forest Regressor',
            'input_summary': {
                'country_name': COUNTRY_MAP.get(int(row['Country Code']), 'International'),
                'city': row['City'],
                'locality': row['Locality'],
                'cuisines': row['Cuisines'],
                'average_cost': float(row['Average Cost for two']),
                'price_range': int(row['Price range']),
                'has_table_booking': row['Has Table booking'],
                'has_online_delivery': row['Has Online delivery'],
                'votes': int(row['Votes'])
            },
            'operational_advice': advice,
            'disclaimer': (
                "Notice: This aggregate rating is an empirical machine-learning estimate "
                "generated by the Random Forest model and does not guarantee specific customer reviews."
            )
        }

_predictor_instance = None

def get_predictor() -> RestaurantRatingPredictor:
    """Singleton getter for the predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = RestaurantRatingPredictor()
    return _predictor_instance

def predict_rating(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience functional wrapper."""
    return get_predictor().predict(raw_input)

if __name__ == '__main__':
    p = get_predictor()
    res = p.predict({
        'Country': 'India',
        'City': 'New Delhi',
        'Locality': 'Connaught Place',
        'Cuisines': 'North Indian, Mughlai',
        'Average Cost for two': 1800,
        'Price range': 3,
        'Has Table booking': 'Yes',
        'Has Online delivery': 'Yes',
        'Votes': 420
    })
    print("Sample Inference Result:")
    print(f"Predicted Rating: {res['predicted_rating']} / 5.0 ({res['rating_label']})")
    print(f"Advice: {res['operational_advice']}")
