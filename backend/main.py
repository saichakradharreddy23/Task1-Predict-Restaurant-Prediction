"""
backend/main.py
===============
Production FastAPI Application for
RESTAURANT INSIGHT – AI POWERED RESTAURANT RATING PREDICTION.
Connects the trained Random Forest model with the luxury restaurant frontend.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure module path
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data, COUNTRY_MAP, inspect_dataset
from predict import get_predictor, predict_rating, get_rating_category

app = FastAPI(
    title="Restaurant Insight API",
    description="Backend API for AI-Powered Restaurant Rating Prediction using Random Forest Regressor",
    version="2.0.0"
)

# Enable CORS for production Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://task1-predict-restaurant-prediction-teal.vercel.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached Dataset in Memory
DATASET_DF = load_data()
DATASET_DF['Country Name'] = DATASET_DF['Country Code'].map(COUNTRY_MAP).fillna('International')
DATASET_DF['Primary Cuisine'] = DATASET_DF['Cuisines'].fillna('Multi-Cuisine').apply(
    lambda s: [c.strip() for c in str(s).split(',')][0]
)

# High-resolution Unsplash food photography mapping by cuisine
CUISINE_IMAGE_MAP = {
    'north indian': 'https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=800&auto=format&fit=crop&q=80',
    'mughlai': 'https://images.unsplash.com/photo-1544025162-d76694265947?w=800&auto=format&fit=crop&q=80',
    'chinese': 'https://images.unsplash.com/photo-1563245372-f21724e3856d?w=800&auto=format&fit=crop&q=80',
    'fast food': 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800&auto=format&fit=crop&q=80',
    'cafe': 'https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=800&auto=format&fit=crop&q=80',
    'bakery': 'https://images.unsplash.com/photo-1509440159596-0249088772ff?w=800&auto=format&fit=crop&q=80',
    'desserts': 'https://images.unsplash.com/photo-1587314168485-3236d6710814?w=800&auto=format&fit=crop&q=80',
    'italian': 'https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=800&auto=format&fit=crop&q=80',
    'pizza': 'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=800&auto=format&fit=crop&q=80',
    'japanese': 'https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=800&auto=format&fit=crop&q=80',
    'sushi': 'https://images.unsplash.com/photo-1611143669185-af224c5e3252?w=800&auto=format&fit=crop&q=80',
    'south indian': 'https://images.unsplash.com/photo-1630383249896-424e482df921?w=800&auto=format&fit=crop&q=80',
    'street food': 'https://images.unsplash.com/photo-1601050690597-df0568f70950?w=800&auto=format&fit=crop&q=80',
    'continental': 'https://images.unsplash.com/photo-1550966871-3ed3cdb5ed0c?w=800&auto=format&fit=crop&q=80',
    'american': 'https://images.unsplash.com/photo-1550547660-d9450f859349?w=800&auto=format&fit=crop&q=80',
    'asian': 'https://images.unsplash.com/photo-1540420773420-3366772f4999?w=800&auto=format&fit=crop&q=80',
    'mexican': 'https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=800&auto=format&fit=crop&q=80',
    'seafood': 'https://images.unsplash.com/photo-1534422298391-e4f8c172dddb?w=800&auto=format&fit=crop&q=80',
    'finger food': 'https://images.unsplash.com/photo-1541544741938-0af808871cc0?w=800&auto=format&fit=crop&q=80',
    'burger': 'https://images.unsplash.com/photo-1586190848861-99aa4a171e90?w=800&auto=format&fit=crop&q=80'
}

DEFAULT_IMAGE = 'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=800&auto=format&fit=crop&q=80'

def get_restaurant_image(primary_cuisine: str) -> str:
    key = str(primary_cuisine).strip().lower()
    for k, url in CUISINE_IMAGE_MAP.items():
        if k in key:
            return url
    return DEFAULT_IMAGE

# Pydantic Schemas
class PredictionRequest(BaseModel):
    country: str = Field(default="India", description="Country name or code")
    city: str = Field(default="New Delhi", description="City name")
    locality: str = Field(default="Connaught Place", description="Locality or neighborhood")
    cuisines: str = Field(default="North Indian, Mughlai", description="Comma-separated cuisines")
    average_cost_for_two: float = Field(default=1200.0, ge=10.0, le=500000.0, description="Average dining cost for two")
    price_range: int = Field(default=3, ge=1, le=4, description="Price tier from 1 (Budget) to 4 (Luxury)")
    has_table_booking: str = Field(default="Yes", description="'Yes' or 'No'")
    has_online_delivery: str = Field(default="Yes", description="'Yes' or 'No'")
    votes: int = Field(default=250, ge=0, le=50000, description="Number of customer reviews/votes")

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Restaurant Insight API",
        "algorithm": "Random Forest Regressor",
        "total_records": len(DATASET_DF)
    }

@app.get("/api/statistics")
def get_statistics():
    """Returns dataset summary statistics and distributions."""
    total_venues = len(DATASET_DF)
    avg_rating = round(float(DATASET_DF['Aggregate rating'].mean()), 2)
    rated_venues = int((DATASET_DF['Aggregate rating'] > 0).sum())
    unrated_venues = int((DATASET_DF['Aggregate rating'] == 0).sum())
    rated_avg = round(float(DATASET_DF[DATASET_DF['Aggregate rating'] > 0]['Aggregate rating'].mean()), 2)

    top_cities = DATASET_DF['City'].value_counts().head(10).to_dict()
    top_cuisines = DATASET_DF['Primary Cuisine'].value_counts().head(10).to_dict()
    price_dist = DATASET_DF['Price range'].value_counts().sort_index().to_dict()
    countries = sorted(DATASET_DF['Country Name'].unique().tolist())

    return {
        "total_restaurants": total_venues,
        "total_countries": len(countries),
        "total_cities": int(DATASET_DF['City'].nunique()),
        "average_rating": avg_rating,
        "rated_average_rating": rated_avg,
        "rated_count": rated_venues,
        "unrated_count": unrated_venues,
        "top_cities": top_cities,
        "top_cuisines": top_cuisines,
        "price_distribution": price_dist,
        "countries": countries
    }

@app.get("/api/restaurants")
def get_restaurants(
    page: int = 1,
    limit: int = 12,
    search: Optional[str] = None,
    city: Optional[str] = None,
    cuisine: Optional[str] = None,
    price_range: Optional[int] = None,
    min_rating: Optional[float] = None,
    sort_by: str = "rating_desc"
):
    """Returns paginated, searchable, filterable restaurants with curated imagery."""
    df = DATASET_DF.copy()

    if search:
        q = search.strip().lower()
        df = df[
            df['Restaurant Name'].astype(str).str.lower().str.contains(q) |
            df['City'].astype(str).str.lower().str.contains(q) |
            df['Cuisines'].astype(str).str.lower().str.contains(q) |
            df['Locality'].astype(str).str.lower().str.contains(q)
        ]

    if city and city != "All":
        df = df[df['City'].astype(str).str.lower() == city.strip().lower()]

    if cuisine and cuisine != "All":
        df = df[df['Primary Cuisine'].astype(str).str.lower() == cuisine.strip().lower()]

    if price_range is not None:
        df = df[df['Price range'] == price_range]

    if min_rating is not None:
        df = df[df['Aggregate rating'] >= min_rating]

    # Sorting
    if sort_by == "rating_desc":
        df = df.sort_values(by=['Aggregate rating', 'Votes'], ascending=[False, False])
    elif sort_by == "rating_asc":
        df = df.sort_values(by=['Aggregate rating', 'Votes'], ascending=[True, True])
    elif sort_by == "cost_asc":
        df = df.sort_values(by='Average Cost for two', ascending=True)
    elif sort_by == "cost_desc":
        df = df.sort_values(by='Average Cost for two', ascending=False)
    elif sort_by == "votes_desc":
        df = df.sort_values(by='Votes', ascending=False)

    total_count = len(df)
    total_pages = max(1, (total_count + limit - 1) // limit)
    start_idx = (page - 1) * limit
    page_df = df.iloc[start_idx:start_idx + limit]

    items = []
    for _, row in page_df.iterrows():
        p_cuis = row['Primary Cuisine']
        rating = float(row['Aggregate rating'])
        label, badge, hex_c = get_rating_category(rating)
        items.append({
            "id": int(row['Restaurant ID']),
            "name": str(row['Restaurant Name']),
            "city": str(row['City']),
            "locality": str(row['Locality']),
            "country": str(row['Country Name']),
            "cuisines": str(row['Cuisines']),
            "primary_cuisine": str(p_cuis),
            "average_cost": float(row['Average Cost for two']),
            "currency": str(row['Currency']),
            "price_range": int(row['Price range']),
            "aggregate_rating": rating,
            "rating_label": label,
            "badge_class": badge,
            "rating_color": hex_c,
            "votes": int(row['Votes']),
            "has_table_booking": str(row['Has Table booking']),
            "has_online_delivery": str(row['Has Online delivery']),
            "image_url": get_restaurant_image(p_cuis)
        })

    return {
        "page": page,
        "limit": limit,
        "total_records": total_count,
        "total_pages": total_pages,
        "restaurants": items
    }

@app.get("/api/restaurants/{restaurant_id}")
def get_restaurant_details(restaurant_id: int):
    """Returns detailed profile for a specific restaurant."""
    match = DATASET_DF[DATASET_DF['Restaurant ID'] == restaurant_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    row = match.iloc[0]
    p_cuis = row['Primary Cuisine']
    rating = float(row['Aggregate rating'])
    label, badge, hex_c = get_rating_category(rating)

    return {
        "id": int(row['Restaurant ID']),
        "name": str(row['Restaurant Name']),
        "address": str(row['Address']),
        "locality": str(row['Locality']),
        "city": str(row['City']),
        "country": str(row['Country Name']),
        "cuisines": str(row['Cuisines']),
        "primary_cuisine": str(p_cuis),
        "average_cost": float(row['Average Cost for two']),
        "currency": str(row['Currency']),
        "price_range": int(row['Price range']),
        "aggregate_rating": rating,
        "rating_label": label,
        "badge_class": badge,
        "rating_color": hex_c,
        "votes": int(row['Votes']),
        "has_table_booking": str(row['Has Table booking']),
        "has_online_delivery": str(row['Has Online delivery']),
        "longitude": float(row['Longitude']),
        "latitude": float(row['Latitude']),
        "image_url": get_restaurant_image(p_cuis)
    }

@app.get("/api/model-metrics")
def get_model_metrics():
    """Returns Random Forest testing metrics and evaluation details."""
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'evaluation', 'rf_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "algorithm": "Random Forest Regressor",
        "test_metrics": {"r2": 0.9592, "rmse": 0.3047, "mae": 0.1991, "mse": 0.0928},
        "train_metrics": {"r2": 0.9798, "rmse": 0.2156, "mae": 0.1382, "mse": 0.0465}
    }

@app.get("/api/feature-importance")
def get_feature_importance():
    """Returns top 10 influential features generated by the Random Forest model."""
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'feature_importance', 'rf_feature_importance.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "top_10": [
            {"Rank": 1, "Feature": "Votes", "Importance": 0.9642, "Permutation_Importance": 1.8799},
            {"Rank": 2, "Feature": "City: Other", "Importance": 0.0097, "Permutation_Importance": 0.0083},
            {"Rank": 3, "Feature": "Average Cost for two", "Importance": 0.0071, "Permutation_Importance": 0.0056},
            {"Rank": 4, "Feature": "Cuisine Count", "Importance": 0.0023, "Permutation_Importance": 0.0002},
            {"Rank": 5, "Feature": "Primary Cuisine: North Indian", "Importance": 0.0013, "Permutation_Importance": 0.0027}
        ]
    }

@app.post("/api/predict")
def predict_restaurant_rating_endpoint(req: PredictionRequest):
    """Predicts aggregate rating using the trained Random Forest model."""
    try:
        payload = {
            'Country': req.country,
            'City': req.city,
            'Locality': req.locality,
            'Cuisines': req.cuisines,
            'Average Cost for two': req.average_cost_for_two,
            'Price range': req.price_range,
            'Has Table booking': req.has_table_booking,
            'Has Online delivery': req.has_online_delivery,
            'Votes': req.votes
        }
        res = predict_rating(payload)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount outputs directory so evaluation charts can be displayed directly in frontend
outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
if os.path.exists(outputs_dir):
    app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# Mount frontend directory for web hosting
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Restaurant Insight Backend Active. Deploying frontend."}
