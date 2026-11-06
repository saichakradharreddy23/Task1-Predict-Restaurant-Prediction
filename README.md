# RESTAURANT INSIGHT – AI POWERED RESTAURANT RATING PREDICTION
### Premium Restaurant Discovery Platform & Machine Learning Rating Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest%20Regressor-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Luxury%20Editorial-38bdf8.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

**RESTAURANT INSIGHT** is a complete, end-to-end Machine Learning web application and luxury restaurant discovery platform designed to predict and analyze the aggregate customer ratings of restaurants based on their operational, geographical, and dining characteristics.

The machine learning system is powered **EXCLUSIVELY by ONE regression algorithm**:
- **Algorithm:** **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`)
- **Dataset:** 9,551 real-world restaurant records across 15 countries and 141 cities from `data/Dataset .csv`.
- **Target Variable:** `Aggregate rating` (scale of `0.0` to `5.0`).

---

## 2. Machine Learning Architecture & Preprocessing

### Leak-Free Preprocessing Pipeline
- **Missing Value Imputation:**
  - Categorical columns (e.g. 9 missing values in `Cuisines`): Mode imputation (`North Indian`).
  - Numerical columns: Median imputation via `SimpleImputer(strategy='median')`.
- **Feature Engineering:**
  - `Primary Cuisine`: Extracted from the primary specialty in comma-separated cuisine tags.
  - `Cuisine Count`: Number of cuisines offered by the restaurant.
  - `Has Table booking` & `Has Online delivery`: Mapped to binary numerical indicators (1/0).
  - High-cardinality management: Top frequent categories for `City` (top 20), `Locality` (top 30), and `Primary Cuisine` (top 25) are preserved; rare categories are grouped into `'Other'` to prevent dimensional explosion.
- **Scaling & Encoding:**
  - Numerical features (`Average Cost for two`, `Price range`, `Votes`, `Cuisine Count`) scaled via `StandardScaler()`.
  - Categorical features (`Country Code`, `City`, `Locality`, `Primary Cuisine`) encoded via `OneHotEncoder(handle_unknown='ignore')`.
  - ColumnTransformer & Feature Pipeline fitted **strictly on the 80% training data** ($n = 7,640$) to prevent any data leakage.
- **Train-Test Split:**
  - 80% Training ($n = 7,640$).
  - 20% Testing ($n = 1,911$).
  - `random_state = 42` for exact reproducibility.

---

## 3. Dedicated Machine Learning Model: Random Forest Regressor

Trained exclusively using **Random Forest Regressor** with systematic cross-validation:

### Hyperparameters
- `n_estimators`: 150 decision trees
- `max_depth`: 16
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `random_state`: 42
- `n_jobs`: -1

### Actual Model Evaluation Results (Testing Dataset)

| Metric | Calculated Test Value | Interpretation |
| :--- | :---: | :--- |
| **R-squared (R²)** | **0.9592** | The Random Forest model explains **95.92%** of the variance in restaurant aggregate ratings. |
| **Root Mean Squared Error (RMSE)** | **0.3047** | On average, predictions deviate by approximately **±0.30 rating points** on a 5-point scale. |
| **Mean Absolute Error (MAE)** | **0.1991** | The average absolute difference between predicted and actual customer ratings is only **~0.20**. |
| **Mean Squared Error (MSE)** | **0.0928** | Low quadratic error on the held-out test set. |
| **5-Fold Cross Validation R²** | **0.9576** ($\pm 0.0026$) | High stability across all 5 cross-validation folds. |

### Overfitting & Generalization Analysis
- **Training R²:** `0.9798` (Training RMSE: `0.2156`, Training MAE: `0.1382`)
- **Testing R²:** `0.9592` (Testing RMSE: `0.3047`, Testing MAE: `0.1991`)
- **Generalization Gap:** Minimal delta of $\approx 0.02$ between training and testing scores confirms that pruning (`max_depth=16`, `min_samples_leaf=2`) successfully prevented overfitting.

---

## 4. Top 10 Feature Importance (Random Forest MDI & Permutation)

| Rank | Feature Name | MDI Importance (%) | Permutation Importance (Test R² Drop) | Strategic Business Role |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `Votes` | **96.42%** | **1.8799** | **Review Volume:** Primary credibility anchor; venues with 0-3 votes account for unrated 0.0 entries. |
| **2** | `City: Other` | **0.97%** | **0.0083** | **Regional Market:** Captures dining variance outside primary metro hubs. |
| **3** | `Average Cost for two` | **0.71%** | **0.0056** | **Spending Tier:** Correlates with premium ingredient sourcing and ambiance. |
| **4** | `Cuisine Count` | **0.23%** | **0.0002** | **Menu Breadth:** Number of culinary specialties offered. |
| **5** | `Primary Cuisine: North Indian` | **0.13%** | **0.0027** | **Culinary Specialty:** High market density cuisine. |
| **6** | `Price range` | **0.13%** | **0.0007** | **Affordability Level:** Higher tiers maintain higher median customer ratings. |
| **7** | `Has Online delivery` | **0.11%** | **0.0008** | **Digital Accessibility:** Online food ordering expands customer reach. |
| **8** | `Country Code: 1 (India)` | **0.09%** | **0.0010** | **Geographic Baseline:** Dominant country cluster in dataset. |
| **9** | `Primary Cuisine: Chinese` | **0.08%** | **0.0006** | **Popular Specialty:** Popular international cuisine. |
| **10** | `Locality: Other` | **0.07%** | **0.0002** | **Neighborhood Effect:** Specific neighborhood dining culture. |

> **Predictive Association vs. Causation:** Feature importance indicates predictive statistical strength in historical data, not a direct causal guarantee. Artificially increasing menu prices or inflating votes without culinary and service excellence will not improve guest satisfaction.

---

## 5. Website Pages & Premium Restaurant Experience

The frontend is crafted with a luxury editorial aesthetic (warm cream, deep charcoal, terracotta, muted gold, soft white) with 7 distinct views:

1. **🏠 Home / Landing Page:**
   - Large hero section with ambient dining photography.
   - Headline: *"Discover the Taste. Predict the Experience."*
   - Interactive CTA buttons: *Explore Restaurants* and *Predict Rating*.
   - Featured top-rated restaurant cards.
   - Popular cuisines interactive visual grid.
   - Real dataset statistics (9,551 venues, 141 cities, 3.44 avg star rating).
2. **🔍 Restaurant Explorer:**
   - Authentic restaurant cards with cuisine-matching imagery, city, price tier, and ratings.
   - Real-time search by name, locality, or cuisine.
   - Filter by City, Cuisine, Price Tier (Budget to Luxury), and Minimum Rating.
   - Sort by highest rating, lowest cost, or most votes.
   - Pagination controls.
3. **📋 Restaurant Details Profile:**
   - Modal view with high-res photo, aggregate rating gauge, verified votes, price information, and service convenience badges (delivery and table booking).
4. **🔮 AI Restaurant Rating Prediction:**
   - Interactive form collecting Country, City, Locality, Cuisines, Avg Cost for two, Price Range (1-4), Table Booking, Online Delivery, and Votes.
   - Input validation and safe unseen category handling.
   - Displays predicted rating on a 0 to 5 scale, color badge (Excellent, Very Good, Good, Average, Poor), visual rating meter, operational recommendations, and disclaimer.
5. **📈 Model Performance:**
   - Displays actual Random Forest metrics: MSE (`0.0928`), RMSE (`0.3047`), MAE (`0.1991`), R² (`0.9592`).
   - 4 diagnostic charts: Actual vs Predicted, Residual Distribution, Residuals vs Predicted, and Absolute Error Distribution.
   - Training vs Testing generalization comparison.
6. **💡 Feature Importance:**
   - Top 10 influential features horizontal bar chart and ranking table.
   - Business insights on review velocity, spending tiers, and digital convenience.
7. **ℹ️ About Project:**
   - Project mission, dataset description, machine learning workflow, and Random Forest architecture documentation.

---

## 6. Project Structure

```
Restaurant_Rating_Prediction/
│
├── data/
│   └── Dataset .csv                   # Empirical dataset (9,551 records)
│
├── backend/
│   ├── main.py                        # FastAPI production application & REST endpoints
│   ├── data_loader.py                 # Dataset loading & structural validation
│   ├── preprocessing.py               # Preprocessing pipeline & 80/20 train-test split
│   ├── train.py                       # Exclusively trains Random Forest Regressor
│   ├── evaluate.py                    # Evaluates Random Forest & generates diagnostics
│   ├── feature_importance.py          # Top 10 feature importance & ranking
│   └── predict.py                     # Inference engine with input validation & advice
│
├── frontend/
│   └── index.html                     # Luxury responsive frontend (Tailwind CSS, Alpine.js)
│
├── models/
│   ├── random_forest_model.pkl        # Trained Random Forest Regressor artifact
│   └── preprocessing_pipeline.pkl     # Fitted ColumnTransformer pipeline
│
├── outputs/
│   ├── eda/                           # 12 EDA diagnostic plots
│   ├── evaluation/                    # 5 Random Forest evaluation charts & metrics JSON
│   └── feature_importance/            # 1 Feature importance chart, ranking CSV & JSON
│
├── requirements.txt                   # All dependencies
└── README.md                          # Full project documentation
```

---

## 7. How to Run the Application on Localhost

### Step 1: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend & Luxury Website
Run the following command from the project root directory:
```powershell
python -m uvicorn backend.main:app --port 8000 --reload
```

### Step 3: Open in Browser
Visit the application in your web browser:
👉 **[http://localhost:8000](http://localhost:8000)**

*(The API documentation is also available at `http://localhost:8000/docs`)*

---

## 8. CLI Execution of Independent Modules

You can also run any backend module directly:
- **Inspect Dataset:** `python backend/data_loader.py`
- **Re-fit Preprocessing Pipeline:** `python backend/preprocessing.py`
- **Train Random Forest Regressor:** `python backend/train.py`
- **Run Model Evaluation & Charts:** `python backend/evaluate.py`
- **Compute Feature Importance:** `python backend/feature_importance.py`
- **Test CLI Prediction Inference:** `python backend/predict.py`
