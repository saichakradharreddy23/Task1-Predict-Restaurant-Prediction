"""
app.py
======
RESTAURANT RATING INTELLIGENCE
Interactive Machine Learning Web Application for Restaurant Aggregate Rating Prediction.
Developed with Streamlit, Plotly, Pandas, Scikit-learn, and Joblib.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src/ is on the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_loader import load_data, COUNTRY_MAP, inspect_dataset
from feature_engineering import RestaurantFeatureEngineer
from predict import predict_restaurant_rating, get_rating_category

# Set page configuration
st.set_page_config(
    page_title="Restaurant Rating Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Styling
st.markdown("""
<style>
    /* Global font & background enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .badge-success { background: #dcfce7; color: #15803d; }
    .badge-info { background: #e0f2fe; color: #0369a1; }
    .badge-purple { background: #f3e8ff; color: #7e22ce; }
    .badge-amber { background: #fef3c7; color: #b45309; }

    .prediction-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .prediction-rating {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #38bdf8;
    }
    .prediction-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1.5rem;
    }

    div[data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data & Artifact Caching
# -------------------------------------------------------------
@st.cache_data
def get_cached_raw_data():
    return load_data()

@st.cache_data
def get_cached_inspection(df):
    return inspect_dataset(df)

@st.cache_data
def get_cached_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'outputs', 'evaluation', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data
def get_cached_ranking():
    csv_path = os.path.join(os.path.dirname(__file__), 'outputs', 'feature_importance', 'feature_ranking.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

raw_df = get_cached_raw_data()
inspection = get_cached_inspection(raw_df)
metrics_data = get_cached_metrics()
ranking_df = get_cached_ranking()

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.markdown("### **RESTAURANT RATING**")
    st.caption("Machine Learning Intelligence Dashboard")
    st.divider()

    navigation = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "🔍 Dataset Explorer",
            "📊 Exploratory Data Analysis",
            "⚙️ Model Training",
            "📈 Model Evaluation",
            "💡 Feature Importance",
            "🔮 Restaurant Rating Prediction"
        ],
        index=0
    )

    st.divider()
    st.markdown("**Platform Status:**")
    st.markdown("🟢 **Model Engine:** Gradient Boosting (`R² = 0.9605`)")
    st.markdown(f"📦 **Records:** {len(raw_df):,} entries")
    st.markdown("⚡ **Pipeline:** Scaled & One-Hot Encoded")

# =============================================================
# PAGE 1: HOME DASHBOARD
# =============================================================
if navigation == "🏠 Home Dashboard":
    st.markdown('<div class="main-title">🍽️ Restaurant Rating Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Machine Learning System for Restaurant Performance & Aggregate Rating Prediction</div>', unsafe_allow_html=True)

    # Key Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Restaurants</div>
            <div class="metric-value">9,551</div>
            <span class="metric-badge badge-info">15 Global Countries</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Features</div>
            <div class="metric-value">21</div>
            <span class="metric-badge badge-purple">9 Selected for ML</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Average Rating</div>
            <div class="metric-value">2.67 / 5.0</div>
            <span class="metric-badge badge-amber">3.44 for Reviewed (n=7,403)</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        best_name = metrics_data.get('best_model', 'Gradient Boosting') if metrics_data else 'Gradient Boosting'
        best_r2 = metrics_data.get('models', {}).get(best_name, {}).get('Test_Metrics', {}).get('R2', 0.9605) if metrics_data else 0.9605
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best ML Model</div>
            <div class="metric-value">{best_r2:.4f} R²</div>
            <span class="metric-badge badge-success">{best_name}</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Project Overview & Architecture
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📌 Project Purpose & Problem Statement")
        st.markdown("""
        In the competitive food and hospitality sector, customer ratings are the single biggest determinant of footfall, revenue, and customer acquisition. 
        
        This enterprise machine learning platform analyzes restaurant operational factors—including **pricing, cuisine specialization, geographical market, reservation availability, online delivery**, and **accumulated review volume**—to predict and interpret a restaurant's **Aggregate Rating**.
        
        **Key Project Highlights:**
        - **100% Real Empirical Data:** Trained strictly on `data/Dataset .csv` (9,551 restaurants across 141 cities globally).
        - **Data Leakage Immunity:** Strict separation of test data with ColumnTransformer and custom Scikit-learn pipelines; removal of post-hoc ratings tags (`Rating color`, `Rating text`).
        - **Superior Predictive Accuracy:** Gradient Boosting and Random Forest achieve **>95.9% R²** with an average absolute prediction error of **~0.20 rating points**.
        - **Explainable AI:** Feature importance rankings and permutation testing to provide actionable insights for restaurateurs.
        """)

    with col_right:
        st.subheader("🏆 Model Leaderboard Highlights")
        if metrics_data and 'models' in metrics_data:
            leaderboard = []
            for m_name, m_info in metrics_data['models'].items():
                leaderboard.append({
                    'Algorithm': m_name,
                    'Test R²': f"{m_info['Test_Metrics']['R2']:.4f}",
                    'Test RMSE': f"{m_info['Test_Metrics']['RMSE']:.4f}",
                    'Test MAE': f"{m_info['Test_Metrics']['MAE']:.4f}"
                })
            st.dataframe(pd.DataFrame(leaderboard), hide_index=True, use_container_width=True)
        else:
            st.info("Train models to populate leaderboard.")

        st.subheader("📊 Dataset Health Check")
        st.markdown(f"""
        - **Duplicates:** {inspection['duplicate_records']} records detected
        - **Missing Records:** 9 entries in `Cuisines` (0.09%) imputed via mode
        - **Unrated Restaurants:** {inspection['target_stats']['zero_rating_count']:,} restaurants with 0.0 rating (low vote volume)
        """)

# =============================================================
# PAGE 2: DATASET EXPLORER
# =============================================================
elif navigation == "🔍 Dataset Explorer":
    st.markdown('<div class="main-title">🔍 Dataset Explorer & Inspections</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect raw and engineered restaurant records, summary statistics, and missing value profiles.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Filter & Search Records", "📊 Column Information & Dtypes", "🛠️ Preprocessed Feature Preview"])

    with tab1:
        st.subheader("Interactive Dataset Filter")
        f1, f2, f3, f4 = st.columns(4)

        countries = ['All'] + sorted(list(raw_df['Country Code'].map(COUNTRY_MAP).dropna().unique()))
        with f1:
            selected_country = st.selectbox("Filter Country", countries)
        with f2:
            price_filter = st.multiselect("Price Range Tier", [1, 2, 3, 4], default=[1, 2, 3, 4])
        with f3:
            min_rating, max_rating = st.slider("Aggregate Rating Range", 0.0, 5.0, (0.0, 5.0), step=0.1)
        with f4:
            search_query = st.text_input("Search Cuisine / City / Name", "")

        filtered_df = raw_df.copy()
        filtered_df['Country Name'] = filtered_df['Country Code'].map(COUNTRY_MAP).fillna('Other')

        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['Country Name'] == selected_country]
        if price_filter:
            filtered_df = filtered_df[filtered_df['Price range'].isin(price_filter)]
        filtered_df = filtered_df[(filtered_df['Aggregate rating'] >= min_rating) & (filtered_df['Aggregate rating'] <= max_rating)]
        if search_query:
            query = search_query.lower()
            filtered_df = filtered_df[
                filtered_df['Restaurant Name'].astype(str).str.lower().str.contains(query) |
                filtered_df['City'].astype(str).str.lower().str.contains(query) |
                filtered_df['Cuisines'].astype(str).str.lower().str.contains(query)
            ]

        st.caption(f"Showing **{len(filtered_df):,}** out of {len(raw_df):,} records:")
        display_cols = ['Restaurant ID', 'Restaurant Name', 'Country Name', 'City', 'Cuisines', 'Average Cost for two', 'Price range', 'Has Table booking', 'Has Online delivery', 'Aggregate rating', 'Votes']
        st.dataframe(filtered_df[display_cols].head(500), use_container_width=True)

    with tab2:
        st.subheader("Column Metadata & Missing Values")
        c_left, c_right = st.columns(2)
        with c_left:
            col_info = []
            for col in raw_df.columns:
                col_info.append({
                    'Column Name': col,
                    'Data Type': str(raw_df[col].dtype),
                    'Non-Null Count': int(raw_df[col].count()),
                    'Missing Count': int(raw_df[col].isnull().sum()),
                    'Unique Values': int(raw_df[col].nunique())
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)

        with c_right:
            st.markdown("#### Numerical Feature Descriptive Statistics")
            st.dataframe(raw_df[['Average Cost for two', 'Price range', 'Votes', 'Aggregate rating']].describe().T, use_container_width=True)

    with tab3:
        st.subheader("Engineered Features Sample")
        st.markdown("Features derived via `RestaurantFeatureEngineer`: `Primary Cuisine`, `Cuisine Count`, and binary numerical indicators.")
        fe = RestaurantFeatureEngineer()
        engineered_sample = fe.fit_transform(raw_df.head(100))
        st.dataframe(engineered_sample.head(20), use_container_width=True)

# =============================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
elif navigation == "📊 Exploratory Data Analysis":
    st.markdown('<div class="main-title">📊 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visualizing distributions, correlations, geographic benchmarks, and operational factors.</div>', unsafe_allow_html=True)

    eda_choice = st.selectbox(
        "Select EDA Dimension to Explore",
        [
            "1. Distribution of Aggregate Ratings",
            "2. Correlation Heatmap",
            "3. Average Rating by Country & City",
            "4. Price Range & Cost vs. Rating",
            "5. Votes vs. Rating Analysis",
            "6. Online Delivery & Table Booking Impact",
            "7. Top Cuisines by Rating",
            "8. Outlier Boxplots"
        ]
    )

    df_eda = raw_df.copy()
    df_eda['Country Name'] = df_eda['Country Code'].map(COUNTRY_MAP).fillna('Other')
    df_eda['Primary Cuisine'] = df_eda['Cuisines'].fillna('Unknown').apply(lambda s: [c.strip() for c in str(s).split(',')][0])

    if eda_choice == "1. Distribution of Aggregate Ratings":
        st.subheader("Distribution of Aggregate Restaurant Ratings")
        fig = px.histogram(
            df_eda, x="Aggregate rating", nbins=30, marginal="box",
            color_discrete_sequence=['#2563eb'], title="Aggregate Rating Histogram & Boxplot"
        )
        fig.add_vline(x=df_eda['Aggregate rating'].mean(), line_dash="dash", line_color="#ef4444", annotation_text=f"Mean: {df_eda['Aggregate rating'].mean():.2f}")
        fig.add_vline(x=df_eda['Aggregate rating'].median(), line_dash="dot", line_color="#10b981", annotation_text=f"Median: {df_eda['Aggregate rating'].median():.2f}")
        fig.update_layout(xaxis_title="Aggregate Rating (0.0 - 5.0)", yaxis_title="Restaurant Count", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Insight:** 2,148 restaurants (22.5%) are unrated with a score of 0.0, corresponding to newly registered venues with 0 to 3 votes. Rated restaurants form a normal distribution centered around 3.44.")

    elif eda_choice == "2. Correlation Heatmap":
        st.subheader("Feature Correlation Heatmap")
        corr_cols = ['Country Code', 'Average Cost for two', 'Price range', 'Votes', 'Aggregate rating']
        corr = df_eda[corr_cols].corr()

        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Blues", aspect="auto",
            title="Pearson Correlation Matrix (Numerical Attributes)"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Insight:** `Votes` (r = 0.31) and `Price range` (r = 0.44) display the strongest direct linear association with Aggregate Rating.")

    elif eda_choice == "3. Average Rating by Country & City":
        st.subheader("Geographical Rating Analysis")
        col1, col2 = st.columns(2)
        with col1:
            country_grp = df_eda.groupby('Country Name')['Aggregate rating'].mean().sort_values(ascending=False).reset_index()
            fig_country = px.bar(
                country_grp, x='Aggregate rating', y='Country Name', orientation='h',
                color='Aggregate rating', color_continuous_scale='Viridis',
                title="Average Rating by Country"
            )
            fig_country.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_country, use_container_width=True)

        with col2:
            top_cities = df_eda['City'].value_counts().head(15).index
            city_grp = df_eda[df_eda['City'].isin(top_cities)].groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).reset_index()
            fig_city = px.bar(
                city_grp, x='Aggregate rating', y='City', orientation='h',
                color='Aggregate rating', color_continuous_scale='Cividis',
                title="Average Rating Across Top 15 Cities by Volume"
            )
            fig_city.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_city, use_container_width=True)

    elif eda_choice == "4. Price Range & Cost vs. Rating":
        st.subheader("Price Tier & Cost for Two vs. Rating")
        col1, col2 = st.columns(2)
        with col1:
            fig_pr = px.box(
                df_eda, x="Price range", y="Aggregate rating", color="Price range",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Rating Distribution Across Price Range (1: Budget to 4: Luxury)"
            )
            fig_pr.update_layout(template="plotly_white")
            st.plotly_chart(fig_pr, use_container_width=True)

        with col2:
            cost_95 = df_eda['Average Cost for two'].quantile(0.95)
            filtered_cost = df_eda[df_eda['Average Cost for two'] <= cost_95]
            fig_cost = px.scatter(
                filtered_cost, x="Average Cost for two", y="Aggregate rating", opacity=0.3,
                trendline="ols", color_discrete_sequence=['#0d9488'],
                title=f"Average Cost for Two vs. Rating (<= 95th Percentile: {cost_95:,.0f})"
            )
            fig_cost.update_layout(template="plotly_white")
            st.plotly_chart(fig_cost, use_container_width=True)

    elif eda_choice == "5. Votes vs. Rating Analysis":
        st.subheader("Votes vs. Aggregate Rating Analysis")
        fig_votes = px.scatter(
            df_eda, x="Votes", y="Aggregate rating", opacity=0.4,
            trendline="ols", color_discrete_sequence=['#3b82f6'],
            title="Total Customer Votes vs. Aggregate Rating"
        )
        fig_votes.update_layout(template="plotly_white")
        st.plotly_chart(fig_votes, use_container_width=True)
        st.info("💡 **Insight:** Restaurants receiving higher vote counts reliably trend towards ratings of 3.8 to 4.8. Zero ratings only occur when votes are between 0 and 3.")

    elif eda_choice == "6. Online Delivery & Table Booking Impact":
        st.subheader("Service Conveniences vs. Rating")
        col1, col2 = st.columns(2)
        with col1:
            fig_del = px.box(
                df_eda, x="Has Online delivery", y="Aggregate rating", color="Has Online delivery",
                color_discrete_sequence=['#f97316', '#06b6d4'],
                title="Online Delivery Availability vs. Rating"
            )
            fig_del.update_layout(template="plotly_white")
            st.plotly_chart(fig_del, use_container_width=True)
        with col2:
            fig_tb = px.box(
                df_eda, x="Has Table booking", y="Aggregate rating", color="Has Table booking",
                color_discrete_sequence=['#64748b', '#ec4899'],
                title="Table Booking Availability vs. Rating"
            )
            fig_tb.update_layout(template="plotly_white")
            st.plotly_chart(fig_tb, use_container_width=True)

    elif eda_choice == "7. Top Cuisines by Rating":
        st.subheader("Top Cuisines Ranked by Mean Rating")
        cuisine_counts = df_eda['Primary Cuisine'].value_counts()
        valid_cuisines = cuisine_counts[cuisine_counts >= 20].index
        top_cuisines = df_eda[df_eda['Primary Cuisine'].isin(valid_cuisines)].groupby('Primary Cuisine')['Aggregate rating'].mean().sort_values(ascending=False).head(15).reset_index()
        fig_cuisines = px.bar(
            top_cuisines, x='Aggregate rating', y='Primary Cuisine', orientation='h',
            color='Aggregate rating', color_continuous_scale='Spectral',
            title="Top 15 Primary Cuisines with Minimum 20 Restaurants"
        )
        fig_cuisines.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_cuisines, use_container_width=True)

    elif eda_choice == "8. Outlier Boxplots":
        st.subheader("Numerical Attributes Outlier Inspection")
        col1, col2 = st.columns(2)
        with col1:
            fig_b1 = px.box(df_eda, y="Average Cost for two", log_y=True, title="Average Cost for Two (Log Scale)", color_discrete_sequence=['#14b8a6'])
            fig_b1.update_layout(template="plotly_white")
            st.plotly_chart(fig_b1, use_container_width=True)
        with col2:
            fig_b2 = px.box(df_eda, y="Votes", log_y=True, title="Votes (Log Scale)", color_discrete_sequence=['#f43f5e'])
            fig_b2.update_layout(template="plotly_white")
            st.plotly_chart(fig_b2, use_container_width=True)

# =============================================================
# PAGE 4: MODEL TRAINING
# =============================================================
elif navigation == "⚙️ Model Training":
    st.markdown('<div class="main-title">⚙️ Machine Learning Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review the 4 benchmarked regression algorithms, hyperparameter tuning grid, and 5-fold cross-validation.</div>', unsafe_allow_html=True)

    if metrics_data and 'models' in metrics_data:
        st.success("✅ **Models Successfully Trained and Verified:** All artifacts are persisted inside `models/`.")

        m_tabs = st.tabs(["Gradient Boosting (Champion)", "Random Forest", "Decision Tree", "Linear Regression"])

        for tab, (m_name, m_info) in zip(m_tabs, metrics_data['models'].items()):
            with tab:
                st.subheader(f"Model Architecture: {m_name}")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("5-Fold Cross-Validation R²", f"{m_info['CV_R2_Mean']:.4f}", f"±{m_info['CV_R2_Std']:.4f}")
                with col_b:
                    st.metric("Test R² Score", f"{m_info['Test_Metrics']['R2']:.4f}")
                with col_c:
                    st.metric("Test RMSE", f"{m_info['Test_Metrics']['RMSE']:.4f}")

                st.write("")
                st.markdown("#### Optimal Tuned Hyperparameters")
                st.json(m_info.get('Best_Hyperparameters', {'status': 'Default / Unconstrained'}))

                st.markdown("#### Training vs Testing Performance Breakdown")
                perf_comp = pd.DataFrame([
                    {'Split': 'Train Set (80%)', 'MAE': m_info['Train_Metrics']['MAE'], 'MSE': m_info['Train_Metrics']['MSE'], 'RMSE': m_info['Train_Metrics']['RMSE'], 'R²': m_info['Train_Metrics']['R2']},
                    {'Split': 'Test Set (20%)', 'MAE': m_info['Test_Metrics']['MAE'], 'MSE': m_info['Test_Metrics']['MSE'], 'RMSE': m_info['Test_Metrics']['RMSE'], 'R²': m_info['Test_Metrics']['R2']}
                ])
                st.table(perf_comp)
    else:
        st.warning("Model metrics file not found. Run `python src/train.py`.")

# =============================================================
# PAGE 5: MODEL EVALUATION
# =============================================================
elif navigation == "📈 Model Evaluation":
    st.markdown('<div class="main-title">📈 Model Evaluation & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comprehensive diagnostic evaluation: MAE, MSE, RMSE, R², residual distributions, and actual vs predicted.</div>', unsafe_allow_html=True)

    # Metric Cards for Best Model
    b_name = metrics_data.get('best_model', 'Gradient Boosting')
    b_test = metrics_data['models'][b_name]['Test_Metrics']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean Absolute Error (MAE)", f"{b_test['MAE']:.4f}", "Lower is Better", delta_color="inverse")
    with c2:
        st.metric("Mean Squared Error (MSE)", f"{b_test['MSE']:.4f}", "Lower is Better", delta_color="inverse")
    with c3:
        st.metric("Root Mean Squared Error (RMSE)", f"{b_test['RMSE']:.4f}", "Lower is Better", delta_color="inverse")
    with c4:
        st.metric("R-Squared (R²)", f"{b_test['R2']:.4f}", "Higher is Better")

    st.write("")
    st.subheader("Model Comparison Leaderboard")
    eval_csv = os.path.join(os.path.dirname(__file__), 'outputs', 'evaluation', 'metrics_summary.csv')
    if os.path.exists(eval_csv):
        st.dataframe(pd.read_csv(eval_csv), use_container_width=True)

    st.write("")
    st.subheader("Visual Diagnostics Gallery")
    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
        "Actual vs Predicted",
        "Residual Distribution",
        "Homoscedasticity (Residuals vs Predicted)",
        "Overfitting Analysis (Train vs Test)"
    ])

    with diag_tab1:
        st.image(os.path.join("outputs", "evaluation", "02_actual_vs_predicted.png"), use_container_width=True)
    with diag_tab2:
        st.image(os.path.join("outputs", "evaluation", "03_residual_distribution.png"), use_container_width=True)
    with diag_tab3:
        st.image(os.path.join("outputs", "evaluation", "04_residuals_vs_predicted.png"), use_container_width=True)
    with diag_tab4:
        st.image(os.path.join("outputs", "evaluation", "05_train_vs_test_performance.png"), use_container_width=True)

# =============================================================
# PAGE 6: FEATURE IMPORTANCE
# =============================================================
elif navigation == "💡 Feature Importance":
    st.markdown('<div class="main-title">💡 Feature Importance & Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Discover the operational and market factors driving restaurant ratings.</div>', unsafe_allow_html=True)

    if ranking_df is not None:
        top10 = ranking_df.head(10).copy()

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.subheader("Top 10 Influential Features (Gradient Boosting)")
            fig_imp = px.bar(
                top10.sort_values(by='Importance_GB', ascending=True),
                x='Importance_GB', y='Feature', orientation='h',
                color='Importance_GB', color_continuous_scale='Blues',
                labels={'Importance_GB': 'Relative Importance (MDI)'}
            )
            fig_imp.update_layout(template="plotly_white")
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_r:
            st.subheader("Top Influential Feature Ranking")
            st.dataframe(top10[['Rank', 'Feature', 'Importance_GB', 'Permutation_Importance_Mean']], use_container_width=True)

        st.subheader("Business Interpretation & Strategic Takeaways")
        st.markdown("""
        > [!IMPORTANT]
        > **Distinction Between Correlation and Causality:** While high review volume and higher price tiers associate with higher ratings in this empirical dataset, merely inflating prices or generating reviews artificially does not cause rating growth. True rating growth is driven by dining experience, food consistency, and service quality.
        
        1. **Review Count / Customer Engagement (`Votes`):**
           - Accounts for the overwhelming majority of predictive signal (~97% MDI).
           - High votes validate credibility and customer engagement, separating active restaurants from unrated venues.
        2. **Affordability & Price Tier (`Price range` & `Average Cost for two`):**
           - Higher price ranges consistently correlate with higher aggregate ratings. Premium venues invest in ambiance, culinary expertise, and customer care.
        3. **Convenience Services (`Has Online delivery` & `Has Table booking`):**
           - Providing table reservations and online food ordering improves accessibility and convenience for diners.
        4. **Cuisine & Geographical Benchmark (`Primary Cuisine` & `City`):**
           - Baseline ratings vary according to local dining culture and the degree of regional restaurant competition.
        """)

        st.write("")
        st.subheader("Feature Importance Comparison: Gradient Boosting vs Random Forest")
        st.image(os.path.join("outputs", "feature_importance", "03_model_feature_comparison.png"), use_container_width=True)

# =============================================================
# PAGE 7: RESTAURANT RATING PREDICTION
# =============================================================
elif navigation == "🔮 Restaurant Rating Prediction":
    st.markdown('<div class="main-title">🔮 Predict Restaurant Aggregate Rating</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your restaurant parameters to predict expected aggregate customer rating using trained regression models.</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("🏢 Restaurant Profile & Operational Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            country_options = list(COUNTRY_MAP.values())
            selected_country = st.selectbox("Country", country_options, index=0)
        with c2:
            top_cities = [
                'New Delhi', 'Gurgaon', 'Noida', 'Faridabad', 'Ghaziabad',
                'Ahmedabad', 'Lucknow', 'Bhubaneshwar', 'Amritsar', 'Guwahati',
                'London', 'Manchester', 'New York City', 'Chicago', 'Dubai', 'Other'
            ]
            city_input = st.selectbox("City", top_cities, index=0)
        with c3:
            locality_input = st.text_input("Locality / Neighborhood", "Connaught Place")

        st.subheader("🍽️ Cuisine Specialization & Pricing")
        c4, c5, c6 = st.columns(3)
        with c4:
            common_cuisines = [
                "North Indian", "Chinese", "Fast Food", "Mughlai", "Cafe",
                "Bakery", "Desserts", "South Indian", "Italian", "Street Food",
                "Continental", "Asian", "Japanese", "Pizza", "Burger"
            ]
            cuisines_input = st.multiselect("Cuisines Offered", common_cuisines, default=["North Indian", "Chinese"])
        with c5:
            avg_cost = st.number_input("Average Cost for Two (Local Currency)", min_value=10, max_value=200000, value=800, step=50)
        with c6:
            price_tier = st.select_slider(
                "Price Range Tier",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {1: "1: Budget ($)", 2: "2: Moderate ($$)", 3: "3: Expensive ($$$)", 4: "4: Luxury ($$$$)"}[x]
            )

        st.subheader("🚀 Services & Customer Engagement")
        c7, c8, c9 = st.columns(3)
        with c7:
            table_booking = st.radio("Table Booking Availability?", ["No", "Yes"], index=0, horizontal=True)
        with c8:
            online_delivery = st.radio("Online Delivery Availability?", ["Yes", "No"], index=0, horizontal=True)
        with c9:
            votes_input = st.number_input("Estimated Review / Vote Count", min_value=0, max_value=20000, value=120, step=10)

        st.subheader("🧠 Select Predictive Machine Learning Model")
        model_selection = st.selectbox(
            "Regression Algorithm",
            [
                ("Gradient Boosting (Best Performer - Recommended)", "best_model"),
                ("Random Forest Regressor", "random_forest"),
                ("Decision Tree Regressor", "decision_tree"),
                ("Linear Regression", "linear_regression")
            ],
            format_func=lambda x: x[0]
        )

        submit_btn = st.form_submit_button("🚀 Predict Aggregate Rating", use_container_width=True)

    if submit_btn:
        cuisines_str = ", ".join(cuisines_input) if cuisines_input else "North Indian"
        features_payload = {
            'Country': selected_country,
            'City': city_input,
            'Locality': locality_input,
            'Cuisines': cuisines_str,
            'Average Cost for two': avg_cost,
            'Price range': price_tier,
            'Has Table booking': table_booking,
            'Has Online delivery': online_delivery,
            'Votes': votes_input
        }

        with st.spinner("Calculating prediction via preprocessing pipeline and machine learning model..."):
            try:
                res = predict_restaurant_rating(features_payload, model_name=model_selection[1])
                pred_rating = res['predicted_rating']
                rating_text = res['rating_text']
                rating_color = res['rating_color']

                st.markdown(f"""
                <div class="prediction-container">
                    <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 0.5rem;">Predicted Aggregate Rating</div>
                    <div class="prediction-rating">{pred_rating:.2f} <span style="font-size: 2rem; color: #94a3b8;">/ 5.0</span></div>
                    <div class="prediction-badge" style="background-color: {rating_color}; color: #ffffff;">{rating_text}</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1;">Model Employed: <strong>{model_selection[0]}</strong></div>
                </div>
                """, unsafe_allow_html=True)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    # Circular Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred_rating,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Aggregate Rating Meter", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': rating_color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 2.5], 'color': '#fee2e2'},
                                {'range': [2.5, 3.5], 'color': '#fef3c7'},
                                {'range': [3.5, 4.0], 'color': '#dcfce7'},
                                {'range': [4.0, 5.0], 'color': '#bbf7d0'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': pred_rating
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_res2:
                    st.subheader("📋 Operational Summary & Recommendations")
                    st.markdown(f"""
                    - **Market Context:** `{city_input}, {selected_country}`
                    - **Cuisines:** `{cuisines_str}` ({len(cuisines_input)} styles)
                    - **Price Positioning:** Tier `{price_tier}` (Avg Cost: `{avg_cost:,.0f}`)
                    - **Digital Enablement:** Online Delivery: `{online_delivery}` | Table Booking: `{table_booking}`
                    - **Customer Feedback Volume:** `{votes_input:,}` votes
                    """)
                    
                    if votes_input < 50:
                        st.warning("⚠️ **Review Volume Opportunity:** Increasing authentic verified reviews through table-top QR codes or delivery inserts can help establish a validated rating score.")
                    if online_delivery == "No":
                        st.info("💡 **Digital Delivery:** Enabling online delivery expands addressable customer reach and review velocity.")
                    if table_booking == "No" and price_tier >= 3:
                        st.info("💡 **Reservation Convenience:** High-tier restaurants benefit from online table booking to boost customer satisfaction.")

                st.markdown(f'<div class="disclaimer-box">⚠️ {res["disclaimer"]}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
