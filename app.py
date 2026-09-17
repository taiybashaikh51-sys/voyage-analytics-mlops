"""
Voyage Analytics — All-in-One Local Streamlit App
Runs all 3 ML models directly from the saved .pkl files in /models
(no Flask API, no internet dependency, no deployment needed).

Place this file inside the `app/` folder of the voyage-analytics-mlops
repo (next to app.py / showcase_app.py) and run:

    streamlit run streamlit_app.py

It expects the following files one level up, in ../models/:
    gender_model.pkl, gender_scaler.pkl, gender_label_encoder.pkl
    flight_price_model.pkl, flight_price_scaler.pkl, flight_price_features.pkl
    hotel_recommender.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (robust regardless of the folder you launch `streamlit run` from)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_models_dir():
    """Locate the `models/` folder regardless of whether this script sits in
    the repo root or inside `app/` — checks the script's own folder, its
    parent, and its parent's parent."""
    candidates = [
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "..", "models"),
        os.path.join(BASE_DIR, "..", "..", "models"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "gender_model.pkl")):
            return c
    # Nothing matched — fall back to the first candidate so the error
    # message below still shows a sensible path.
    return candidates[0]


MODELS_DIR = _find_models_dir()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    def _load(name):
        with open(os.path.join(MODELS_DIR, name), "rb") as f:
            return pickle.load(f)

    artifacts = {
        "gender_model": _load("gender_model.pkl"),
        "gender_scaler": _load("gender_scaler.pkl"),
        "gender_label_encoder": _load("gender_label_encoder.pkl"),
        "flight_model": _load("flight_price_model.pkl"),
        "flight_scaler": _load("flight_price_scaler.pkl"),
        "flight_features": _load("flight_price_features.pkl"),
        "hotel_artifacts": _load("hotel_recommender.pkl"),
    }
    return artifacts


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Voyage Analytics — All-in-One",
    page_icon="🌍",
    layout="wide",
)

try:
    artifacts = load_artifacts()
    load_error = None
except Exception as e:  # missing files, version mismatch, etc.
    artifacts = None
    load_error = e

with st.sidebar:
    st.title("🌍 Voyage Analytics")
    st.caption("Gender · Flight Price · Hotel Recommendation")
    st.markdown("---")
    st.markdown("**Mode:** fully local — models are loaded straight from `../models/*.pkl`, no API call.")
    st.markdown("---")
    if artifacts is not None:
        st.success("Models loaded ✅")
    else:
        st.error("Models failed to load ❌")

st.title("🌍 Voyage Analytics — All-in-One Dashboard")
st.caption("Every prediction below runs the trained model locally, in this process.")

if load_error is not None:
    st.error(
        "Could not load the model files. Make sure this script sits in the `app/` folder "
        "of the repo (next to `app.py`) so that `../models/` resolves correctly, and that "
        f"scikit-learn/xgboost versions match `requirements.txt`.\n\nDetails: {load_error}"
    )
    st.stop()

tab_overview, tab_gender, tab_price, tab_hotel = st.tabs(
    ["📖 Overview", "🧑 Gender Prediction", "✈️ Flight Price", "🏨 Hotel Recommendation"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Models", "3", "Classification · Regression · Recommendation")
    col2.metric("Mode", "Local", "No API / no internet needed")
    col3.metric("Data source", "3 CSVs", "users · flights · hotels")

    st.markdown("---")
    st.subheader("What this app does")
    st.markdown(
        """
        1. **Gender Prediction** *(Classification)* — infers a user's gender from travel behavior.
        2. **Flight Price Prediction** *(Regression)* — predicts ticket price from route, class, and agency.
        3. **Hotel Recommendation** *(Collaborative Filtering)* — suggests destinations based on similar travelers.

        This version loads the saved model/scaler/encoder `.pkl` files directly with `pickle`
        and runs inference in-process — the same logic as the Flask API in `api/app.py`,
        just without needing the API running.
        """
    )

# ---------------------------------------------------------------------------
# Gender Prediction
# ---------------------------------------------------------------------------
with tab_gender:
    st.subheader("🧑 Gender Prediction")
    st.caption("Inputs: age, total flights, avg flight price, avg distance, avg hotel price, avg stay, company (encoded).")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 70, 30)
        total_flights = st.number_input("Total flights taken", 0, 500, 150)
        avg_flight_price = st.number_input("Average flight price", 0.0, 2000.0, 900.0)
    with c2:
        avg_distance = st.number_input("Average distance (km)", 0.0, 1000.0, 500.0)
        avg_hotel_price = st.number_input("Average hotel price/night", 0.0, 1500.0, 600.0)
        avg_days = st.number_input("Average stay (days)", 0.0, 10.0, 2.5)

    company_encoded = st.number_input(
        "Company (encoded)", min_value=0, max_value=50, value=0, step=1,
        help="Same integer encoding used at training time for the 'company' column.",
    )

    if st.button("Predict Gender", type="primary"):
        features = np.array([[
            age, total_flights, avg_flight_price,
            avg_distance, avg_hotel_price, avg_days,
            company_encoded,
        ]])
        try:
            features_scaled = artifacts["gender_scaler"].transform(features)
            pred = artifacts["gender_model"].predict(features_scaled)
            gender_label = artifacts["gender_label_encoder"].inverse_transform(pred)[0]
            st.success(f"Predicted gender: **{gender_label}**")
            st.caption("Documented finding: this model performs near the random-guess baseline for 3 balanced classes.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Flight Price Prediction
# ---------------------------------------------------------------------------
with tab_price:
    st.subheader("✈️ Flight Price Prediction")
    st.caption("Tuned XGBoost regressor.")

    c1, c2 = st.columns(2)
    with c1:
        distance = st.slider("Distance (km)", 100, 1000, 500)
        flight_type = st.selectbox("Flight Type", ["economic", "premium", "firstClass"])
    with c2:
        agency = st.selectbox("Agency", ["CloudFy", "FlyingDrops", "Rainbow"])

    if st.button("Predict Flight Price", type="primary"):
        payload = {
            "distance": distance,
            "flightType_firstClass": 1 if flight_type == "firstClass" else 0,
            "flightType_premium": 1 if flight_type == "premium" else 0,
            "agency_FlyingDrops": 1 if agency == "FlyingDrops" else 0,
            "agency_Rainbow": 1 if agency == "Rainbow" else 0,
        }
        try:
            input_df = pd.DataFrame([payload])
            input_df = input_df.reindex(columns=artifacts["flight_features"], fill_value=0)
            input_scaled = artifacts["flight_scaler"].transform(input_df)
            pred_price = artifacts["flight_model"].predict(input_scaled)[0]
            st.success(f"Predicted price: **₹{float(pred_price):.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Hotel Recommendation
# ---------------------------------------------------------------------------
with tab_hotel:
    st.subheader("🏨 Hotel Recommendation")
    st.caption("Collaborative filtering with a popularity fallback for unknown users.")

    hotel = artifacts["hotel_artifacts"]
    user_place_matrix = hotel["user_place_matrix"]
    user_similarity_df = hotel["user_similarity_df"]
    place_profile = hotel["place_profile"]
    top_popular_places = hotel["top_popular_places"]

    c1, c2 = st.columns(2)
    with c1:
        user_code = st.number_input("User Code", min_value=0, max_value=5000, value=8, step=1)
    with c2:
        top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Get Recommendations", type="primary"):
        try:
            if int(user_code) not in user_similarity_df.index:
                recommendations = top_popular_places[:top_n]
                st.info("Unknown user code — showing popularity-based fallback recommendations.")
            else:
                similar_users = user_similarity_df[int(user_code)].sort_values(ascending=False)[1:11].index
                scores = user_place_matrix.loc[similar_users].sum().sort_values(ascending=False)
                recommendations = scores.head(top_n).index.tolist()

            st.success("Recommended places:")
            for i, place in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {place}")

            st.subheader("Place Details")
            details = place_profile[place_profile["place"].isin(recommendations)]
            st.dataframe(details, use_container_width=True)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")