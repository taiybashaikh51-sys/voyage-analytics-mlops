"""
Voyage Analytics — Project Showcase App
A single interactive Streamlit dashboard that demos all 3 ML models
by calling the LIVE deployed Flask API. Built for interview demos.
"""

import streamlit as st
import requests

API_BASE = "https://voyage-analytics-api-uc0p.onrender.com"

st.set_page_config(page_title="Voyage Analytics — Project Showcase", page_icon="🌍", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🌍 Voyage Analytics")
    st.caption("Integrating MLOps in Travel")
    st.markdown("---")
    st.markdown("**Live API**")
    st.code(API_BASE, language=None)
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("`Flask` · `Docker` · `GitHub Actions` · `MLFlow` · `Render`")
    st.markdown("---")
    st.markdown("[📂 GitHub Repo](https://github.com/taiybashaikh51-sys/voyage-analytics-mlops)")

st.title("🌍 Voyage Analytics — Project Showcase")
st.caption("Every prediction below is a real call to the live, Dockerized, deployed API — not a local mock.")

tab_overview, tab_gender, tab_price, tab_hotel, tab_arch = st.tabs(
    ["📖 Overview", "🧑 Gender Prediction", "✈️ Flight Price", "🏨 Hotel Recommendation", "⚙️ Architecture"]
)

# ---------------- Overview Tab ----------------
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Models Built", "3", "Classification · Regression · Recommendation")
    col2.metric("Flight Price R²", "0.843", "Tuned XGBoost")
    col3.metric("Deployment", "Live ✅", "Docker on Render")

    st.markdown("---")
    st.subheader("What this project does")
    st.markdown(
        """
        Three real travel datasets — **users**, **flights**, and **hotels** — power three ML problems,
        all served through a single production-grade pipeline:

        1. **Gender Prediction** *(Classification)* — tests whether a user's gender can be inferred from travel behavior
        2. **Flight Price Prediction** *(Regression)* — predicts ticket price from route, class, and agency
        3. **Hotel Recommendation** *(Collaborative Filtering)* — suggests destinations based on similar travelers

        Every model is wrapped in a **Flask REST API**, **containerized with Docker**, **tested automatically via
        GitHub Actions CI/CD**, and **deployed live on Render** — this app talks to that exact live API.
        """
    )

    st.markdown("---")
    st.subheader("Key EDA Insight")
    st.info(
        "Age, flight spend, and hotel spend are almost **identically distributed** across gender groups — "
        "which is *why* the gender model deliberately reports a near-baseline result rather than a forced 'good' number."
    )

# ---------------- Gender Prediction Tab ----------------
with tab_gender:
    st.subheader("🧑 Gender Prediction (Classification)")
    st.caption("Honest result: ~35% accuracy — at the random-guess baseline for 3 balanced classes. This is a genuine finding, not a bug.")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 70, 30)
        total_flights = st.number_input("Total flights taken", 0, 500, 150)
        avg_flight_price = st.number_input("Average flight price", 0.0, 2000.0, 900.0)
    with c2:
        avg_distance = st.number_input("Average distance (km)", 0.0, 1000.0, 500.0)
        avg_hotel_price = st.number_input("Average hotel price/night", 0.0, 1500.0, 600.0)
        avg_days = st.number_input("Average stay (days)", 0.0, 10.0, 2.5)

    if st.button("Predict Gender", type="primary"):
        payload = {
            "age": age, "total_flights": total_flights, "avg_flight_price": avg_flight_price,
            "avg_distance": avg_distance, "avg_hotel_price": avg_hotel_price,
            "avg_days": avg_days, "company_encoded": 0
        }
        try:
            with st.spinner("Calling live API... (first request may take ~30s if the server is waking up)"):
                r = requests.post(f"{API_BASE}/predict-gender", json=payload, timeout=60)
                r.raise_for_status()
                result = r.json()
            st.success(f"Predicted gender: **{result['predicted_gender']}**")
            st.caption("Remember: this prediction is expected to be near-random — that's the documented finding.")
        except Exception as e:
            st.error(f"API call failed: {e}")

# ---------------- Flight Price Tab ----------------
with tab_price:
    st.subheader("✈️ Flight Price Prediction (Regression)")
    st.caption("Tuned XGBoost — RMSE ≈ 143.8, R² ≈ 0.843")

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
            with st.spinner("Calling live API..."):
                r = requests.post(f"{API_BASE}/predict-flight-price", json=payload, timeout=60)
                r.raise_for_status()
                result = r.json()
            st.success(f"Predicted price: **₹{result['predicted_price']:.2f}**")
        except Exception as e:
            st.error(f"API call failed: {e}")

# ---------------- Hotel Recommendation Tab ----------------
with tab_hotel:
    st.subheader("🏨 Hotel Recommendation (Collaborative Filtering)")
    st.caption("Try user codes 0-1339 — recommendations are personalized per user via similarity, with a popularity fallback for unknown users.")

    c1, c2 = st.columns(2)
    with c1:
        user_code = st.number_input("User Code", min_value=0, max_value=5000, value=8, step=1)
    with c2:
        top_n = st.slider("Number of recommendations", 1, 5, 3)

    if st.button("Get Recommendations", type="primary"):
        payload = {"user_code": int(user_code), "top_n": int(top_n)}
        try:
            with st.spinner("Calling live API..."):
                r = requests.post(f"{API_BASE}/recommend-hotel", json=payload, timeout=60)
                r.raise_for_status()
                result = r.json()
            st.success("Recommended places:")
            for i, place in enumerate(result["recommended_places"], 1):
                st.markdown(f"**{i}.** {place}")
        except Exception as e:
            st.error(f"API call failed: {e}")

# ---------------- Architecture Tab ----------------
with tab_arch:
    st.subheader("⚙️ End-to-End MLOps Architecture")
    st.markdown(
        """
        ```
         Data (CSV)        EDA & Feature      Model Training      Flask REST API
        users/flights/  →   Engineering    →   (3 notebooks)   →   (3 endpoints)
         hotels                                                          │
                                                                          ▼
                             Cloud Deploy   ←     CI/CD        ←      Docker
                              (Render)      (GitHub Actions)      (containerized)
        ```
        """
    )
    st.markdown("#### Pipeline stages")
    st.table({
        "Stage": ["Model Serving", "Experiment Tracking", "Containerization", "CI/CD", "Deployment"],
        "Tool": ["Flask", "MLFlow", "Docker", "GitHub Actions", "Render"],
        "Purpose": [
            "REST endpoints for all 3 models",
            "Logs hyperparameters & metrics",
            "Packages API + models into one image",
            "Auto-builds & health-checks on every push",
            "Hosts the live, public API",
        ]
    })
    st.markdown("This dashboard itself is calling the **live** endpoint above — not a local copy.")