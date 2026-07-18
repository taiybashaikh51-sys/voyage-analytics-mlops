from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---------- Load all saved models/artifacts ----------
with open('../models/gender_model.pkl', 'rb') as f:
    gender_model = pickle.load(f)
with open('../models/gender_scaler.pkl', 'rb') as f:
    gender_scaler = pickle.load(f)
with open('../models/gender_label_encoder.pkl', 'rb') as f:
    gender_label_encoder = pickle.load(f)

with open('../models/flight_price_model.pkl', 'rb') as f:
    flight_model = pickle.load(f)
with open('../models/flight_price_scaler.pkl', 'rb') as f:
    flight_scaler = pickle.load(f)
with open('../models/flight_price_features.pkl', 'rb') as f:
    flight_features = pickle.load(f)

with open('../models/hotel_recommender.pkl', 'rb') as f:
    recommender_artifacts = pickle.load(f)

user_place_matrix = recommender_artifacts['user_place_matrix']
user_similarity_df = recommender_artifacts['user_similarity_df']
place_profile = recommender_artifacts['place_profile']
top_popular_places = recommender_artifacts['top_popular_places']


# ---------- Home route ----------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Voyage Analytics API is running",
        "endpoints": ["/predict-gender", "/predict-flight-price", "/recommend-hotel"]
    })


# ---------- 1. Gender Prediction ----------
@app.route('/predict-gender', methods=['POST'])
def predict_gender():
    data = request.get_json()
    # Expected input: age, total_flights, avg_flight_price, avg_distance, avg_hotel_price, avg_days, company_encoded
    features = np.array([[
        data['age'], data['total_flights'], data['avg_flight_price'],
        data['avg_distance'], data['avg_hotel_price'], data['avg_days'],
        data['company_encoded']
    ]])
    features_scaled = gender_scaler.transform(features)
    pred = gender_model.predict(features_scaled)
    gender_label = gender_label_encoder.inverse_transform(pred)[0]
    return jsonify({"predicted_gender": gender_label})


# ---------- 2. Flight Price Prediction ----------
@app.route('/predict-flight-price', methods=['POST'])
def predict_flight_price():
    data = request.get_json()
    # Build a DataFrame row matching the saved feature order (one-hot columns included)
    input_df = pd.DataFrame([data])
    # Reindex to match training feature columns, filling missing (e.g. missing dummy cols) with 0
    input_df = input_df.reindex(columns=flight_features, fill_value=0)
    input_scaled = flight_scaler.transform(input_df)
    pred_price = flight_model.predict(input_scaled)[0]
    return jsonify({"predicted_price": round(float(pred_price), 2)})


# ---------- 3. Hotel Recommendation ----------
@app.route('/recommend-hotel', methods=['POST'])
def recommend_hotel():
    data = request.get_json()
    user_code = int(data.get('user_code'))
    top_n = int(data.get('top_n', 5))

    if user_code not in user_similarity_df.index:
        recommendations = top_popular_places[:top_n]
    else:
        similar_users = user_similarity_df[user_code].sort_values(ascending=False)[1:11].index
        scores = user_place_matrix.loc[similar_users].sum().sort_values(ascending=False)
        recommendations = scores.head(top_n).index.tolist()

    return jsonify({"user_code": user_code, "recommended_places": recommendations})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)