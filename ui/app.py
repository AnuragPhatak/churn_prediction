import streamlit as st
import requests

st.title("📊 Customer Churn Prediction")

st.write("Provide customer details to check churn likelihood:")

# Input fields (only the 5 features)
recency = st.number_input("Recency (days since last activity)", min_value=0.0, value=3.0)
frequency = st.number_input("Frequency (number of interactions)", min_value=0.0, value=5.0)
engagement_duration = st.number_input("Engagement Duration (minutes)", min_value=0.0, value=120.0)
inactivity_streak = st.number_input("Inactivity Streak (days)", min_value=0.0, value=2.0)
engagement_per_interaction = st.number_input("Engagement per Interaction", min_value=0.0, value=24.0)

# Prepare payload
payload = {
    "recency": recency,
    "frequency": frequency,
    "engagement_duration": engagement_duration,
    "inactivity_streak": inactivity_streak,
    "engagement_per_interaction": engagement_per_interaction
}

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is NOT likely to churn")
    else:
        st.error("API request failed")
