import streamlit as st
import requests

# API_URL = "http://127.0.0.1:8000/predict"  
API_URL = "https://churn-prediction-1-9dvv.onrender.com/predict"


st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📊 Customer Churn Prediction")


with st.form("churn_form"):
    recency = st.number_input("Recency", value=0, step=1)
    frequency = st.number_input("Frequency", value=0, step=1)
    engagement_duration = st.number_input("Engagement Duration", value=0, step=1)
    inactivity_streak = st.number_input("Inactivity Streak", value=0, step=1)
    engagement_per_interaction = st.number_input("Engagement per Interaction", value=0, step=1)

    submitted = st.form_submit_button("Predict Churn")



if submitted:
    payload = {
        "recency": recency,
        "frequency": frequency,
        "engagement_duration": engagement_duration,
        "inactivity_streak": inactivity_streak,
        "engagement_per_interaction": engagement_per_interaction,
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            prediction = response.json()["churn_prediction"]
            if prediction == 1:
                st.error("⚠️ This customer is likely to churn!")
            else:
                st.success("✅ This customer is not likely to churn.")
        else:
            st.error(f"API error: {response.status_code}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
