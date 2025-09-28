from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "registered", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_data_new_features.csv")

# ---------------------------
# Load Model
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Load feature names dynamically from CSV
# ---------------------------
df = pd.read_csv(DATA_PATH)
id_cols = [col for col in df.columns if "customer_id" in col.lower()]
feature_names = [col for col in df.columns if col not in id_cols + ["churn"]]

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="API to predict customer churn using trained XGBoost model",
    version="1.0"
)

class CustomerData(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API 🚀"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert request → DataFrame
    input_df = pd.DataFrame([data.features])

    # Align columns with training features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "used_features": feature_names
    }
