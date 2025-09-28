from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

MODEL_NAME = "ChurnPredictionModel"
MODEL_STAGE = "Production"  
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")


app = FastAPI(title="Churn Prediction API", version="1.0")


class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    engagement_duration: float
    inactivity_streak: float
    engagement_per_interaction: float

def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(features: CustomerFeatures):
    data = [[
        features.recency,
        features.frequency,
        features.engagement_duration,
        features.inactivity_streak,
        features.engagement_per_interaction,
    ]]
    prediction = model.predict(data)[0]
    return {"churn_prediction": int(prediction)}
