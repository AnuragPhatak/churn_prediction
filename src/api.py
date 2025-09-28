from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("file:///C:/Anurag/loylty_rewardz/churn_prediction/src/mlruns")

app = FastAPI(title="Churn Prediction API")

# MLflow Model Registry details
MODEL_NAME = "churn_prediction_model"

# ✅ Fetch the latest version of the model
client = MlflowClient()
latest_versions = client.get_latest_versions(MODEL_NAME)
latest_version = latest_versions[0].version  # get latest version number

# Load model from latest version
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest_version}")


# Define input schema with Pydantic
class ChurnFeatures(BaseModel):
    recency: float
    frequency: float
    engagement_duration: float
    inactivity_streak: float
    engagement_per_interaction: float


@app.post("/predict")
def predict(features: ChurnFeatures):
    """
    Predict churn likelihood for given customer features.
    """
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
