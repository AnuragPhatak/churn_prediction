import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from xgboost import XGBClassifier
from mlflow.tracking import MlflowClient

# -----------------------------
# 1. Load config
# -----------------------------
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv(config["data"]["features_path"])
X = df[config["features"]["input_features"]]
y = df[config["features"]["target_col"]]

# -----------------------------
# 3. Best hyperparameters (from YAML)
# -----------------------------
best_params = config["training"]["xgb"]
best_params["random_state"] = config["training"]["random_state"]

# -----------------------------
# 4. Train final model on all data
# -----------------------------
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# -----------------------------
# 5. Log to MLflow Experiment + Register model
# -----------------------------
mlflow.set_experiment("churn_final")

model_name = config["registry"]["model_name"]

with mlflow.start_run(run_name="XGBoost_final_full_data"):
    mlflow.log_params(best_params)  # log hyperparameters
    
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name=model_name
    )

print("✅ Final model retrained on full data and logged to MLflow Registry")

# -----------------------------
# 6. Promote to Production
# -----------------------------
client = MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🚀 Model {model_name} v{latest_version} promoted to Production")
