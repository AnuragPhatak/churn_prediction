import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from mlflow.tracking import MlflowClient

# -----------------------------
# 1. Load full dataset
# -----------------------------
df = pd.read_csv("../data/processed/customer_data_new_features.csv")  
X = df.drop("churn", axis=1)
y = df["churn"]

# -----------------------------
# 2. Best hyperparameters (from MLflow experiment results)
# -----------------------------
best_params = {
    "n_estimators": 50,
    "max_depth": 5,
    "learning_rate": 0.1706869478158034,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42
}

# -----------------------------
# 3. Train final model on all data
# -----------------------------
final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

# -----------------------------
# 4. Log to MLflow Experiment + Register model
# -----------------------------
mlflow.set_experiment("churn_final_1")   # ✅ ensures runs don’t go to Default

model_name = "ChurnPredictionModel"

with mlflow.start_run(run_name="XGBoost_final_full_data"):
    # log hyperparameters for traceability
    mlflow.log_params(best_params)
    
    # log and register model
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name=model_name
    )

print("✅ Final model retrained on full data and logged to registry")

# -----------------------------
# 5. Promote to Production
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
