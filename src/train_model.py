import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# MLflow setup
mlflow.set_tracking_uri("file:///C:/Anurag/loylty_rewardz/churn_prediction/src/mlruns")
EXPERIMENT_NAME = "churn_prediction_demo"
MODEL_NAME = "churn_prediction_model"

# Ensure all runs log under this experiment
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data(path):
    """Load dataset and split features/target."""
    df = pd.read_csv(path)

    # Drop churn + customer_id if exists
    drop_cols = [col for col in ["churn", "customer_id"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["churn"]

    return X, y


def get_best_run(experiment_name):
    """Fetch the best run (highest accuracy) from MLflow experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}, Accuracy: {best_run.data.metrics['accuracy']:.4f}")
    return best_run


def retrain_and_register(X, y, best_run):
    """Retrain best model on full dataset and register in Model Registry."""
    params = best_run.data.params
    algo = best_run.data.tags.get("mlflow.runName", "Unknown")

    if algo.startswith("RandomForest"):
        model = RandomForestClassifier(
            n_estimators=int(float(params["n_estimators"])),
            max_depth=int(float(params["max_depth"])),
            min_samples_split=int(float(params["min_samples_split"])),
            random_state=42,
            n_jobs=-1
        )
    elif algo.startswith("XGBoost"):
        model = XGBClassifier(
            n_estimators=int(float(params["n_estimators"])),
            max_depth=int(float(params["max_depth"])),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown algorithm name found in best run: {algo}")

    # Retrain on full dataset
    print(f"🔄 Retraining {algo} with best hyperparameters on full dataset...")
    model.fit(X, y)

    # Log & register model
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=f"{algo}_final"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", best_run.data.metrics["accuracy"])
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Register in Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)

    print(f"✅ Registered {algo} model as '{MODEL_NAME}' in Model Registry")


if __name__ == "__main__":
    X, y = load_data("../data/processed/customer_data_new_features.csv")
    best_run = get_best_run(EXPERIMENT_NAME)
    retrain_and_register(X, y, best_run)
