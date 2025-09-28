import pandas as pd
import pickle
import mlflow.sklearn

def load_data(path="../data/processed/customer_data_new_features.csv"):
    df = pd.read_csv(path)
    id_cols = [col for col in df.columns if "customer_id" in col.lower()]
    df = df.drop(columns=id_cols, errors="ignore")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    return X, y

def train_and_save(model_name="churn_model", stage="None", version=1, output_path="../models/registered/model.pkl"):
    X, y = load_data()

    if stage: 
        model_uri = f"models:/{model_name}/{stage}"
    else:     
        model_uri = f"models:/{model_name}/{version}"

    print(f"🔍 Loading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)


    print("🚀 Training model on full dataset...")
    model.fit(X, y)

    
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Final model saved at {output_path}")


if __name__ == "__main__":
    
    train_and_save(model_name="churn_model", version=1)
