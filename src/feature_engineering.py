import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    if 'nps_score' in df.columns:
        df['nps_score'] = df['nps_score'].fillna(df['nps_score'].median())
    return df

def encode_categorical(df):
    if 'plan_tier' in df.columns:
        df = pd.get_dummies(df, columns=['plan_tier'], drop_first=True)
    return df

def create_features(df):
    if {'engagement_duration', 'frequency'}.issubset(df.columns):
        df['engagement_per_interaction'] = df['engagement_duration'] / (df['frequency'] + 1e-5)
    

    if {'support_tickets', 'billing_issues'}.issubset(df.columns):
        df['total_issues'] = df['support_tickets'] + df['billing_issues']
    
    return df

def scale_features(df, columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def feature_engineering_pipeline(input_csv, output_csv):
    df = load_data(input_csv)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = create_features(df)
    
    numerical_cols = ['account_age', 'recency', 'frequency', 'engagement_duration', 
                      'feature_usage_count', 'marketing_ctr', 'nps_score', 'inactivity_streak']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    df = scale_features(df, numerical_cols)
    
    df.to_csv(output_csv, index=False)
    print(f"Feature engineered data saved to {output_csv}")
    return df

if __name__ == "__main__":
    input_csv = "data/processed/cleaned_customer_churn_data.csv"
    output_csv = "data/processed/customer_data_new_features.csv"
    df_final = feature_engineering_pipeline(input_csv, output_csv)
