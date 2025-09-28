import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
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
    
    # Drop customer_id if it's there
    if 'customer_id' in df.columns:
        cols_to_scale = [col for col in columns if col in df.columns and col != 'customer_id']
    else:
        cols_to_scale = [col for col in columns if col in df.columns]
    
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def select_features(df, target_col, k=5):
    """
    Select top k features using multiple techniques:
    - Mutual Information
    - Chi2 (for classification tasks, categorical-friendly)
    - ANOVA F-test
    - RandomForest feature importances
    Returns reduced dataframe with top k features.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
    from sklearn.ensemble import RandomForestClassifier

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_scores = pd.DataFrame(index=X.columns)

    mi = mutual_info_classif(X, y, discrete_features='auto')
    feature_scores['mutual_info'] = mi

    X_chi = X.copy()
    X_chi[X_chi < 0] = 0  
    chi2_scores, _ = chi2(X_chi, y)
    feature_scores['chi2'] = chi2_scores

    f_scores, _ = f_classif(X, y)
    feature_scores['anova_f'] = f_scores

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_scores['rf_importance'] = rf.feature_importances_

    feature_scores['rank'] = feature_scores.rank(method='average', ascending=False).mean(axis=1)

    top_features = feature_scores.sort_values("rank", ascending=True).head(k).index.tolist()
    print(f"Selected top {k} features: {top_features}")

    return df[top_features + [target_col]]


def feature_engineering_pipeline(input_csv, output_csv, target_col="churn", k=5):
    df = load_data(input_csv)
    
    # 🚨 Drop customer_id if present
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    
    df = encode_categorical(df)
    df = create_features(df)
    
    numerical_cols = ['account_age', 'recency', 'frequency', 'engagement_duration', 
                      'feature_usage_count', 'marketing_ctr', 'nps_score', 'inactivity_streak']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    df = scale_features(df, numerical_cols)

    # Feature Selection
    if target_col in df.columns:
        df = select_features(df, target_col, k=k)

    df.to_csv(output_csv, index=False)
    print(f"Feature engineered + selected data saved to {output_csv}")
    return df


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save features CSV")
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    df_final = feature_engineering_pipeline(
        input_csv, output_csv, target_col="churn", k=5
    )

