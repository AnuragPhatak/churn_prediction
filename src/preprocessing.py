import pandas as pd
import numpy as np

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="customer_id", keep="first")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)
    return df

def fix_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df.loc[df[col] < 0, col] = np.nan
    df = handle_missing_values(df)  # re-impute after fixing
    return df

def fix_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    if "recency" in df.columns and "frequency" in df.columns:
        inconsistent_mask = (df["recency"] > df["recency"].quantile(0.95)) & \
                            (df["frequency"] > df["frequency"].quantile(0.95))
        df.loc[inconsistent_mask, "frequency"] = df["frequency"].median()
    return df

def fix_categorical_typos(df: pd.DataFrame) -> pd.DataFrame:
    if "plan_tier" in df.columns:
        mapping = {
            "Basic": "basic",
            "BAsic": "basic",
            "basic": "basic",
            "Premium": "premium",
            "Premiuum": "premium",
            "premium": "premium",
            "Gold": "gold",
            "gold": "gold"
        }
        df["plan_tier"] = df["plan_tier"].map(mapping).fillna("unknown")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = fix_negative_values(df)
    df = fix_inconsistencies(df)
    df = fix_categorical_typos(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/customer_churn_data.csv")
    clean_df = preprocess_data(df)
    clean_df.to_csv("data/processed/cleaned_customer_churn_data.csv", index=False)

    print("✅ Preprocessing complete. Cleaned file saved to data/processed/")
