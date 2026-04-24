import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df

def drop_high_missing(df, threshold=0.4):
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    print(f"Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing: {drop_cols}")
    return df.drop(columns=drop_cols)

def impute_features(df, strategy="median"):
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return df_imputed, imputer

def scale_features(X_train, X_val, X_test, scaler_type="robust"):
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    return X_train_s, X_val_s, X_test_s, scaler

def split_features_target(df, target_col="SepsisLabel"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
