import pandas as pd
import numpy as np

def add_shock_index(df):
    if "HR" in df.columns and "SBP" in df.columns:
        df["ShockIndex"] = df["HR"] / (df["SBP"] + 1e-6)
    return df

def add_sofa_proxies(df):
    if "Creatinine" in df.columns:
        df["RenalScore"] = pd.cut(df["Creatinine"],
                                   bins=[0, 1.2, 2.0, 3.5, 5.0, np.inf],
                                   labels=[0, 1, 2, 3, 4]).astype(float)
    if "Bilirubin_total" in df.columns:
        df["LiverScore"] = pd.cut(df["Bilirubin_total"],
                                   bins=[0, 1.2, 2.0, 6.0, 12.0, np.inf],
                                   labels=[0, 1, 2, 3, 4]).astype(float)
    return df

def add_time_features(df, time_col="ICULOS"):
    if time_col in df.columns:
        df["HourOfStay"] = df[time_col]
        df["EarlyICU"]   = (df[time_col] <= 6).astype(int)
    return df

def add_vital_interactions(df):
    if "HR" in df.columns and "Resp" in df.columns:
        df["HR_Resp_ratio"] = df["HR"] / (df["Resp"] + 1e-6)
    if "SBP" in df.columns and "MAP" in df.columns:
        df["Pulse_Pressure"] = df["SBP"] - df["MAP"]
    return df

def engineer_all(df):
    df = add_shock_index(df)
    df = add_sofa_proxies(df)
    df = add_time_features(df)
    df = add_vital_interactions(df)
    return df
