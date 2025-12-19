# telecom_customer_churn_prediction/features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch

def encode_categorical(df, categorical_cols):
    """
    Encode categorical columns using LabelEncoder or OneHotEncoder.
    Returns transformed DataFrame and encoders.
    """
    df_encoded = df.copy()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders

def one_hot_encode(df, categorical_cols):
    """
    One-hot encode selected categorical columns using pandas.get_dummies
    """
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def scale_features(X, scaler=None):
    """
    Standard scale features.
    If scaler is provided, uses it; otherwise, fits a new scaler.
    Returns scaled array and scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def df_to_tensor(X_df, dtype=torch.float32):
    """
    Convert pandas DataFrame to PyTorch tensor.
    """
    return torch.tensor(X_df.values, dtype=dtype)

def preprocess_features(df, categorical_cols=None, scale=True, one_hot=False, scaler=None):
    """
    Full preprocessing pipeline:
    - Encode categorical variables
    - Optionally one-hot encode
    - Optionally scale
    Returns: preprocessed DataFrame or array, scaler, encoders
    """
    df_processed = df.copy()
    encoders = {}

    if categorical_cols:
        df_processed, encoders = encode_categorical(df_processed, categorical_cols)
        if one_hot:
            df_processed = one_hot_encode(df_processed, categorical_cols)

    X_processed = df_processed.values

    if scale:
        X_processed, scaler = scale_features(X_processed, scaler)

    return X_processed, scaler, encoders
