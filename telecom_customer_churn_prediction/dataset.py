import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def load_features_labels(X_file, y_file=None):
    X_df = pd.read_csv(X_file)
    X = X_df.values
    y = None
    if y_file:
        y_df = pd.read_csv(y_file)
        y = y_df.values
    return X, y

def preprocess_features(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    return X_tensor, scaler

def create_dataloader(X_tensor, y_tensor=None, batch_size=32, shuffle=True):
    if y_tensor is not None:
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
