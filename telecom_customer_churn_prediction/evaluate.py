# telecom_customer_churn_prediction/evaluate.py

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow

def load_model(model_class, model_path, input_dim):
    """Load trained PyTorch model."""
    model = model_class(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def compute_metrics(model, X_tensor, y_tensor, log_mlflow=True, experiment_name="Churn_ANN_Evaluation"):
    """
    Compute evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    X_tensor: torch.FloatTensor
    y_tensor: torch.FloatTensor
    """
    with torch.no_grad():
        y_pred_prob = model(X_tensor).numpy()
        y_pred_class = (y_pred_prob >= 0.5).astype(int)

    y_true = y_tensor.numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class),
        "recall": recall_score(y_true, y_pred_class),
        "f1_score": f1_score(y_true, y_pred_class),
        "roc_auc": roc_auc_score(y_true, y_pred_prob)
    }

    if log_mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metrics(metrics)

    return metrics, y_pred_prob, y_pred_class
