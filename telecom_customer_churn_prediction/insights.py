# telecom_customer_churn_prediction/insights.py

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import torch

def generate_insights(model, X_tensor, customers_df, scaler=None, models_dir='./models', reports_dir='./reports',
                      log_mlflow=True, experiment_name="Churn_ANN_Business_Insights"):
    """
    Generates business insights:
    - Predict churn probabilities
    - Segment customers into Low/Medium/High risk
    - Save high-risk customers CSV
    - Plot distributions
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Apply scaler if provided
    if scaler is not None:
        X_scaled = scaler.transform(X_tensor)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predictions
    with torch.no_grad():
        y_pred_prob = model(X_tensor).numpy()
        y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # Merge with customer dataframe
    customers = customers_df.copy()
    customers['churn_prob'] = y_pred_prob
    customers['churn_pred'] = y_pred_class

    # Segment risk tiers (quantiles)
    customers['risk_tier'] = pd.qcut(customers['churn_prob'], 3, labels=['Low', 'Medium', 'High'])

    # Save CSVs
    customers.to_csv(os.path.join(reports_dir, 'customers_with_churn_prob.csv'), index=False)
    high_risk = customers[customers['risk_tier'] == 'High'].sort_values(by='churn_prob', ascending=False)
    high_risk.to_csv(os.path.join(reports_dir, 'high_risk_customers.csv'), index=False)

    # Plot churn probability distribution
    plt.figure(figsize=(8,5))
    sns.histplot(customers['churn_prob'], bins=20, kde=True)
    plt.title("Predicted Churn Probability Distribution")
    plt.xlabel("Churn Probability")
    plt.ylabel("Count")
    plt.show()

    # Risk tier distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='risk_tier', data=customers, order=['Low','Medium','High'])
    plt.title("Customer Risk Tier Distribution")
    plt.show()

    # Optional: Log artifacts to MLflow
    if log_mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_artifact(os.path.join(reports_dir, 'customers_with_churn_prob.csv'))
            mlflow.log_artifact(os.path.join(reports_dir, 'high_risk_customers.csv'))

    return customers, high_risk
