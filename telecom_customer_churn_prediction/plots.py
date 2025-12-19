# telecom_customer_churn_prediction/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_churn_distribution(customers_df, column='churn_prob', save_path=None):
    plt.figure(figsize=(8,5))
    sns.histplot(customers_df[column], bins=20, kde=True)
    plt.title("Churn Probability Distribution")
    plt.xlabel("Churn Probability")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_risk_tiers(customers_df, column='risk_tier', save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(x=column, data=customers_df, order=['Low','Medium','High'])
    plt.title("Customer Risk Tier Distribution")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
