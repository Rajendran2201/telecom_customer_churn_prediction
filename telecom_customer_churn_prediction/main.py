# main.py

import os
import torch
import pandas as pd
from telecom_customer_churn_prediction import config, dataset, features, model, train, evaluate, insights, plots

def main():
    # -----------------------------
    # 1️⃣ Load raw features & labels
    # -----------------------------
    X_train_file = os.path.join(config.DATA_DIR, "X_train.csv")
    X_test_file = os.path.join(config.DATA_DIR, "X_test.csv")
    y_train_file = os.path.join(config.DATA_DIR, "y_train.csv")
    y_test_file = os.path.join(config.DATA_DIR, "y_test.csv")

    X_train_df, y_train = dataset.load_features_labels(X_train_file, y_train_file)
    X_test_df, y_test = dataset.load_features_labels(X_test_file, y_test_file)

    # -----------------------------
    # 2️⃣ Preprocess features
    # -----------------------------
    categorical_cols = X_train_df.select_dtypes(include='object').columns.tolist()

    # Full preprocessing: encode, one-hot, scale
    X_train_processed, scaler, encoders = features.preprocess_features(
        X_train_df, categorical_cols=categorical_cols, one_hot=True, scale=True
    )
    X_test_processed, _, _ = features.preprocess_features(
        X_test_df, categorical_cols=categorical_cols, one_hot=True, scale=True, scaler=scaler
    )

    # Convert labels to tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # -----------------------------
    # 3️⃣ Initialize model
    # -----------------------------
    input_dim = X_train_tensor.shape[1]
    churn_model = model.ChurnANN(input_dim)

    # -----------------------------
    # 4️⃣ Train model
    # -----------------------------
    trained_model = train.train_model(
        model=churn_model,
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        X_val_tensor=X_test_tensor,
        y_val_tensor=y_test_tensor
    )

    # -----------------------------
    # 5️⃣ Evaluate model
    # -----------------------------
    metrics, y_pred_prob, y_pred_class = evaluate.compute_metrics(
        trained_model,
        X_test_tensor,
        y_test_tensor
    )

    print("\n--- Model Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------
    # 6️⃣ Business insights
    # -----------------------------
    # Load the original customer data for insights
    customers_file = os.path.join(config.DATA_DIR, "raw", "customer_churn.csv")
    customers_df = pd.read_csv(customers_file)

    customers, high_risk_customers = insights.generate_insights(
        model=trained_model,
        X_tensor=X_test_df.values,  # Use unscaled for mapping back to customer rows
        customers_df=customers_df,
        scaler=scaler
    )

    # -----------------------------
    # 7️⃣ Optional: plots
    # -----------------------------
    plots.plot_churn_distribution(customers)
    plots.plot_risk_tiers(customers)

    print("\nBusiness insights and high-risk customer CSV saved in:", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
