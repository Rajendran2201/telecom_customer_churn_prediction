# telecom_customer_churn_prediction/train.py

import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from telecom_customer_churn_prediction import dataset, features, config

def create_dataloader(X_tensor, y_tensor=None, batch_size=32, shuffle=True):
    """
    Create PyTorch DataLoader
    """
    if y_tensor is not None:
        ds = TensorDataset(X_tensor, y_tensor)
    else:
        ds = TensorDataset(X_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
                batch_size=config.BATCH_SIZE, num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE, patience=config.PATIENCE,
                models_dir=config.MODELS_DIR, experiment_name=config.MLFLOW_EXPERIMENT_TRAIN):
    """
    Train ANN model with early stopping and MLflow logging
    """
    os.makedirs(models_dir, exist_ok=True)

    # DataLoaders
    train_loader = create_dataloader(X_train_tensor, y_train_tensor, batch_size=batch_size)
    val_loader = create_dataloader(X_val_tensor, y_val_tensor, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    counter = 0

    # Start MLflow run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "input_dim": X_train_tensor.shape[1],
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })

        for epoch in range(num_epochs):
            # ---- Training ----
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
            train_loss = running_loss / len(train_loader.dataset)

            # ---- Validation ----
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    val_loss_total += val_loss.item() * X_val.size(0)
            val_loss_avg = val_loss_total / len(val_loader.dataset)

            # MLflow metrics logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss_avg, step=epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f}")

            # ---- Early Stopping ----
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(models_dir, 'best_churn_model.pth'))
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_churn_model.pth')))
                    break

        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "churn_ann_model")

    return model
