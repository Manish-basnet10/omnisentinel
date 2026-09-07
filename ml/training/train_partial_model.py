import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib

from ml.serving.feature_contract import PARTIAL_MODEL_FEATURES
from ml.models.partial_model import PartialFlowMLP

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_partial_model():
    print(f"Loading data from {PROC_DIR / 'cleaned_full.parquet'}")
    try:
        df = pd.read_parquet(PROC_DIR / "cleaned_full.parquet")
    except Exception as e:
        print("Could not load dataset. Make sure cleaned_full.parquet exists.", e)
        return

    print("Extracting partial model features (strictly no target leakage):")
    X = df[PARTIAL_MODEL_FEATURES].copy()
    
    # We load the label mappings to get target
    with open(PROC_DIR / "label_encoding.json") as f:
        encoding = json.load(f)
    
    # We must use label_multiclass if it exists, or encode Label
    if "label_multiclass" in df.columns:
        y = df["label_multiclass"].values
    else:
        y = df["Label"].map(encoding["class2idx"]).values

    # Drop NaNs just in case
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.values.astype(np.float32)

    # Train, val, test split (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler immediately
    scaler_path = MODEL_DIR / "partial_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved partial model scaler to {scaler_path}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Model definition
    model = PartialFlowMLP(input_size=len(PARTIAL_MODEL_FEATURES), num_classes=15)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 1024
    epochs = 15

    print("Starting training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size()[0])
        total_loss = 0
        for i in range(0, X_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            _, predicted = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_val_t.numpy(), predicted.numpy())

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(X_train_t):.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    # Test set evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs, 1)
        test_acc = accuracy_score(y_test_t.numpy(), test_predicted.numpy())
        test_f1 = f1_score(y_test_t.numpy(), test_predicted.numpy(), average='weighted')
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test F1-Score: {test_f1:.4f}")

    # Save checkpoint
    chk_path = MODEL_DIR / "partial_model.pth"
    torch.save(best_model_state, chk_path)
    print(f"Saved partial PyTorch model to {chk_path}")

    # Save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "num_features": len(PARTIAL_MODEL_FEATURES)
    }
    with open(PROC_DIR / "partial_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_partial_model()
