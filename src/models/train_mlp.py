# MLP for accident severity classification (4-class: Severity 1-4)
# outside class algorithm

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

np.random.seed(14)
torch.manual_seed(14)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATH = "../../data/processed/data_processed.csv"

BOOLEAN_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
]

WEATHER_CATEGORIES = ["Clear", "Cloudy", "Rain_Drizzle", "Snow", "Fog", "Thunderstorm", "Other"]

FEATURE_COLS = [
    "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Distance(mi)",
    "Hour", "DayOfWeek", "Month", "Duration_min",
    "Sunrise_Sunset_enc", "Weather_enc",
    "Start_Lat", "Start_Lng",
] + BOOLEAN_COLS


def load_and_engineer(path):
    df = pd.read_csv(path, low_memory=False)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month
    df["Duration_min"] = (
        (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    ).clip(0, 1440)

    df["Sunrise_Sunset_enc"] = (df["Sunrise_Sunset"] == "Night").astype(int)

    weather_map = {w: i for i, w in enumerate(WEATHER_CATEGORIES)}
    df["Weather_enc"] = df["Weather_Condition"].map(weather_map).fillna(6).astype(int)

    for col in BOOLEAN_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
            df[col] = df[col].fillna(0).astype(int)

    return df


def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  figure -> {path}")


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    print("loading data...")
    df = load_and_engineer(data_path)

    cols = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[cols + ["Severity"]].dropna()
    X = df_clean[cols].values.astype(np.float32)
    y = (df_clean["Severity"].values - 1).astype(np.int64)  # severity 1-4 -> 0-3

    print(f"train size: {len(X):,}  features: {len(cols)}")
    print("class distribution:")
    for i in range(4):
        cnt = (y == i).sum()
        print(f"  Severity {i+1}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # softer class weights using sqrt to avoid over-correcting minority classes
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / np.sqrt(class_counts.astype(float))
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)

    BATCH_SIZE = 8192
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, torch.tensor(y_test)),
                             batch_size=BATCH_SIZE * 4, shuffle=False)

    model = MLP(len(cols), 4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    max_epochs = 30
    patience = 7
    best_val_loss = float('inf')
    patience_ctr = 0
    best_state = None
    train_losses, val_losses, val_accs = [], [], []

    print(f"training up to {max_epochs} epochs (early stopping patience={patience})...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(Xb)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss_sum += criterion(logits, yb).item() * len(Xb)
                all_preds.append(logits.argmax(dim=1).cpu())
        val_loss = val_loss_sum / len(test_loader.dataset)
        val_losses.append(val_loss)
        y_pred_epoch = torch.cat(all_preds).numpy()
        val_acc = accuracy_score(y_test, y_pred_epoch)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch:02d}/{max_epochs}  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  val acc: {val_acc*100:.2f}%  lr: {lr:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # final predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            all_preds.append(model(Xb.to(device)).argmax(dim=1).cpu())
    y_pred = torch.cat(all_preds).numpy()

    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n--- MLP Results ---")
    print(f"test accuracy: {test_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=[f"Severity {i+1}" for i in range(4)]))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=[f"Sev {i+1}" for i in range(4)]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("MLP - Confusion Matrix")
    save_fig("mlp_cm")

    # loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train", color="steelblue")
    ax.plot(val_losses, label="val", color="orange", linestyle="--")
    ax.set_title("MLP - Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    save_fig("mlp_loss")

    # save model
    model_path = os.path.join(MODEL_DIR, "mlp.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model_state": model.state_dict(), "input_dim": len(cols), "scaler": scaler}, f)
    print(f"  model -> {model_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()
    train(args.data)
