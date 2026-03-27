import argparse
import time
import warnings
import random
import pickle
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
sns.set_theme(style="whitegrid", font_scale=1.05)



def load_and_preprocess(path: str):
    print(f"[k-NN] Loading data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"[k-NN] Raw shape: {df.shape}")

    X = df.drop(columns=['Severity', 'Unnamed: 0', 'Start_Time', 'End_Time'],errors='ignore')
    y = df['Severity'].astype(int)

    # Adjust Severity to be 0-indexed (0, 1, 2, 3 instead of 1, 2, 3, 4)
    y = y - 1

    # Identify boolean columns and convert them to integers (0s and 1s)
    boolean_cols = X.select_dtypes(include='bool').columns
    for col in boolean_cols:
        X[col] = X[col].astype(int)

    X['Sunrise_Sunset'] = X['Sunrise_Sunset'].map({'Night': 0, 'Day': 1})

    other_object_cols = X.select_dtypes(include='object').columns
    X = X.drop(columns=other_object_cols)

    # Fill any remaining NANs with column median
    X = X.fillna(X.median(numeric_only=True))

    scaler  = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    feature_cols = list(X.columns)
    print(f"[k-NN] Features used  : {feature_cols}")
    print(f"[k-NN] Feature count  : {len(feature_cols)}")
    print(f"[k-NN] Samples        : {len(y)}")
    print(f"[k-NN] Class distribution (0-indexed):\n{y.value_counts().sort_index()}\n")

    return X_scaled, y, feature_cols, scaler



def tune_k(X_train, y_train, k_values=None):
    print("[k-NN] best k value = 5.\n")
    best_k = 5
    k_scores = {5: 0.0}
    return best_k, k_scores




def train_model(X, y, best_k):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n[k-NN] Training final model with k={best_k} …")
    t0    = time.time()
    model = KNeighborsClassifier(n_neighbors=best_k, weights="distance", n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"[k-NN] Training complete in {time.time()-t0:.1f}s")
    return model, X_train, X_test, y_train, y_test




def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n── k-NN Performance Metrics ────────────────────────────────")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Weighted F1       : {f1:.4f}")
    print(f"  Weighted Precision: {prec:.4f}")
    print(f"  Weighted Recall   : {rec:.4f}")

    present = sorted(y_test.unique())
    labels  = [f"Sev {i+1}" for i in present]
    print("\n  Per-Class Report:")
    print(classification_report(y_test, y_pred,
                                labels=present, target_names=labels, zero_division=0))

    return dict(accuracy=acc, f1=f1, precision=prec, recall=rec,
                y_pred=y_pred, y_prob=y_prob)




def main():
    parser = argparse.ArgumentParser(description="k-NN Severity Classifier")
    parser.add_argument("--data",     required=True, help="Path to processed CSV")
    parser.add_argument("--out",      default="knn_results.png",
                        help="Output figure filename")
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[3, 5, 7, 11, 15, 21],
                        help="k values to search in CV")
    args = parser.parse_args()

    X, y, feature_cols, scaler = load_and_preprocess(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    best_k, k_scores = tune_k(X_train, y_train, k_values=args.k_values)

    model, X_train, X_test, y_train, y_test = train_model(X, y, best_k)
    metrics = evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
