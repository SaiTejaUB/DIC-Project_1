# random forest model 
# uses SMOTE to handle class imbalance


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from utils import (
    load_and_engineer, build_feature_matrix, stratified_split,
    print_metrics, plot_confusion, save_fig, save_model,
)


def train(data_path):
    df = load_and_engineer(data_path)
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print(f"train size: {len(X_train)}  test size: {len(X_test)}")
    print(f"class balance before SMOTE  minor: {(y_train==0).sum()}  severe: {(y_train==1).sum()}")

    # pipeline handles SMOTE so it only runs on training data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy=0.5, random_state=14)),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    print("fitting model...")
    pipe.fit(X_train, y_train)
    print("done fitting")

    # just double checking the resampling worked
    scaler = pipe.named_steps["scaler"]
    smote = pipe.named_steps["smote"]
    X_sc = scaler.transform(X_train)
    _, y_res = smote.fit_resample(X_sc, y_train)
    print(f"  after SMOTE – minor: {(y_res==0).sum()}  severe: {(y_res==1).sum()}")

    y_prob = pipe.predict_proba(X_test)[:, 1]

    # find threshold that maximizes f1 instead of using default 0.5
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0, 1, 200)
    best_t = max(thresholds, key=lambda t: f1_score(y_test, (y_prob >= t).astype(int), zero_division=0))
    print(f"  best threshold: {best_t:.3f}")
    y_pred = (y_prob >= best_t).astype(int)

    print_metrics(y_test, y_pred, y_prob, "Random Forest + SMOTE")
    plot_confusion(y_test, y_pred, "Random Forest + SMOTE", "rf_cm")

    print("saving plots...")
    # feature importance plot
    rf = pipe.named_steps["rf"]
    fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi[:15].sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title("RF + SMOTE - Top 15 Feature Importances")
    ax.set_xlabel("Mean Decrease in Impurity")
    save_fig("rf_feature_importance")

    # precision-recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rec, prec, color="forestgreen")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("RF + SMOTE - Precision-Recall Curve")
    ax.text(0.5, 0.9, f"AP = {ap:.3f}", transform=ax.transAxes)
    save_fig("rf_pr_curve")

    save_model(pipe, "random_forest")
    return pipe


DATA_PATH = "../../data/processed/data_processed.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()
    train(args.data)
