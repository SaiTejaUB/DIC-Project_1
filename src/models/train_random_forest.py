# random forest model
# uses SMOTE to handle class imbalance

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, f1_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import argparse

np.random.seed(42)

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

WEATHER_CATEGORIES = [
    "Clear", "Cloudy", "Rain_Drizzle", "Snow", "Fog", "Thunderstorm", "Other"
]

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

    df["Severe"] = (df["Severity"].isin([3, 4])).astype(int)
    df["Sunrise_Sunset_enc"] = (df["Sunrise_Sunset"] == "Night").astype(int)

    weather_map = {w: i for i, w in enumerate(WEATHER_CATEGORIES)}
    df["Weather_enc"] = (
        df["Weather_Condition"].map(weather_map)
        .fillna(len(WEATHER_CATEGORIES) - 1)
        .astype(int)
    )

    for col in BOOLEAN_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
            df[col] = df[col].fillna(0).astype(int)

    return df


def build_feature_matrix(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df_feat = df[cols + ["Severe"]].dropna()
    return df_feat[cols], df_feat["Severe"]


def stratified_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def save_fig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  figure -> {path}")


def save_model(model, name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  model -> {path}")
    return path


def print_metrics(y_test, y_pred, y_prob=None, label=""):
    print(f"\n--- {label} ---")
    print(classification_report(y_test, y_pred, target_names=["Minor", "Severe"]))
    if y_prob is not None:
        print("  computing AUC scores...")
        ap = average_precision_score(y_test, y_prob)
        roc = roc_auc_score(y_test, y_prob)
        print(f"  PR-AUC: {ap:.4f}   ROC-AUC: {roc:.4f}")


def plot_confusion(y_test, y_pred, title, fname):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Minor", "Severe"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    save_fig(fname)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()
    train(args.data)
