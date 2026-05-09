# xgboost model
# uses SMOTE to handle class imbalance

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, f1_score,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

np.random.seed(14)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_processed.csv")

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

    # binary target
    df["Severe"] = (df["Severity"].isin([3, 4])).astype(int)

    # day/night encoding
    df["Sunrise_Sunset_enc"] = (
        df["Sunrise_Sunset"] == "Night"
    ).astype(int)

    # weather encoding
    weather_map = {w: i for i, w in enumerate(WEATHER_CATEGORIES)}

    df["Weather_enc"] = (
        df["Weather_Condition"]
        .map(weather_map)
        .fillna(len(WEATHER_CATEGORIES) - 1)
        .astype(int)
    )

    # boolean cleanup
    for col in BOOLEAN_COLS:
        if col in df.columns:

            if df[col].dtype == object:
                df[col] = df[col].map({
                    "TRUE": 1,
                    "FALSE": 0,
                    True: 1,
                    False: 0
                })

            df[col] = df[col].fillna(0).astype(int)

    return df


def build_feature_matrix(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]

    df_feat = df[cols + ["Severe"]].dropna()

    return df_feat[cols], df_feat["Severe"]


def stratified_split(X, y, test_size=0.2):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=14,
        stratify=y
    )


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

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Minor", "Severe"]
        )
    )

    if y_prob is not None:

        ap = average_precision_score(y_test, y_prob)
        roc = roc_auc_score(y_test, y_prob)

        print(f"  PR-AUC: {ap:.4f}")
        print(f"  ROC-AUC: {roc:.4f}")


def plot_confusion(y_test, y_pred, title, fname):

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["Minor", "Severe"]
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    disp.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues"
    )

    ax.set_title(title)

    save_fig(fname)


def train(data_path):

    df = load_and_engineer(data_path)

    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print(f"train size: {len(X_train)}  test size: {len(X_test)}")

    print(
        f"class balance before SMOTE  "
        f"minor: {(y_train==0).sum()}  "
        f"severe: {(y_train==1).sum()}"
    )

    # pipeline
    pipe = Pipeline([

        ("scaler", StandardScaler()),

        ("smote", SMOTE(
            sampling_strategy=0.5,
            random_state=14
        )),

        ("xgb", xgb.XGBClassifier(

            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,

            subsample=0.8,
            colsample_bytree=0.8,tree_method="hist", device="cuda",

            objective="binary:logistic",

            eval_metric="logloss",

            n_jobs=-1,

            random_state=42,
        )),
    ])

    print("fitting model...")

    pipe.fit(X_train, y_train)

    print("done fitting")

    # verify SMOTE
    scaler = pipe.named_steps["scaler"]
    smote = pipe.named_steps["smote"]

    X_sc = scaler.transform(X_train)

    _, y_res = smote.fit_resample(X_sc, y_train)

    print(
        f"  after SMOTE – "
        f"minor: {(y_res==0).sum()}  "
        f"severe: {(y_res==1).sum()}"
    )

    # probabilities
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # optimize threshold
    thresholds = np.linspace(0, 1, 200)

    best_t = max(
        thresholds,
        key=lambda t: f1_score(
            y_test,
            (y_prob >= t).astype(int),
            zero_division=0
        )
    )

    print(f"  best threshold: {best_t:.3f}")

    y_pred = (y_prob >= best_t).astype(int)

    # metrics
    print_metrics(
        y_test,
        y_pred,
        y_prob,
        "XGBoost + SMOTE"
    )

    # confusion matrix
    plot_confusion(
        y_test,
        y_pred,
        "XGBoost + SMOTE",
        "xgb_cm"
    )

    # feature importance
    print("saving plots...")

    xgb_model = pipe.named_steps["xgb"]

    fi = pd.Series(
        xgb_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    fi[:15].sort_values().plot.barh(
        ax=ax,
        color="darkorange"
    )

    ax.set_title("XGBoost + SMOTE - Top 15 Feature Importances")

    ax.set_xlabel("Importance Score")

    save_fig("xgb_feature_importance")

    # precision recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)

    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(rec, prec, color="darkred")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_title("XGBoost + SMOTE - Precision-Recall Curve")

    ax.text(
        0.5,
        0.9,
        f"AP = {ap:.3f}",
        transform=ax.transAxes
    )

    save_fig("xgb_pr_curve")

    # save model
    save_model(pipe, "xgboost_model")

    return pipe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=DATA_PATH)

    args = parser.parse_args()

    train(args.data)
