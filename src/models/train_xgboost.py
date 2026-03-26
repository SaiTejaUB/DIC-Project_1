import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb



from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,average_precision_score
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

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy=0.5, random_state=14)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print("fitting model...")
    pipe.fit(X_train, y_train)
    print("done fitting")



    y_prob = pipe.predict_proba(X_test)[:, 1]



DATA_PATH = "../../data/processed/data_processed.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()
    train(args.data)
