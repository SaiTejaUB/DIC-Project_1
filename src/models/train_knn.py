import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
file_path = os.path.join(BASE_DIR, "data", "processed", "data_processed.csv")




def load_and_preprocess(path: str):
    print(f"[Preprocessing] Loading data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"[Preprocessing] Raw shape: {df.shape}")

    y_raw = df["Severity"].astype(int)

    # Create a binary target variable: 0 for minor (1, 2) and 1 for major (3, 4)
    y = y_raw.apply(lambda x: 0 if x in [1, 2] else 1)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"]= pd.to_datetime(df["End_Time"], errors="coerce")
    df["Duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60

    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsPeakHour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # Encode categoricals
    cat_cols = ["Weather_Condition", "Sunrise_Sunset", "State"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    bool_cols = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
        "No_Exit", "Railway", "Roundabout", "Station", "Stop",
        "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    feature_cols = [
        "Start_Lat", "Start_Lng",
        "Distance(mi)",
        "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)",
        "Weather_Condition", "Sunrise_Sunset", "State",
        "Hour", "DayOfWeek", "Month", "IsWeekend", "IsPeakHour",
        "Duration_min",
    ] + bool_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    print(f"[Preprocessing] Features: {X.shape[1]}  |  Samples: {len(y)}")
    print(f"[Preprocessing] Class distribution:\n{y_raw.value_counts().sort_index()}\n")
    print(f"[Preprocessing] Binary target distribution:\n{y.value_counts().sort_index()}\n")
    return X, y, feature_cols

X, y, feature_cols = load_and_preprocess(file_path)

# Combine X and y for consistent sampling
data_combined = pd.concat([X, y.rename('target')], axis=1)

#ONLY TAKING 500K ROWS DUE TO HIGH PREDICTION TIME FOR kNN
if len(data_combined) > 500000:
    sampled_data = resample(data_combined, n_samples=500000, random_state=42, stratify=data_combined['target'])
else:
    sampled_data = data_combined

X_sampled = sampled_data.drop('target', axis=1)
y_sampled = sampled_data['target']

print(f"Original data shape: {X.shape}")
print(f"Sampled data shape (X_sampled): {X_sampled.shape}")
print(f"Sampled target shape (y_sampled): {y_sampled.shape}")


# Split the sampled data into training and testing sets
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42, stratify=y_sampled)

print(f"KNN X_train shape: {X_train_knn.shape}")
print(f"KNN X_test shape: {X_test_knn.shape}")
print(f"KNN y_train shape: {y_train_knn.shape}")
print(f"KNN y_test shape: {y_test_knn.shape}")


# Using k=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# Oversample minority class before fitting using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_knn, y_train_knn)

knn_model.fit(X_train_res, y_train_res)
knn_model.fit(X_train_knn, y_train_knn)

# Make predictions on the test set, No training in KNN
y_pred_knn = knn_model.predict(X_test_knn)

# Evaluate the model
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
report_knn = classification_report(y_test_knn, y_pred_knn)

print(f"\nKNN Model Accuracy: {accuracy_knn:.4f}")
print("\nKNN Classification Report:")
print(report_knn)

