import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_processed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Dataset shape:", df.shape)

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df = df.dropna(subset=['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour.astype('int8')
df['Month'] = df['Start_Time'].dt.month.astype('int8')

x_colums = [
    'Distance(mi)', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'State', 'Weather_Condition', 'Sunrise_Sunset', 'Hour', 'Month'
]
y_colums = 'Severity'

df = df[x_colums + [y_colums]].dropna()
print("After dropna:", df.shape)
print(df[y_colums].value_counts().sort_index())

x = df[x_colums]
y = df[y_colums]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("x_train:", x_train.shape, "  x_test:", x_test.shape)

numeric = ['Distance(mi)', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Hour', 'Month']
categorical = ['State', 'Weather_Condition', 'Sunrise_Sunset']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(C=0.1, solver='saga', max_iter=150, random_state=42))
])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Weighted F1:", f1_score(y_test, y_pred, average='weighted'))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipe, f)
print(f"Model saved to {model_path}")
