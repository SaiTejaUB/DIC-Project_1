import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_processed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Shape:", df.shape)

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
df['Month'] = df['Start_Time'].dt.month

features = [
    'Distance(mi)', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'State', 'Weather_Condition', 'Sunrise_Sunset', 'Hour', 'Month'
]
target = 'Severity'

df = df[features + [target]].dropna()
print("After dropna:", df.shape)
print(df[target].value_counts().sort_index())

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print("x_train:", x_train.shape, "  x_test:", x_test.shape)

sample_size = 400000
x_train_sample, _, y_train_sample, _ = train_test_split(
    x_train, y_train, train_size=sample_size, random_state=42, stratify=y_train
)

categorical = ['State', 'Weather_Condition', 'Sunrise_Sunset']
numeric = ['Distance(mi)', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Hour', 'Month']

pre = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', 'passthrough', numeric)
])

pipe = Pipeline([('prep', pre), ('model', DecisionTreeClassifier(random_state=43))])

param_grid = {
    'model__max_depth': [5, 10, 15, 20],
    'model__min_samples_split': [2, 10, 20],
    'model__min_samples_leaf': [1, 5, 10]
}
grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')
grid.fit(x_train_sample, y_train_sample)
print("Best params:", grid.best_params_)
best = grid.best_params_

final = Pipeline([
    ('prep', pre),
    ('model', DecisionTreeClassifier(
        max_depth=best['model__max_depth'],
        min_samples_split=best['model__min_samples_split'],
        min_samples_leaf=best['model__min_samples_leaf'],
        random_state=42
    ))
])
final.fit(x_train, y_train)
y_pred = final.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Weighted F1:", f1_score(y_test, y_pred, average='weighted'))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

model_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final, f)
print(f"Model saved to {model_path}")
