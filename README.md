# US Accidents Severity Analysis
EAS 587 – Data Intensive Computing

## Project Overview
This project analyzes the US Accidents dataset combined with HPMS (Highway Performance Monitoring System) road infrastructure data to predict accident severity. The project covers data cleaning, machine learning modeling, and scalable big data processing across three phases.

---

## Dataset Access

**US_Accidents_March23.csv** — not included in repo due to file size.
Download from Kaggle: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

**Phase 2 processed dataset (data_processed.csv):**
https://buffalo.app.box.com/file/2159530636364

**Phase 3 combined dataset (us_accidents_hpms_final_100m.csv):**
https://buffalo.box.com/s/ygyw80z3n803ls70li7t7dmbf7dhbxil

---

## Repository Structure

```
DIC-Project_1/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── databricks/
│       ├── --- PRIMARY NOTEBOOKS (run these for grading) ---
│       ├── 01_bronze_layer_hpms.ipynb      <- Bronze: accidents + HPMS
│       ├── 02_silver_layer_hpms.ipynb      <- Silver: accidents + HPMS
│       ├── 03_gold_layer_hpms.ipynb        <- Gold: accidents + HPMS
│       ├── Random_Forest_HPMS.ipynb        <- MLlib Random Forest
│       ├── final..logisticregression.ipynb <- MLlib Logistic Regression
│       ├── 04_MLlib_DecisionTree.ipynb     <- MLlib Decision Tree
│       ├── --- SUPPLEMENTARY NOTEBOOKS (medallion on original dataset) ---
│       ├── 01_bronze_layer_t.ipynb         <- Bronze: original accidents only
│       ├── 02_silver_layer_t.ipynb         <- Silver: original accidents only
│       └── 03_gold_layer_t.ipynb           <- Gold: original accidents only
└── src/
    ├── data_collection.py
    ├── data_cleaning.py
    ├── eda.py
    ├── models/
    │   ├── train_random_forest.py
    │   ├── train_xgboost.py
    │   ├── train_mlp.py
    │   └── train_knn.py
    └── mcp/
        ├── server.py
        └── README.md
```

---

## Setup

```
pip install -r requirements.txt
```

---

## Phase 3 – Databricks, Delta Lake, and Spark MLlib

Estimated run time for the primary pipeline: about 10 minutes.

### Before You Start – One-Time Setup

Sign in to Databricks and run the following in any notebook to create the required folders:

```python
dbutils.fs.mkdirs("/Volumes/workspace/default/uploads/sparkml_cache")
dbutils.fs.mkdirs("/Volumes/workspace/default/uploads/models")
dbutils.fs.mkdirs("/Volumes/workspace/default/mama/sparkml_cache")
dbutils.fs.mkdirs("/Volumes/workspace/default/mama/models")
```

Then upload the combined dataset:
1. On the left sidebar go to Data Ingestion, then Upload files to a volume
2. Create a volume with catalog: workspace, schema: default, name: uploads
3. Also create a second volume with catalog: workspace, schema: default, name: mama
4. Upload us_accidents_hpms_final_100m.csv to the uploads volume

---

## Primary Notebooks – Run These for Grading

These notebooks implement the full Phase 3 requirements: medallion architecture on the combined accidents + HPMS dataset and three Spark MLlib models.

Run order:

```
01_bronze_layer_hpms
        |
        v
02_silver_layer_hpms
        |
        v
03_gold_layer_hpms
        |
        v (run independently, any order)
Random_Forest_HPMS   |   04_MLlib_DecisionTree   |   final..logisticregression
```

---

### 01_bronze_layer_hpms.ipynb

What it does:
- Reads the raw CSV from /Volumes/workspace/default/uploads/us_accidents_hpms_final_100m.csv
- Renames columns with special characters: Temperature(F) to Temperature_F, Humidity(%) to Humidity_pct, Visibility(mi) to Visibility_mi, Wind_Speed(mph) to Wind_Speed_mph, Distance(mi) to Distance_mi, Pressure(in) to Pressure_in, Precipitation(in) to Precipitation_in
- Saves as Delta table with no other changes

Tables created: workspace.default.bronze_accidents_hpms

Expected output:
```
Loaded [number] rows
Bronze layer done! Rows: [number]
```

---

### 02_silver_layer_hpms.ipynb

Prerequisite: 01_bronze_layer_hpms must run first.

What it does:
- Reads from workspace.default.bronze_accidents_hpms
- Extracts hour, month, day_of_week from Start_Time
- Drops rows with nulls in: Severity, Start_Time, Start_Lat, Start_Lng, Temperature_F, Visibility_mi, SPEED_LIMIT, AADT
- Filters to top 25 weather conditions and standardizes into 7 groups: Rainy_Thunder, Rain_Drizzle, Snow, Clear_Fair, Cloudy_Windy, Cloudy, other
- Removes outliers using IQR for Wind_Speed_mph, Temperature_F, AADT, and SPEED_LIMIT
- Removes duplicates
- Saves cleaned data

Tables created: workspace.default.silver_accidents_hpms

Expected output:
```
Loaded [number] rows from Bronze
Time features extracted
Weather conditions standardized
After outlier removal: [number] rows
Silver layer done! Rows: [number]
```

---

### 03_gold_layer_hpms.ipynb

Prerequisite: 02_silver_layer_hpms must run first.

What it does:
- Reads from workspace.default.silver_accidents_hpms
- Creates three aggregated Gold tables

gold_accidents_hpms: accidents grouped by State, Severity, Weather_Condition, Sunrise_Sunset with averages for temperature, visibility, wind, speed limit, AADT, and lane count

gold_speed_severity: accidents grouped by SPEED_LIMIT and Severity showing how road speed limits relate to accident severity and traffic volume

gold_traffic_severity: accidents bucketed by AADT into Low/Medium/High/Very High showing how traffic volume relates to accident severity

Tables created:
- workspace.default.gold_accidents_hpms
- workspace.default.gold_speed_severity
- workspace.default.gold_traffic_severity

Expected output:
```
Loaded [number] rows from Silver
Gold layer 1 done!
Gold layer 2 done - Speed limit vs Severity
Gold layer 3 done - Traffic volume vs Severity
Top 10 States by Accident Count: [table]
```

---

### Random_Forest_HPMS.ipynb

Prerequisite: 02_silver_layer_hpms must run first.

What it does:
- Sets SPARKML_TEMP_DFS_PATH to /Volumes/workspace/default/uploads/sparkml_cache to prevent 1GB cache overflow
- Reads from workspace.default.silver_accidents_hpms
- Limits to 50,000 rows to stay under the serverless 1GB cache limit
- Uses 14 features: Distance_mi, Temperature_F, Visibility_mi, Wind_Speed_mph, Start_Lat, Start_Lng, hour, day_of_week, month, AADT, SPEED_LIMIT, THROUGH_LANES, F_SYSTEM, ACCESS_CONTROL
- Binary label: Severity 1-2 = 0 (minor), Severity 3-4 = 1 (severe)
- Splits 80/20 into train and test before training (no data leakage)
- Pipeline: VectorAssembler, StandardScaler, RandomForestClassifier
- CrossValidator: numTrees=50, maxDepth=10, 3 folds, parallelism=1
- Logs to MLflow under Phase3_RandomForest_HPMS
- Generates confusion matrix and feature importance plots
- Saves model to /Volumes/workspace/default/uploads/models/phase3_random_forest

Expected output:
```
Train: ~40000  Test: ~10000
Training Random Forest with CrossValidator...
Best numTrees: 50  Best maxDepth: 10
F1 Score: ~0.84   Accuracy: ~0.85
Predictions count: ~10000
Test df count: ~10000
```

Phase 2 vs Phase 3 comparison:
- Phase 2 F1: 0.59 (sklearn, full dataset, no HPMS features)
- Phase 3 F1: 0.84 (MLlib, 50k sample, with HPMS features)

Top features by importance: Distance_mi (32%), F_SYSTEM (13%), Start_Lat (12%), AADT (5%), SPEED_LIMIT (4%)

---

### 04_MLlib_DecisionTree.ipynb

What it does:
- Sets SPARKML_TEMP_DFS_PATH to /Volumes/us-accidents/default/temp/sparkml_cache
- Reads from the silver accidents table
- Uses 3 features: Distance_mi, Temperature_F, Hour
- Binary label: Severity 1-2 = 0, Severity 3-4 = 1
- Limits to 50,000 rows
- Pipeline: VectorAssembler, StandardScaler, DecisionTreeClassifier
- CrossValidator: maxDepth in [3, 5, 7], minInstancesPerNode in [1, 5, 10], 3 folds
- Logs to MLflow under Phase3_DecisionTree_MLlib
- Saves model to /Volumes/us-accidents/default/temp/models/phase3_decision_tree

Note for graders: This notebook uses catalog us-accidents which is a specific workspace setup. Update the silver table path and Volume paths at the top of the notebook to match your workspace before running. Change /Volumes/us-accidents/default/temp/ to /Volumes/workspace/default/uploads/ if needed.

Expected output:
```
Data ready for training: [number] rows processed.
Training Decision Tree with CrossValidator...
Best maxDepth: [value]
Best minInstancesPerNode: [value]
Decision Tree training complete!
```

---

### final..logisticregression.ipynb

What it does:
- Sets SPARKML_TEMP_DFS_PATH to /Volumes/workspace/default/mama/sparkml_cache
- Reads from workspace.default.us_accidents_hpms_final_100_m_features (pre-engineered feature table)
- Limits to 500,000 rows
- Uses 13 features: Distance_mi, Temperature_F, Humidity_pct, Visibility_mi, Wind_Speed_mph, Precipitation_in, Hour, Month, AADT, SPEED_LIMIT, THROUGH_LANES, URBAN_ID, hpms_distance_m
- 4-class severity target: label = Severity - 1
- Splits 80/20 into train and test
- Pipeline: VectorAssembler, StandardScaler, LogisticRegression (multinomial, maxIter=20)
- CrossValidator: regParam in [0.01, 0.1, 1.0], elasticNetParam in [0.0, 0.5, 1.0], 3 folds
- Logs to MLflow under Phase3_LogisticRegression_HPMS_4class_500k
- Saves model to /Volumes/workspace/default/mama/models/logistic_regression_hpms_4class_500k

Note for graders: This notebook requires the pre-engineered feature table workspace.default.us_accidents_hpms_final_100_m_features and the mama volume. Create the mama volume under workspace.default before running.

Expected output:
```
rows ready: 386010
train: 308874   test: 77136
training...
best regParam: 0.01
best elasticNetParam: 0.0
Accuracy: 0.7413
F1 Score: 0.7032
Weighted Precision: 0.7044
Weighted Recall: 0.7413
saved to: /Volumes/workspace/default/mama/models/logistic_regression_hpms_4class_500k
```

---

## Supplementary Notebooks – Medallion Architecture on Original Dataset

These notebooks show the medallion architecture applied to the original US Accidents dataset without HPMS. They are included for reference but are not required to run for grading.

01_bronze_layer_t.ipynb — loads original US_Accidents_March23.csv into a Bronze Delta table

02_silver_layer_t.ipynb — cleans and transforms the original accidents data into a Silver Delta table

03_gold_layer_t.ipynb — creates business aggregates from the original accidents Silver table

---

## Troubleshooting

Py4JSecurityException on VectorAssembler:
- Create a brand new notebook manually in Databricks (do not import the ipynb file)
- Copy and paste the cell contents directly into the new notebook
- This is a known Databricks Free Edition environment versioning issue

ML_CACHE_SIZE_OVERFLOW_EXCEPTION (1GB cache limit):
- Click Run then Clear state and run all to reset the session cache
- The notebooks redirect cache using SPARKML_TEMP_DFS_PATH but the session must be fresh

Catalog or schema not found:
- Primary notebooks use workspace.default as catalog and schema
- If your workspace uses a different name, update the table references at the top of each notebook

Column not found in Silver layer:
- Bronze must run first as Silver reads from workspace.default.bronze_accidents_hpms

Volume not found:
- Create the uploads volume under workspace.default for the main pipeline
- Create the mama volume under workspace.default for the Logistic Regression notebook

---

## Phase 2 – Machine Learning Models

### How to Run

```
cd src/models
```

Random Forest:
```
python train_random_forest.py
```

XGBoost:
```
python train_xgboost.py
```

MLP Neural Network:
```
python train_mlp.py
```

Note: MLP is slow on CPU. Use Colab for GPU training.

k-NN:
```
python train_knn.py
```

Each script saves the trained model to models/ and figures to figures/.

### MCP Server

The Random Forest model is deployed as an MCP server for Claude Desktop. See src/mcp/README.md for full setup instructions. Model pkl files are not included in the repo — run the training scripts first to generate them.

---

## Phase 1 – Exploratory Data Analysis

### How to Run

1. Place US_Accidents_March23.csv in data/raw/
2. Run data cleaning:
```
python src/data_cleaning.py
```
3. Run EDA:
```
python src/eda.py
```

---

## Dependencies

See requirements.txt. Key packages:
- pandas, numpy — data processing
- scikit-learn, imbalanced-learn — Phase 2 ML models
- xgboost — gradient boosting
- torch — MLP neural network
- mcp — MCP server
- matplotlib, seaborn, plotly — visualization
- pyspark — Phase 3 Spark MLlib
- mlflow — Phase 3 experiment tracking

---

## Reproducibility

Random seeds are set to 42 for Phase 2 models and 14 for the Phase 3 Random Forest. The Logistic Regression and Decision Tree use seed 42. The full Phase 3 primary pipeline was verified on a fresh Databricks workspace following the steps above.
