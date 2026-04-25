# US Accidents Severity Analysis
EAS 587 – Data Intensive Computing

## Project Overview
This project performs exploratory data analysis (EDA) and machine learning modeling on the US Accidents (March 2023) dataset. The purpose is to understand patterns of accidents in terms of severity, time, weather conditions, and states, and to build predictive models for accident severity.

## Dataset
US_Accidents_March23.csv — not included in the repo due to size.
Download from Kaggle: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

Cleaned dataset (data_processed.csv): https://buffalo.app.box.com/file/2159530636364

Important columns selected from the dataset:
- Severity
- Start_Time
- State, City, County
- Weather_Condition
- Temperature(F)
- Humidity(%)
- Pressure(in)
- Visibility(mi)
- Wind_Speed(mph)
- Precipitation(in)
- Distance(mi)

## Repository Structure
```
DIC-Project_1/
├── data/
│   ├── raw/                  # place US_Accidents_March23.csv here
│   └── processed/            # data_processed.csv goes here
├── figures/                  # output plots saved here
├── models/                   # trained model pkl files saved here
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── models/
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   ├── train_mlp.py
│   │   └── train_knn.py
│   └── mcp/
│       ├── server.py
│       └── README.md
├── requirements.txt
└── README.md
```

## Setup
```
pip install -r requirements.txt
```

---

## Phase 1 – Exploratory Data Analysis

### How to Run
1. Place `US_Accidents_March23.csv` in `data/raw/`
2. Run EDA:
   ```
   python src/eda.py
   ```
3. Run data cleaning (outputs `data_processed.csv` to `data/processed/`):
   ```
   python src/data_cleaning.py
   ```

### What Was Done

**1. Data Loading and Checking**
- Loaded the dataset using pandas
- Checked the shape of the dataset
- Viewed first few rows
- Selected useful columns for analysis

**2. Severity Distribution**
- Used value_counts() to analyze severity levels
- Severity 2 dominates most of the dataset
- The dataset is highly imbalanced
- This indicates that if predictive modeling is performed later, class imbalance must be considered

**3. Missing Value Analysis**
- Calculated missing values and their percentages
- Precipitation shows higher missing percentage
- Wind Speed also has some missing values
- Core variables like State and City have minimal missing data

**4. Visibility Analysis**
- Plotted histogram of Visibility
- Most accidents occur when visibility is between 5 and 20 miles
- Accidents also occur during high visibility conditions
- Used describe() to check summary statistics

**5. Temperature Analysis**
- Plotted temperature distribution
- Most values lie between 0°F and 100°F
- Extreme values such as -89°F were observed

**6. Visibility vs Severity**
- Created a boxplot comparing visibility across severity levels
- No strong visible difference observed across severity groups

**7. State-Level Analysis**
- Plotted top 10 states by accident count
- California has the highest accident count
- This may be influenced by population size and traffic volume

**8. Correlation Analysis**
- Created correlation heatmap for numeric weather variables
- Most relationships are weak
- Negative correlation observed between humidity and temperature
- No strong linear relationships were found

### Conclusion
- Accident severity is highly imbalanced
- Morning and evening hours have higher accident frequency
- Seasonal patterns are visible
- Weather variables show mostly weak correlations with severity

---

## Phase 2 – Machine Learning & MCP Deployment

### How to Run
All model scripts are in `src/models/`. Run from inside that directory:

```
cd src/models
```

**Random Forest (outside class):**
```
python train_random_forest.py
```

**XGBoost (outside class):**
```
python train_xgboost.py
```

**MLP – Neural Network (outside class):**
```
python train_mlp.py
```
Note: MLP training on the full dataset is slow on CPU. A Colab version is available for GPU training.

**k-NN:**
```
python train_knn.py
```

Each script saves:
- Trained model to `models/`
- Figures (confusion matrix, feature importance, etc.) to `figures/`

### MCP Server
The trained Random Forest model is deployed as an MCP server for Claude Desktop. See `src/mcp/README.md` for full setup instructions.

Note: The model pkl files are not included in the repo due to size. Run the training scripts first to generate them before starting the MCP server.

---

## Dependencies
See `requirements.txt`. Key packages:
- scikit-learn, imbalanced-learn — ML models
- torch — MLP neural network
- xgboost — XGBoost model
- mcp — MCP server
- pandas, numpy, matplotlib, seaborn — data processing and visualization
## Phase 3 – Databricks, Delta Lake & Spark MLlib

### How to Run
1. Upload the combined dataset to Databricks as a table:
## Phase 3 – Databricks, Delta Lake & Spark MLlib

### How to Run
1. Upload the combined dataset to Databricks as a table
2. Run the Databricks notebooks in this order
3. Run the Spark MLlib logistic regression notebook on the combined dataset.
4. The final saved Spark model was stored at:/Volumes/workspace/default/mama/models/logistic_regression_hpms_4class_500k

### What Was Done

**1. Databricks Setup**
- Created Databricks notebooks for Phase 3
- Loaded the project data into Databricks
- Used Delta tables for Bronze, Silver, and Gold layers
- Built the Spark MLlib workflow in Databricks

**2. Bronze Layer**
- Loaded the raw accident dataset into Databricks
- Saved the raw dataset as a Delta table
- No cleaning was done in this step

**3. Silver Layer**
- Read the data from the Bronze table
- Applied cleaning and preprocessing
- Fixed data types
- Handled missing values
- Removed duplicates
- Standardized selected values

**4. Gold Layer**
- Created the final model-ready dataset
- Included accident, weather, and time variables
- Included roadway variables from the second dataset
- Used this Gold table for Spark MLlib modeling

**5. Additional Data Source**
- Added **HPMS Spatial NHS Sections – 2024** as the second dataset
- This dataset was useful because it adds road and traffic information that is not in the original US Accidents dataset
- Important added features included:
- `AADT`
- `SPEED_LIMIT`
- `THROUGH_LANES`
- `ACCESS_CONTROL`
- `FACILITY_TYPE`
- `F_SYSTEM`
- `URBAN_ID`
- `hpms_distance_m`

**6. Combining the Datasets**
- Used a nearest-road spatial join
- Each accident point was matched to the nearest HPMS road segment
- Only matches within **100 meters** were kept
- The HPMS road features were added to the accident record
- This created the final combined dataset used for Phase 3 modeling

**7. Spark MLlib Logistic Regression**
- Re-implemented logistic regression in Databricks using Spark MLlib
- Used a Spark pipeline with:
- `VectorAssembler`
- `StandardScaler`
- `LogisticRegression`
- `CrossValidator`
- Kept the original **4 severity classes**
- Used a **500,000-row subset** to keep the Spark run manageable
- Final model results:
- **Accuracy = 0.7413**
- **F1 Score = 0.7032**
- **Weighted Precision = 0.7044**
- **Weighted Recall = 0.7413**

**8. Comparison with Earlier Work**
- Earlier Phase 2 baseline result:
- **Accuracy = 0.8279**
- **Weighted F1 = 0.7698**
- **Macro F1 = 0.2701**
- The Spark MLlib logistic regression did not improve the overall metrics
- However, the earlier model was also affected by class imbalance and mainly performed better on the dominant class
- The main Phase 3 contribution was:
- scalable Databricks implementation
- successful Spark MLlib pipeline
- integration of the second dataset
- multi-source severity prediction workflow

**9. Insights from the Combined Data**
- Severity depends not only on weather and time, but also on roadway context
- Traffic volume, speed limit, and lane count added useful information
- The severity target is still highly imbalanced, so rare classes are still difficult
- HPMS strengthened the feature space even though it did not fully solve the minority-class prediction problem

### Conclusion
- Phase 3 moved the project into Databricks using Spark, Delta Lake, and Spark MLlib
- The US Accidents dataset was combined with HPMS to create a stronger multi-source dataset
- Logistic regression was successfully implemented in Spark MLlib on the combined dataset
- The main result of Phase 3 was the scalable re-implementation of the workflow and the use of the second dataset in Databricks




