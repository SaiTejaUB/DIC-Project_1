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
