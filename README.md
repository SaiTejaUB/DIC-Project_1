Project Description
This project performs exploratory data analysis (EDA) on the US Accidents (March 2023) dataset. The purpose of this project is to understand the patterns of accidents in terms of severity, time, weather conditions, and states.
The project mainly focuses on analyzing the structure of the dataset and identifying patterns before applying any modeling techniques.
Dataset Used
US_Accidents_March23.csv
Important columns selected from the dataset:
Severity
Start_Time
State, City, County
Weather_Condition
Temperature(F)
Humidity(%)
Pressure(in)
Visibility(mi)
Wind_Speed(mph)
Precipitation(in)
Distance(mi)
The dataset is large and is not uploaded to the repository.
What Was Done in This Project

1. Data Loading and Checking
Loaded the dataset using pandas
Checked the shape of the dataset
Viewed first few rows
Selected useful columns for analysis
2. Severity Distribution
Used value_counts() to analyze severity levels
Severity 2 dominates most of the dataset
The dataset is highly imbalanced
This indicates that if predictive modeling is performed later, class imbalance must be considered.
3. Missing Value Analysis
Calculated missing values and their percentages
Precipitation shows higher missing percentage
Wind Speed also has some missing values
Core variables like State and City have minimal missing data
4. Visibility Analysis
Plotted histogram of Visibility
Most accidents occur when visibility is between 5 and 20 miles
Accidents also occur during high visibility conditions
Used describe() to check summary statistics
5. Temperature Analysis
Plotted temperature distribution
Most values lie between 0°F and 100°F
Extreme values such as -89°F were observed
6. Visibility vs Severity
Created a boxplot comparing visibility across severity levels
No strong visible difference observed across severity groups
7. State-Level Analysis
Plotted top 10 states by accident count
California has the highest accident count
This may be influenced by population size and traffic volume
8. Correlation Analysis
Created correlation heatmap for numeric weather variables
Most relationships are weak
Negative correlation observed between humidity and temperature
No strong linear relationships were found
Libraries Used
Python 3
Pandas
NumPy
Matplotlib
Seaborn
How to Run
pip install pandas numpy matplotlib seaborn folium
Place the dataset file US_Accidents_March23.csv in the project folder.
Update the file path if needed.
Run the script: python untitled197.py
Conclusion
From this exploratory analysis:
Accident severity is highly imbalanced.
Morning and evening hours have higher accident frequency.
Seasonal patterns are visible.
Weather variables show mostly weak correlations.

## How to Run (Step-by-Step)

1. Manually unzip the data file.  
   (data_collection.py will only work with Kaggle API)

2. Save the dataset in the data/raw folder.  
   Example:  
   data/raw/US_Accidents_March23.csv  

3. Run EDA with the correct path to the dataset:  
   python eda.py  

4. Run data cleaning with the correct path to the dataset:  
   python data_cleaning.py  

