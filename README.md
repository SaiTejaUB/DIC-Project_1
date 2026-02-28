Project Description
In this project I performed exploratory data analysis (EDA) on the US Accidents (March 2023) dataset. The purpose of this project is to understand the patterns of accidents in terms of severity, time, weather conditions, and states.
This project mainly focuses on analyzing the structure of the dataset and finding patterns before doing any modeling.
Dataset Used
US_Accidents_March23.csv
From the dataset I selected important columns such as:
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
The dataset is large, so it is not uploaded to the repository.
What I Did in This Project

1. Data Loading and Checking
Loaded the dataset using pandas.
Checked the shape of the dataset.
Viewed first few rows.
Selected only useful columns for analysis.

2. Severity Distribution
I checked how accidents are distributed across severity levels using value_counts().
Observation:
Severity 2 dominates most of the dataset.
Other severity levels (1, 3, 4) are much smaller.
So the dataset is highly imbalanced.
This means if we build a model later, it may get biased toward severity 2.

3. Missing Value Analysis
I calculated missing values and percentage of missing data.
Findings:
Precipitation has high missing percentage.
Wind Speed also has some missing values.
Core variables like State and City have very low missing data.
So some weather variables may need cleaning before modeling.

4. Visibility Analysis
Plotted histogram of Visibility.
Most accidents occur when visibility is between 5 and 20 miles.
Some accidents also happen even at high visibility.
Used describe() to check summary statistics.
This shows that accidents do not happen only during low visibility.

5. Temperature Analysis
Plotted temperature distribution.
Most values are between 0°F and 100°F.
Found extreme values like -89°F which seem unrealistic.
These extreme values may be outliers or data errors.
So temperature needs cleaning if we use it further.

6. Visibility vs Severity
I created a boxplot to compare visibility across severity levels.
Observation:
Visibility distribution looks similar across severity groups.
No strong visible difference from basic inspection.

7. State-Level Analysis
Plotted top 10 states by accident count.
California has highest accident count.
This may be due to larger population and more vehicles.

8. Correlation Analysis
Created correlation heatmap for numeric weather variables.
Most variables show weak linear relationship.
Negative correlation observed between humidity and temperature.
No very strong relationships were found.

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
