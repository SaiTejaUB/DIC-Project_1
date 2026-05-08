# -*- coding: utf-8 -*-
# Exploratory analysis of the US Accidents dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "data/raw/US_Accidents_March23.csv"

# Preview a subset of the data to get a quick sense of structure and types
df_preview = pd.read_csv(PATH, nrows=100000, low_memory=False)

# Select a subset of columns that are relevant for initial analysis
selected_colums = [
    "Severity", "Start_Time",
    "State", "City", "County",
    "Weather_Condition", "Sunrise_Sunset",
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Distance(mi)"
]

df = pd.read_csv(PATH, usecols=selected_colums, low_memory=False)
print("Shape of the dataset", df.shape)
df.head()

# Check distribution of severity levels to understand class balance
severity_counting = df["Severity"].value_counts()
print(severity_counting)

# The severity distribution is highly imbalanced.
# Severity 2 dominates (around 80% of the data), while 1, 3, and 4 are much smaller.
# Any modeling will need to account for this imbalance or it may bias towards the majority class.

# Calculate missing values and missing percentages per column
missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "count of missing": missing,
    "Missing percentage": missing_percentage
})

missing_df = missing_df.sort_values("Missing percentage", ascending=False)
missing_df

# Precipitation has about 28.5% missing values and is effectively incomplete.
# Wind_Speed has around 7.39% missing, while most other columns are near 2%.
# Core variables like City and State have relatively low missingness, which is useful.

# Sample a subset to speed up plotting and EDA
df_samples = df.sample(n=900000, random_state=45)

# Distribution of visibility
plt.hist(df_samples["Visibility(mi)"].dropna())
plt.title("Distribution of the visibility")
plt.xlabel("Number of visible miles")
plt.ylabel("Frequency")
plt.show()

# Visibility is concentrated roughly between 5 and 20 miles.
# Most accidents happen in this range, but there are also accidents at higher visibility,
# which suggests accidents are not limited to low-visibility conditions.

df["Visibility(mi)"].describe()

# The 25th, 50th, and 75th percentiles of visibility are around 10 miles.
# The maximum reported visibility looks high and needs to be checked during cleaning.

# Distribution of temperature
plt.hist(df_samples["Temperature(F)"].dropna())
plt.title("Distribution of the temperature")
plt.xlabel("Temperature (F)")
plt.ylabel("Frequency")
plt.show()

# Most temperatures fall between about 30F and 100F, with a peak around 60–90F.
# Values below 0F or extreme highs may be outliers or rare weather conditions
# and should be checked in the cleaning step.

df["Temperature(F)"].describe()

# The mean temperature is around 61F and the median is about 64F.
# The minimum of -89F and other extreme values are likely errors or outliers,
# so they should be handled during cleaning.

# Boxplot of visibility by severity
df_samples.boxplot(column="Visibility(mi)", by="Severity")
plt.title("Visibility by the Severity")
plt.suptitle("")
plt.xlabel("Severity")
plt.ylabel("Visibility")
plt.show()

# Visibility is centered around ~10 miles across severity levels.
# This suggests visibility alone may not strongly differentiate severity,
# or there may be reporting issues in the data.

# Convert Start_Time to datetime and extract useful time features
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

df["Hours"] = df["Start_Time"].dt.hour

# Accidents by hour of day
counts = df["Hours"].value_counts().sort_index()
counts.plot(kind="line")
plt.title("Accidents by the Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of the Accidents")
plt.show()

# Accidents peak between about 5–7 AM and again in the evening.
# This lines up with commute times when traffic volume is high.
# Around 2 AM has the fewest accidents, likely due to low traffic.

df["Month"] = df["Start_Time"].dt.month

# Accidents by month
month_counting = df["Month"].value_counts().sort_index()

month_counting.plot(kind="line")
plt.title("Accidents by the Months")
plt.xlabel("Months")
plt.ylabel("Number of the Accidents")
plt.show()

# Accidents are lower in the middle months and increase towards the end of the year.
# This may relate to winter conditions, holidays, and increased travel in bad weather,
# suggesting seasonality and weather patterns influence accident counts.

# Top 10 states by accident count
state_counts = df["State"].value_counts().head(10)
state_counts.plot(kind="bar")
plt.title("Top 10 States by the number of Accidents")
plt.xlabel("State")
plt.ylabel("Number of the Accidents")
plt.show()

# California has the highest number of accidents in this dataset.
# This may reflect its larger population, more vehicles, and higher traffic volume.

import seaborn as sns

# Correlation among key continuous variables
cols = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Distance(mi)"
]

corr = df[cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# The heatmap shows mostly weak linear relationships between these variables.
# There is noticeable negative correlation between humidity and temperature,
# which matches the idea that higher humidity often goes with lower temperature
# and can reduce visibility in real-world conditions.
