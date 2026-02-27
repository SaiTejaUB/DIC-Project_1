import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
PATH = "/content/drive/MyDrive/US_Accidents_March23.csv"
selected_colums = [
    "Severity","Start_Time",
    "State", "City","County",
    "Weather_Condition", "Sunrise_Sunset",
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)","Wind_Speed(mph)","Precipitation(in)",
    "Distance(mi)"
]
df = pd.read_csv(PATH, usecols=selected_colums, low_memory=False)
print("Shape of the dataset",df.shape)
df.head()
severity_counting = df["Severity"].value_counts()
print(severity_counting)
# the distibution of the Severity is highly imbalanced it is dominated by the severity 2 with almost 80 percent of the distibution having severity 2 the other severity 1,3,4 are only small proportion of the data. modeling will also get influnce the data imbalance may get biased towards the single class
missing=df.isnull().sum()
missing_percentage=(missing/len(df))*100
missing_df=missing_df.sort_values("Missing percentage",ascending=False)
missing_df
# the values show that 28.5 percentage of the Precipitation so we can say it is a incomplete variable in the dataset and the other colums wind_speed had 7.39 percentage of the missing values while the other colums have only 2 percentage missing. while the other variables whicvha re core like city etc are not
df_samples= df.sample(n=900000,random_state=45)
plt.hist(df_samples["Visibility(mi)"].dropna())
plt.title("Distribution of the visibility")
plt.xlabel("number of Visibile miles")
plt.ylabel("Frequency")
plt.show()
# the distibution of the visibility is concentrated between the 5 and 20 which means the most accidents took place in this range with some values streaching into above 20 miles which means that accidents doesent take place only in the low visibilty but can also take place in high visibility
df["Visibility(mi)"].describe()
#i found that the visibility of the most accidents 25% and 50 percentage and 75 percentage  is in the 10 miles range it can be maximum reported value be 10 miles
plt.hist(df_samples["Temperature(F)"].dropna())
plt.title("Distribution of the temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()
# the the distibution of the 30f and the 100f which mean most accidents take place in this distibution and the peak is between the 60 and the 90f most values are between the 0f and the 100f and below the 0f we can say that that temp is outliner or that weather conditon is rare
df["Temperature(F)"].describe()
#the mean temp of the thing 61 percentage  and 64 is the median where as the min is -89 which i feel is impossible and it is very extream this may an outline or error same with the extream temp so duning cleaning we have to remove them as it is too extream for temperature

