import pandas as pd           
import numpy as np           
import matplotlib.pyplot as plt 
import seaborn as sns        

def day_or_night(start_time):
    hour = start_time.hour
    if (hour >= 6 and hour <= 17):
      return 'Day'
    else:
      return 'Night'

df = pd.read_csv('/workspaces/DIC-Project_1/data/raw/placeholder_raw.txt')

df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')


# Handling missing values for sunrise_sunset column
nan_mask = df['Sunrise_Sunset'].isna()
calculated_twilight_periods = df.loc[nan_mask, 'Start_Time'].apply(day_or_night)
df.loc[nan_mask, 'Sunrise_Sunset'] = calculated_twilight_periods
print(df['Sunrise_Sunset'].value_counts())

#We cant do much with these columns 
df.drop(columns=['ID','Source','Weather_Timestamp','Airport_Code','Timezone','Country','Description','End_Lat','End_Lng','Wind_Chill(F)','Wind_Direction','Civil_Twilight','Humidity(%)','Pressure(in)','Precipitation(in)','Nautical_Twilight','Astronomical_Twilight'],inplace=True)

#filling the NaN values in Wind Speed and Visibility
weather_condition_summary = df.dropna(subset=['Weather_Condition']).groupby('Weather_Condition')[['Wind_Speed(mph)', 'Visibility(mi)']].mean()


df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(df['Weather_Condition'].map(weather_condition_summary['Wind_Speed(mph)']))
df['Visibility(mi)'] = df['Visibility(mi)'].fillna(df['Weather_Condition'].map(weather_condition_summary['Visibility(mi)']))

print("Null values after filling 'Wind_Speed(mph)' and 'Visibility(mi)' by Weather_Condition mean:")
print(f"Wind_Speed(mph): {df['Wind_Speed(mph)'].isna().sum()}")
print(f"Visibility(mi): {df['Visibility(mi)'].isna().sum()}")

df.dropna(subset=['Weather_Condition','Street','City','Wind_Speed(mph)','Temperature(F)'],inplace=True)