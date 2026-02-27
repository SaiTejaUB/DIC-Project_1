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

df = pd.read_csv('../data/raw/US_Accidents_March23.csv')

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


df['State'] = df['State'].str.strip()
df['Street'] = df['Street'].str.strip()
df['County'] = df['County'].str.strip()
df['City'] = df['City'].str.strip()





#Weater Conditions category reduction

top_25_weather_conditions = (df['Weather_Condition'].value_counts().head(25).index.tolist())
df = df[df['Weather_Condition'].isin(top_25_weather_conditions)]


df['Weather_Condition'] = df['Weather_Condition'].str.strip()
df['Weather_Condition'] = df['Weather_Condition'].str.replace(r'[/\\-]', ' ', regex=True)
df['Weather_Condition'] = df['Weather_Condition'].str.replace(r'\s+', ' ', regex=True)
df['Weather_Condition'] = df['Weather_Condition'].str.lower()


def consolidate_weather_conditions(condition):
    if any(x in condition for x in ['light rain with thunder', 'thunder','t storm']):
        return 'Rainy_Thunder'
    elif any(x in condition for x in ['light rain', 'light drizzle', 'rain']):
        return 'Rain_Drizzle'
    elif any(x in condition for x in ['light snow', 'snow']):
        return 'Snow'
    elif any(x in condition for x in ['fair', 'clear']):
        return 'Clear_Fair'
    elif any(x in condition for x in ['mostly cloudy windy', 'cloudy windy']):
        return 'Cloudy_Windy'
    elif any(x in condition for x in ['mostly cloudy', 'cloudy', 'partly cloudy', 'overcast', 'scattered clouds']):
        return 'Cloudy'
    else:
        return condition

df['Weather_Condition'] = df['Weather_Condition'].apply(consolidate_weather_conditions)

#remove duplicates if any
df.drop_duplicates(inplace=True)


#remove outliers 
# Calculate Q1, Q3, and IQR for wind_speed
q1_wind_speed = df['Wind_Speed(mph)'].quantile(0.25)
q3_wind_speed = df['Wind_Speed(mph)'].quantile(0.75)
iqr_wind_speed = q3_wind_speed - q1_wind_speed

lower_bound_wind_speed = q1_wind_speed - 1.5 * iqr_wind_speed
upper_bound_wind_speed = q3_wind_speed + 1.5 * iqr_wind_speed

# Calculate Q1, Q3, and IQR for temp
q1_temperature = df['Temperature(F)'].quantile(0.25)
q3_temperature = df['Temperature(F)'].quantile(0.75)
iqr_temperature = q3_temperature - q1_temperature

lower_bound_temperature = q1_temperature - 3.0 * iqr_temperature
upper_bound_temperature = q3_temperature + 1.5 * iqr_temperature

df_no_outliers = df[
    (df['Wind_Speed(mph)'] >= lower_bound_wind_speed) & (df['Wind_Speed(mph)'] <= upper_bound_wind_speed) &
    (df['Temperature(F)'] >= lower_bound_temperature) & (df['Temperature(F)'] <= upper_bound_temperature)
].copy()
df = df_no_outliers

#final cleaned dataset saved as csv
df.to_csv('../data/processed/data_processed.csv')