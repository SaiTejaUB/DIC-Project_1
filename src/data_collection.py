import kaggle
import pandas as pd

# Authenticate with the Kaggle API using the credentials stored in the .kaggle/kaggle.json file
kaggle.api.authenticate()

# Download the dataset from kaggle and unzip it to the specified path
kaggle.api.dataset_download_files('sobhanmoosavi/us-accidents', path='data/raw', unzip=True)

# check the first few rows of the dataset
df = pd.read_csv('data/raw/US_Accidents_March23.csv')

print(df.head())