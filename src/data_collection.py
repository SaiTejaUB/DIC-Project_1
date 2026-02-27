import kaggle
import pandas as pd
import kagglehub
import os
import zipfile
# Authenticate with the Kaggle API using the credentials stored in the .kaggle/kaggle.json file
# kaggle.api.authenticate()


# kaggle.api.dataset_download_files('sobhanmoosavi/us-accidents', path='data/raw', unzip=True)



# setting the environment variable to specify the cache directory for kagglehub
os.environ['KAGGLEHUB_CACHE'] = 'DIC-Project_1/data/raw'

# Downloading the dataset using kagglehub
dataset = kagglehub.dataset_download('sobhanmoosavi/us-accidents')

# checking if the dataset was downloaded successfully
print(f"Dataset successfully downloaded to: {dataset}")

#unzipping the downloaded file
zip_path = os.path.join(dataset, "US_Accidents_March23.csv.zip")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset)

# check the first few rows of the dataset
df = pd.read_csv('data/raw/datasets/US_Accidents_March23.csv')

print(df.head())
