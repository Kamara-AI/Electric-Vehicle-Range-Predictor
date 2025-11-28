#cell 1
import pandas as pd
import numpy as np
from pathlib import Path

# defining the location of our csv file
DATA_FILE_NAME = "electric_vehicles_spec_2025.csv"
 
 #defining the path to the directory containg the data
 # path(__file__).resolve() -> utils.py file
 # .parent -> the directory containing utils.py(which is the project root)
DATA_DIR = Path(__file__).resolve().parent

#combine the directory path and the file name
DATA_PATH = DATA_DIR/DATA_FILE_NAME

#creating a re-usable functtion for storing the clean data
def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    try:
        #code that might fail goes here
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {file_path}")
        return pd.DataFrame()
    print (f"Successfully loaded data with shape: {df.shape}")
    print ("--Start Initial Data Cleaning--")
    
    #Handling missing numeriacal data
    print("imputing missing values in numerical columns...")

    #indentifying numerical columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    for col in numerical_cols:
        #check if the column has any missing (NaN) values
        if df[col].isnull().any():
            #calculate the median of the non-missing values
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True) #imputing missing values with median
            print(f"Imputed missing values in column '{col}' with median value {median_val:.2f}")

    # 2. IMPUTATION FOR CATEGORICAL/STRING COLUMNS (using 'Missing' label)
    print("Handling missing CATEGORICAL data ('Missing' label imputation)...")
    # Identify string/object columns (these are typically categorical)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        #check if the column has any missing (NaN) values
        if df[col].isnull().any():
            #fill missing values with "Missing" label
            #makes NAN value a seprate category
            df[col].fillna("Missing", inplace=True) #imputing missing values with 'Missing' label
            print(f"Imputed missing values in column '{col}' with label 'Missing'")
#counting the total missing value after imputation
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values after imputation: {total_missing}")
    print("--Data Cleaning Completed--")
# 5. Return the fully cleaned DataFrame
    return df

# 2. SECOND FUNCTION: FEATURE ENGINEERING
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering steps, primarily converting categorical data 
    into a numerical format suitable for modeling.
    """
    print("Starting Feature Engineering...")

#--- 1. One-Hot Encoding for Categorical Data ---
# Columns that are not numerical will be converted using one-hot encoding
    categorical_cols = df.select_dtypes(include=["object"]).columns
    print(f"One-Hot Encoding the following categorical columns: {list(categorical_cols)}")
    #Use pd.get_dummies to perform one-hot encoding
    #drop_first=True prevents multicollinearity by dropping one redundant category column
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Orignal columns: {len(df.columns)}, New columns after encoding: {len(df_encoded.columns)}")
    print(f"Feature Engineering Completed. New shape of data: {df_encoded.shape}")
    return df_encoded