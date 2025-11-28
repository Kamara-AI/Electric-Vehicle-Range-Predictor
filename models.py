import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from utils import load_and_clean_data, engineer_features, DATA_PATH

# Save aritifact
def save_artifacts(model, scaler, feature_names):
    """Saves the trained model, scaler, and feature list to disk."""
    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved 'model_rf.pkl'")

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved 'scaler.pkl'")
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("Saved 'feature_names.pkl'")
    
    # We also need a file containing the unique categories for the app's dropdowns
    # We can rely on the engineer_features function to get the data
    df_clean = load_and_clean_data(DATA_PATH)
    unique_categories = {
        col: df_clean[col].dropna().unique().tolist()
        for col in df_clean.select_dtypes(include=['object']).columns
    }
    with open('categories.pkl', 'wb') as f:
        pickle.dump(unique_categories, f)
    print("Saved 'categories.pkl' with unique values for the web app.")


    #Training Function
def train_and_evaluate_models(file_path):
    # 1. Load, clean, and feature engineer data
    df_clean = load_and_clean_data(file_path)
    df_final = engineer_features(df_clean)

    # 2. Define features (X) and target (y)
    X = df_final.drop("range_km", axis=1)
    y = df_final ["range_km"]

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Scaling
    #Initialize the scaler
    scaler = StandardScaler() 
    # Identify numerical columns to scale (all features are now numerical after encoding)
    # We exclude the encoded binary columns, but since the scaler handles 0s and 1s well,
    # we can safely scale all columns in X_train and X_test here for simplicity.

    # FIT the scaler ONLY on the training data (CRITICAL step to prevent data leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    # TRANSFORM the test data using the statistics learned from the training data
    X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easy use 
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Feature scaling completed.")

    # Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, 
                                                         random_state=42),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5)
    } 
    results = {}
    print("\n--- Starting Model Training and Evaluation ---")
    
    #Train and evaluation loop
    best_midel = None
    best_r2 = -float("inf")

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        #Train the model
        model.fit(X_train_scaled, y_train)
        #Make predictions
        y_pred = model.predict(X_test_scaled)
        #Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        #Store only what you need
        results[model_name] = {
            "MAE": mae, 
            "RMSE" : rmse,
            "R2": r2
            }
        
        #Track the best model based on R2 score 
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name

        print(f"{model_name}" )
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")
        print("_"*40) 

    # Extract the actual best model object
    final_model = models[best_model]
    print(f"\nBest Model: {best_model} with R2: {best_r2:.2f}")

    # Save the best model and scaler for web app
    save_artifacts(final_model, scaler, X.columns.tolist())

    # return scaler for future use
    return results, X_test, y_test, X_test_scaled, scaler 

if __name__ == "__main__":
    #Execute the pipeline when models.py is run directly
    all_results, X_test, y_test, X_test_scaled, scaler = train_and_evaluate_models(DATA_PATH)

    print("\n---Summary of model perfomance---")
    for name, metrics in all_results.items():
        print(f"{name}: MAE={metrics["MAE"]:.2f}, RMSE={metrics["RMSE"]:.2f}, R2={metrics["R2"]:.2f} ")
