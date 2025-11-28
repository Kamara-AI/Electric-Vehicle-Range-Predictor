Electric Vehicle Range Predictor

Project Overview

This is a full-stack machine learning project designed to predict the real-world driving range (in kilometers) of electric vehicles (EVs) based on key technical specifications. The solution uses a Random Forest Regressor model trained on a comprehensive dataset of modern electric cars.

The application is deployed live on Streamlit Cloud, allowing users to interactively test how changes to battery capacity, efficiency, and drivetrain impact the estimated range.

🔗 Live Application Demo

Click the link below to access the deployed application:

https://electric-vehicle-range-predictor-4rfvlaog8yp8fw2tneib98.streamlit.app/

⚙️ Technology Stack

Category

Tools/Libraries

Model

Random Forest Regressor (Scikit-learn)

Data Processing

Pandas, NumPy, StandardScaler

Web App

Streamlit

Deployment

Streamlit Cloud, GitHub

Language

Python

📊 Model Performance

The model was trained to minimize the error between the predicted range and the actual observed range.

Model: Random Forest Regressor

Key Metric: Mean Absolute Error (MAE)

Result: MAE = 14.93 km
(On average, the predicted range is within 15 km of the actual range, demonstrating high accuracy.)

💡 How the Prediction Works

The prediction pipeline includes several critical steps to ensure accurate results:

Data Ingestion: User inputs (e.g., Battery Capacity, Drivetrain) are collected via the Streamlit UI.

Feature Alignment: The few user inputs are transformed via One-Hot Encoding to match the 100+ features the model was trained on. Unprovided features (like other brands) are set to zero.

Scaling: The aligned features are scaled using the saved StandardScaler.

Prediction: The scaled features are fed into the pickled model_rf.pkl to generate the final range estimate.

🚀 Getting Started Locally

To clone and run this project on your machine, follow these steps:

Prerequisites

Python 3.8+

The following ML artifacts must be present in the root directory: model_rf.pkl, scaler.pkl, feature_names.pkl, categories.pkl.

Installation

Clone the repository:

git clone- https://github.com/Kamara-AI/Electric-Vehicle-Range-Predictor


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install dependencies:

pip install -r requirements.txt


Running the App

After installation, start the Streamlit application:

streamlit run app.py


The application will open automatically in your browser.
