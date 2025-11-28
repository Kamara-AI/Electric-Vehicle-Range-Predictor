import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Configurations
st.set_page_config(
    page_title = "EV Range Predictor",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Load artifacts
#Define path to be saved from artifacts
MODEL_PATH = Path("model_rf.pkl")
SCALER_PATH = Path("scaler.pkl")
FEATURES_PATH = Path("feature_names.pkl")
CATEGORIES_PATH = Path("categories.pkl")

# Use st.cache_data to load files only once, speeding up the app
@st.cache_data(show_spinner="Loading Model and Data...")
def load_artifacts():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        with open(CATEGORIES_PATH, 'rb') as f:
            unique_categories = pickle.load(f)
        return model, scaler, feature_names, unique_categories
    except FileNotFoundError as e:
        st.error(f"Required artifact not found: {e.filename}. Please run 'models.py' first.")
        return None, None, None, None

model, scaler, feature_names, unique_categories = load_artifacts()

# Check if artifacts loaded successfully
if model is None:
    st.stop()

# --- 2. Define the Prediction Function ---
def predict_range(input_data: dict, model, scaler, feature_names):
    
    # 1. Create a DataFrame from the input data
    # The input_data dict keys correspond to the original column names
    df_raw = pd.DataFrame([input_data])
    
    # 2. Re-create the feature engineering steps for the single input
    
    # Identify categorical columns that were encoded (based on keys in categories.pkl)
    APP_CATEGORICAL_COLS = ['drivetrain', 'car_body_type']
    
    # Perform One-Hot Encoding (drop_first=True must match training)
    df_encoded = pd.get_dummies(df_raw, columns=APP_CATEGORICAL_COLS, drop_first=True)
    
    # 3. Align Columns (Crucial Step!)
    # Create an empty DataFrame with all 100+ feature names used during training
    X_predict = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in the values from the current encoded input
    for col in df_encoded.columns:
        if col in X_predict.columns:
            X_predict.loc[0, col] = df_encoded[col].values[0]
        
    # 4. Scale the input using the pre-trained scaler
    X_scaled = scaler.transform(X_predict)
    
    # 5. Make Prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction

# --- 3. Streamlit UI ---

st.title("⚡ EV Range Predictor ")
st.markdown("Use the features below to predict the EPA/WLTP range (in km) of a hypothetical electric vehicle.")
st.markdown("---")

# Dictionary to hold all user inputs
inputs = {}

# Layout the input fields side-by-side using columns
col1, col2, col3 = st.columns(3)

# --- COLUMN 1: Key Performance Metrics ---
with col1:
    st.header("Performance & Power")
    # Note: These values should cover the range found in your EDA
    inputs['battery_capacity_kWh'] = st.number_input(
        "Battery Capacity (kWh)", 
        min_value=10.0, max_value=200.0, value=75.0, step=5.0,
        help="The total energy capacity of the battery."
    )
    inputs['top_speed_kmh'] = st.number_input(
        "Top Speed (km/h)", 
        min_value=120, max_value=300, value=200, step=5
    )
    inputs['acceleration_0_100_s'] = st.number_input(
        "0-100 km/h (seconds)", 
        min_value=1.5, max_value=20.0, value=7.0, step=0.1, format="%.1f"
    )
    inputs['fast_charging_power_kw_dc'] = st.number_input(
        "Fast Charging Power (kW DC)", 
        min_value=50, max_value=350, value=150, step=10
    )

# --- COLUMN 2: Body & Drive Train ---
with col2:
    st.header("Physical Specifications")
    # Use unique_categories to populate dropdowns
    inputs['drivetrain'] = st.selectbox(
        "Drivetrain", 
        options=sorted(unique_categories.get('drivetrain', ['AWD', 'RWD', 'FWD'])),
        index=0
    )
    inputs['car_body_type'] = st.selectbox(
        "Car Body Type", 
        options=sorted(unique_categories.get('car_body_type', ['SUV', 'Sedan', 'Hatchback'])),
        index=0
    )
    
    # Example continuous features that were not cleaned/encoded
    inputs['length_mm'] = st.number_input("Length (mm)", min_value=3000, max_value=6000, value=4800, step=50)
    inputs['width_mm'] = st.number_input("Width (mm)", min_value=1500, max_value=2200, value=1900, step=25)
    inputs['height_mm'] = st.number_input("Height (mm)", min_value=1400, max_value=2000, value=1600, step=25)


# --- COLUMN 3: Other Specs (Must include all features the model expects) ---
with col3:
    st.header("Efficiency & Capacity")
    
    # Efficiency is a strong predictor
    inputs['efficiency_wh_per_km'] = st.number_input(
        "Energy Efficiency (Wh/km)", 
        min_value=100, max_value=300, value=180, step=5,
        help="Lower is better. Represents energy consumption per kilometer."
    )
    inputs['torque_nm'] = st.number_input("Torque (Nm)", min_value=100, max_value=1500, value=500, step=50)
    inputs['cargo_volume_l'] = st.number_input("Cargo Volume (L)", min_value=100, max_value=1000, value=500, step=10)
    inputs['seats'] = st.number_input("Number of Seats", min_value=2, max_value=7, value=5, step=1)
    
    # Set default values for features that were imputed as 'Missing' in the original data
    # These are placeholders required by the model, but we won't ask the user for them.
    # The clean_data and engineer_features steps handle these, so we need to ensure 
    # the feature names that resulted from encoding these show up correctly in X_predict.
    
    # We must ensure all other features that the model was trained on are initialized (e.g., brand_Missing, model_Missing, etc.)
    # The 'predict_range' function's Alignment Step handles initializing all non-user-provided features to 0.

st.markdown("---")

# --- 4. Prediction Button and Output ---
if st.button("Predict Driving Range", type="primary"):
    
    # Ensure all numerical values are correct data types (important for safety)
    try:
        # The categorical columns must be in the input dict to pass to get_dummies
        
        with st.spinner('Calculating Range...'):
            predicted_range = predict_range(inputs, model, scaler, feature_names)
            
            st.success("✅ Prediction Complete")
            
            st.metric(
                label="Estimated Driving Range (km)", 
                value=f"{predicted_range:,.0f} km",
                delta=f"MAE: {14.93} km" # Display the average error as a reference
            )
            
            st.markdown(
                """
                **Note on Accuracy:** This prediction has an average error (MAE) of **14.93 km**. 
                The actual range could be slightly higher or lower.
                """
            )
            
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")