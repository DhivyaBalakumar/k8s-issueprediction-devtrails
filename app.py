import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load the trained model
model = joblib.load("xgboost_model.pkl")  # Ensure model is saved with this name

# Streamlit UI
st.title("XGBoost Model Predictor")
st.write("Upload a CSV file with test data to get predictions.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read uploaded CSV
    test_data = pd.read_csv(uploaded_file)
    
    # Ensure features are present
    required_features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    if all(feat in test_data.columns for feat in required_features):
        # Make predictions
        predictions = model.predict(test_data[required_features])
        
        # Display results
        test_data["Predictions"] = predictions
        st.write("### Predictions:")
        st.dataframe(test_data)
        
        # Provide CSV Download Option
        csv = test_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    
    else:
        st.error(f"Missing required features: {set(required_features) - set(test_data.columns)}")

# Deploy this on Streamlit Cloud to get a public URL
