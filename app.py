import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
# Set page configuration for the Streamlit app
st.set_page_config(page_title="Personalized Healthcare Recommendation", layout="centered")

# --- Load the Model Pipeline ---
# This assumes 'recommendation_pipeline.joblib' is in the same directory as app.py
try:
    # The pipeline encapsulates both the preprocessor (StandardScaler) and the classifier
    model_pipeline = joblib.load('recommendation_pipeline.joblib')
    st.sidebar.success("Model pipeline loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'recommendation_pipeline.joblib' not found. Please ensure your main training script has been run successfully to save the model.")
    st.stop() # Stop the app if the model file is missing
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()


# --- Define Input Mappings and Recommendation Labels ---
# These must exactly match your training script's feature order and classification
feature_names = ['Recency', 'Frequency', 'Monetary', 'Time']

# Define the recommendation mapping based on your 'Class' labels
# 0: 'No immediate action needed', 1: 'Recommendation: Consult with a healthcare professional for further evaluation/check-up.'
recommendation_labels = {
    0: 'No immediate action needed; maintain current health routine.',
    1: 'Recommendation: Consult with a healthcare professional for further evaluation/check-up.'
}


# --- Streamlit App Header ---
st.title("💡 Personalized Healthcare Recommendation System")
st.markdown("Enter patient RFMT (Recency, Frequency, Monetary, Time) data to get a tailored healthcare recommendation.")
st.write("---") # Visual separator


# --- Sidebar for User Input ---
st.sidebar.header("📊 Patient RFMT Data Input")
st.sidebar.markdown("Adjust the sliders below for patient characteristics:")

# Create input widgets for Recency, Frequency, Monetary, Time
# Use approximate ranges/default values based on your data.describe() output
# Your dataset had:
# Recency: min 0, max 74, mean ~9.5
# Frequency: min 1, max 50, mean ~5.5
# Monetary: min 250, max 12500, mean ~1378
# Time: min 2, max 99, mean ~34

input_recency = st.sidebar.slider("Recency (Time since last interaction)", 0, 74, 9)
input_frequency = st.sidebar.slider("Frequency (Total interactions)", 1, 50, 5)
input_monetary = st.sidebar.slider("Monetary (Total value/contribution)", 250, 12500, 1378)
input_time = st.sidebar.slider("Time (Total duration/time period)", 2, 99, 34)

st.sidebar.write("---")


# --- Main Content Area for Display & Prediction ---

# Display the collected user inputs in the main area
st.subheader("Current Patient Input:")
user_input = pd.DataFrame([[input_recency, input_frequency, input_monetary, input_time]],
                          columns=feature_names) # Create DataFrame with correct column names

st.dataframe(user_input) # Use st.dataframe for nicer table display


# --- Prediction Button and Logic ---
if st.button("Generate Recommendation"):
    try:
        # The model_pipeline handles scaling internally, just pass the raw input DataFrame
        prediction_raw = model_pipeline.predict(user_input)
        predicted_class_id = int(prediction_raw[0]) # Get the integer class ID from the prediction
        
        # Get the human-readable recommendation
        recommendation_text = recommendation_labels.get(predicted_class_id, "Unknown Recommendation Type")

        st.success("--- **Healthcare Recommendation** ---")
        st.markdown(f"## :sparkles: **{recommendation_text}**")
        
        # Optional: Display raw prediction info
        st.write("---")
        st.info(f"Model Predicted Class ID: **{predicted_class_id}**")

        # Visual feedback (balloons are fun!)
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values or model loading process.")


st.markdown("---")
st.markdown("Developed for Personalized Healthcare Recommendations project using ML.")