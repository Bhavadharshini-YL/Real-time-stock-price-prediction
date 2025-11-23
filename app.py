import streamlit as st
import pandas as pd
import joblib
import os
import json
from model import train_model

st.title("📈 Real-Time Stock Price Prediction")
st.write("Upload your dataset or use the built-in Netflix dataset")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# If user uploads a file → use it
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(df.head())

    df.to_csv("temp_data.csv", index=False)
    csv_path = "temp_data.csv"

else:
    st.write("Using built-in Netflix dataset")
    csv_path = "data/nflx_2014_2023.csv"
    df = pd.read_csv(csv_path)
    st.dataframe(df.head())

# Train button
if st.button("Train Model"):
    st.write("Training model... please wait 🔄")
    results = train_model(csv_path)

    # Save results to JSON so prediction section can use them
    with open("model_info.json", "w") as f:
        json.dump(results, f)

    st.success("Model trained successfully! 🎉")

    st.write("### Model Performance")
    st.write(f"**MAE:** {results['mae']}")
    st.write(f"**MSE:** {results['mse']}")
    st.write(f"**R² Score:** {results['r2']}")

# Prediction section
st.write("## Make Prediction")

# Load model and features if available
if os.path.exists("trained_model.pkl") and os.path.exists("model_info.json"):
    model = joblib.load("trained_model.pkl")

    with open("model_info.json", "r") as f:
        results = json.load(f)

    st.write("Enter stock indicators:")
    feature_values = {}

    for feature in results["features"]:
        value = st.number_input(f"Enter {feature}:", value=0.0)
        feature_values[feature] = value

    if st.button("Predict Next Day Close"):
        input_df = pd.DataFrame([feature_values])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Next-Day Closing Price: **${prediction:.2f}**")

else:
    st.warning("Train the model first.")
