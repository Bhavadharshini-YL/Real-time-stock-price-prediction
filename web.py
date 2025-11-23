import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Real-Time Stock Price Prediction")

# Input box for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (Example: AAPL, TSLA, INFY, TCS):")

if ticker:
    # Download stock data
    data = yf.download(ticker, period="1y")
    
    if not data.empty:
        st.subheader("📊 Stock Closing Price - Last 1 Year")
        st.line_chart(data['Close'])

        # Prepare data for prediction
        data['Days'] = np.arange(len(data))
        X = data[['Days']]
        y = data['Close']

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next 7 days
        future_days = np.arange(len(data), len(data) + 7)
        future_pred = model.predict(future_days.reshape(-1, 1))

        # Show predictions
        st.subheader("🔮 Predicted Prices for Next 7 Days")
        pred_df = pd.DataFrame({
            "Day": future_days,
            "Predicted Price": future_pred.flatten()
        })
        st.dataframe(pred_df)

        # Plot predictions
        fig, ax = plt.subplots()
        ax.plot(data['Days'], data['Close'], label="Past Prices")
        ax.plot(future_days, future_pred, label="Predicted", linestyle="dashed")
        st.pyplot(fig)

    else:
        st.error("Invalid ticker or no data found.")
