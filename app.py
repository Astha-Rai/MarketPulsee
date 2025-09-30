import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import pipeline
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# ------------------------------
# Title
# ------------------------------
st.title("📈 MarketPulse: AI-Powered Stock Sentiment Analyzer")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# ------------------------------
# Load Stock Price Data
# ------------------------------
st.subheader(f"Stock Price Data for {ticker}")
try:
    data = yf.download(ticker, start=start_date, end=end_date)
except Exception as e:
    st.error(f"❌ Error fetching stock data: {e}")
    st.stop()

if data.empty:
    st.error("❌ No stock data found. Please try another ticker or date range.")
    st.stop()

st.write("📊 Data preview:", data.head())

# ------------------------------
# Preprocess Data for LSTM
# ------------------------------
df_close = data[["Close"]].reset_index()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_close["Close"].values.reshape(-1, 1))

# ------------------------------
# Load LSTM Model
# ------------------------------
MODEL_PATH = "marketpulse_model.keras"
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.warning(f"⚠️ Could not load LSTM model: {e}")
        model = None
else:
    st.warning("⚠️ Model file not found. Using dummy forecast instead.")
    model = None

# Forecast next value (dummy if model missing)
if model and len(scaled_data) >= 60:
    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    predicted_price = model.predict(last_60)
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]
else:
    predicted_price = float(df_close["Close"].iloc[-1]) * (1 + np.random.uniform(-0.02, 0.02))

st.metric("📌 Predicted Next Closing Price", f"${predicted_price:.2f}")

# ------------------------------
# Sentiment Analysis with FinBERT
# ------------------------------
st.subheader("📑 Sentiment Analysis (FinBERT)")

# Example headlines (replace with real news in production)
headlines = [
    "Federal Reserve raises interest rates amid inflation concerns",
    f"{ticker} reports strong quarterly earnings beating estimates",
    f"{ticker} stock downgraded by analysts due to market volatility"
]

try:
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")
    sentiments = classifier(headlines)
except Exception as e:
    st.warning(f"⚠️ Could not load FinBERT model: {e}")
    st.info("Using dummy sentiment scores instead.")
    sentiments = [{"label": "neutral", "score": 0.5} for _ in headlines]

# Display sentiment table
sentiment_df = pd.DataFrame(sentiments)
sentiment_df["headline"] = headlines
sentiment_df = sentiment_df[["headline", "label", "score"]]
st.write(sentiment_df)

# Convert sentiment labels to numeric values
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
sentiment_df["sentiment_value"] = sentiment_df["label"].map(sentiment_map)
avg_sentiment = sentiment_df["sentiment_value"].mean()
st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

# ------------------------------
# Merge Stock + Sentiment (constant for demo)
# ------------------------------
st.subheader("📊 Stock Price vs Average Sentiment")
combined = df_close.copy()
combined["Sentiment"] = avg_sentiment

# Plot stock price only (sentiment is constant)
try:
    st.line_chart(combined.set_index("Date")[["Close"]])
except Exception as e:
    st.error(f"❌ Plotting error: {e}")
    st.write("Available columns:", list(combined.columns))

# ------------------------------
# End
# ------------------------------
st.success("✅ Dashboard Ready!")
