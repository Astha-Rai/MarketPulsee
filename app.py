import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------
# Load Model + FinBERT
# ---------------------------
model = load_model("marketpulse_model.keras")

finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["Negative", "Neutral", "Positive"]
    return dict(zip(labels, scores[0].detach().numpy()))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📈 MarketPulse: AI-Powered Stock Sentiment Analyzer")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
if st.button("Analyze"):
    df = yf.download(ticker, period="1y", interval="1d")
    data = df[['Close']]
    st.line_chart(data)

    # LSTM Prediction
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data)

    def create_dataset(dataset, time_step=60):
        X = []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step), 0])
        return np.array(X)

    last_60 = scaled[-60:]
    X_input = last_60.reshape(1, 60, 1)
    pred = model.predict(X_input)
    pred_price = scaler.inverse_transform(pred)[0][0]

    st.subheader(f"🔮 Predicted Next Close Price: ${pred_price:.2f}")

    # Sentiment Input
    st.subheader("📰 Financial News Sentiment")
    news = st.text_area("Paste recent financial news about the company:")
    if st.button("Analyze Sentiment"):
        result = analyze_sentiment(news)
        st.write(result)
