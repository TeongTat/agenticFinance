import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import groq

# === PAGE CONFIG ===
st.set_page_config(page_title="S&P 500 Stock Forecast", layout="wide")

# === API KEYS ===
TWELVE_DATA_API_KEY = "your_twelve_data_api_key"
GROQ_API_KEY = "your_groq_api_key"

# === LOAD S&P 500 LIST ===
@st.cache_data
def load_sp500_symbols():
    return pd.read_csv("sp500_symbols.csv")

sp500_df = load_sp500_symbols()
symbol_name = st.selectbox("Choose an S&P 500 Company:", sp500_df["Name"])
symbol = sp500_df[sp500_df["Name"] == symbol_name]["Symbol"].values[0]

# === FETCH HISTORICAL STOCK DATA ===
def fetch_stock_data(symbol):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 100,
        "apikey": TWELVE_DATA_API_KEY,
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "values" not in data:
        return None

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["close"] = pd.to_numeric(df["close"])
    df = df[["datetime", "close"]].sort_values("datetime")
    return df

# === PREDICT NEXT 5 DAYS USING ARIMA ===
def forecast_next_5_days(data):
    model = ARIMA(data["close"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    forecast_dates = [data["datetime"].iloc[-1] + timedelta(days=i) for i in range(1, 6)]
    return pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})

# === AI COMMENTARY WITH GROQ ===
def generate_commentary(symbol_name, recent_close, forecast_df):
    prompt = (
        f"You are a financial analyst. The recent stock price for {symbol_name} is ${recent_close:.2f}. "
        f"The next 5-day ARIMA forecast is:\n{forecast_df.to_string(index=False)}\n\n"
        f"Give an investor-friendly, concise summary on this trend and what they should consider."
    )
    client = groq.Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return chat_completion.choices[0].message.content

# === MAIN APP ===
st.title("📈 S&P 500 Stock Predictor with AI Commentary")

data = fetch_stock_data(symbol)

if data is None:
    st.error("⚠️ Failed to fetch data. Check your API key or symbol.")
else:
    st.subheader(f"Recent Closing Prices for {symbol}")
    st.line_chart(data.set_index("datetime")["close"])

    st.subheader("5-Day Forecast (ARIMA)")
    forecast_df = forecast_next_5_days(data)
    st.dataframe(forecast_df)

    # Combine actual and forecast for plot
    full_plot = pd.concat([data[["datetime", "close"]].rename(columns={"close": "Price"}), 
                           forecast_df.rename(columns={"Date": "datetime", "Forecast": "Price"})])
    full_plot = full_plot.set_index("datetime")

    st.line_chart(full_plot)

    # MiniChat Groq Commentary
    st.subheader("🧠 AI Financial Commentary")
    with st.spinner("Thinking..."):
        ai_text = generate_commentary(symbol_name, data['close'].iloc[-1], forecast_df)
        st.success("Done!")
        st.write(ai_text)
