import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import openai
import os

# --- Set API keys ---
TWELVE_DATA_API_KEY = "your_twelve_data_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"

# Set keys as env vars if not hardcoding
openai.api_key = OPENAI_API_KEY

# --- Sidebar ---
st.sidebar.title("📈 Stock Forecast with Multi-Agent AI")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 5)
interval = st.sidebar.selectbox("Interval", ["1day", "1h"])
user_question = st.sidebar.text_input("Ask AI something about the stock...")

# --- Fetch data ---
def fetch_stock_data(symbol, interval="1day", outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": TWELVE_DATA_API_KEY,
        "outputsize": outputsize
    }
    r = requests.get(url, params=params).json()
    if "values" not in r:
        st.error("Failed to fetch data. Check symbol or API key.")
        return None
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df["close"] = pd.to_numeric(df["close"])
    return df

# --- Forecast ---
def forecast_arima(df, steps=5):
    try:
        model = ARIMA(df["close"], order=(5,1,0))
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        st.error(f"ARIMA error: {e}")
        return None

# --- Groq agent (Mixtral / LLaMA3) ---
def groq_respond(prompt, model="mixtral-8x7b-32768"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# --- MiniChat (ChatGPT) ---
def openai_respond(prompt, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- MAIN ---
st.title("📊 Stock Forecast & Multi-Agent AI Insights")

df = fetch_stock_data(symbol, interval)
if df is not None:
    st.subheader(f"Recent {symbol} Prices")
    st.line_chart(df["close"])

    forecast = forecast_arima(df, forecast_days)
    if forecast is not None:
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
        forecast_df = pd.DataFrame({"Forecast": forecast}, index=future_dates)

        st.subheader("Forecast")
        fig, ax = plt.subplots()
        df["close"].plot(ax=ax, label="Historical")
        forecast_df["Forecast"].plot(ax=ax, label="Forecast", color="orange")
        plt.legend()
        plt.title(f"{symbol} Forecast for {forecast_days} Days")
        st.pyplot(fig)

        st.dataframe(forecast_df)

    # --- Multi-Agent AI Section ---
    if user_question:
        st.subheader("🤖 Multi-Agent AI Responses")
        context = f"The following stock data and ARIMA forecast is available:\n{df.tail().to_string()}\nForecast: {forecast_df.to_string()}\n\nQuestion: {user_question}"

        with st.spinner("MiniChat-GPT thinking..."):
            openai_reply = openai_respond(context)

        with st.spinner("Groq agent thinking..."):
            groq_reply = groq_respond(context)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧠 MiniChat-GPT")
            st.write(openai_reply)
        with col2:
            st.markdown("### ⚡ Groq (Mixtral)")
            st.write(groq_reply)

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, ARIMA, Twelve Data, MiniChat-GPT, and Groq Multi-Agent AI")
