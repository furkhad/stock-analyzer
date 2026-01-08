import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Stock Analyzer", layout="wide")

# --- 1. SIDEBAR INPUTS ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- 2. CACHED FUNCTION TO LOAD DATA ---
# This prevents downloading the same data 10 times if you just change the chart view
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

# --- LOAD DATA ---
data_load_state = st.text('Loading data...')
df = load_data(ticker, start_date, end_date)
data_load_state.text('') # Clear the loading text

# --- MAIN PAGE LOGIC ---
if df is not None and not df.empty:
    st.title(f"Analysis for {ticker} 📈")

    # --- 3. CALCULATE INDICATORS (Moving Averages) ---
    # SMA 50 = Short term trend, SMA 200 = Long term trend
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # --- 4. INTERACTIVE PLOTLY CHART ---
    tab1, tab2 = st.tabs(["Chart", "Raw Data"])

    with tab1:
        fig = go.Figure()

        # Candlestick (The actual stock price)
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Price'
        ))

        # Add Moving Averages
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='50 Day SMA'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='200 Day SMA'))

        fig.update_layout(
            title=f"{ticker} Share Price",
            yaxis_title="Stock Price (USD)",
            xaxis_rangeslider_visible=False,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("Recent Data:")
        st.dataframe(df.tail())
        
        # --- 5. DOWNLOAD BUTTON ---
        # Allow users to download the data as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Data as CSV",
            data=csv,
            file_name=f"{ticker}_data.csv",
            mime="text/csv",
        )

else:
    st.error("Error: Could not find stock data. Please check the ticker symbol.")
