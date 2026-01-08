import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Global Stock Analyzer", layout="wide")

# --- 2. THE "DATABASE" OF POPULAR STOCKS ---
# This serves as a quick starter list for users.
COMMON_STOCKS = {
    "--- US Tech ---": "Skip", # Header
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT", 
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    
    "--- Indian Market (NSE) ---": "Skip",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Svcs (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Zomato (ZOMATO.NS)": "ZOMATO.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",

    "--- Crypto ---": "Skip",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    
    "--- Indices ---": "Skip",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nifty 50 (^NSEI)": "^NSEI",
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Configuration")

# Input Method Selection
input_method = st.sidebar.radio("Search Method:", ["Select from List", "Enter Custom Ticker"])

ticker = "AAPL" # Default
stock_name = "Apple Inc."

if input_method == "Select from List":
    selected_option = st.sidebar.selectbox("Choose a Stock:", list(COMMON_STOCKS.keys()))
    # Logic to handle headers (if user clicks a header, default to first real stock)
    if COMMON_STOCKS[selected_option] == "Skip":
        st.sidebar.warning("Please select a valid stock, not a category header.")
        ticker = "AAPL"
    else:
        ticker = COMMON_STOCKS[selected_option]
        stock_name = selected_option
else:
    # Custom Manual Entry
    raw_ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g. MSFT, RELIANCE.NS):", "AAPL")
    ticker = raw_ticker.upper().strip()
    stock_name = ticker

# Date Selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.today())

# --- 4. DATA LOADING FUNCTION (CACHED) ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        # Download data
        df = yf.download(symbol, start=start, end=end)
        
        # Check if data is empty (wrong ticker)
        if df.empty:
            return None
        
        # Reset index so 'Date' becomes a column
        df.reset_index(inplace=True)
        
        # Flatten MultiIndex columns if they exist (common yfinance issue)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- 5. MAIN DASHBOARD LOGIC ---
st.title(f"📊 {stock_name} Analysis")

# Load the data
data_state = st.text("Fetching data from Yahoo Finance...")
df = load_data(ticker, start_date, end_date)
data_state.empty() # Clear loading text

if df is not None:
    # --- CALCULATIONS ---
    # Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate simple return metrics
    current_price = df['Close'].iloc[-1]
    start_price = df['Close'].iloc[0]
    pct_change = ((current_price - start_price) / start_price) * 100
    
    # --- METRICS ROW ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"{current_price:.2f}")
    m2.metric("Start Price", f"{start_price:.2f}")
    m3.metric("Change", f"{pct_change:.2f}%", delta=f"{pct_change:.2f}%")
    
    # --- PLOTLY CHART ---
    st.subheader("Price History & Moving Averages")
    
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1.5), name='50-Day SMA'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1.5), name='200-Day SMA'))

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark", # Looks "Pro"
        title_text=f"{ticker} Daily Chart",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- RAW DATA & DOWNLOAD ---
    with st.expander("📥 View & Download Raw Data"):
        st.dataframe(df.sort_values(by="Date", ascending=False).head(10))
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{ticker}_history.csv",
            mime="text/csv",
        )

else:
    st.error(f"❌ Could not find data for **{ticker}**.")
    st.info("Tips: \n- For Indian stocks, add `.NS` (e.g., `RELIANCE.NS`). \n- For Crypto, add `-USD` (e.g., `BTC-USD`). \n- For US stocks, just use the symbol (e.g., `AAPL`).")
