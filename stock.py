import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Global Stock Search", layout="wide")

# --- 2. THE "SEARCH THE WORLD" FUNCTION ---
# This hits Yahoo Finance's hidden API to find ANY ticker globally
def search_yahoo(query):
    try:
        # The API endpoint Yahoo uses for its own search bar
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        
        # We need a fake "User-Agent" so Yahoo doesn't block us
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Extract the list of companies found
        results = {}
        if 'quotes' in data:
            for quote in data['quotes']:
                # We only want Stocks and ETFs (not news articles)
                if 'symbol' in quote and 'shortname' in quote:
                    symbol = quote['symbol']
                    name = quote['shortname']
                    exch = quote.get('exchange', 'Unknown')
                    label = f"{name} ({symbol}) - {exch}"
                    results[label] = symbol
        return results
    except Exception as e:
        return {}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Find a Stock")

# SEARCH BOX
search_query = st.sidebar.text_input("Type Company Name (e.g., Toyota, Shell, Tencent)", "Apple")

# SEARCH LOGIC
ticker = "AAPL" # Fallback default
stock_name = "Apple Inc."

if search_query:
    # 1. Search the world for this string
    search_results = search_yahoo(search_query)
    
    if search_results:
        # 2. Let user choose the correct one from the dropdown
        selected_label = st.sidebar.selectbox("Select Result:", list(search_results.keys()))
        ticker = search_results[selected_label]
        stock_name = selected_label
    else:
        st.sidebar.warning("No results found. Try a different name.")

# DATE CONTROLS
st.sidebar.markdown("---")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- 4. DATA LOADING & ANALYSIS (Same as before) ---
@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty: return None
        df.reset_index(inplace=True)
        # Fix for multi-level columns if they appear
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except:
        return None

# --- MAIN PAGE ---
st.title(f"Analysis: {stock_name}")

data_load_state = st.text(f"Fetching data for {ticker}...")
df = load_data(ticker, start_date, end_date)
data_load_state.empty()

if df is not None:
    # Indicators
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
    
    fig.update_layout(height=600, title=f"{ticker} - {stock_name}", template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(10))

else:
    st.error(f"Could not load data for {ticker}. The exchange might be closed or data unavailable.")
