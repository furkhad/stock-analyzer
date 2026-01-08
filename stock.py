import streamlit as st
import yfinance as yf

st.title("Ahmed's Stock Analyzer 📈")

# 1. Simple Input for the user
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, RELIANCE.NS)", "AAPL")

# 2. Get Data
if ticker:
    stock = yf.Ticker(ticker)
    # Get history
    history = stock.history(period="1y")

    # 3. Show Data & Chart
    st.write(f"Showing data for: **{ticker}**")
    st.line_chart(history['Close'])
    
    # Show raw data if they want
    if st.checkbox("Show Raw Data"):
        st.dataframe(history)
