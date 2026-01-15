import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
import time
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant AI Master v2.0", layout="wide", page_icon="🛡️")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #303030; padding: 20px; border-radius: 10px; text-align: center;}
    .trust-score-high {color: #00ff41; font-weight: bold;}
    .trust-score-med {color: #ffa500; font-weight: bold;}
    .trust-score-low {color: #ff2b2b; font-weight: bold;}
    .stProgress .st-bo {background-color: #00ff41;}
</style>
""", unsafe_allow_html=True)

# --- ROBUST DATA ENGINE ---

@st.cache_data(ttl=300, show_spinner=False) # Cache for 5 minutes
def get_stock_data_robust(ticker, period="2y"):
    """
    Attempts to download data with retries. 
    Prevents 'Unable to fetch' errors by trying 3 times.
    """
    attempts = 0
    max_retries = 3
    while attempts < max_retries:
        try:
            df = yf.download(ticker, period=period, progress=False)
            
            # Fix for yfinance MultiIndex issue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df.reset_index(inplace=True)
            
            if df.empty:
                raise ValueError("Empty Dataframe")
                
            return df
        except Exception as e:
            attempts += 1
            time.sleep(1) # Wait 1 second before retry
            
    return pd.DataFrame() # Return empty if all fails

def search_yahoo_robust(query):
    """Safely searches for a ticker."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        results = {}
        if 'quotes' in data:
            for quote in data['quotes']:
                if 'symbol' in quote:
                    name = quote.get('shortname', quote.get('longname', 'Unknown'))
                    label = f"{name} ({quote['symbol']})"
                    results[label] = quote['symbol']
        return results
    except:
        return {}

# --- ADVANCED MATH & AI ---

def add_technical_indicators(df):
    """Adds professional-grade indicators for the AI to learn from."""
    df = df.copy()
    
    # 1. Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Bollinger Bands (Volatility)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)
    
    # Clean NaN values created by rolling windows
    df.dropna(inplace=True)
    return df

def train_robust_model(df):
    """
    Trains a Random Forest model with a 'Confidence Threshold'.
    """
    df = add_technical_indicators(df)
    
    # Features (X) and Target (y)
    # We predict if Tomorrow's Close > Today's Close
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
    
    # Target: 1 if Up, 0 if Down
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Remove last row (no target for it)
    data_model = df[:-1].copy()
    
    if len(data_model) < 100:
        return None, None, 0, 0 # Not enough data
    
    X = data_model[features]
    y = data_model['Target']
    
    # Split Data (80% Train, 20% Test)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Initialize Model (More trees = more stable)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Prediction for Tomorrow (using the latest available data)
    last_row = df.iloc[-1:][features]
    prediction = model.predict(last_row)[0]
    probabilities = model.predict_proba(last_row)[0] # [prob_down, prob_up]
    
    return prediction, probabilities, acc, model

# --- UI LAYOUT ---

# Sidebar
st.sidebar.title("🔍 Asset Search")
query = st.sidebar.text_input("Type Company Name", "Apple")
search_results = search_yahoo_robust(query)

if search_results:
    ticker_display = st.sidebar.selectbox("Select Stock", list(search_results.keys()))
    ticker = search_results[ticker_display]
else:
    ticker = "AAPL"
    st.sidebar.info("Using default: Apple Inc.")

# Main Page
st.title(f"🛡️ Quant AI: {ticker}")
st.markdown("### reliable. data-driven. automated.")

# Fetch Data
with st.spinner(f"Connecting to global exchanges for {ticker}..."):
    raw_data = get_stock_data_robust(ticker)

if not raw_data.empty:
    
    # Run AI
    pred, probs, accuracy, model = train_robust_model(raw_data)
    
    # -- DASHBOARD HEADER --
    col1, col2, col3, col4 = st.columns(4)
    current_price = raw_data['Close'].iloc[-1]
    prev_close = raw_data['Close'].iloc[-2]
    change = current_price - prev_close
    pct_change = (change / prev_close) * 100
    
    col1.metric("Current Price", f"${current_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    
    # -- AI DECISION LOGIC --
    # We only show a signal if confidence is > 60%
    confidence_up = probs[1] * 100
    confidence_down = probs[0] * 100
    
    decision = "HOLD / NEUTRAL"
    color = "off"
    
    if confidence_up > 60:
        decision = "BUY SIGNAL 🚀"
        color = "green"
    elif confidence_down > 60:
        decision = "SELL SIGNAL 🔻"
        color = "red"
        
    with col2:
        st.markdown(f"**AI Recommendation**")
        if color == "green":
            st.success(decision)
        elif color == "red":
            st.error(decision)
        else:
            st.warning(decision)
            
    with col3:
        st.metric("AI Confidence", f"{max(confidence_up, confidence_down):.1f}%")
        
    with col4:
        st.metric("Model Reliability", f"{accuracy*100:.1f}%", help="Accuracy on unseen data (last 6 months)")

    st.divider()

    # -- TABS --
    tab1, tab2, tab3 = st.tabs(["📊 Deep Analysis", "🧠 AI Logic Explained", "🎓 Beginner's Guide"])
    
    with tab1:
        # Professional Chart with Bollinger Bands
        df_chart = add_technical_indicators(raw_data)
        
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(x=df_chart['Date'],
                        open=df_chart['Open'], high=df_chart['High'],
                        low=df_chart['Low'], close=df_chart['Close'], name='Price'))
        
        # Bollinger Bands (Area)
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['BB_Upper'], 
                                 line=dict(color='rgba(255, 255, 255, 0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['BB_Lower'], 
                                 fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', 
                                 line=dict(color='rgba(255, 255, 255, 0)'), name='Volatility Band'))
        
        # Moving Averages
        fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['SMA_50'], line=dict(color='orange', width=1), name='50-Day Trend'))
        
        fig.update_layout(height=600, template="plotly_dark", title="Institutional Grade Chart")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Why did the AI make this decision?")
        
        # Feature Importance
        if model:
            importances = model.feature_importances_
            feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            st.write("The model looked at these factors mainly to decide:")
            st.bar_chart(feat_df.set_index('Feature'))
            
            st.info("""
            **Interpretation:**
            * If **Volume** is high, the AI is watching trading activity.
            * If **RSI** is high, it's checking if the stock is 'Overbought'.
            * If **SMA** is high, it's looking at the long-term trend.
            """)
            
    with tab3:
        st.subheader("🎓 Trading for Beginners")
        st.markdown("""
        **1. What is the 'AI Confidence'?**
        Stocks are chaotic. The AI analyzes patterns. If confidence is 50-55%, it's guessing. We only show a BUY signal if it is >60% sure.
        
        **2. What is RSI?**
        * **RSI > 70:** The stock might be too expensive (Overbought). Price might drop.
        * **RSI < 30:** The stock might be cheap (Oversold). Price might rise.
        
        **3. What are the Blue Bands (Bollinger)?**
        These measure volatility. If the bands squeeze tight, a big move (up or down) is often coming soon.
        
        **4. Disclaimer**
        *No algorithm can predict the future 100%. Always use this as a helper, not a master. Never invest money you cannot afford to lose.*
        """)

else:
    st.error("⚠️ Massive Data Failure. The stock exchange is not responding. Please try a different ticker or wait 60 seconds.")
