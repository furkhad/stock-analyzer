import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant AI Stock Master", layout="wide", page_icon="🧠")

# --- CUSTOM CSS FOR "HACKER" VIBE ---
st.markdown("""
<style>
    .metric-card {background-color: #0e1117; border: 1px solid #303030; padding: 20px; border-radius: 10px; text-align: center;}
    .stAlert {background-color: #0e1117; border: 1px solid #303030;}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def search_yahoo(query):
    """Searches for a stock ticker globally."""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        data = response.json()
        results = {}
        if 'quotes' in data:
            for quote in data['quotes']:
                if 'symbol' in quote and 'shortname' in quote:
                    label = f"{quote['shortname']} ({quote['symbol']}) - {quote.get('exchange', 'Unknown')}"
                    results[label] = quote['symbol']
        return results
    except:
        return {}

def get_news_sentiment(ticker_symbol):
    """Fetches news and calculates average sentiment (Positive/Negative)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        news = stock.news
        sentiment_score = 0
        headlines = []
        
        if not news:
            return 0, []
            
        for article in news[:5]: # Analyze last 5 articles
            title = article.get('title', '')
            link = article.get('link', '')
            blob = TextBlob(title)
            sentiment_score += blob.sentiment.polarity
            headlines.append((title, link))
            
        avg_sentiment = sentiment_score / len(news[:5])
        return avg_sentiment, headlines
    except:
        return 0, []

def predict_stock_movement(df):
    """
    TRAINS AN AI MODEL LIVE.
    Uses Random Forest to predict if Tomorrow's Close > Today's Close.
    """
    df = df.copy()
    
    # 1. Feature Engineering (Creating variables for the AI to study)
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    
    df = df.dropna()
    
    # 2. Define the Target: 1 if price went UP next day, 0 if DOWN
    # We shift -1 so 'y' matches the NEXT day's result to TODAY's data
    X = df[['Open-Close', 'High-Low', 'SMA_10', 'SMA_50', 'RSI', 'Price_Change']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Remove the last row (since it has no "tomorrow" yet)
    X = X[:-1]
    y = y[:-1]
    
    if len(X) < 100:
        return None, None, 0 # Not enough data
        
    # 3. Train the Model
    split = int(0.8 * len(df))
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(X.iloc[:split], y[:split]) # Train on past data
    
    # 4. Test the Model
    preds = model.predict(X.iloc[split:])
    acc = accuracy_score(y[split:], preds)
    
    # 5. Predict for TOMORROW using the absolute latest data
    latest_data = df.iloc[-1:][['Open-Close', 'High-Low', 'SMA_10', 'SMA_50', 'RSI', 'Price_Change']]
    prediction = model.predict(latest_data)
    probability = model.predict_proba(latest_data)
    
    return prediction[0], probability[0], acc

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- SIDEBAR SEARCH ---
st.sidebar.title("🔍 Deep Search")
query = st.sidebar.text_input("Search Ticker", "Apple")
search_results = search_yahoo(query)

if search_results:
    ticker_options = list(search_results.keys())
    selected_label = st.sidebar.selectbox("Select Asset", ticker_options)
    ticker = search_results[selected_label]
else:
    ticker = "AAPL"
    st.sidebar.warning("Ticker not found, defaulting to AAPL")

# --- MAIN DATA LOADING ---
st.title(f"🧠 AI Quant Analysis: {ticker}")

# Load 2 years of data for the AI
data = yf.download(ticker, period="2y")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
data.reset_index(inplace=True)

if not data.empty:
    current_price = data['Close'].iloc[-1]
    
    # --- TABS FOR ANALYSIS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 AI Prediction", "📊 Deep Technicals", "📰 News & Sentiment", "🏢 Fundamentals"])

    # === TAB 1: AI PREDICTION ===
    with tab1:
        st.subheader("🤖 Artificial Intelligence Forecast")
        st.write("I am training a Random Forest model on the last 2 years of data to find patterns...")
        
        prediction, probability, accuracy = predict_stock_movement(data)
        
        if prediction is not None:
            col1, col2, col3 = st.columns(3)
            
            # Prediction Card
            with col1:
                st.markdown("### AI Forecast for Tomorrow")
                if prediction == 1:
                    st.success("🚀 UP / BULLISH")
                else:
                    st.error("🔻 DOWN / BEARISH")
            
            # Probability Card
            with col2:
                st.markdown("### Confidence")
                prob_up = probability[1] * 100
                st.metric("Probability of Rise", f"{prob_up:.1f}%")
                
            # Accuracy Card
            with col3:
                st.markdown("### Model Accuracy")
                st.metric("Backtest Score", f"{accuracy*100:.1f}%", help="How often this model was right in the last 6 months")

            if accuracy < 0.5:
                st.warning("⚠️ Warning: This stock is highly volatile. The AI is struggling to find a clear pattern.")
            else:
                st.info("ℹ️ The model has detected a tradable pattern in recent movements.")
        else:
            st.error("Not enough historical data to train the AI.")

    # === TAB 2: TECHNICALS ===
    with tab2:
        # Calculate Indicators
        data['SMA50'] = data['Close'].rolling(50).mean()
        data['SMA200'] = data['Close'].rolling(200).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA50'], line=dict(color='orange', width=1), name='SMA 50'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA200'], line=dict(color='blue', width=1), name='SMA 200'))
        
        fig.update_layout(height=600, template="plotly_dark", title=f"{ticker} Technical Chart")
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 3: NEWS & SENTIMENT ===
    with tab3:
        st.subheader("📰 Market Sentiment Analysis")
        sentiment, headlines = get_news_sentiment(ticker)
        
        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            st.metric("Sentiment Score", f"{sentiment:.2f}")
            if sentiment > 0.1:
                st.success("The news is generally POSITIVE 😄")
            elif sentiment < -0.1:
                st.error("The news is generally NEGATIVE 😡")
            else:
                st.warning("The news is NEUTRAL 😐")
        
        with s_col2:
            st.write("### Latest Headlines")
            for title, link in headlines:
                st.markdown(f"• [{title}]({link})")

    # === TAB 4: FUNDAMENTALS ===
    with tab4:
        st.subheader("🏢 Company Health (Deep Dive)")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
            f2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            f3.metric("Revenue Growth", f"{info.get('revenueGrowth', 0)*100:.1f}%")
            f4.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 0)}")
            
            st.markdown("### Business Summary")
            st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            st.markdown("### Major Holders")
            st.dataframe(stock.major_holders)
            
        except:
            st.error("Could not retrieve fundamental data.")

else:
    st.error("Data could not be loaded. Please check the ticker.")
