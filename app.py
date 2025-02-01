import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from textblob import TextBlob
import requests

st.set_page_config(layout="wide", page_title="Saudi Stock Market News Analyzer")

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL"
    confidence = abs(polarity) * 100
    return sentiment, confidence

# Function to fetch stock data
def get_stock_data(ticker, period="1mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None, "No data available"
        return df, None
    except Exception as e:
        return None, str(e)

# Function to plot stock data
def plot_stock_analysis(df, company_name, company_symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name=f'{company_symbol}'))
    fig.update_layout(title=f"{company_name} Stock Performance",
                      xaxis_title='Date',
                      yaxis_title='Price (SAR)',
                      xaxis_rangeslider_visible=False)
    return fig

# Function to analyze technical indicators
def analyze_technical_indicators(df):
    indicators = []
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
    
    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
        indicators.append(("Moving Average Crossover", "BULLISH", "Short-term trend is positive"))
    else:
        indicators.append(("Moving Average Crossover", "BEARISH", "Short-term trend is negative"))
    
    if df['RSI'].iloc[-1] < 30:
        indicators.append(("RSI", "BULLISH", "Stock is oversold"))
    elif df['RSI'].iloc[-1] > 70:
        indicators.append(("RSI", "BEARISH", "Stock is overbought"))
    else:
        indicators.append(("RSI", "NEUTRAL", "RSI in normal range"))
    
    return indicators

# Function to fetch news articles (Dummy Data for Now)
def fetch_news():
    return [
        {"title": "Saudi Aramco announces record profits", "description": "Saudi Aramco reported a 20% increase in profits.", "url": "https://example.com/aramco", "source": "Reuters", "published_at": "2025-02-01T12:00:00Z"},
        {"title": "STC expands its 5G network", "description": "Saudi Telecom Company (STC) continues to expand its 5G network.", "url": "https://example.com/stc", "source": "Bloomberg", "published_at": "2025-02-01T10:00:00Z"},
    ]

# Function to find mentioned companies
def find_companies_in_text(text, companies_df):
    mentioned = []
    for _, row in companies_df.iterrows():
        if row['name'].lower() in text.lower() or row['symbol'].lower() in text.lower():
            mentioned.append(row.to_dict())
    return mentioned

# Load company data (Dummy Data for Now)
companies_df = pd.DataFrame([
    {"name": "Saudi Aramco", "symbol": "ARAMCO", "code": "2222.SR"},
    {"name": "STC", "symbol": "STC", "code": "7010.SR"},
])

# Display News & Analysis
def display_article(article, companies_df):
    """Display news article with sentiment and technical analysis"""
    title = article.get("title", "No title")
    description = article.get("description", "No description")
    url = article.get("url", "#")
    source = article.get("source", "Unknown")
    published_at = article.get("published_at", "")
    
    # Use a unique key for the article
    st.markdown(f"## {title}", key=f"title_{title[:20]}")
    
    # Display source and date
    st.write(f"Source: {source} | Published: {published_at[:16]}", key=f"source_{source}")
    
    # Display description
    st.write(description[:200] + "..." if len(description) > 200 else description, key=f"desc_{title[:20]}")
    
    # Sentiment Analysis
    sentiment, confidence = analyze_sentiment(title + " " + description)
    
    # Create columns for analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Analysis", key=f"sentiment_header_{title[:20]}")
        st.write(f"**Sentiment:** {sentiment}", key=f"sentiment_{title[:20]}")
        st.write(f"**Confidence:** {confidence:.2f}%", key=f"confidence_{title[:20]}")
    
    # Find mentioned companies
    mentioned_companies = find_companies_in_text(title + " " + description, companies_df)
    
    if mentioned_companies:
        st.markdown("### Companies Mentioned", key=f"companies_header_{title[:20]}")
        for company in mentioned_companies:
            st.markdown(f"- {company['name']} ({company['symbol']})", key=f"company_{company['symbol']}")
        
        st.markdown("### Stock Analysis", key=f"stock_header_{title[:20]}")
        # Create tabs for each company
        tabs = st.tabs([company['name'] for company in mentioned_companies])
        
        for tab, company in zip(tabs, mentioned_companies):
            with tab:
                # Get stock data and technical analysis
                df, error = get_stock_data(company['code'])
                if error:
                    st.error(f"Error fetching stock data: {error}", key=f"error_{company['symbol']}")
                    continue
                
                if df is not None:
                    # Show current stock price
                    latest_price = df['Close'][-1]
                    prev_price = df['Close'][-2]
                    price_change = ((latest_price - prev_price)/prev_price*100)
                    
                    # Price metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"{latest_price:.2f} SAR", 
                                f"{price_change:.2f}%", key=f"price_{company['symbol']}")
                    with col2:
                        st.metric("Day High", f"{df['High'][-1]:.2f} SAR", key=f"high_{company['symbol']}")
                    with col3:
                        st.metric("Day Low", f"{df['Low'][-1]:.2f} SAR", key=f"low_{company['symbol']}")
                    
                    # Plot stock chart
                    fig = plot_stock_analysis(df, company['name'], company['symbol'])
                    st.plotly_chart(fig, key=f"plot_{company['symbol']}")
                    
                    # Technical Analysis Signals
                    st.markdown("### Technical Analysis Signals", key=f"tech_header_{company['symbol']}")
                    signals = analyze_technical_indicators(df)
                    
                    # Create a clean table for signals
                    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                    st.table(signal_df, key=f"table_{company['symbol']}")
                    
                    # Combined Analysis
                    st.markdown("### Combined Analysis", key=f"combined_header_{company['symbol']}")
                    tech_sentiment = sum(1 if signal[1] == "BULLISH" else -1 if signal[1] == "BEARISH" else 0 for signal in signals)
                    news_sentiment_score = 1 if sentiment == "POSITIVE" else -1 if sentiment == "NEGATIVE" else 0
                    
                    combined_score = (tech_sentiment + news_sentiment_score) / (len(signals) + 1)
                    
                    if combined_score > 0.3:
                        st.success("🟢 Overall Bullish: Technical indicators and news sentiment suggest positive momentum", key=f"bullish_{company['symbol']}")
                    elif combined_score < -0.3:
                        st.error("🔴 Overall Bearish: Technical indicators and news sentiment suggest negative pressure", key=f"bearish_{company['symbol']}")
                    else:
                        st.warning("🟡 Neutral: Mixed signals from technical indicators and news sentiment", key=f"neutral_{company['symbol']}")
                    
                    # Volume Analysis
                    avg_volume = df['Volume'].mean()
                    latest_volume = df['Volume'][-1]
                    volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                    
                    st.markdown("### Volume Analysis", key=f"volume_header_{company['symbol']}")
                    st.metric("Trading Volume", f"{int(latest_volume):,}", 
                             f"{volume_change:.1f}% vs 30-day average", key=f"volume_{company['symbol']}")
    
    # Article link
    st.markdown(f"[Read full article]({url})", key=f"link_{title[:20]}")
    st.markdown("---", key=f"divider_{title[:20]}")
