import streamlit as st
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objs as go
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import uuid
from datetime import datetime, timedelta

# Function to fetch news (dummy data for now)
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

# Function to analyze sentiment
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        return "POSITIVE", (compound_score + 1) / 2 * 100
    elif compound_score <= -0.05:
        return "NEGATIVE", (-compound_score + 1) / 2 * 100
    else:
        return "NEUTRAL", 50

# Function to get stock data
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        return df, None
    except Exception as e:
        return None, str(e)

# Function to plot stock analysis
def plot_stock_analysis(df, company_name, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=20).mean(), name='20 Day MA', line=dict(color='orange')))
    
    fig.update_layout(
        title=f'{company_name} ({symbol}) Stock Price',
        yaxis_title='Price (SAR)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Function to analyze technical indicators
def analyze_technical_indicators(df):
    signals = []
    
    # MACD
    macd = MACD(close=df['Close'])
    if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]:
        signals.append(("MACD", "BULLISH", "MACD line above signal line"))
    else:
        signals.append(("MACD", "BEARISH", "MACD line below signal line"))
    
    # RSI
    rsi = RSIIndicator(close=df['Close'])
    rsi_value = rsi.rsi().iloc[-1]
    if rsi_value > 70:
        signals.append(("RSI", "BEARISH", "Overbought condition (RSI > 70)"))
    elif rsi_value < 30:
        signals.append(("RSI", "BULLISH", "Oversold condition (RSI < 30)"))
    else:
        signals.append(("RSI", "NEUTRAL", f"RSI at {rsi_value:.2f}"))
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    if df['Close'].iloc[-1] > bb.bollinger_hband().iloc[-1]:
        signals.append(("Bollinger Bands", "BEARISH", "Price above upper band"))
    elif df['Close'].iloc[-1] < bb.bollinger_lband().iloc[-1]:
        signals.append(("Bollinger Bands", "BULLISH", "Price below lower band"))
    else:
        signals.append(("Bollinger Bands", "NEUTRAL", "Price within bands"))
    
    return signals

# Function to display article
def display_article(article, companies_df):
    unique_id = str(uuid.uuid4())
    title = article.get("title", "No title")
    description = article.get("description", "No description")
    url = article.get("url", "#")
    source = article.get("source", "Unknown")
    published_at = article.get("published_at", "")
    
    st.markdown(f"## {title}", key=f"title_{unique_id}")
    st.write(f"Source: {source} | Published: {published_at[:16]}", key=f"source_{unique_id}")
    st.write(description[:200] + "..." if len(description) > 200 else description, key=f"desc_{unique_id}")
    
    sentiment, confidence = analyze_sentiment(title + " " + description)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Analysis", key=f"sentiment_header_{unique_id}")
        st.write(f"**Sentiment:** {sentiment}", key=f"sentiment_{unique_id}")
        st.write(f"**Confidence:** {confidence:.2f}%", key=f"confidence_{unique_id}")
    
    mentioned_companies = find_companies_in_text(title + " " + description, companies_df)
    
    if mentioned_companies:
        st.markdown("### Companies Mentioned", key=f"companies_header_{unique_id}")
        for company in mentioned_companies:
            st.markdown(f"- {company['name']} ({company['symbol']})", key=f"company_{company['symbol']}_{unique_id}")
        
        st.markdown("### Stock Analysis", key=f"stock_header_{unique_id}")
        tabs = st.tabs([company['name'] for company in mentioned_companies])
        
        for tab, company in zip(tabs, mentioned_companies):
            with tab:
                df, error = get_stock_data(company['symbol'])
                if error:
                    st.error(f"Error fetching stock data: {error}", key=f"error_{company['symbol']}_{unique_id}")
                    continue
                
                if df is not None and not df.empty:
                    latest_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    price_change = ((latest_price - prev_price)/prev_price*100)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"{latest_price:.2f} SAR", 
                                f"{price_change:.2f}%", key=f"price_{company['symbol']}_{unique_id}")
                    with col2:
                        st.metric("Day High", f"{df['High'].iloc[-1]:.2f} SAR", key=f"high_{company['symbol']}_{unique_id}")
                    with col3:
                        st.metric("Day Low", f"{df['Low'].iloc[-1]:.2f} SAR", key=f"low_{company['symbol']}_{unique_id}")
                    
                    fig = plot_stock_analysis(df, company['name'], company['symbol'])
                    st.plotly_chart(fig, key=f"plot_{company['symbol']}_{unique_id}")
                    
                    st.markdown("### Technical Analysis Signals", key=f"tech_header_{company['symbol']}_{unique_id}")
                    signals = analyze_technical_indicators(df)
                    
                    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                    st.table(signal_df)
                    
                    st.markdown("### Combined Analysis", key=f"combined_header_{company['symbol']}_{unique_id}")
                    tech_sentiment = sum(1 if signal[1] == "BULLISH" else -1 if signal[1] == "BEARISH" else 0 for signal in signals)
                    news_sentiment_score = 1 if sentiment == "POSITIVE" else -1 if sentiment == "NEGATIVE" else 0
                    
                    combined_score = (tech_sentiment + news_sentiment_score) / (len(signals) + 1)
                    
                    if combined_score > 0.3:
                        st.success("🟢 Overall Bullish: Technical indicators and news sentiment suggest positive momentum", key=f"bullish_{company['symbol']}_{unique_id}")
                    elif combined_score < -0.3:
                        st.error("🔴 Overall Bearish: Technical indicators and news sentiment suggest negative pressure", key=f"bearish_{company['symbol']}_{unique_id}")
                    else:
                        st.warning("🟡 Neutral: Mixed signals from technical indicators and news sentiment", key=f"neutral_{company['symbol']}_{unique_id}")
                    
                    avg_volume = df['Volume'].mean()
                    latest_volume = df['Volume'].iloc[-1]
                    volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                    
                    st.markdown("### Volume Analysis", key=f"volume_header_{company['symbol']}_{unique_id}")
                    st.metric("Trading Volume", f"{int(latest_volume):,}", 
                             f"{volume_change:.1f}% vs 30-day average", key=f"volume_{company['symbol']}_{unique_id}")
                else:
                    st.error(f"No data available for {company['name']}", key=f"no_data_{company['symbol']}_{unique_id}")
    
    st.markdown(f"[Read full article]({url})", key=f"link_{unique_id}")
    st.markdown("---", key=f"divider_{unique_id}")

# Main function
def main():
    st.set_page_config(page_title="Saudi Stock Market News", page_icon="📈", layout="wide")
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # File uploader for companies CSV
    uploaded_file = st.file_uploader("Upload companies file (optional)", type="csv")
    if uploaded_file is not None:
        companies_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(companies_df)} companies")
    else:
        # Use dummy data if no file is uploaded
        companies_df = pd.DataFrame({
            'name': ['Saudi Aramco', 'SABIC', 'Al Rajhi Bank'],
            'symbol': ['2222.SR', '2010.SR', '1120.SR']
        })
