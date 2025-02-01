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
def display_article(article):
    title = article['title']
    description = article['description']
    url = article['url']
    source = article['source']
    published_at = article['published_at']
    
    st.markdown(f"## {title}")
    st.write(f"**Source:** {source} | **Published:** {published_at[:16]}")
    st.write(description)
    
    sentiment, confidence = analyze_sentiment(title + " " + description)
    st.write(f"### Sentiment Analysis: {sentiment} ({confidence:.2f}%)")
    
    mentioned_companies = find_companies_in_text(title + " " + description, companies_df)
    if mentioned_companies:
        for company in mentioned_companies:
            st.markdown(f"#### 📈 {company['name']} ({company['symbol']}) Analysis")
            df, error = get_stock_data(company['code'])
            if error:
                st.error(f"Error fetching stock data: {error}")
                continue
            
            latest_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = ((latest_price - prev_price) / prev_price) * 100
            st.metric("Current Price", f"{latest_price:.2f} SAR", f"{change:.2f}%")
            
            fig = plot_stock_analysis(df, company['name'], company['symbol'])
            st.plotly_chart(fig, use_container_width=True)
            
            signals = analyze_technical_indicators(df)
            st.table(pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason']))
    
    st.markdown(f"[Read full article]({url})")
    st.markdown("---")

# Main Application
def main():
    st.title("📊 Saudi Stock Market News & Analysis")
    
    news_articles = fetch_news()
    for article in news_articles:
        display_article(article)

if __name__ == "__main__":
    main()
