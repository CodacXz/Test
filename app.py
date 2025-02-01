import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
GITHUB_CSV_URL = "https://raw.githubusercontent.com/saudistocks/stock_data/main/data/companies.csv"

def main():
    # Must be the first Streamlit command
    st.set_page_config(page_title="Saudi Stock Market News", page_icon="📈", layout="wide")
    
    # Load custom CSS
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Get API token from secrets
    API_TOKEN = st.secrets.get("MARKETAUX_TOKEN", "")
    
    st.title("Saudi Stock Market News")
    st.markdown("Real-time news analysis for Saudi stock market")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload companies file (optional)", 
                                       type=['csv'], 
                                       help="CSV format: Company_Code,Company_Name")
        
        # Number of articles slider
        num_articles = st.slider("Number of articles", 1, 10, 3)
        
        # Version info
        st.markdown("App Version: 1.0.5")
        
        # Show API token status
        if API_TOKEN:
            st.success("✅ API Token loaded")
        else:
            st.error("❌ API Token missing")
        
        # Help section
        with st.expander("How to use company data"):
            st.write("""
            Option 1: Upload CSV file using the uploader above
            Option 2: Add file to GitHub and update GITHUB_CSV_URL
            
            CSV file format:
            ```
            Company_Code,Company_Name
            1010,Riyad Bank
            1020,Bank Aljazira
            ...
            ```
            """)
    
    # Load companies data
    companies_df = load_company_data(uploaded_file)
    if not companies_df.empty:
        st.success(f"✅ Successfully loaded {len(companies_df)} companies")
    
    # Date picker for news
    published_after = st.date_input("Show news published after:", 
                                  value=datetime.now() - timedelta(days=7),
                                  max_value=datetime.now())
    
    # Fetch and display news
    articles = fetch_news(API_TOKEN, published_after.strftime("%Y/%m/%d"), num_articles)
    
    if articles:
        st.write(f"\nFound {len(articles)} articles\n")
        
        # Articles Summary
        st.markdown("### 📰 News Summary")
        for i, article in enumerate(articles):
            title = article.get("title", "No title")
            source = article.get("source", "Unknown")
            published_at = article.get("published_at", "")[:16]
            
            # Get sentiment
            sentiment, confidence = analyze_sentiment(title + " " + article.get("description", ""))
            sentiment_color = {
                "POSITIVE": "green",
                "NEGATIVE": "red",
                "NEUTRAL": "gray"
            }.get(sentiment, "black")
            
            # Find companies
            mentioned_companies = find_companies_in_text(
                title + " " + article.get("description", ""), 
                companies_df
            )
            
            # Create summary card
            with st.container():
                st.markdown(f"#### {i+1}. {title}")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**Source:** {source} | **Published:** {published_at}")
                with col2:
                    st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment}] ({confidence:.1f}%)")
                with col3:
                    if mentioned_companies:
                        company_names = [f"`{c['symbol']}`" for c in mentioned_companies]
                        st.markdown(f"**Companies:** {', '.join(company_names)}")
                st.markdown("---")
        
        # Detailed Article Analysis
        st.markdown("### 📊 Detailed Analysis")
        article_tabs = st.tabs([f"Article {i+1}" for i in range(len(articles))])
        
        for tab, article in zip(article_tabs, articles):
            with tab:
                display_article(article, companies_df)
    else:
        st.info("No articles found for the selected date range")

def load_company_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(GITHUB_CSV_URL)
        return df
    except Exception as e:
        return pd.DataFrame()

def fetch_news(api_token, published_after, limit=3):
    params = {
        "api_token": api_token,
        "countries": "sa",
        "language": "en",
        "limit": limit,
        "published_after": published_after
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        return []

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "POSITIVE"
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    confidence = abs(compound) * 100
    return sentiment, confidence

def find_companies_in_text(text, companies_df):
    if companies_df.empty:
        return []
    
    mentioned_companies = []
    text = text.lower()
    
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        if company_name in text:
            mentioned_companies.append({
                'code': row['Company_Code'],
                'name': row['Company_Name'],
                'symbol': f"{row['Company_Code']}.SR"
            })
    
    return mentioned_companies

def get_stock_data(code):
    try:
        symbol = f"{code}.SR"
        stock = yf.Ticker(symbol)
        df = stock.history(period="1mo")
        return df, None
    except Exception as e:
        return None, str(e)

def analyze_technical_indicators(df):
    signals = []
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    if macd.iloc[-1] > signal.iloc[-1]:
        signals.append(["MACD", "BULLISH", "MACD line above signal line"])
    else:
        signals.append(["MACD", "BEARISH", "MACD line below signal line"])
    
    # RSI
    rsi = RSIIndicator(df['Close']).rsi()
    current_rsi = rsi.iloc[-1]
    
    if current_rsi > 70:
        signals.append(["RSI", "BEARISH", "Overbought condition (RSI > 70)"])
    elif current_rsi < 30:
        signals.append(["RSI", "BULLISH", "Oversold condition (RSI < 30)"])
    else:
        signals.append(["RSI", "NEUTRAL", f"Normal range (RSI: {current_rsi:.2f})"])
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'])
    current_price = df['Close'].iloc[-1]
    upper_band = bb.bollinger_hband().iloc[-1]
    lower_band = bb.bollinger_lband().iloc[-1]
    
    if current_price > upper_band:
        signals.append(["Bollinger Bands", "BEARISH", "Price above upper band"])
    elif current_price < lower_band:
        signals.append(["Bollinger Bands", "BULLISH", "Price below lower band"])
    else:
        signals.append(["Bollinger Bands", "NEUTRAL", "Price within bands"])
    
    return signals

def plot_stock_analysis(df, company_name, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=(f'{company_name} ({symbol})', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False
    )
    
    return fig

def get_combined_analysis(signals, sentiment, confidence):
    # Count technical signals
    tech_signals = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    for signal in signals:
        tech_signals[signal[1]] += 1
    
    # Convert sentiment to signal
    sentiment_signal = {
        "POSITIVE": "BULLISH",
        "NEGATIVE": "BEARISH",
        "NEUTRAL": "NEUTRAL"
    }[sentiment]
    
    # Calculate technical score (60% weight)
    tech_score = (tech_signals["BULLISH"] - tech_signals["BEARISH"]) / len(signals)
    
    # Calculate sentiment score (40% weight)
    sentiment_score = {
        "BULLISH": 1,
        "BEARISH": -1,
        "NEUTRAL": 0
    }[sentiment_signal] * (confidence / 100)
    
    # Combine scores with weights
    final_score = (tech_score * 0.6) + (sentiment_score * 0.4)
    
    # Determine overall signal
    if final_score > 0.2:
        signal = "🟢 Overall Bullish"
        detail = "Technical indicators and news sentiment suggest positive momentum"
    elif final_score < -0.2:
        signal = "🔴 Overall Bearish"
        detail = "Technical indicators and news sentiment suggest negative pressure"
    else:
        signal = "🟡 Neutral"
        detail = "Mixed signals from technical indicators and news sentiment"
    
    return signal, detail

def display_article(article, companies_df):
    title = article.get("title", "No title")
    description = article.get("description", "No description")
    url = article.get("url", "#")
    source = article.get("source", "Unknown")
    published_at = article.get("published_at", "")
    
    # Article header with metadata
    st.markdown(f"## {title}")
    st.markdown(f"**Source:** {source} | **Published:** {published_at[:16]}")
    
    # Article description in a quote block
    st.markdown("> " + (description[:200] + "..." if len(description) > 200 else description))
    
    # Analysis container
    with st.container():
        # Two columns: Sentiment and Companies Overview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Sentiment Analysis card
            st.markdown("### 📊 Sentiment Analysis")
            sentiment, confidence = analyze_sentiment(title + " " + description)
            
            sentiment_color = {
                "POSITIVE": "green",
                "NEGATIVE": "red",
                "NEUTRAL": "gray"
            }.get(sentiment, "black")
            
            st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment}]")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
        
        # Find mentioned companies
        mentioned_companies = find_companies_in_text(title + " " + description, companies_df)
        
        if mentioned_companies:
            with col2:
                st.markdown("### 🏢 Companies Overview")
                # Create a metric row for each company's current status
                metrics_cols = st.columns(len(mentioned_companies))
                for i, company in enumerate(mentioned_companies):
                    df, error = get_stock_data(company['code'])
                    if df is not None and not error:
                        latest_price = df['Close'][-1]
                        prev_price = df['Close'][-2]
                        price_change = ((latest_price - prev_price)/prev_price*100)
                        with metrics_cols[i]:
                            st.metric(
                                f"{company['name']} ({company['symbol']})",
                                f"{latest_price:.2f} SAR",
                                f"{price_change:.2f}%"
                            )
        
        # Detailed company analysis
        if mentioned_companies:
            st.markdown("### 📈 Detailed Analysis")
            
            # Create tabs for each company
            company_tabs = st.tabs([company['name'] for company in mentioned_companies])
            
            for tab, company in zip(company_tabs, mentioned_companies):
                with tab:
                    df, error = get_stock_data(company['code'])
                    if error:
                        st.error(f"Error fetching stock data: {error}")
                        continue
                    
                    if df is not None:
                        # Technical Analysis
                        signals = analyze_technical_indicators(df)
                        
                        # Main content columns
                        chart_col, analysis_col = st.columns([2, 1])
                        
                        with chart_col:
                            # Price Chart
                            fig = plot_stock_analysis(df, company['name'], company['symbol'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Price metrics row
                            metrics_cols = st.columns(4)
                            with metrics_cols[0]:
                                st.metric("Day High", f"{df['High'][-1]:.2f} SAR")
                            with metrics_cols[1]:
                                st.metric("Day Low", f"{df['Low'][-1]:.2f} SAR")
                            with metrics_cols[2]:
                                avg_volume = df['Volume'].mean()
                                latest_volume = df['Volume'][-1]
                                volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                                st.metric("Volume", f"{int(latest_volume):,}", 
                                        f"{volume_change:.1f}% vs avg")
                            with metrics_cols[3]:
                                volatility = df['Close'].pct_change().std() * 100
                                st.metric("Volatility", f"{volatility:.1f}%")
                        
                        with analysis_col:
                            # Technical Signals
                            st.markdown("#### Technical Signals")
                            signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                            st.table(signal_df)
                            
                            # Combined Analysis
                            st.markdown("#### Combined Analysis")
                            signal, detail = get_combined_analysis(signals, sentiment, confidence)
                            st.info(f"**{signal}**\n\n{detail}")
    
    # Article link
    st.markdown("---")
    st.markdown(f"[Read full article]({url})")

if __name__ == "__main__":
    main()
