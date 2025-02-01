import requests
import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = "W737FzQuSSOm3MyYYLJ1kt7csT8NOwxl2WL7Gl6x"  # Temporary direct key for testing

# GitHub raw file URL for the companies CSV file
GITHUB_CSV_URL = "https://raw.githubusercontent.com/CodacXz/Test/main/saudi_companies.csv?raw=true"

@st.cache_data
def load_company_data(uploaded_file=None):
    """Load and cache company data from either uploaded file or GitHub"""
    try:
        if uploaded_file is not None:
            # Load from uploaded file
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', engine='python')
        else:
            # Load from GitHub
            response = requests.get(GITHUB_CSV_URL, timeout=10)
            response.raise_for_status()
            # Use pandas read_csv with explicit parameters
            df = pd.read_csv(
                GITHUB_CSV_URL,
                encoding='utf-8',
                sep=',',
                engine='python',
                on_bad_lines='skip'  # Skip problematic lines
            )
        
        # Convert company names and codes to lowercase for better matching
        df['Company_Name_Lower'] = df['Company_Name'].str.lower()
        # Convert company codes to strings and ensure they're padded to 4 digits
        df['Company_Code'] = df['Company_Code'].astype(str).str.zfill(4)
        
        # Log the number of companies loaded
        st.sidebar.success(f"✅ Successfully loaded {len(df)} companies")
        return df
    except Exception as e:
        st.error(f"Error loading company data: {e}")
        # Print more detailed error information
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def find_companies_in_text(text, companies_df):
    """Find all companies mentioned in the text"""
    if companies_df.empty:
        return []
    
    text = text.lower()
    mentioned_companies = []
    
    # Common name variations
    name_variations = {
        'al rajhi': '1120',
        'alrajhi': '1120',
        'rajhi': '1120',
        'saudi fransi': '1050',
        'banque saudi fransi': '1050',
        'bsf': '1050',
        'aljazira': '1020',
        'al jazira': '1020',
        'anb': '1080',
        'arab national': '1080',
        'arab national bank': '1080'
    }
    
    # First check for name variations
    for variation, code in name_variations.items():
        if variation in text:
            company = companies_df[companies_df['Company_Code'] == code].iloc[0]
            mentioned_companies.append({
                'code': str(company['Company_Code']).zfill(4),
                'name': company['Company_Name'],
                'symbol': f"{str(company['Company_Code']).zfill(4)}.SR"
            })
    
    # Then check for exact matches from the dataframe
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code']).zfill(4)
        
        # Skip if already added through variations
        if company_code in [c['code'] for c in mentioned_companies]:
            continue
        
        # Check for exact company name or code
        if company_name in text or company_code in text:
            mentioned_companies.append({
                'code': company_code,
                'name': row['Company_Name'],
                'symbol': f"{company_code}.SR"
            })
    
    return mentioned_companies

def find_company_code(text, companies_df):
    """Find company code from news text"""
    if companies_df.empty:
        return None, None
    
    text_lower = text.lower()
    
    # Try to find any company name in the text
    for _, row in companies_df.iterrows():
        if row['Company_Name_Lower'] in text_lower:
            return row['Company_Code'], row['Company_Name']
    
    return None, None

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores['compound']
    
    # Convert compound score to sentiment
    if compound >= 0.05:
        sentiment = "POSITIVE"
        confidence = (compound + 1) / 2 * 100  # Convert to percentage
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
        confidence = (-compound + 1) / 2 * 100
    else:
        sentiment = "NEUTRAL"
        confidence = 50
    
    return sentiment, confidence

def fetch_news(published_after, limit=3):
    """Fetch news articles from MarketAux API"""
    params = {
        "api_token": API_TOKEN,
        "countries": "sa",
        "filter_entities": "true",
        "limit": limit,
        "published_after": published_after
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            st.error(f"API Error: {data['error']['message']}")
            return []
        return data.get("data", [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def get_stock_data(symbol, period='1mo'):
    """Fetch stock data and calculate technical indicators"""
    try:
        # Format symbol for Saudi market
        symbol = str(symbol).zfill(4) + ".SR"  # Saudi market uses 4 digits + .SR
        
        # Get stock data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        if df.empty:
            return None, f"No stock data available for {symbol}"
        
        # Calculate technical indicators
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # RSI
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def analyze_technical_indicators(df):
    """Analyze technical indicators and generate trading signals"""
    signals = []
    
    # MACD Analysis
    last_macd = df['MACD'].iloc[-1]
    last_signal = df['MACD_Signal'].iloc[-1]
    macd_diff = last_macd - last_signal
    
    if macd_diff > 0.1:  # Add threshold to avoid noise
        signals.append(("MACD", "BULLISH", "MACD line above signal line"))
    elif macd_diff < -0.1:
        signals.append(("MACD", "BEARISH", "MACD line below signal line"))
    else:
        signals.append(("MACD", "NEUTRAL", "MACD and signal lines are close"))

    # RSI Analysis
    last_rsi = df['RSI'].iloc[-1]
    if last_rsi > 75:  # Increased threshold for stronger signal
        signals.append(("RSI", "BEARISH", f"Strongly overbought (RSI: {last_rsi:.2f})"))
    elif last_rsi > 70:
        signals.append(("RSI", "NEUTRAL", f"Potentially overbought (RSI: {last_rsi:.2f})"))
    elif last_rsi < 25:  # Increased threshold for stronger signal
        signals.append(("RSI", "BULLISH", f"Strongly oversold (RSI: {last_rsi:.2f})"))
    elif last_rsi < 30:
        signals.append(("RSI", "NEUTRAL", f"Potentially oversold (RSI: {last_rsi:.2f})"))
    else:
        signals.append(("RSI", "NEUTRAL", f"Normal range (RSI: {last_rsi:.2f})"))

    # Bollinger Bands Analysis
    last_close = df['Close'].iloc[-1]
    last_upper = df['BB_upper'].iloc[-1]
    last_lower = df['BB_lower'].iloc[-1]
    bb_width = (last_upper - last_lower) / df['Close'].mean()
    
    # Calculate percentage distance from bands
    upper_dist = (last_upper - last_close) / last_close * 100
    lower_dist = (last_close - last_lower) / last_close * 100
    
    if upper_dist < -1:  # Price is more than 1% above upper band
        signals.append(("Bollinger Bands", "BEARISH", "Price significantly above upper band"))
    elif lower_dist < -1:  # Price is more than 1% below lower band
        signals.append(("Bollinger Bands", "BULLISH", "Price significantly below lower band"))
    else:
        signals.append(("Bollinger Bands", "NEUTRAL", "Price within normal range of bands"))
    
    return signals

def plot_stock_analysis(df, company_name, symbol):
    """Create an interactive plot with price and indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                            line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                            line=dict(color='gray', dash='dash')))
    
    fig.update_layout(
        title=f'{company_name} ({symbol}) - Price and Technical Indicators',
        yaxis_title='Price (SAR)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def display_article(article, companies_df):
    """Display news article with sentiment and technical analysis"""
    title = article.get("title", "No title")
    description = article.get("description", "No description")
    url = article.get("url", "#")
    source = article.get("source", "Unknown")
    published_at = article.get("published_at", "")
    
    st.markdown(f"## {title}")
    st.write(f"Source: {source} | Published: {published_at[:16]}")
    st.write(description[:200] + "..." if len(description) > 200 else description)
    
    # Sentiment Analysis
    sentiment, confidence = analyze_sentiment(title + " " + description)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sentiment Analysis")
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
        st.markdown("### Companies Mentioned")
        for company in mentioned_companies:
            st.markdown(f"- {company['name']} ({company['symbol']})")
        
        st.markdown("### Stock Analysis")
        tabs = st.tabs([company['name'] for company in mentioned_companies])
        
        for tab, company in zip(tabs, mentioned_companies):
            with tab:
                df, error = get_stock_data(company['code'])
                if error:
                    st.error(f"Error fetching stock data: {error}")
                    continue
                
                if df is not None:
                    # Price Analysis
                    latest_price = df['Close'][-1]
                    prev_price = df['Close'][-2]
                    price_change = ((latest_price - prev_price)/prev_price*100)
                    
                    # Price metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"{latest_price:.2f} SAR", 
                                f"{price_change:.2f}%")
                    with col2:
                        st.metric("Day High", f"{df['High'][-1]:.2f} SAR")
                    with col3:
                        st.metric("Day Low", f"{df['Low'][-1]:.2f} SAR")
                    
                    # Technical Analysis
                    signals = analyze_technical_indicators(df)
                    
                    # Count bullish vs bearish signals
                    signal_counts = {
                        "BULLISH": len([s for s in signals if s[1] == "BULLISH"]),
                        "BEARISH": len([s for s in signals if s[1] == "BEARISH"]),
                        "NEUTRAL": len([s for s in signals if s[1] == "NEUTRAL"])
                    }
                    
                    # Technical Overview
                    st.markdown("### Technical Overview")
                    tech_cols = st.columns(3)
                    with tech_cols[0]:
                        st.metric("Bullish Signals", signal_counts["BULLISH"])
                    with tech_cols[1]:
                        st.metric("Bearish Signals", signal_counts["BEARISH"])
                    with tech_cols[2]:
                        st.metric("Neutral Signals", signal_counts["NEUTRAL"])
                    
                    # Plot stock chart
                    fig = plot_stock_analysis(df, company['name'], company['symbol'])
                    st.plotly_chart(fig)
                    
                    # Technical Analysis Signals
                    st.markdown("### Technical Analysis Signals")
                    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                    st.table(signal_df)
                    
                    # Combined Analysis
                    st.markdown("### Combined Analysis")
                    tech_score = (signal_counts["BULLISH"] - signal_counts["BEARISH"]) / len(signals)
                    news_score = 1 if sentiment == "POSITIVE" else -1 if sentiment == "NEGATIVE" else 0
                    
                    # Weight technical analysis more heavily (70/30 split)
                    combined_score = (0.7 * tech_score) + (0.3 * news_score)
                    
                    if combined_score > 0.3:
                        st.success("🟢 Overall Bullish: Technical indicators and news sentiment suggest positive momentum")
                    elif combined_score < -0.3:
                        st.error("🔴 Overall Bearish: Technical indicators and news sentiment suggest negative pressure")
                    else:
                        st.warning("🟡 Neutral: Mixed signals from technical indicators and news sentiment")
                    
                    # Volume Analysis
                    avg_volume = df['Volume'].mean()
                    latest_volume = df['Volume'][-1]
                    volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                    
                    st.markdown("### Volume Analysis")
                    st.metric("Trading Volume", f"{int(latest_volume):,}", 
                             f"{volume_change:.1f}% vs 30-day average")
    
    st.markdown(f"[Read full article]({url})")
    st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # File upload option in sidebar
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload companies file (optional)", type=['csv'])
    
    # Load company data
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.warning("⚠️ No company data loaded. Either upload a CSV file or update the GitHub URL in the code.")
    else:
        st.sidebar.success(f"✅ Loaded {len(companies_df)} companies")

    # Rest of the settings
    limit = st.sidebar.slider("Number of articles", 1, 3, 3)
    
    # Date selection
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input("Show news published after:", value=default_date)
    published_after_iso = published_after.isoformat() + "T00:00:00"

    # Fetch and display news
    if st.button("Fetch News"):
        with st.spinner("Fetching latest news..."):
            news_articles = fetch_news(published_after_iso, limit)
            
            if news_articles:
                st.success(f"Found {len(news_articles)} articles")
                for article in news_articles:
                    display_article(article, companies_df)
            else:
                st.warning("No news articles found for the selected date range")

    # App information
    st.sidebar.markdown("---")
    st.sidebar.write("App Version: 1.0.5")
    
    # API status
    st.sidebar.success("✅ API Token loaded")
    
    # Add GitHub information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### How to use company data:
    1. **Option 1:** Upload CSV file using the uploader above
    2. **Option 2:** Add file to GitHub and update `GITHUB_CSV_URL`
    
    CSV file format:
    ```
    Company_Code,Company_Name
    1010,Riyad Bank
    1020,Bank Aljazira
    ...
    ```
    """)

if __name__ == "__main__":
    main()
