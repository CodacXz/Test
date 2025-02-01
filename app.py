import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"

# Get API key from secrets with fallback
try:
    API_TOKEN = st.secrets["general"]["MARKETAUX_API_KEY"]
except Exception as e:
    st.error("Error loading API key. Please check your secrets.toml file.")
    st.stop()

@st.cache_data
def load_company_data(uploaded_file=None):
    """Load company data from uploaded file or default GitHub URL"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            github_url = "https://raw.githubusercontent.com/CodacXz/Test/main/saudi_companies.csv?raw=true"
            df = pd.read_csv(github_url)
        
        # Clean and prepare data
        df['Company_Name'] = df['Company_Name'].str.strip()
        df['Company_Code'] = df['Company_Code'].astype(str).str.zfill(4)
        
        return df
    except Exception as e:
        st.error(f"Error loading company data: {str(e)}")
        return pd.DataFrame()

def find_companies_in_text(text, companies_df):
    """Find unique companies mentioned in the text"""
    if not text or companies_df.empty:
        return []
    
    text = text.lower()
    seen_companies = set()  # Track unique companies
    mentioned_companies = []
    
    for _, row in companies_df.iterrows():
        company_name = str(row['Company_Name']).lower()
        company_code = str(row['Company_Code'])
        
        # Only add each company once
        if (company_name in text or company_code in text) and company_code not in seen_companies:
            seen_companies.add(company_code)
            mentioned_companies.append({
                'name': row['Company_Name'],
                'code': company_code,
                'symbol': f"{company_code}.SR"
            })
    
    return mentioned_companies

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    confidence = (abs(compound) * 100)  # Convert to percentage
    
    if compound >= 0.05:
        sentiment = "🟢 Positive"
    elif compound <= -0.05:
        sentiment = "🔴 Negative"
    else:
        sentiment = "⚪ Neutral"
    
    return sentiment, confidence

def fetch_news(published_after, limit=3):
    """Fetch news articles"""
    params = {
        "api_token": API_TOKEN,
        "countries": "sa",
        "filter_entities": "true",
        "limit": limit,
        "published_after": published_after,
        "language": "en",
        "must_have_entities": "true",  # Only get articles with entities
        "group_similar": "true"  # Group similar articles to save API calls
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache stock data for 1 hour
def get_stock_data(symbol, period='1mo'):
    """Fetch stock data and calculate technical indicators"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            return None, f"No stock data available for {symbol}"
        
        # Calculate indicators
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def display_article_preview(article, article_idx):
    """Display just the article without analysis"""
    title = article.get('title', 'No title')
    description = article.get('description', 'No description')
    url = article.get('url', '#')
    source = article.get('source', 'Unknown')
    published_at = article.get('published_at', '')
    
    # Create unique key prefix from title
    unique_key = f"{hash(title)}_{article_idx}"
    
    with st.container():
        # Title row with skip button
        title_col, skip_col = st.columns([5, 1])
        with title_col:
            st.markdown(f"## {title}", key=f"title_{unique_key}")
        with skip_col:
            if st.button("⏭️ Skip", key=f"skip_{unique_key}"):
                st.session_state[f'skip_{unique_key}'] = True
                st.experimental_rerun()
                return False
        
        # Check if this article should be skipped
        if st.session_state.get(f'skip_{unique_key}', False):
            return False
            
        st.write(f"Source: {source} | Published: {published_at[:16]}", key=f"meta_{unique_key}")
        st.write(description, key=f"desc_{unique_key}")
        st.markdown(f"[Read full article]({url})", key=f"url_{unique_key}")
        st.markdown("---", key=f"divider_{unique_key}")
        
        return True

def analyze_articles(articles, companies_df):
    """Analyze all articles at once"""
    st.header("📊 Articles Analysis")
    
    for idx, article in enumerate(articles):
        if not st.session_state.get(f'skip_{idx}', False):
            title = article.get('title', 'No title')
            description = article.get('description', 'No description')
            text = f"{title} {description}"
            
            st.subheader(f"Analysis for: {title}")
            
            # Sentiment Analysis
            sentiment, confidence = analyze_sentiment(text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Sentiment Analysis")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Get unique companies from entities
            entities = article.get('entities', [])
            seen_symbols = set()
            unique_companies = []
            
            for entity in entities:
                symbol = entity.get('symbol')
                if symbol and symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    unique_companies.append(entity)
            
            if unique_companies:
                st.write("### Companies Mentioned")
                for company in unique_companies:
                    st.write(f"**{company.get('name')} ({company.get('symbol')})**")
                
                # Create tabs for each company's analysis
                if len(unique_companies) > 0:
                    company_tabs = st.tabs([f"{company.get('name', 'Company')} Analysis" for company in unique_companies])
                    
                    for tab_idx, (tab, company) in enumerate(zip(company_tabs, unique_companies)):
                        with tab:
                            try:
                                symbol = company.get('symbol')
                                df, error = get_stock_data(f"{symbol}.SR")
                                
                                if error:
                                    st.error(error)
                                elif df is not None and not df.empty:
                                    latest_price = df['Close'][-1]
                                    price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
                                    
                                    # Use unique keys for metrics
                                    metrics_cols = st.columns(3)
                                    with metrics_cols[0]:
                                        st.metric(
                                            "Current Price",
                                            f"{latest_price:.2f} SAR",
                                            f"{price_change:.2f}%",
                                            key=f"price_{idx}_{tab_idx}"
                                        )
                                    with metrics_cols[1]:
                                        st.metric(
                                            "Day High",
                                            f"{df['High'][-1]:.2f} SAR",
                                            key=f"high_{idx}_{tab_idx}"
                                        )
                                    with metrics_cols[2]:
                                        st.metric(
                                            "Day Low",
                                            f"{df['Low'][-1]:.2f} SAR",
                                            key=f"low_{idx}_{tab_idx}"
                                        )
                                    
                                    # Technical Analysis
                                    st.subheader("Technical Analysis Signals")
                                    
                                    # MACD Analysis
                                    macd_signal = "BULLISH" if df['MACD'][-1] > df['MACD_Signal'][-1] else "BEARISH"
                                    macd_reason = "MACD line above signal line" if macd_signal == "BULLISH" else "MACD line below signal line"
                                    
                                    # RSI Analysis
                                    rsi_value = df['RSI'][-1]
                                    if rsi_value > 70:
                                        rsi_signal = "BEARISH"
                                        rsi_reason = "Overbought condition (RSI > 70)"
                                    elif rsi_value < 30:
                                        rsi_signal = "BULLISH"
                                        rsi_reason = "Oversold condition (RSI < 30)"
                                    else:
                                        rsi_signal = "NEUTRAL"
                                        rsi_reason = "RSI in neutral zone"
                                    
                                    # Bollinger Bands Analysis
                                    if df['Close'][-1] > df['BB_upper'][-1]:
                                        bb_signal = "BEARISH"
                                        bb_reason = "Price above upper band"
                                    elif df['Close'][-1] < df['BB_lower'][-1]:
                                        bb_signal = "BULLISH"
                                        bb_reason = "Price below lower band"
                                    else:
                                        bb_signal = "NEUTRAL"
                                        bb_reason = "Price within bands"
                                    
                                    signals_df = pd.DataFrame({
                                        'Indicator': ['MACD', 'RSI', 'Bollinger Bands'],
                                        'Signal': [macd_signal, rsi_signal, bb_signal],
                                        'Reason': [macd_reason, rsi_reason, bb_reason]
                                    })
                                    
                                    st.dataframe(signals_df, key=f"signals_{idx}_{tab_idx}")
                                    
                                    # Create stock chart with unique key
                                    fig = go.Figure()
                                    fig.add_trace(go.Candlestick(
                                        x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name='Price'
                                    ))
                                    
                                    fig.update_layout(
                                        title=None,
                                        yaxis_title='Price (SAR)',
                                        xaxis_title='Date',
                                        template='plotly_dark',
                                        height=400,
                                        margin=dict(t=0)
                                    )
                                    
                                    st.plotly_chart(fig, key=f"chart_{idx}_{tab_idx}", use_container_width=True)
                                    
                            except Exception as e:
                                st.error(f"Error analyzing {company.get('name')}: {str(e)}")
            
            st.markdown("---")

def check_api_credits():
    """Check remaining API credits"""
    try:
        params = {
            "api_token": API_TOKEN
        }
        response = requests.get("https://api.marketaux.com/v1/usage", params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("credits", {})
    except Exception as e:
        st.error(f"Error checking API credits: {str(e)}")
        return None

def main():
    st.title("Saudi Stock Market News", key="main_title")
    st.write("Real-time news analysis for Saudi stock market", key="main_desc")
    
    # Initialize session state for API calls
    if 'api_calls_today' not in st.session_state:
        st.session_state.api_calls_today = 0
    
    # Check and display API credits
    credits = check_api_credits()
    if credits:
        st.sidebar.write("### API Credits")
        st.sidebar.write(f"Used: {credits.get('used', 'N/A')}")
        st.sidebar.write(f"Remaining: {credits.get('remaining', 'N/A')}")
        st.sidebar.write(f"Limit: {credits.get('limit', 'N/A')}")
    
    # Add reset button in sidebar
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            if key.startswith('skip_') or key == 'api_calls_today':
                del st.session_state[key]
        st.experimental_rerun()
    
    # Load company data
    uploaded_file = st.sidebar.file_uploader(
        "Upload company data (optional)",
        type=['csv'],
        key="file_uploader"
    )
    
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.error("Failed to load company data")
        return
    
    # Date range selector
    days_ago = st.sidebar.slider(
        "Days of news to fetch",
        min_value=1,
        max_value=30,
        value=1,
        key="days_slider"
    )
    
    published_after = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Number of articles
    article_limit = st.sidebar.number_input(
        "Number of articles to fetch",
        min_value=1,
        max_value=3,
        value=3,
        key="article_limit"
    )
    
    # Fetch news
    if st.button("Fetch News", key="fetch_button", use_container_width=True):
        with st.spinner("Fetching latest news..."):
            news_data = fetch_news(published_after, limit=article_limit)
            
            if not news_data:
                st.error("No news articles found")
                return
            
            # First show all articles
            st.header("📰 Latest News")
            for idx, article in enumerate(news_data):
                with st.container():
                    title = article.get('title', 'No title')
                    description = article.get('description', 'No description')
                    url = article.get('url', '#')
                    source = article.get('source', 'Unknown')
                    published_at = article.get('published_at', '')
                    
                    st.markdown(f"## {title}")
                    st.write(f"Source: {source} | Published: {published_at[:16]}")
                    st.write(description)
                    st.markdown(f"[Read full article]({url})")
                    st.markdown("---")
            
            # Then analyze all articles
            analyze_articles(news_data, companies_df)

if __name__ == "__main__":
    main()
