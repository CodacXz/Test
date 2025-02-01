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
        if not st.session_state.get(f'skip_{hash(article.get("title", ""))}_preview_{idx}', False):
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
            
            # Find mentioned companies
            mentioned_company = None
            for _, row in companies_df.iterrows():
                company_name = str(row['Company_Name']).lower()
                company_code = str(row['Company_Code'])
                if company_name in text.lower() or company_code in text.lower():
                    mentioned_company = {
                        'name': row['Company_Name'],
                        'code': company_code,
                        'symbol': f"{company_code}.SR"
                    }
                    break
            
            if mentioned_company:
                st.write("### Company Analysis")
                st.write(f"**{mentioned_company['name']} ({mentioned_company['symbol']})**")
                
                try:
                    df, error = get_stock_data(mentioned_company['symbol'])
                    if error:
                        st.error(error)
                    else:
                        if df is not None and not df.empty:
                            latest_price = df['Close'][-1]
                            price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
                            
                            metrics_cols = st.columns(3)
                            with metrics_cols[0]:
                                st.metric(
                                    "Current Price", 
                                    f"{latest_price:.2f} SAR",
                                    f"{price_change:.2f}%"
                                )
                            with metrics_cols[1]:
                                st.metric(
                                    "Day High", 
                                    f"{df['High'][-1]:.2f} SAR"
                                )
                            with metrics_cols[2]:
                                st.metric(
                                    "Day Low", 
                                    f"{df['Low'][-1]:.2f} SAR"
                                )
                            
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error analyzing {mentioned_company['name']}: {str(e)}")
            
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
    
    # Initialize session state for API calls and articles
    if 'api_calls_today' not in st.session_state:
        st.session_state.api_calls_today = 0
    if 'fetched_articles' not in st.session_state:
        st.session_state.fetched_articles = []
    if 'analyzed_articles' not in st.session_state:
        st.session_state.analyzed_articles = []
    
    # Check and display API credits
    credits = check_api_credits()
    if credits:
        with st.sidebar:
            st.write("### API Credits")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Used", credits.get('used', 'N/A'))
            with cols[1]:
                st.metric("Remaining", credits.get('remaining', 'N/A'))
            with cols[2]:
                st.metric("Limit", credits.get('limit', 'N/A'))
    
    # Add reset button in sidebar
    if st.sidebar.button("🔄 Reset Session", help="Clear all cached articles and reset session"):
        for key in list(st.session_state.keys()):
            if key.startswith('skip_') or key in ['api_calls_today', 'fetched_articles', 'analyzed_articles']:
                del st.session_state[key]
        st.experimental_rerun()
    
    # Load company data
    uploaded_file = st.sidebar.file_uploader(
        "Upload company data (optional)",
        type=['csv'],
        help="Upload a custom CSV file with company data",
        key="file_uploader"
    )
    
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.error("Failed to load company data. Please check your connection or try uploading a valid CSV file.")
        return
    
    # Date range selector with better defaults
    col1, col2 = st.sidebar.columns(2)
    with col1:
        days_ago = st.number_input(
            "Days of news",
            min_value=1,
            max_value=30,
            value=1,
            help="Number of days to look back for news",
            key="days_input"
        )
    with col2:
        article_limit = st.number_input(
            "Articles to fetch",
            min_value=1,
            max_value=3,
            value=3,
            help="Number of articles to fetch (max 3)",
            key="article_limit"
        )
    
    published_after = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Fetch news with progress tracking
    if st.button("🔄 Fetch Latest News", key="fetch_button", use_container_width=True):
        progress_text = "Fetching latest news articles..."
        progress_bar = st.progress(0, text=progress_text)
        
        try:
            with st.spinner(progress_text):
                news_data = fetch_news(published_after, limit=article_limit)
                progress_bar.progress(50, text="Processing articles...")
                
                if not news_data:
                    st.warning("No news articles found for the selected date range. Try adjusting the date range or try again later.")
                    progress_bar.empty()
                    return
                
                st.session_state.fetched_articles = news_data
                progress_bar.progress(100, text="Done!")
                progress_bar.empty()
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            progress_bar.empty()
            return
    
    # Display fetched articles
    if st.session_state.fetched_articles:
        articles_tab, analysis_tab = st.tabs(["📰 Articles", "📊 Analysis"])
        
        with articles_tab:
            st.header("Latest News Articles")
            for idx, article in enumerate(st.session_state.fetched_articles):
                with st.container():
                    title = article.get('title', 'No title')
                    description = article.get('description', 'No description')
                    url = article.get('url', '#')
                    source = article.get('source', 'Unknown')
                    published_at = article.get('published_at', '')
                    
                    col1, col2 = st.columns([5,1])
                    with col1:
                        st.markdown(f"### {title}")
                    with col2:
                        if st.button("⏭️ Skip", key=f"skip_{idx}", help="Skip this article from analysis"):
                            st.session_state[f'skip_{idx}'] = True
                            st.experimental_rerun()
                    
                    if not st.session_state.get(f'skip_{idx}', False):
                        st.markdown(f"**Source:** {source} | **Published:** {published_at[:16]}")
                        st.write(description)
                        st.markdown(f"[🔗 Read full article]({url})")
                        st.divider()
        
        with analysis_tab:
            st.header("Articles Analysis")
            
            if not any(not st.session_state.get(f'skip_{idx}', False) for idx in range(len(st.session_state.fetched_articles))):
                st.info("No articles selected for analysis. Uncheck the skip button on articles you want to analyze.")
                return
            
            for idx, article in enumerate(st.session_state.fetched_articles):
                if st.session_state.get(f'skip_{idx}', False):
                    continue
                
                with st.expander(f"Analysis: {article.get('title', 'No title')}", expanded=True):
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    entities = article.get('entities', [])
                    
                    # Sentiment Analysis
                    if entities:
                        entity = entities[0]
                        sentiment_score = entity.get('sentiment_score', 0)
                        sentiment = "🟢 Positive" if sentiment_score > 0 else "🔴 Negative" if sentiment_score < 0 else "⚪ Neutral"
                        confidence = abs(sentiment_score * 100)
                    else:
                        sentiment, confidence = analyze_sentiment(text)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Sentiment Analysis")
                        st.metric("Sentiment", sentiment)
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Company Analysis
                    symbol = entities[0].get('symbol') if entities else None
                    if symbol:
                        with col2:
                            st.markdown("#### Company Details")
                            st.write(f"**{entities[0].get('name')} ({symbol})**")
                        
                        try:
                            df, error = get_stock_data(f"{symbol}.SR")
                            if error:
                                st.error(error)
                            elif df is not None and not df.empty:
                                latest_price = df['Close'][-1]
                                price_change = ((latest_price - df['Close'][-2])/df['Close'][-2]*100)
                                
                                metrics_cols = st.columns(3)
                                with metrics_cols[0]:
                                    st.metric(
                                        "Current Price", 
                                        f"{latest_price:.2f} SAR",
                                        f"{price_change:.2f}%"
                                    )
                                with metrics_cols[1]:
                                    st.metric(
                                        "Day High", 
                                        f"{df['High'][-1]:.2f} SAR"
                                    )
                                with metrics_cols[2]:
                                    st.metric(
                                        "Day Low", 
                                        f"{df['Low'][-1]:.2f} SAR"
                                    )
                                
                                # Create stock chart
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
                                
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error analyzing stock data: {str(e)}")
                    
                    st.divider()

if __name__ == "__main__":
    main()
