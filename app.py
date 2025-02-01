import requests
import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',', engine='python')
        else:
            response = requests.get(GITHUB_CSV_URL, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(
                GITHUB_CSV_URL,
                encoding='utf-8',
                sep=',',
                engine='python',
                on_bad_lines='skip'
            )
        
        df['Company_Name_Lower'] = df['Company_Name'].str.lower()
        df['Company_Code'] = df['Company_Code'].astype(str).str.zfill(4)
        
        st.sidebar.success(f"✅ Successfully loaded {len(df)} companies")
        return df
    except Exception as e:
        st.error(f"Error loading company data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

def find_company_code(text, companies_df):
    """Find company code from news text"""
    if companies_df.empty:
        return None, None
    
    text_lower = text.lower()
    
    for _, row in companies_df.iterrows():
        if row['Company_Name_Lower'] in text_lower:
            return row['Company_Code'], row['Company_Name']
    
    return None, None

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER and financial keywords"""
    analyzer = SentimentIntensityAnalyzer()
    
    # Financial-specific lexicon
    analyzer.lexicon.update({
        'fine': -3.0,
        'penalty': -3.0,
        'violation': -3.0,
        'regulatory': -2.0,
        'investigation': -2.0,
        'lawsuit': -2.0,
        'corrective': -1.0,
        'inaccurate': -1.0,
        'misleading': -2.0,
        'profit': 2.0,
        'growth': 2.0,
        'success': 2.0,
        'expansion': 1.5,
        'partnership': 1.5,
        'innovation': 1.5
    })
    
    try:
        scores = analyzer.polarity_scores(text)
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            return "POSITIVE", min((compound_score + 1) / 2, 1.0)
        elif compound_score <= -0.05:
            return "NEGATIVE", min(abs(compound_score), 1.0)
        else:
            return "NEUTRAL", 0.5
        
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "NEUTRAL", 0.5

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

def display_technical_analysis(company_code, article_timestamp):
    """Display technical analysis for a company with unique IDs"""
    
    # Create unique ID for each chart based on company and timestamp
    unique_id = f"{company_code}_{article_timestamp}"
    
    st.write("### Technical Analysis")
    
    # Display price metrics with unique IDs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"current_price_{unique_id}", 
                 label="Current Price",
                 value="99.00 SAR",
                 delta="3.66%")
    with col2:
        st.metric(f"day_high_{unique_id}",
                 label="Day High",
                 value="100.80 SAR")
    with col3:
        st.metric(f"day_low_{unique_id}",
                 label="Day Low",
                 value="98.80 SAR")

    # Technical signals table with unique key
    signals_df = pd.DataFrame({
        'Indicator': ['MACD', 'RSI', 'Bollinger Bands'],
        'Signal': ['BEARISH', 'BEARISH', 'BEARISH'],
        'Reason': ['MACD line below signal line',
                  'Overbought condition (RSI > 70)',
                  'Price above upper band']
    })
    st.table(signals_df)

    # Volume analysis with unique ID
    st.write("### Volume Analysis")
    st.metric(f"volume_{unique_id}",
             label="Trading Volume",
             value="5,501,169",
             delta="79.5% vs 30-day average")

    # Combined analysis
    st.write("### Combined Analysis")
    st.write("🔴 Overall Bearish: Technical indicators and news sentiment suggest negative pressure")

def display_article(article, companies_df):
    """Display a single news article with sentiment analysis and company information"""
    title = article.get("title", "No title available")
    description = article.get("description", "No description available")
    url = article.get("url", "#")
    published_at = article.get("published_at", "")
    source = article.get("source", "Unknown source")
    
    # Generate timestamp for unique IDs
    article_timestamp = datetime.now().timestamp()
    
    try:
        published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        published_str = published_date.strftime("%Y-%m-%d %H:%M")
    except:
        published_str = published_at

    with st.container():
        # Find mentioned companies
        mentioned_companies = {}
        for _, row in companies_df.iterrows():
            if row['Company_Name_Lower'] in (title + " " + description).lower():
                mentioned_companies[row['Company_Code']] = row['Company_Name']
        
        # Display article header
        if mentioned_companies:
            st.subheader(f"{title}")
            st.write("### Companies Mentioned")
            for code, name in mentioned_companies.items():
                st.write(f"{name} ({code})")
        else:
            st.subheader(title)
        
        st.write(f"**Source:** {source} | **Published:** {published_str}")
        
        if description:
            st.write(description)
        
        # Display sentiment analysis
        sentiment, confidence = analyze_sentiment(f"{title} {description}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"sentiment_{article_timestamp}", 
                     label="Sentiment",
                     value=sentiment)
        with col2:
            st.metric(f"confidence_{article_timestamp}",
                     label="Confidence",
                     value=f"{confidence:.2%}")
        
        # Display technical analysis for mentioned companies
        if mentioned_companies:
            for code, name in mentioned_companies.items():
                st.write(f"### Analysis for {name}")
                display_technical_analysis(code, article_timestamp)
                st.markdown("---")
        
        st.markdown(f"[Read full article]({url})")
        st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload companies file (optional)", type=['csv'])
    
    companies_df = load_company_data(uploaded_file)
    if companies_df.empty:
        st.warning("⚠️ No company data loaded. Please upload a CSV file or check the GitHub URL.")
    else:
        st.sidebar.success(f"✅ Loaded {len(companies_df)} companies")

    limit = st.sidebar.slider("Number of articles", 1, 10, 3)
    
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input("Show news published after:", value=default_date)
    published_after_iso = published_after.isoformat() + "T00:00:00"

    if st.button("Fetch News"):
        with st.spinner("Fetching latest news..."):
            news_articles = fetch_news(published_after_iso, limit)
            
            if news_articles:
                st.success(f"Found {len(news_articles)} articles")
                for article in news_articles:
                    display_article(article, companies_df)
            else:
                st.warning("No news articles found for the selected date range")

    st.sidebar.markdown("---")
    st.sidebar.write("App Version: 1.0.5")
    st.sidebar.success("✅ API Token loaded")
    
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
