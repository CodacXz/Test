import requests
import streamlit as st
from datetime import datetime, timedelta
from textblob import TextBlob

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob and financial keywords"""
    # Financial negative keywords
    negative_keywords = ['fine', 'penalty', 'violation', 'failed', 'failure', 'loss', 'decline', 
                        'debt', 'investigation', 'lawsuit', 'regulatory action', 'corrective', 
                        'inaccurate', 'misleading']
    
    # Financial positive keywords
    positive_keywords = ['profit', 'growth', 'increase', 'success', 'expansion', 'dividend', 
                        'earnings', 'upgrade', 'innovation', 'partnership']
    
    # Count keyword occurrences
    text_lower = text.lower()
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    
    try:
        # Get TextBlob sentiment
        analysis = TextBlob(text)
        base_polarity = analysis.sentiment.polarity
        
        # Adjust polarity based on financial keywords
        keyword_adjustment = (positive_count - negative_count) * 0.2
        final_polarity = base_polarity + keyword_adjustment
        
        # Convert adjusted polarity to label and score
        if final_polarity > 0:
            confidence = (final_polarity + 1) / 2
            return "POSITIVE", min(confidence, 1.0)
        elif final_polarity < 0:
            return "NEGATIVE", min(abs(final_polarity), 1.0)
        return "NEUTRAL", 0.5
        
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "NEUTRAL", 0.5

def fetch_news(published_after, limit=10):
    """Fetch news articles from the API"""
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": limit,
        "published_after": published_after,
        "api_token": API_TOKEN
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def display_article(article):
    """Display a single news article with sentiment analysis"""
    title = article.get("title", "No title available")
    description = article.get("description", "No description available")
    url = article.get("url", "#")
    published_at = article.get("published_at", "")
    source = article.get("source", "Unknown source")
    
    # Format published date
    try:
        published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S.%fZ")
        published_str = published_date.strftime("%Y-%m-%d %H:%M")
    except:
        published_str = published_at

    # Article container
    with st.container():
        st.subheader(title)
        st.write(f"**Source:** {source} | **Published:** {published_str}")
        
        # Analyze both title and description
        combined_text = f"{title} {description}"
        sentiment, score = analyze_sentiment(combined_text)
        
        # Display description and sentiment
        if description:
            st.write(description)
        
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", sentiment)
        col2.metric("Confidence", f"{score:.2%}")
        
        # Link to full article
        st.markdown(f"[Read full article]({url})")
        st.markdown("---")

def main():
    st.title("Saudi Stock Market News")
    st.write("Real-time news analysis for Saudi stock market")

    # Sidebar configuration
    st.sidebar.title("Settings")
    limit = st.sidebar.slider("Number of articles", 1, 20, 10)
    
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
                    display_article(article)
            else:
                st.warning("No news articles found for the selected date range")

    # App information
    st.sidebar.markdown("---")
    st.sidebar.write("App Version: 1.0.1")
    if API_TOKEN:
        st.sidebar.success("API Token loaded successfully")
    else:
        st.sidebar.error("API Token not found")

if __name__ == "__main__":
    main()
