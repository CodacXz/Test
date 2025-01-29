import requests
import streamlit as st
from datetime import datetime, timedelta
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]

@st.cache_resource
def load_sentiment_model():
    """
    Loads and caches the sentiment analysis model
    """
    try:
        # Specify a specific model for sentiment analysis
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_saudi_stock_news(published_after, page=1):
    """
    Fetches Saudi stock market news from the StockData API.
    """
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": 2,
        "page": page,
        "published_after": published_after,
        "api_token": API_TOKEN
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "data" not in data:
            st.error("Unexpected API response format.")
            return []
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch news: {e}")
        return []

def analyze_sentiment(text, sentiment_analyzer):
    """
    Analyze sentiment using the provided sentiment analyzer.
    """
    if sentiment_analyzer is None:
        return "N/A", 0.0
    
    try:
        result = sentiment_analyzer(text)[0]
        return result["label"], result["score"]
    except Exception as e:
        st.warning(f"Sentiment analysis failed: {e}")
        return "ERROR", 0.0

def display_news_articles(news_articles, sentiment_analyzer):
    """
    Displays news articles with titles, summaries, links, and sentiment analysis.
    """
    if not news_articles:
        st.warning("No news articles found.")
        return
        
    st.success(f"Found {len(news_articles)} articles.")
    for article in news_articles:
        title = article["title"]
        summary = article.get("summary") or article.get("description") or article.get("snippet") or "No summary available."
        url = article["url"]
        
        st.subheader(title)
        if summary and summary != "No summary available.":
            st.write(f"**Summary:** {summary}")
            # Perform sentiment analysis on the summary
            sentiment, confidence = analyze_sentiment(summary, sentiment_analyzer)
            if sentiment != "ERROR":
                st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
        st.write(f"[Read More]({url})")
        st.write("---")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Saudi Stock Market News")
    
    # Load sentiment analyzer
    sentiment_analyzer = load_sentiment_model()
    if sentiment_analyzer is None:
        st.error("Failed to load sentiment analysis model. Continuing without sentiment analysis.")
    
    # Allow user to specify a date range
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input("Show news published after:", value=default_date)
    published_after_iso = published_after.isoformat() + "T00:00:00"
    
    # Initialize session state for pagination
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "news_articles" not in st.session_state:
        st.session_state.news_articles = []
    
    col1, col2 = st.columns(2)
    
    # Fetch news on button click
    if col1.button("Fetch Saudi News"):
        with st.spinner("Fetching news..."):
            st.session_state.page = 1
            st.session_state.news_articles = fetch_saudi_stock_news(published_after_iso, st.session_state.page)
        display_news_articles(st.session_state.news_articles, sentiment_analyzer)
    
    # Load more news on button click
    if col2.button("Load More"):
        with st.spinner("Loading more news..."):
            st.session_state.page += 1
            new_articles = fetch_saudi_stock_news(published_after_iso, st.session_state.page)
            st.session_state.news_articles.extend(new_articles)
        display_news_articles(st.session_state.news_articles, sentiment_analyzer)

if __name__ == "__main__":
    main()
