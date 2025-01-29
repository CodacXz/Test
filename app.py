import requests
import streamlit as st
from datetime import datetime, timedelta
from transformers import pipeline

# Use Streamlit secrets for API token
NEWS_API_URL = "https://api.marketaux.com/v1/news/all?countries=sa&filter_entities=true&limit=10&published_after=2025-01-28T14:44&api_token=YOUR_API_TOKEN"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]  # Ensure you set this in Streamlit secrets

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_saudi_stock_news(published_after, page=1):
    """
    Fetches Saudi stock market news from the StockData API.
    """
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": 2,  # Match your plan's limit
        "page": page,  # Add pagination
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

def analyze_sentiment(text):
    """
    Analyze sentiment using Hugging Face Transformers.
    Returns a sentiment label (POSITIVE, NEGATIVE) and confidence score.
    """
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]

def display_news_articles(news_articles):
    """
    Displays news articles with titles, summaries, links, and sentiment analysis.
    """
    if news_articles:
        st.success(f"Found {len(news_articles)} articles.")
        for article in news_articles:
            title = article["title"]
            summary = article.get("summary") or article.get("description") or article.get("snippet") or "No summary available."
            url = article["url"]
            
            st.subheader(title)
            if summary and summary != "No summary available.":
                st.write(f"**Summary:** {summary}")
                # Perform sentiment analysis on the summary
                sentiment, confidence = analyze_sentiment(summary)
                st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2f})")
            st.write(f"[Read More]({url})")
            st.write("---")
    else:
        st.warning("No news articles found.")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Saudi Stock Market News")
    
    # Allow user to specify a date range
    default_date = datetime.now() - timedelta(days=7)
    published_after = st.date_input("Show news published after:", value=default_date)
    published_after_iso = published_after.isoformat() + "T00:00:00"
    
    # Initialize session state for pagination
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "news_articles" not in st.session_state:
        st.session_state.news_articles = []
    
    # Fetch news on button click
    if st.button("Fetch Saudi News"):
        with st.spinner("Fetching news..."):
            st.session_state.page = 1
            st.session_state.news_articles = fetch_saudi_stock_news(published_after_iso, st.session_state.page)
        display_news_articles(st.session_state.news_articles)
    
    # Load more news on button click
    if st.button("Load More"):
        with st.spinner("Loading more news..."):
            st.session_state.page += 1
            new_articles = fetch_saudi_stock_news(published_after_iso, st.session_state.page)
            st.session_state.news_articles.extend(new_articles)
        display_news_articles(st.session_state.news_articles)

if __name__ == "__main__":
    main()
