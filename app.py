import requests
import streamlit as st
from datetime import datetime, timedelta
import os

# Use environment variables or Streamlit secrets for API token
NEWS_API_URL = "https://api.stockdata.org/v1/news/all"
API_TOKEN = os.getenv("STOCKDATA_API_TOKEN", "bS2jganHVlFYtAly7ttdHYLrTB0s6BmONWmFEApD")

def fetch_saudi_stock_news(published_after, page=1):
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
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch news: {e}")
        return []

def display_news_articles(news_articles):
    if news_articles:
        st.success(f"Found {len(news_articles)} articles.")
        for article in news_articles:
            title = article["title"]
            summary = article.get("summary", "No summary available.")
            url = article["url"]
            st.subheader(title)
            st.write(f"**Summary:** {summary}")
            st.write(f"[Read More]({url})")
            st.write("---")
    else:
        st.warning("No news articles found.")

def main():
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
    
    if st.button("Fetch Saudi News"):
        st.session_state.page = 1
        st.session_state.news_articles = fetch_saudi_stock_news(published_after_iso, st.session_state.page)
        display_news_articles(st.session_state.news_articles)
    
    if st.button("Load More"):
        st.session_state.page += 1
        new_articles = fetch_saudi_stock_news(published_after_iso, st.session_state.page)
        st.session_state.news_articles.extend(new_articles)
        display_news_articles(st.session_state.news_articles)

if __name__ == "__main__":
    main()
