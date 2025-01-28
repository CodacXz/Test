import requests
import streamlit as st
from datetime import datetime, timedelta
import os

# Use environment variables or Streamlit secrets for API token
NEWS_API_URL = "https://api.stockdata.org/v1/news/all"
API_TOKEN = os.getenv("STOCKDATA_API_TOKEN", "bS2jganHVlFYtAly7ttdHYLrTB0s6BmONWmFEApD")

def fetch_saudi_stock_news(published_after):
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": 10,
        "published_after": published_after,
        "api_token": API_TOKEN
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        news_articles = response.json().get("data", [])
        
        # Debug: Print the full API response
        st.write("API Response:", response.json())
        
        return news_articles
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
    
    if st.button("Fetch Saudi News"):
        with st.spinner("Fetching news articles..."):
            news_articles = fetch_saudi_stock_news(published_after_iso)
            display_news_articles(news_articles)

if __name__ == "__main__":
    main()
