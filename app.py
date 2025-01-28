import requests
import streamlit as st

NEWS_API_URL = "https://api.stockdata.org/v1/news/all"
API_TOKEN = "bS2jganHVlFYtAly7ttdHYLrTB0s6BmONWmFEApD"

def fetch_saudi_stock_news():
    params = {
        "countries": "sa",
        "filter_entities": "true",
        "limit": 10,
        "published_after": "2025-01-20T15:03",
        "api_token": API_TOKEN
    }
    
    response = requests.get(NEWS_API_URL, params=params)
    
    if response.status_code == 200:
        news_articles = response.json().get("data", [])
        return news_articles
    else:
        error_message = response.text
        st.error(f"Failed to fetch news: {error_message}")
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
    
    if st.button("Fetch Saudi News"):
        news_articles = fetch_saudi_stock_news()
        display_news_articles(news_articles)

if __name__ == "__main__":
    main()
