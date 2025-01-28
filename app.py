import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up the API keys
STOCKDATA_API_TOKEN = "bS2jganHVlFYtAly7ttdHYLrTB0s6BmONWmFEApD"

# Define the StockData.org API endpoint
NEWS_API_URL = "https://api.stockdata.org/v1/news"

def fetch_saudi_stock_news(api_token, symbols="TADAWUL"):
    """
    Fetch news articles related to the Saudi stock market from StockData.org.
    """
    params = {
        "symbols": symbols,
        "api_token": api_token
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        st.error("Failed to fetch news articles. Please check your API token or network connection.")
        return []

def analyze_sentiment(news_title):
    """
    Perform sentiment analysis on the news title using VaderSentiment.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(news_title)
    compound_score = sentiment_score["compound"]

    if compound_score >= 0.05:
        return "Positive", compound_score
    elif compound_score <= -0.05:
        return "Negative", compound_score
    else:
        return "Neutral", compound_score

def main():
    st.title("Saudi Stock Market News and Sentiment Analysis")

    # Fetch news from the API
    st.sidebar.header("Settings")
    st.sidebar.write("You can adjust the API settings or fetch news for a specific symbol.")
    symbol = st.sidebar.text_input("Enter Stock Symbol", "TADAWUL").strip()
    
    if st.button("Fetch News"):
        st.info("Fetching news articles...")
        news_articles = fetch_saudi_stock_news(STOCKDATA_API_TOKEN, symbols=symbol)

        if news_articles:
            st.success(f"Found {len(news_articles)} articles related to {symbol}")
            for article in news_articles:
                title = article["title"]
                summary = article.get("summary", "No summary available.")
                url = article["url"]

                sentiment_label, sentiment_score = analyze_sentiment(title)
                
                st.subheader(title)
                st.write(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")
                st.write(summary)
                st.write(f"[Read More]({url})")
                st.write("---")
        else:
            st.warning(f"No news articles found for the symbol {symbol}.")

if __name__ == "__main__":
    main()
