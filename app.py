import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Add a text input for stock symbol
symbol = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA):", "AAPL").upper()

# Fetch Stock Data using Alpha Vantage (optional)
ALPHA_VANTAGE_API_KEY = ["2YU3MJIM5Y36NIAM"]
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

alpha_params = {
    "function": "GLOBAL_QUOTE",
    "symbol": symbol,
    "apikey": ALPHA_VANTAGE_API_KEY
}

alpha_response = requests.get(ALPHA_VANTAGE_URL, params=alpha_params)
alpha_data = alpha_response.json()

# Display stock data
if "Global Quote" in alpha_data:
    st.subheader(f"Stock Data for {symbol}")
    st.write(f"**Price:** {alpha_data['Global Quote']['05. price']}")
    st.write(f"**Change:** {alpha_data['Global Quote']['09. change']}")
    st.write(f"**Change Percent:** {alpha_data['Global Quote']['10. change percent']}")
else:
    st.error("Failed to fetch stock data. Please check the symbol and try again.")

# Fetch News using StockData API
STOCKDATA_API_TOKEN = ["bS2jganHVlFYtAly7ttdHYLrTB0s6BmONWmFEApDn"]
STOCKDATA_URL = "https://api.stockdata.org/v1/news/all"

params = {
    "countries": "us",  # Filter by country (e.g., "us" for United States)
    "filter_entities": "true",  # Filter by entities (e.g., stocks)
    "limit": 10,  # Number of articles to fetch
    "published_after": "2023-01-01T00:00",  # Fetch news after this date
    "api_token": STOCKDATA_API_TOKEN
}

# Add symbol to the query if provided
if symbol:
    params["search"] = symbol  # Search for news related to the stock symbol

response = requests.get(STOCKDATA_URL, params=params)
data = response.json()

# Check if news data is available
if "data" in data:
    news_data = []
    for article in data["data"]:
        title = article["title"]
        sentiment = analyzer.polarity_scores(title)
        news_data.append({
            "Title": title,
            "Description": article["description"],
            "URL": article["url"],
            "Positive": sentiment['pos'],
            "Neutral": sentiment['neu'],
            "Negative": sentiment['neg'],
            "Compound": sentiment['compound']
        })

    df = pd.DataFrame(news_data)

    # Display Dashboard
    st.title("Stock Market News Aggregator")
    st.write(f"Sentiment analysis of recent news for {symbol}.")

    st.dataframe(df)

    st.subheader("Top Positive News")
    positive_news = df[df["Compound"] > 0.5]
    st.dataframe(positive_news)

    st.subheader("Top Negative News")
    negative_news = df[df["Compound"] < -0.5]
    st.dataframe(negative_news)
else:
    st.error("Failed to fetch news data. Please check the API token and try again.")