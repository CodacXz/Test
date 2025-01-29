import requests
import streamlit as st
from datetime import datetime, timedelta
from textblob import TextBlob

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]

st.title("Saudi Stock Market News")
st.write("Welcome to the Saudi Stock Market News Analyzer!")

# Debug section to verify API token
st.sidebar.write("Debug Information:")
if API_TOKEN:
    st.sidebar.success("API Token loaded successfully")
else:
    st.sidebar.error("API Token not found")

# Date selector
default_date = datetime.now() - timedelta(days=7)
published_after = st.date_input("Show news published after:", value=default_date)
published_after_iso = published_after.isoformat() + "T00:00:00"

# Test API connection
if st.button("Test API Connection"):
    try:
        params = {
            "countries": "sa",
            "filter_entities": "true",
            "limit": 1,
            "published_after": published_after_iso,
            "api_token": API_TOKEN
        }
        
        with st.spinner("Testing API connection..."):
            response = requests.get(NEWS_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                st.success("API connection successful!")
                st.json(response.json())
            else:
                st.error(f"API Error: Status code {response.status_code}")
                st.write(response.text)
    except Exception as e:
        st.error(f"Connection Error: {e}")

# Display basic instructions
st.markdown("""
### How to use:
1. Select a date to show news from
2. Click 'Test API Connection' to verify the connection
3. The app will display news articles with sentiment analysis
""")

# Show app version
st.sidebar.markdown("---")
st.sidebar.write("App Version: 1.0.0")
