import requests
import streamlit as st
from datetime import datetime, timedelta
from textblob import TextBlob

# API Configuration
NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
API_TOKEN = st.secrets["STOCKDATA_API_TOKEN"]

# Download TextBlob corpora on first run
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@st.cache_resource
def load_sentiment_analyzer():
    """
    Loads TextBlob for sentiment analysis
    """
    return TextBlob

# Rest of your code remains the same...
