def display_article(article, companies_df):
    article_id = str(uuid.uuid4())
    title = article.get("title", "No title")
    description = article.get("description", "No description")
    url = article.get("url", "#")
    source = article.get("source", "Unknown")
    published_at = article.get("published_at", "")
    
    st.markdown(f"## {title}", key=f"title_{article_id}")
    st.write(f"Source: {source} | Published: {published_at[:16]}", key=f"source_{article_id}")
    st.write(description[:200] + "..." if len(description) > 200 else description, key=f"desc_{article_id}")
    
    sentiment, confidence = analyze_sentiment(title + " " + description)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Analysis", key=f"sentiment_header_{article_id}")
        st.write(f"**Sentiment:** {sentiment}", key=f"sentiment_{article_id}")
        st.write(f"**Confidence:** {confidence:.2f}%", key=f"confidence_{article_id}")
    
    mentioned_companies = find_companies_in_text(title + " " + description, companies_df)
    
    if mentioned_companies:
        st.markdown("### Companies Mentioned", key=f"companies_header_{article_id}")
        for company in mentioned_companies:
            st.markdown(f"- {company['name']} ({company['symbol']})", key=f"company_{company['symbol']}_{article_id}")
        
        st.markdown("### Stock Analysis", key=f"stock_header_{article_id}")
        tabs = st.tabs([company['name'] for company in mentioned_companies])
        
        for i, (tab, company) in enumerate(zip(tabs, mentioned_companies)):
            with tab:
                df, error = get_stock_data(company['symbol'])
                if error:
                    st.error(f"Error fetching stock data: {error}", key=f"error_{company['symbol']}_{article_id}")
                    continue
                
                if df is not None and not df.empty:
                    latest_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    price_change = ((latest_price - prev_price)/prev_price*100)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"{latest_price:.2f} SAR", 
                                f"{price_change:.2f}%", key=f"price_{company['symbol']}_{article_id}")
                    with col2:
                        st.metric("Day High", f"{df['High'].iloc[-1]:.2f} SAR", key=f"high_{company['symbol']}_{article_id}")
                    with col3:
                        st.metric("Day Low", f"{df['Low'].iloc[-1]:.2f} SAR", key=f"low_{company['symbol']}_{article_id}")
                    
                    fig = plot_stock_analysis(df, company['name'], company['symbol'])
                    st.plotly_chart(fig, key=f"plot_{company['symbol']}_{article_id}_{i}", use_container_width=True)
                    
                    st.markdown("### Technical Analysis Signals", key=f"tech_header_{company['symbol']}_{article_id}")
                    signals = analyze_technical_indicators(df)
                    
                    signal_df = pd.DataFrame(signals, columns=['Indicator', 'Signal', 'Reason'])
                    st.table(signal_df)
                    
                    st.markdown("### Combined Analysis", key=f"combined_header_{company['symbol']}_{article_id}")
                    tech_sentiment = sum(1 if signal[1] == "BULLISH" else -1 if signal[1] == "BEARISH" else 0 for signal in signals)
                    news_sentiment_score = 1 if sentiment == "POSITIVE" else -1 if sentiment == "NEGATIVE" else 0
                    
                    combined_score = (tech_sentiment + news_sentiment_score) / (len(signals) + 1)
                    
                    if combined_score > 0.3:
                        st.success("🟢 Overall Bullish: Technical indicators and news sentiment suggest positive momentum", key=f"bullish_{company['symbol']}_{article_id}")
                    elif combined_score < -0.3:
                        st.error("🔴 Overall Bearish: Technical indicators and news sentiment suggest negative pressure", key=f"bearish_{company['symbol']}_{article_id}")
                    else:
                        st.warning("🟡 Neutral: Mixed signals from technical indicators and news sentiment", key=f"neutral_{company['symbol']}_{article_id}")
                    
                    avg_volume = df['Volume'].mean()
                    latest_volume = df['Volume'].iloc[-1]
                    volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                    
                    st.markdown("### Volume Analysis", key=f"volume_header_{company['symbol']}_{article_id}")
                    st.metric("Trading Volume", f"{int(latest_volume):,}", 
                             f"{volume_change:.1f}% vs 30-day average", key=f"volume_{company['symbol']}_{article_id}")
                else:
                    st.error(f"No data available for {company['name']}", key=f"no_data_{company['symbol']}_{article_id}")
    
    st.markdown(f"[Read full article]({url})", key=f"link_{article_id}")
    st.markdown("---", key=f"divider_{article_id}")
