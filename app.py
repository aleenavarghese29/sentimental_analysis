# app.py
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import udf
from textblob import TextBlob
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 1. Streamlit UI
st.set_page_config(layout="centered")
st.title("📰 Real-Time News Sentiment Analysis with PySpark + TextBlob")

# 2. User Inputs
query = st.text_input("Enter topic to search news about:", "India")
api_key = st.text_input("Enter your GNews API Key:", type="password")

if st.button("Fetch & Analyze"):
    if not api_key:
        st.error("❌ API key is required.")
        st.stop()

    with st.spinner("🔎 Fetching news articles..."):
        url = f'https://gnews.io/api/v4/search?q={query}&lang=en&apikey={api_key}'
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            st.success(f"✅ Retrieved {len(articles)} articles.")
        else:
            st.error(f"❌ Failed to fetch news: {response.status_code}")
            st.stop()

    # 3. Create Spark Session
    spark = SparkSession.builder.appName("NewsSentimentStreamlit").getOrCreate()

    # 4. Convert API response to Spark DataFrame
    schema = StructType([
        StructField("title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("publishedAt", StringType(), True),
        StructField("url", StringType(), True)
    ])

    news_data = [
        (a.get("title"), a.get("description"), a.get("publishedAt"), a.get("url"))
        for a in articles
    ]

    df = spark.createDataFrame(news_data, schema)

    # 5. Sentiment Analysis UDF
    def get_sentiment(text):
        if text:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0:
                return "Positive"
            elif polarity < 0:
                return "Negative"
        return "Neutral"

    sentiment_udf = udf(get_sentiment, StringType())
    df_sentiment = df.withColumn("sentiment", sentiment_udf(df["title"]))

    # 6. Display results
    pd_df = df_sentiment.select("title", "sentiment").toPandas()
    st.subheader("🧾 News Headlines with Sentiment")
    st.dataframe(pd_df, use_container_width=True)

    # 7. Visualization
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = pd_df["sentiment"].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", color=["green", "red", "gray"], ax=ax)
    ax.set_ylabel("Number of Articles")
    ax.set_xlabel("Sentiment")
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    # 8. Optional download
    csv = pd_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="news_sentiments.csv", mime="text/csv")

    # Stop Spark
    spark.stop()
