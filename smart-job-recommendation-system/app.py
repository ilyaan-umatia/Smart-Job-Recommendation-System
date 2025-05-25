import streamlit as st
import pandas as pd
import joblib
import requests
import os
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Loading pre-trained model and data
@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    knn = joblib.load("knn_model.pkl")
    df = pd.read_csv("jobs_cleaned.csv")
    return vectorizer, knn, df

# Fetch jobs from JSearch API
def fetch_live_jobs(query, limit=5):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
         'x-rapidapi-key': os.environ.get("JSEARCH_API_KEY"),  # Replace with your actual API key
         'x-rapidapi-host': "jsearch.p.rapidapi.com"
    }
    params = {"query": query, "page": "1", "num_pages": "1"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])[:limit]
    else:
        return []


st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 AI-Based Job Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get job recommendations from both trained AI and live data sources.</p>", unsafe_allow_html=True)

query = st.text_input(placeholder="🔍 Enter your skills, keywords, or preferred job title", label="Job Query")

if query:
    with st.spinner("Finding best job matches for you..."):
        vectorizer, knn, df = load_model()
        user_vec = vectorizer.transform([query])
        distances, indices = knn.kneighbors(user_vec)

        st.subheader("📦 Recommended Jobs from Offline AI Model")
        for i in indices[0]:
            row = df.iloc[i]
            st.markdown(f"### {row['title']}")
            st.markdown(f"📍 Location:** {row.get('location', 'N/A')}")
            st.markdown(f"💼 Description:** {row['description'][:500]}...")
            st.markdown("---")

        st.subheader("🌐 Live Jobs from the Internet (JSearch API)")
        live_jobs = fetch_live_jobs(query)
        if live_jobs:
            for job in live_jobs:
                st.markdown(f"### 🔗 [{job['job_title']}]({job['job_apply_link']})")
                st.markdown(f"*Company:* {job.get('employer_name', 'N/A')}")
                st.markdown(f"*Location:* {job.get('job_city', '')}, {job.get('job_country', '')}")
                date = job.get('job_posted_at_datetime_utc', '')
                if date:
                    try:
                        posted = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%b %d, %Y")
                        st.markdown(f"*Posted:* {posted}")
                    except:
                        pass
                st.markdown(f"*Description:* {job.get('job_description', '')[:500]}...")
                st.markdown("---")
        else:
            st.info("No live jobs found for this query.")