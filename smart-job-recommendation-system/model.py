import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# This will load and preprocess data
df = pd.read_csv("Jobs_posting.csv")
df.dropna(subset=['title', 'description'], inplace=True)
df['text'] = df['title'] + " " + df['description']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['text'])

# KNN Model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(tfidf_matrix)

# It will save model and vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(knn, 'knn_model.pkl')
df.to_csv('jobs_cleaned.csv', index=False)

print("Model and vectorizer saved successfully.")