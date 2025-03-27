from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("./movie_dataset.csv")

# Ensure necessary columns exist (modify as needed)
if "title" not in movies.columns or "genres" not in movies.columns:
    raise ValueError("Dataset must have 'title' and 'genres' columns.")

# Process genres into a single string for each movie
movies["genres"] = movies["genres"].fillna("").astype(str)

# TF-IDF Vectorizer to convert genres into feature vectors
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genres"]].to_dict(orient="records")

# Flask API route
@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title")
    recommendations = get_recommendations(title)
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
