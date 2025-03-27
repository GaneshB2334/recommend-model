from flask import Flask, request, jsonify, render_template
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
    sim_scores = sim_scores[1:11] # Exclude the movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genres"]].to_dict(orient="records")

def get_recommendations_by_genre(genre):
    # Filter movies containing the genre
    genre_movies = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    
    if genre_movies.empty:
        return []

    # Compute similarity scores for the filtered movies
    genre_indices = genre_movies.index.tolist()
    sim_scores = []

    for idx in genre_indices:
        scores = list(enumerate(cosine_sim[idx]))
        sim_scores.extend(scores)

    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:10]  # Top 10 recommendations

    # Get unique movie indices
    movie_indices = list(set([i[0] for i in sim_scores]))
    
    return movies.iloc[movie_indices][["title", "genres"]].to_dict(orient="records")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_post():
    data = request.json
    title = data.get("title")
    genre = data.get("genre")
    if title:
        recommendations = get_recommendations(title)
    elif genre:
        recommendations = get_recommendations_by_genre(genre)
    else:
        return jsonify({"error": "Missing 'title' or 'genre' parameter"})

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
