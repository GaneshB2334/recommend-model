import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_language_names

# Load dataset
movies = pd.read_csv("./movie_dataset.csv")

# Ensure necessary columns exist
if "title" not in movies.columns or "genres" not in movies.columns:
    raise ValueError("Dataset must have 'title' and 'genres' columns.")

# Preprocess data
movies["title"] = movies["title"].str.lower()
movies["genres"] = movies["genres"].fillna("").astype(str).str.lower()

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def get_movie_by_title(title):
    title = title.lower()
    if title not in indices:
        return None
    idx = indices[title]
    movie = movies.iloc[idx][["id", "title", "genres", "overview", "cast", "director", "runtime", "vote_average", "spoken_languages"]].copy()
    movie["spoken_languages"] = extract_language_names(movie["spoken_languages"])
    movie["genres"] = movie["genres"].split(" ")
    movie = movie.replace({np.nan: None})

    return movie.to_dict()

def get_recommendations(title):
    title = title.lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][["id", "title", "genres", "overview", "cast", "director", "runtime", "vote_average", "spoken_languages"]].copy()
    recommendations["spoken_languages"] = recommendations["spoken_languages"].apply(extract_language_names)
    recommendations["genres"] = recommendations["genres"].str.split(" ")

    return recommendations.to_dict(orient="records")

def get_recommendations_by_genre(genre):
    genre = genre.lower()
    genre_movies = movies[movies["genres"].str.contains(genre, case=False, na=False)]
    if genre_movies.empty:
        return []

    genre_indices = genre_movies.index.tolist()
    sim_scores = sorted([(i, cosine_sim[i][i]) for i in genre_indices], key=lambda x: x[1], reverse=True)[:10]
    movie_indices = list(set([i[0] for i in sim_scores]))

    recommendations = movies.iloc[movie_indices][["id", "title", "genres", "overview", "cast", "director", "runtime", "vote_average", "spoken_languages"]].copy()
    recommendations["spoken_languages"] = recommendations["spoken_languages"].apply(extract_language_names)
    recommendations["genres"] = recommendations["genres"].str.split(" ")

    return recommendations.to_dict(orient="records")

def get_recommendation_by_id(movie_id):
    try:
        idx = movies[movies["id"] == movie_id].index[0]
    except IndexError:
        return [], None

    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][["id", "title", "genres", "overview", "cast", "director", "runtime", "vote_average", "spoken_languages"]].copy()
    recommendations["spoken_languages"] = recommendations["spoken_languages"].apply(extract_language_names)
    recommendations["genres"] = recommendations["genres"].str.split(" ")
    recommendations = recommendations.replace({np.nan: None})
    recommendations = recommendations[recommendations["id"] != movie_id]

    movie = movies.iloc[idx][["id", "title", "genres", "overview", "cast", "director", "runtime", "vote_average", "spoken_languages"]].copy()
    movie["spoken_languages"] = extract_language_names(movie["spoken_languages"])
    movie["genres"] = movie["genres"].split(" ")
    movie = movie.replace({np.nan: None})

    return recommendations.to_dict(orient="records"), movie.to_dict()

def get_top_movies():
    top_movies = movies.sort_values(by="vote_average", ascending=False).head(20)
    top_movies = top_movies[["id", "title", "genres", "overview", "cast", "director", "runtime", "vote_average", "spoken_languages"]].copy()
    top_movies["spoken_languages"] = top_movies["spoken_languages"].apply(extract_language_names)
    top_movies["genres"] = top_movies["genres"].str.split(" ")
    top_movies = top_movies.replace({np.nan: None})
    return top_movies.to_dict(orient="records")
