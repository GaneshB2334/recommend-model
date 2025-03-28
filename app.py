from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from recommendations import get_recommendations, get_recommendations_by_genre, get_top_movies, get_recommendation_by_id, get_movie_by_title

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080/","https://cine-match-ochre.vercel.app/"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    title = data.get("title")
    if not title:
        return jsonify({"error": "Missing 'title' parameter"}), 400
    return jsonify(get_movie_by_title(title))

@app.route("/recommend_by_title", methods=["POST"])
def recommend_post():
    data = request.json
    title = data.get("title")
    if not title:
        return jsonify({"error": "Missing 'title' parameter"}), 400
    return jsonify(get_recommendations(title))

@app.route("/recommend_by_genre", methods=["POST"])
def recommend_by_genre():
    data = request.json
    genre = data.get("genre")
    if not genre:
        return jsonify({"error": "Missing 'genre' parameter"}), 400
    return jsonify(get_recommendations_by_genre(genre))

@app.route("/top_movies", methods=["GET"])
def top_movies():
    return jsonify(get_top_movies())

@app.route("/recommend/<int:id>", methods=["GET"])
def recommend_by_id(id):
    recommendations, movie = get_recommendation_by_id(id)
    if not movie:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify({"recommendations": recommendations, "movie": movie})


if __name__ == "__main__":
    app.run(debug=True)
