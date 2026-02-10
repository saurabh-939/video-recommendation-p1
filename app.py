from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pickle
import re

app = FastAPI()

# -----------------------------
# Load artifacts
# -----------------------------
try:
    interaction_matrix = pickle.load(open("interaction_matrix.pkl", "rb"))
    user_similarity = pickle.load(open("user_similarity.pkl", "rb"))
    user_encoder, video_encoder = pickle.load(open("encoders.pkl", "rb"))

    print("Artifacts loaded successfully")
    print("Sample users:", user_encoder.classes_[:5])

except Exception as e:
    interaction_matrix = None
    user_similarity = None
    user_encoder = None
    video_encoder = None
    print("Error loading artifacts:", e)


# -----------------------------
# Validate User ID (U0001 format)
# -----------------------------
def is_valid_user(user_id):
    return re.match(r"^U\d{4}$", user_id)


# -----------------------------
# Recommendation Logic
# -----------------------------
def recommend_videos(user_id, top_k=5):

    if interaction_matrix is None:
        return ["Model not loaded"]

    if not is_valid_user(user_id):
        return ["Invalid User ID format. Use U0001"]

    try:
        numeric_id = int(user_id[1:])   # U0003 → 3

        # Try both formats
        if numeric_id in user_encoder.classes_:
            encoded_user = numeric_id
        elif user_id in user_encoder.classes_:
            encoded_user = user_id
        else:
            return ["User not found"]

        user_idx = user_encoder.transform([encoded_user])[0]

        similarity_scores = list(enumerate(user_similarity[user_idx]))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )[1:6]

        similar_users = [i[0] for i in similarity_scores]

        video_scores = interaction_matrix.iloc[similar_users].mean(axis=0)

        watched_videos = interaction_matrix.iloc[user_idx]
        unwatched_videos = video_scores[watched_videos == 0]

        if len(unwatched_videos) == 0:
            unwatched_videos = video_scores

        recommendations = (
            unwatched_videos.sort_values(ascending=False).head(top_k)
        )

        videos = video_encoder.inverse_transform(
            recommendations.index
        ).tolist()

        return videos

    except Exception as e:
        return [str(e)]


# -----------------------------
# UI Page
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Video Recommendation System</title>
    </head>
    <body style="font-family: Arial; text-align:center; margin-top:50px;">
        <h2>Video Recommendation System</h2>

        <input id="userId" placeholder="Enter User ID (ex: U0003)" />
        <button onclick="getRecommendations()">Recommend</button>

        <h3 id="result"></h3>

        <script>
        async function getRecommendations() {
            const userId = document.getElementById("userId").value;
            const response = await fetch(`/recommend/${userId}`);
            const data = await response.json();

            document.getElementById("result").innerText =
                "User: " + userId + " | Recommendations: " +
                (data.recommendations || data.error);
        }
        </script>
    </body>
    </html>
    """


# -----------------------------
# API Endpoint
# -----------------------------
@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_k: int = 5):
    recs = recommend_videos(user_id, top_k)
    return {"user": user_id, "recommendations": recs}
