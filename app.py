from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# ---- CORS FIX ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model artifacts ----
interaction_matrix = pickle.load(open("interaction_matrix.pkl", "rb"))
user_similarity = pickle.load(open("user_similarity.pkl", "rb"))
user_encoder, video_encoder = pickle.load(open("encoders.pkl", "rb"))


def recommend_videos(user_id, top_k=5):
    user_id = user_id.upper()  # prevents case issues

    if user_id not in user_encoder.classes_:
        return []

    user_idx = user_encoder.transform([user_id])[0]

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

    recommendations = unwatched_videos.sort_values(ascending=False).head(top_k)

    return video_encoder.inverse_transform(recommendations.index).tolist()


@app.get("/")
def home():
    return {"message": "Recommendation API Running"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_k: int = 5):
    return {
        "user": user_id,
        "recommendations": recommend_videos(user_id, top_k)
    }
