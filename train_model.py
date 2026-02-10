import pandas as pd
import numpy as np
import pickle
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
df = pd.read_csv("video_recommendation_sample_dataset.csv")

# -----------------------------
# Validation
# -----------------------------
if df.empty:
    raise ValueError("Dataset is empty")

if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values")

print("Dataset loaded successfully")


# -----------------------------
# Normalize USER ID FORMAT
# Ensures U0001 format everywhere
# -----------------------------
def normalize_user_id(uid):
    uid = str(uid).strip()
    if not uid.startswith("U"):
        uid = "U" + uid
    number = int(re.sub(r"\D", "", uid))
    return f"U{number:04d}"

df["user_id"] = df["user_id"].apply(normalize_user_id)


# -----------------------------
# Feature Engineering
# -----------------------------
print("Creating interaction score...")

df["interaction_score"] = (
    0.4 * (df["watch_duration"] / 100) +
    0.2 * df["liked"] +
    0.2 * df["commented"] +
    0.2 * df["subscribed_after_watching"]
)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek


# -----------------------------
# Encoding
# -----------------------------
print("Encoding users and videos...")

user_encoder = LabelEncoder()
video_encoder = LabelEncoder()

df["user_encoded"] = user_encoder.fit_transform(df["user_id"])
df["video_encoded"] = video_encoder.fit_transform(df["video_id"])

print("Sample encoded users:", user_encoder.classes_[:5])


# -----------------------------
# Interaction Matrix
# -----------------------------
print("Creating interaction matrix...")

interaction_matrix = df.pivot_table(
    index="user_encoded",
    columns="video_encoded",
    values="interaction_score",
    fill_value=0
)


# -----------------------------
# Similarity Computation
# -----------------------------
print("Computing user similarity...")
user_similarity = cosine_similarity(interaction_matrix)


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_videos(user_id, top_n=5):

    if user_id not in user_encoder.classes_:
        return ["User not found"]

    user_idx = user_encoder.transform([user_id])[0]

    similarity_scores = user_similarity[user_idx]
    similar_users = similarity_scores.argsort()[::-1][1:6]

    recommended_scores = interaction_matrix.iloc[similar_users].mean(axis=0)

    watched_videos = interaction_matrix.iloc[user_idx]
    unwatched_videos = recommended_scores[watched_videos == 0]

    if len(unwatched_videos) == 0:
        unwatched_videos = recommended_scores

    top_videos = (
        unwatched_videos.sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return video_encoder.inverse_transform(top_videos)


# -----------------------------
# Evaluation
# -----------------------------
print("Evaluating model...")

users = df["user_id"].unique()[:20]
scores = []

for u in users:
    try:
        actual_videos = df[df["user_id"] == u] \
            .sort_values("interaction_score", ascending=False)["video_id"] \
            .head(5).tolist()

        recommended = recommend_videos(u, 5)
        score = len(set(recommended) & set(actual_videos)) / 5
        scores.append(score)
    except:
        pass

print("Average Precision@5:", np.mean(scores))


# -----------------------------
# Save Artifacts
# -----------------------------
print("Saving artifacts...")

pickle.dump(interaction_matrix, open("interaction_matrix.pkl", "wb"))
pickle.dump(user_similarity, open("user_similarity.pkl", "wb"))
pickle.dump((user_encoder, video_encoder), open("encoders.pkl", "wb"))

print("Training completed successfully!")


# -----------------------------
# Example Recommendation
# -----------------------------
sample_user = df["user_id"].iloc[0]
print("\nSample recommendation for user:", sample_user)
print(recommend_videos(sample_user))
