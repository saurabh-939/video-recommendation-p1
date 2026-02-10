import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
df = pd.read_csv("video_recommendation_sample_dataset.csv")

print("Feature engineering...")
df["interaction_score"] = (
    0.4 * df["watch_duration"] +
    15 * df["liked"] +
    20 * df["commented"] +
    25 * df["subscribed_after_watching"]
)

user_encoder = LabelEncoder()
video_encoder = LabelEncoder()

df["user_encoded"] = user_encoder.fit_transform(df["user_id"])
df["video_encoded"] = video_encoder.fit_transform(df["video_id"])

interaction_matrix = df.pivot_table(
    index="user_encoded",
    columns="video_encoded",
    values="interaction_score",
    fill_value=0
)

print("Computing similarity...")
user_similarity = cosine_similarity(interaction_matrix)

print("Saving artifacts...")
pickle.dump(interaction_matrix, open("interaction_matrix.pkl", "wb"))
pickle.dump(user_similarity, open("user_similarity.pkl", "wb"))
pickle.dump((user_encoder, video_encoder), open("encoders.pkl", "wb"))

print("Training completed successfully!")
