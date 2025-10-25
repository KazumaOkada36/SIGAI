from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import os, joblib

# ---- Load CLIP ----
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Load labels ----
df = pd.read_csv("labels.csv")  # ad_id, emotion, tone, audience

# ---- Extract embeddings ----
def get_clip_embedding(img_path):
    image = Image.open(img_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

X, y_emotion, y_tone, y_audience = [], [], [], []
for _, row in df.iterrows():
    path = os.path.join("image_ads", row["ad_id"])
    if os.path.exists(path):
        emb = get_clip_embedding(path)
        X.append(emb)
        y_emotion.append(row["emotion"])
        y_tone.append(row["tone"])
        y_audience.append(row["audience"])

X = np.array(X)

# ---- Encode labels ----
le_emotion = LabelEncoder().fit(y_emotion)
le_tone = LabelEncoder().fit(y_tone)
le_audience = LabelEncoder().fit(y_audience)

y_emotion = le_emotion.transform(y_emotion)
y_tone = le_tone.transform(y_tone)
y_audience = le_audience.transform(y_audience)

# ---- Split data ----
X_train, X_test, y_train, y_test = train_test_split(X, y_emotion, test_size=0.2, random_state=42)

# ---- Train classifier ----
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_emotion.classes_))

# ---- Save everything ----
joblib.dump(clf, "outputs/emotion_classifier.pkl")
joblib.dump(le_emotion, "outputs/label_encoder.pkl")
np.save("outputs/clip_embeddings.npy", X)
print("✅ Model saved!")
