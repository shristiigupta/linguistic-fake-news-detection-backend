import pickle
import sys
import json
import pandas as pd
from feature_extractor import extract_features

# Get input text from command line argument
if len(sys.argv) < 2:
    print(json.dumps({"error": "No text provided"}))
    sys.exit(1)

text = sys.argv[1]

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

# Feature names in the same order used during training
feature_names = [
    "certainty_ratio",
    "hedging_ratio",
    "emotion_ratio",
    "subjectivity",
    "polarity",
    "avg_sentence_length",
    "pronoun_ratio",
    "sensational_ratio",
    "headline_exclamations",
    "headline_questions",
    "capital_word_ratio"
]

# Extract features
features_list = extract_features(text)

# Convert list to dict (for insights)
features = dict(zip(feature_names, features_list))

# Prepare DataFrame in exact order for prediction
X = pd.DataFrame([features_list], columns=feature_names)

# Predict label and probability
pred_label = model.predict(X)[0]
pred_prob = model.predict_proba(X)[0].max()

# Map label to human-readable credibility
label_map = {0: "Likely Real", 1: "Likely Fake"}
credibility = label_map[pred_label]

# Simple risk assessment
risk = "Low" if pred_label == 0 else "High"

# Build insights for frontend
insights = [{"feature": k.replace("_", " ").title(), "value": v} for k, v in features.items()]

# Build explanation based on features
explanation = ""
if pred_label == 0:
    explanation = "The text shows characteristics of reliable content with balanced language and factual assertions."
else:
    explanation = "The text contains indicators often found in unreliable or sensationalized content."

# Build JSON output
output = {
    "credibility": credibility,
    "confidence": round(pred_prob * 100, 2),
    "features": features,
    "insights": insights,
    "risk": risk,
    "explanation": explanation
}

# Print JSON output
print(json.dumps(output))
