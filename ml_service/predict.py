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

if not text or text.strip() == "":
    print(json.dumps({"error": "Empty text provided"}))
    sys.exit(1)

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

# Feature names - must match training exactly (18 features)
feature_names = [
    "certainty_ratio",
    "hedging_ratio",
    "emotion_ratio",
    "subjectivity",
    "polarity",
    "avg_sentence_length",
    "pronoun_ratio",
    "sensational_ratio_title",
    "sensational_ratio_body",
    "headline_exclamations",
    "headline_questions",
    "capital_word_ratio_title",
    "capital_word_ratio_body",
    "neg_emotion_ratio",
    "pos_emotion_ratio",
    "objective_ratio",
    "body_exclamations",
    "body_questions",
]

# Extract features using the same method as training
# Pass empty title since we only have text from the frontend
features_list = extract_features(text, "")

# Convert list to dict for insights
features = dict(zip(feature_names, features_list))

# Prepare DataFrame for prediction
X = pd.DataFrame([features_list], columns=feature_names)

# Predict label and probability
pred_label = model.predict(X)[0]
pred_prob = model.predict_proba(X)[0].max()

# Map label to human-readable credibility
# Note: In training, 0=Fake, 1=Real
label_map = {0: "Likely Fake", 1: "Likely Real"}
credibility = label_map[pred_label]

# Risk assessment
risk = "Low" if pred_label == 1 else "High"

# Build insights for frontend
insights = [
    {"feature": k.replace("_", " ").title(), "value": round(v, 4)} 
    for k, v in features.items()
]

# Build explanation based on prediction
if pred_label == 1:
    explanation = "The text shows characteristics of reliable news content with balanced language and factual assertions."
else:
    explanation = "The text contains indicators often found in unreliable or sensationalized content. Consider verifying with additional sources."

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
