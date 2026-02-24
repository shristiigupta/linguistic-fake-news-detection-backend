import pickle
import sys
import json
import os
import pandas as pd
from feature_extractor import extract_features

# Ensure UTF-8 output (important on Render)
sys.stdout.reconfigure(encoding='utf-8')

try:
    # Get input text from command line argument
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No text provided"}))
        sys.exit(1)

    text = sys.argv[1]

    if not text or text.strip() == "":
        print(json.dumps({"error": "Empty text provided"}))
        sys.exit(1)

    # Absolute base directory (VERY IMPORTANT on Render)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load trained ML model
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Feature names — MUST match training EXACTLY (18)
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

    # Extract features (same logic as training)
    # Title is empty because frontend sends only body text
    features_list = extract_features(text, "")

    if len(features_list) != len(feature_names):
        raise ValueError("Feature count mismatch with trained model")

    # Convert to DataFrame
    X = pd.DataFrame([features_list], columns=feature_names)

    # Predict
    pred_label = int(model.predict(X)[0])
    pred_prob = float(model.predict_proba(X)[0].max())

    # Map prediction
    label_map = {0: "Likely Fake", 1: "Likely Real"}
    credibility = label_map[pred_label]
    risk = "Low" if pred_label == 1 else "High"

    # Build feature dictionary
    features = dict(zip(feature_names, features_list))

    # Insights for frontend
    insights = [
        {
            "feature": k.replace("_", " ").title(),
            "value": round(float(v), 4)
        }
        for k, v in features.items()
    ]

    # Explanation
    if pred_label == 1:
        explanation = (
            "The text demonstrates linguistic and emotional patterns commonly found "
            "in reliable and factual news content."
        )
    else:
        explanation = (
            "The text exhibits linguistic indicators frequently associated with "
            "sensational or misleading news content. Verification is recommended."
        )

    # Final output
    output = {
        "credibility": credibility,
        "confidence": round(pred_prob * 100, 2),
        "risk": risk,
        "features": features,
        "insights": insights,
        "explanation": explanation
    }

    # ALWAYS print JSON only
    print(json.dumps(output))

except Exception as e:
    # Never crash silently — always return JSON
    print(json.dumps({
        "error": "Prediction failed",
        "details": str(e)
    }))
    sys.exit(1)