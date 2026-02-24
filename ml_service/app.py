from flask import Flask, request, jsonify
import sys
import os
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob

# Download required NLTK data
try:
    for pkg in ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'vader_lexicon']:
        nltk.download(pkg, quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# ============== WORD LISTS (must match training) ==============
certainty_words = set([
    "always", "never", "definitely", "certainly", "undeniable", "proven",
    "fact", "clearly", "everyone", "nobody", "certain", "sure", "absolutely",
    "without doubt", "guaranteed", "obvious", "proof", "undoubtedly", "irrefutable",
    "unquestionable", "conclusively", "must", "will", "impossible", "inevitably"
])

hedging_words = set([
    "may", "might", "could", "possibly", "allegedly", "reported", "appears",
    "suggests", "likely", "unlikely", "apparently", "presumably", "according",
    "claims", "said to", "believed", "considered", "seems", "thought to",
    "estimated", "expected", "potential", "possible", "perhaps", "sometimes",
    "often", "usually", "generally", "typically", "tends", "indicate"
])

sensational_words = set([
    "shocking", "breaking", "unbelievable", "exposed", "truth", "secret",
    "revealed", "miracle", "amazing", "guaranteed", "exclusive", "dramatic",
    "incredible", "unprecedented", "alert", "urgent", "bizarre", "astonishing",
    "warning", "revolutionary", "bombshell", "explosive", "scandalous",
    "outrageous", "stunning", "terrifying", "alarming", "catastrophic",
    "devastating", "horrifying", "jaw-dropping", "mind-blowing", "earth-shattering",
    "game-changing", "must-see", "viral", "hidden", "suppressed", "forbidden",
    "banned", "censored", "conspiracy", "hoax", "rigged", "corrupt"
])

negative_emotion_words = set([
    "shocking", "terrible", "disaster", "horrible", "corrupt", "angry",
    "hate", "fear", "crisis", "danger", "horrific", "devastating", "evil",
    "disgusting", "outrageous", "despicable", "vile", "wicked", "monstrous",
    "atrocious", "appalling", "dreadful", "alarming", "threatening", "dangerous"
])

positive_emotion_words = set([
    "amazing", "incredible", "fantastic", "great", "excellent", "happy",
    "joy", "success", "win", "love", "wonderful", "brilliant", "outstanding",
    "remarkable", "extraordinary", "superb", "magnificent", "glorious",
    "triumphant", "victory", "heroic", "inspiring", "uplifting"
])

pronoun_words = set([
    "we", "they", "you", "he", "she", "them", "us", "our", "him", "her",
    "i", "me", "my", "their", "your", "his", "hers", "ours", "theirs"
])

objective_words = set([
    "reported", "confirmed", "according", "stated", "official", "data",
    "evidence", "statistics", "record", "announcement", "declaration",
    "study", "research", "analysis", "report", "survey", "findings",
    "investigation", "source", "spokesperson", "authority"
])

# Load model at startup
model = None
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


def extract_features(text, title=""):
    """Extract 18 features from text"""
    if not text or text == "":
        return [0] * 18

    text = str(text)
    
    # Title proxy logic
    if not title or str(title).strip() == "":
        try:
            sentences = sent_tokenize(text)
            title = sentences[0] if sentences else text[:100]
        except:
            title = text[:100]
    else:
        title = str(title)

    text_lower = text.lower()
    title_lower = title.lower()

    words = text_lower.split()
    title_words = title_lower.split()
    total_words = max(len(words), 1)
    total_title_words = max(len(title_words), 1)

    # Features
    certainty_count = sum(1 for w in words if w in certainty_words)
    certainty_ratio = certainty_count / total_words

    hedging_count = sum(1 for w in words if w in hedging_words)
    hedging_ratio = hedging_count / total_words

    try:
        blob = TextBlob(text)
        emotion_ratio = (blob.sentiment.polarity + 1) / 2
        subjectivity = blob.sentiment.subjectivity
        polarity = blob.sentiment.polarity
    except:
        emotion_ratio = 0.5
        subjectivity = 0.5
        polarity = 0.0

    try:
        sentences = sent_tokenize(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    except:
        avg_sentence_length = 15.0

    pronoun_count = sum(1 for w in words if w in pronoun_words)
    pronoun_ratio = pronoun_count / total_words

    sensational_count_title = sum(1 for w in title_words if w in sensational_words)
    sensational_ratio_title = sensational_count_title / total_title_words

    sensational_count_body = sum(1 for w in words if w in sensational_words)
    sensational_ratio_body = sensational_count_body / total_words

    headline_exclamations = title.count("!")
    headline_questions = title.count("?")

    capital_count_title = sum(1 for w in title.split() if w.isupper() and len(w) > 1)
    capital_word_ratio_title = capital_count_title / total_title_words

    capital_count_body = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    capital_word_ratio_body = capital_count_body / total_words

    neg_emotion_count = sum(1 for w in words if w in negative_emotion_words)
    neg_emotion_ratio = neg_emotion_count / total_words

    pos_emotion_count = sum(1 for w in words if w in positive_emotion_words)
    pos_emotion_ratio = pos_emotion_count / total_words

    objective_count = sum(1 for w in words if w in objective_words)
    objective_ratio = objective_count / total_words

    body_exclamations = text.count("!")
    body_questions = text.count("?")

    return [
        certainty_ratio, hedging_ratio, emotion_ratio, subjectivity, polarity,
        avg_sentence_length, pronoun_ratio, sensational_ratio_title, sensational_ratio_body,
        headline_exclamations, headline_questions, capital_word_ratio_title, capital_word_ratio_body,
        neg_emotion_ratio, pos_emotion_ratio, objective_ratio, body_exclamations, body_questions
    ]


def predict_text(text):
    """Make prediction on text"""
    global model
    
    if model is None:
        # Try loading again
        try:
            model = pickle.load(open("model.pkl", "rb"))
        except Exception as e:
            return {"error": f"Model not loaded: {str(e)}"}
    
    feature_names = [
        "certainty_ratio", "hedging_ratio", "emotion_ratio", "subjectivity", "polarity",
        "avg_sentence_length", "pronoun_ratio", "sensational_ratio_title", "sensational_ratio_body",
        "headline_exclamations", "headline_questions", "capital_word_ratio_title", "capital_word_ratio_body",
        "neg_emotion_ratio", "pos_emotion_ratio", "objective_ratio", "body_exclamations", "body_questions"
    ]
    
    features_list = extract_features(text, "")
    features = dict(zip(feature_names, features_list))
    
    X = pd.DataFrame([features_list], columns=feature_names)
    
    pred_label = model.predict(X)[0]
    pred_prob = model.predict_proba(X)[0].max()
    
    label_map = {0: "Likely Fake", 1: "Likely Real"}
    credibility = label_map[pred_label]
    risk = "Low" if pred_label == 1 else "High"
    
    insights = [{"feature": k.replace("_", " ").title(), "value": round(v, 4)} for k, v in features.items()]
    
    if pred_label == 1:
        explanation = "The text shows characteristics of reliable news content with balanced language and factual assertions."
    else:
        explanation = "The text contains indicators often found in unreliable or sensationalized content. Consider verifying with additional sources."
    
    return {
        "credibility": credibility,
        "confidence": round(pred_prob * 100, 2),
        "features": features,
        "insights": insights,
        "risk": risk,
        "explanation": explanation
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text missing"}), 400

    result = predict_text(text)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
