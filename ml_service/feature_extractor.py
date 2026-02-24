import re
import nltk
import pandas as pd
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('vader_lexicon')
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# ============== EXPANDED WORD LISTS (must match training) ==============

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


def extract_features(text, title=""):
    """
    Extract 18 features from text and title.
    Must match training exactly in fake_news_detection.py
    
    Returns list of features in this order:
    0.  certainty_ratio
    1.  hedging_ratio
    2.  emotion_ratio
    3.  subjectivity
    4.  polarity
    5.  avg_sentence_length
    6.  pronoun_ratio
    7.  sensational_ratio_title
    8.  sensational_ratio_body
    9.  headline_exclamations
    10. headline_questions
    11. capital_word_ratio_title
    12. capital_word_ratio_body
    13. neg_emotion_ratio
    14. pos_emotion_ratio
    15. objective_ratio
    16. body_exclamations
    17. body_questions
    """
    if pd.isna(text) or text == "":
        return [0] * 18

    text = str(text)
    
    # ----- Title proxy logic -----
    # When no title is provided, use first sentence as proxy
    if not title or str(title).strip() == "":
        try:
            sentences = nltk.sent_tokenize(text)
            title = sentences[0] if sentences else text[:100]
        except:
            title = text[:100]
    else:
        title = str(title)

    # Lowercase versions for word matching
    text_lower = text.lower()
    title_lower = title.lower()

    words = text_lower.split()
    title_words = title_lower.split()
    total_words = max(len(words), 1)
    total_title_words = max(len(title_words), 1)

    # ---- FEATURE 1: Certainty ratio (body) ----
    certainty_count = sum(1 for w in words if w in certainty_words)
    certainty_ratio = certainty_count / total_words

    # ---- FEATURE 2: Hedging ratio (body) ----
    hedging_count = sum(1 for w in words if w in hedging_words)
    hedging_ratio = hedging_count / total_words

    # ---- FEATURE 3 & 4: TextBlob emotion & subjectivity (body) ----
    try:
        blob = TextBlob(text)
        emotion_ratio = (blob.sentiment.polarity + 1) / 2
        subjectivity = blob.sentiment.subjectivity
        polarity = blob.sentiment.polarity
    except:
        emotion_ratio = 0.5
        subjectivity = 0.5
        polarity = 0.0

    # ---- FEATURE 5: Average sentence length (body) ----
    try:
        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    except:
        avg_sentence_length = 15.0

    # ---- FEATURE 6: Pronoun ratio (body) ----
    pronoun_count = sum(1 for w in words if w in pronoun_words)
    pronoun_ratio = pronoun_count / total_words

    # ---- FEATURE 7: Sensational ratio (TITLE) ----
    sensational_count_title = sum(1 for w in title_words if w in sensational_words)
    sensational_ratio_title = sensational_count_title / total_title_words

    # ---- FEATURE 8: Sensational ratio (BODY) ----
    sensational_count_body = sum(1 for w in words if w in sensational_words)
    sensational_ratio_body = sensational_count_body / total_words

    # ---- FEATURE 9: Headline exclamations ----
    headline_exclamations = title.count("!")

    # ---- FEATURE 10: Headline questions ----
    headline_questions = title.count("?")

    # ---- FEATURE 11: Capital word ratio (TITLE) ----
    capital_count_title = sum(1 for w in title.split() if w.isupper() and len(w) > 1)
    capital_word_ratio_title = capital_count_title / total_title_words

    # ---- FEATURE 12: Capital word ratio (BODY) ----
    capital_count_body = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    capital_word_ratio_body = capital_count_body / total_words

    # ---- FEATURE 13: Negative emotion word ratio (body) ----
    neg_emotion_count = sum(1 for w in words if w in negative_emotion_words)
    neg_emotion_ratio = neg_emotion_count / total_words

    # ---- FEATURE 14: Positive emotion word ratio (body) ----
    pos_emotion_count = sum(1 for w in words if w in positive_emotion_words)
    pos_emotion_ratio = pos_emotion_count / total_words

    # ---- FEATURE 15: Objective language ratio (body) ----
    objective_count = sum(1 for w in words if w in objective_words)
    objective_ratio = objective_count / total_words

    # ---- FEATURE 16: Exclamation marks in body ----
    body_exclamations = text.count("!")

    # ---- FEATURE 17: Question marks in body ----
    body_questions = text.count("?")

    return [
        certainty_ratio,           # 0
        hedging_ratio,             # 1
        emotion_ratio,             # 2
        subjectivity,              # 3
        polarity,                  # 4
        avg_sentence_length,       # 5
        pronoun_ratio,             # 6
        sensational_ratio_title,  # 7
        sensational_ratio_body,   # 8
        headline_exclamations,    # 9
        headline_questions,        # 10
        capital_word_ratio_title, # 11
        capital_word_ratio_body,  # 12
        neg_emotion_ratio,         # 13
        pos_emotion_ratio,         # 14
        objective_ratio,           # 15
        body_exclamations,         # 16
        body_questions,            # 17
    ]


# For backwards compatibility
def extract_features_single(text):
    """Extract features from just the text (no title)"""
    return extract_features(text, "")
