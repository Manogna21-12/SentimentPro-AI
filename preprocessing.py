import pandas as pd
import re

# ── Confidence thresholds for neutral assignment ──────────────────────────
# Samples whose heuristic confidence falls BETWEEN these values are Neutral.
NEUTRAL_LOW = 0.55
NEUTRAL_HIGH = 0.80

# ── Keyword sets for confidence scoring ───────────────────────────────────
POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "amazing", "perfect", "love",
    "wonderful", "fantastic", "best", "happy", "nice", "beautiful", "superb",
    "brilliant", "outstanding", "pleased", "enjoyed", "recommend", "favorite",
}
NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "awful", "poor", "hate", "horrible",
    "disappointing", "useless", "waste", "broken", "annoying", "frustrating",
    "dreadful", "pathetic", "rubbish", "disgusting", "regret", "refund",
}
HEDGE_WORDS = {
    "but", "however", "although", "though", "except", "despite",
    "unfortunately", "mixed", "sometimes", "okay", "ok", "decent",
    "average", "fair", "alright",
}


def compute_sentiment_confidence(text: str) -> float:
    """
    Compute a heuristic confidence score (0–1) for how clearly a text
    expresses a single sentiment polarity.
      High (> NEUTRAL_HIGH)  → clearly positive or negative
      Mid  (NEUTRAL_LOW–HIGH) → ambiguous / mixed → Neutral
      Low  (< NEUTRAL_LOW)   → weak signal → trust original label
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))

    pos_count   = len(words & POSITIVE_WORDS)
    neg_count   = len(words & NEGATIVE_WORDS)
    hedge_count = len(words & HEDGE_WORDS)

    total_signal = pos_count + neg_count

    if total_signal == 0:
        # No clear sentiment keywords found
        if hedge_count > 0:
            return 0.50  # Only hedge words → likely ambiguous
        return 0.85      # No keywords at all → trust original label

    # Confidence = how lopsided the sentiment signal is
    dominant   = max(pos_count, neg_count)
    confidence = dominant / total_signal

    # Penalise confidence when hedge/contrast words are present
    if hedge_count > 0:
        penalty    = min(0.4, 0.12 * hedge_count)
        confidence *= (1.0 - penalty)

    return round(min(max(confidence, 0.0), 1.0), 4)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'text': 'review_text'})
    df['review_text'] = df['review_text'].astype(str).str.lower()

    # Compute a per-sample confidence score
    df['_confidence'] = df['review_text'].apply(compute_sentiment_confidence)

    # Original labels: 0=Negative, 1=Positive
    # Re-map to 3 classes:  0=Negative, 1=Neutral, 2=Positive
    def remap(row):
        conf = row['_confidence']
        if NEUTRAL_LOW <= conf <= NEUTRAL_HIGH:
            return 1          # Neutral (ambiguous zone)
        elif row['label'] == 0:
            return 0          # Negative
        else:
            return 2          # Positive

    df['label'] = df.apply(remap, axis=1)
    df = df.drop(columns=['_confidence'])

    print("3-class label distribution:\n", df['label'].value_counts().sort_index())
    return df
