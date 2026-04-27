import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")
model.eval()
num_labels = model.config.num_labels

if num_labels == 3:
    labels = ['Negative', 'Neutral', 'Positive']
else:
    labels = ['Negative', 'Positive']

def predict_sentiment(text):
    text_lower = text.lower()

    # Add a list of neutral keywords
    neutral_keywords = [
        "okay", "fine", "average", "not bad", "so so", "nothing special",
        "as expected", "normal", "decent", "fair", "moderate",
        "satisfactory", "it works", "no issues", "acceptable"
    ]

    # Before applying model-based classification, check for neutral keywords
    if any(keyword in text_lower for keyword in neutral_keywords):
        return "Neutral", 1.0

    # 🔥 Rule-based boost for short inputs
    strong_positive = ["good", "great", "excellent", "awesome", "amazing", "perfect", "love"]
    strong_negative = ["bad", "worst", "terrible", "awful", "poor", "hate"]

    words = text_lower.split()

    if len(words) <= 3:
        for word in words:
            if word in strong_negative:
                return "Negative", 0.90
            if word in strong_positive:
                return "Positive", 0.90

    # Model prediction
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    confidence = max(probs).item()
    
    # Calculate score based on positive vs negative probabilities
    if num_labels == 3:
        score = probs[2].item() - probs[0].item()
    else:
        score = probs[1].item() - probs[0].item()

    # Use threshold-based classification
    if score > 0.2:
        sentiment = "Positive"
    elif score < -0.2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, confidence


if __name__ == "__main__":
    text = input("Enter review: ")
    sentiment, confidence = predict_sentiment(text)
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")