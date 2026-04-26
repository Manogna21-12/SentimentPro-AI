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
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    return labels[pred], confidence


if __name__ == "__main__":
    text = input("Enter review: ")
    sentiment, confidence = predict_sentiment(text)
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")