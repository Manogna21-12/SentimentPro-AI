import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import Dataset
from data_loader import load_data
from preprocessing import preprocess_dataframe


if __name__ == "__main__":
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")
    model.eval()

    df = load_data()
    df = preprocess_dataframe(df)

    dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(batch['review_text'], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Predictions
    preds = []
    true = []

    with torch.no_grad():
        for item in dataset:
            inputs = {
                "input_ids": item['input_ids'].unsqueeze(0),
                "attention_mask": item['attention_mask'].unsqueeze(0)
            }
            outputs = model(**inputs)
            pred = np.argmax(outputs.logits.numpy())

            preds.append(pred)
            true.append(item['label'].item())

    # Report
    print(classification_report(true, preds))

    # Confusion Matrix
    cm = confusion_matrix(true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()