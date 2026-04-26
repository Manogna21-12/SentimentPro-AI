import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import Dataset

from data_loader import load_data
from preprocessing import preprocess_dataframe

# Load + preprocess
df = load_data()
df = preprocess_dataframe(df)

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HF dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['review_text'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ✅ Binary model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    problem_type="single_label_classification"
)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    learning_rate=2e-5
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()

# Save
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")