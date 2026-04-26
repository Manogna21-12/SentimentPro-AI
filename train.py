import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from data_loader import load_data
from preprocessing import preprocess_dataframe

# ── Load + preprocess ──────────────────────────────────────────────────────
df = load_data()
df = preprocess_dataframe(df)

# ── Split ──────────────────────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

# ── Per-split class counts ────────────────────────────────────────────────
label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
    counts = split_df['label'].value_counts().sort_index()
    print(f"\n{split_name} split class counts:")
    for label_id in sorted(label_names.keys()):
        count = counts.get(label_id, 0)
        print(f"  {label_names[label_id]}: {count}")
        if count < 100:
            warnings.warn(
                f"⚠️  {split_name} split has only {count} '{label_names[label_id]}' "
                f"samples (< 100). Model may underperform on this class."
            )

# ── HuggingFace datasets ───────────────────────────────────────────────────
train_dataset = Dataset.from_pandas(train_df[['review_text', 'label']])
test_dataset  = Dataset.from_pandas(test_df[['review_text',  'label']])

# ── Tokenizer ─────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['review_text'], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize,  batched=True)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format( 'torch', columns=['input_ids', 'attention_mask', 'label'])

# ── Model — 3 classes ─────────────────────────────────────────────────────
# 0 = Negative  |  1 = Neutral  |  2 = Positive
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification",
)

# ── Training args ─────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

# ── Trainer ───────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
print("✅ 3-class model saved to sentiment_model/")
