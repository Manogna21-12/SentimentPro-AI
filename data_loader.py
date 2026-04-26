from datasets import load_dataset
import pandas as pd

def load_data():
    dataset = load_dataset("racro/sentiment-analysis-finetune")

    df = pd.DataFrame(dataset['train'])

    print("Columns:", df.columns)  # debug once
    print(df.head())

    return df