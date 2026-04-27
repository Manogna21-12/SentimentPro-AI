from datasets import load_dataset
import pandas as pd

def load_data():
    dataset = load_dataset("racro/sentiment-analysis-finetune")

    df = pd.DataFrame(dataset['train'])

    print("Columns:", df.columns.tolist())
    print("Label distribution:\n", df['label'].value_counts())
    print(df.head())

    return df
