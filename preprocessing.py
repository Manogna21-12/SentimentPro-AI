def preprocess_dataframe(df):

    # Rename columns (adjust if needed after print)
    df = df.rename(columns={
        'text': 'review_text',
        'label': 'label'
    })

    # Lowercase text
    df['review_text'] = df['review_text'].astype(str).str.lower()

    # Ensure labels are integers (0 = negative, 1 = positive)
    df['label'] = df['label'].astype(int)

    return df