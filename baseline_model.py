from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import preprocess_dataframe


if __name__ == "__main__":
    df = load_data()
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df['review_text'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("Baseline Model Results:")
    print(classification_report(y_test, preds))