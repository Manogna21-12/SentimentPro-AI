# SentimentPro-AI (Sentiment Studio Pro)

A premium sentiment analysis application featuring a beautiful Streamlit interface, powered by a fine-tuned transformer model (DistilBERT).

## ✨ Features
- **Transformer-based Inference**: Uses Hugging Face's DistilBERT for highly accurate sequence classification.
- **Three-Class Sentiment**: Accurately detects **Positive**, **Neutral**, and **Negative** sentiments, complete with confidence scoring and full probability distributions.
- **Hybrid Prediction Logic**: Combines robust machine learning with heuristic rule-based overrides to catch obvious neutral keywords and strong short inputs.
- **Modern Web Interface**: A premium Streamlit dashboard featuring glassmorphism, dynamic animations, and beautifully styled Light (pastel) and Dark themes.
- **Batch Processing**: Upload CSV/TXT files or paste multiple lines of text to process hundreds of reviews in seconds.
- **Prediction History**: Automatically logs predictions with CSV export functionality.

## 📂 Project Structure
- `app.py`: Main Streamlit web application.
- `predict.py`: CLI script for quick sentiment inference and testing.
- `train.py`: Script to fine-tune the transformer model.
- `baseline_model.py`: Uses TF-IDF and Logistic Regression to establish baseline metrics.
- `preprocessing.py`: Handles data cleaning and heuristic confidence-based re-labeling for the Neutral class.
- `evaluate.py`: Model evaluation script.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Running the App
To launch the Sentiment Studio Pro web interface, simply run:
```bash
py -m streamlit run app.py
```
This will open the application in your default web browser (usually at `localhost:8502`).

### Quick CLI Testing
If you want to test the model quickly in your terminal without the UI:
```bash
py predict.py
```

## 🎨 UI Highlights
The application features a fully custom-styled CSS layout overriding default Streamlit elements to provide a "wow" factor, utilizing gradients, rounded corners, drop shadows, and responsive hover states.
