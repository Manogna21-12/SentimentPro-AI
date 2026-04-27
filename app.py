import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ================= CONFIG =================
st.set_page_config(page_title="Sentiment Studio Pro", layout="wide")

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
num_labels = model.config.num_labels

# Adapt labels to actual model output
if num_labels == 3:
    labels = ['Negative', 'Neutral', 'Positive']
else:
    labels = ['Negative', 'Positive']

def extract_probs(probs):
    """Extract neg/neu/pos probabilities, adapting to 2-class or 3-class model."""
    if num_labels == 3:
        neg = probs[0].item()
        neu = probs[1].item()
        pos = probs[2].item()
    else:
        neg = probs[0].item()
        pos = probs[1].item()
        neu = 0.0  # No native neutral class
    return neg, neu, pos

def get_sentiment(pred, confidence):
    """Map prediction index to label."""
    if num_labels == 3:
        return labels[pred]
    # 2-class model: return predicted label directly instead of defaulting to Neutral
    return labels[pred]

def run_inference(text):
    text_lower = text.lower()
    strong_positive = ["good", "great", "excellent", "awesome", "amazing", "perfect", "love"]
    strong_negative = ["bad", "worst", "terrible", "awful", "poor", "hate"]

    words = text_lower.split()

    if len(words) <= 3:
        for word in words:
            if word in strong_negative:
                if num_labels == 3: return "Negative", 0.90, 0.90, 0.05, 0.05
                else: return "Negative", 0.90, 0.90, 0.0, 0.10
            if word in strong_positive:
                if num_labels == 3: return "Positive", 0.90, 0.05, 0.05, 0.90
                else: return "Positive", 0.90, 0.10, 0.0, 0.90

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    sentiment = get_sentiment(pred, confidence)
    neg, neu, pos = extract_probs(probs)
    return sentiment, confidence, neg, neu, pos

# ================= STATE =================
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ================= SIDEBAR =================
with st.sidebar:
    st.title("UI Controls")
    
    # Theme Toggle
    theme_choice = st.radio("Theme", ["dark", "light"], index=0 if st.session_state.get("theme", "dark") == "dark" else 1, horizontal=True, label_visibility="collapsed")
    if theme_choice != st.session_state.get("theme", "dark"):
        st.session_state.theme = theme_choice
        st.rerun()
    
    st.markdown("---")
    
    # Example input Section
    st.subheader("Example input")
    example = st.selectbox(
        "",
        [
            "None",
            # Strong positive
            "This product is amazing!",
            "Absolutely love it, best purchase ever!",
            # Strong negative
            "Worst experience ever",
            "Completely useless, total waste of money",
            # Neutral (mixed / factual / ambiguous)
            "Good quality but terrible service",
            "The package arrived on the expected date",
            "Some features work well, others do not",
            "It has pros and cons like everything else",
            "I have mixed feelings about this product",
            "Not what I expected but not terrible either",
        ],
        key="example_select"
    )
    
    st.markdown("---")
    
    # About Section
    st.subheader("About")
    st.write("3-class sentiment classification using a locally loaded transformer model.")
    st.write("• Task: Text polarity")
    st.write("• Output: Class + confidence")
    st.write("• Runtime: Local inference")
    
    st.markdown("---")
    
    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ================= SIDEBAR CSS =================
theme = st.session_state.get("theme", "dark")

if theme == "dark":
    st.markdown("""
    <style>
    /* Hide Streamlit toolbar / top bar */
    [data-testid="stToolbar"] { display: none !important; }
    header[data-testid="stHeader"] {
        background: #0f172a !important;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b, #0f172a) !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* App background */
    .stApp { background: radial-gradient(circle at top, #0f172a, #020617); color: white; }

    /* Glass card */
    .glass {
        background: rgba(255,255,255,0.05);
        padding: 25px; border-radius: 18px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Header banner */
    .header {
        background: linear-gradient(135deg, #1e293b, #0f4a46);
        padding: 35px; border-radius: 20px; color: white; margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    /* Info cards */
    .card {
        background: rgba(255,255,255,0.05);
        padding: 20px; border-radius: 15px;
        text-align: center; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Buttons */
    .stButton>button {
        width: 100%; border-radius: 10px; padding: 12px;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white; border: none;
    }

    /* Text area */
    textarea {
        background: #020617 !important; color: white !important;
        border-radius: 12px !important; border: 1px solid #334155 !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    /* ── Streamlit chrome ── */
    [data-testid="stToolbar"] { display: none !important; }
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    /* Deploy button text */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] span { color: #1e293b !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #dbeafe, #ccfbf1) !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] * { color: #1e293b !important; }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
    }

    /* ── App background ── */
    .stApp { background: #f8fafc !important; color: #1e293b !important; }

    /* ── Main content area ── */
    [data-testid="stAppViewContainer"] > section:nth-child(2) {
        background: #f8fafc !important;
    }

    /* ── Glass card ── */
    .glass {
        background: linear-gradient(135deg, #dbeafe, #ccfbf1);
        padding: 25px; border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }

    /* ── Header banner ── */
    .header {
        background: linear-gradient(135deg, #dbeafe, #ccfbf1);
        padding: 35px; border-radius: 20px; color: #0f172a !important; margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .header h1, .header p { color: #0f172a !important; }

    /* ── Info cards ── */
    .card {
        background: linear-gradient(135deg, #dbeafe, #ccfbf1);
        padding: 20px; border-radius: 15px;
        text-align: center; font-weight: 600;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        color: #1e293b !important;
    }

    /* ── Buttons ── */
    .stButton>button {
        width: 100%; border-radius: 10px; padding: 12px;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important; border: none;
    }

    /* ── Text area ── */
    textarea {
        background: #ffffff !important; color: #1e293b !important;
        border-radius: 12px !important; border: 1px solid #cbd5e1 !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploadDropzone"] {
        background: #f0f7ff !important;
        border: 2px dashed #3b82f6 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploadDropzone"] * { color: #1e293b !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] button { color: #475569 !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #3b82f6 !important; }

    /* ── Selectbox / dropdowns ── */
    [data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] { background: #ffffff; border-radius: 12px; padding: 12px; border: 1px solid #e2e8f0; }
    [data-testid="stMetricLabel"] * { color: #64748b !important; }
    [data-testid="stMetricValue"] * { color: #1e293b !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* ── General text ── */
    p, span, label, div { color: #1e293b !important; }
    h1, h2, h3, h4 { color: #0f172a !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header">
    <h1>Sentiment Studio Pro</h1>
    <p>Premium sentiment analysis interface for product feedback, support tickets, and social comments.</p>
</div>
""", unsafe_allow_html=True)

# ================= TOP CARDS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">MODEL<br><b>DistilBERT</b></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">CLASSES<br><b>Positive / Neutral / Negative</b></div>', unsafe_allow_html=True)

with col3:
    theme_display = "🌙 Dark" if st.session_state.theme == "dark" else "☀️ Light"
    st.markdown(f'<div class="card">THEME<br><b>{theme_display}</b></div>', unsafe_allow_html=True)

st.write("")

# ================= TABS =================
tab1, tab2 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis"])

with tab1:
    # ================= MAIN SECTION =================
    left, right = st.columns([2,1])

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Analyze Text")

        # Use example text as default if selected, always show text area
        default_text = example if example != "None" else ""
        review = st.text_area("", height=140, placeholder="Enter text to analyze sentiment...", value=default_text)

        colA, colB = st.columns(2)
        analyze = colA.button("⚡ Analyze", use_container_width=True)
        clear = colB.button("Clear", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Quick Guide")
        st.write("1. Enter text in the analyzer panel.")
        st.write("2. Click Analyze to run inference.")
        st.write("3. Review confidence and class distribution.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= ANALYSIS =================
    if analyze and review.strip():

        sentiment, confidence, neg, neu, pos = run_inference(review)

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": review,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "positive_probability": round(pos, 4),
            "neutral_probability": round(neu, 4),
            "negative_probability": round(neg, 4),
            "neutral_flag": sentiment == 'Neutral',
            "word_count": len(review.split()),
            "character_count": len(review)
        }

        st.session_state.history.append(result)
        
        # Display Results Section
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")
        
        # Top metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if sentiment == 'Positive':
                sentiment_color = "🟢"
            elif sentiment == 'Negative':
                sentiment_color = "🔴"
            else:
                sentiment_color = "🟡"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 14px; opacity: 0.7; margin-bottom: 8px;">SENTIMENT</div>
                <div style="font-size: 28px; font-weight: bold;">{sentiment_color} {sentiment}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 14px; opacity: 0.7; margin-bottom: 8px;">CONFIDENCE</div>
                <div style="font-size: 28px; font-weight: bold;">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 14px; opacity: 0.7; margin-bottom: 8px;">WORD COUNT</div>
                <div style="font-size: 28px; font-weight: bold;">{result['word_count']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Pie chart and probability details
        col_chart, col_details = st.columns([1.2, 1])
        
        with col_chart:
            fig, ax = plt.subplots(figsize=(6, 5))
            sizes      = [neg * 100, neu * 100, pos * 100]
            labels_pie = ['Negative', 'Neutral', 'Positive']
            colors     = ['#ef4444', '#f59e0b', '#22c55e']

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels_pie,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(13)
                autotext.set_weight('bold')
            for text in texts:
                text.set_fontsize(13)
                text.set_weight('bold')

            ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col_details:
            st.markdown("**PROBABILITY BREAKDOWN**")

            st.write(f"🟢 Positive: **{pos*100:.2f}%**")
            st.progress(pos)

            st.write(f"🟡 Neutral: **{neu*100:.2f}%**")
            st.progress(neu)

            st.write(f"🔴 Negative: **{neg*100:.2f}%**")
            st.progress(neg)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📥 Batch Analysis Input")
    
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"], horizontal=True)
    
    df_input = None
    
    if input_method == "Paste Text":
        batch_text = st.text_area("Paste reviews (one per line)", height=200, placeholder="Enter multiple reviews here...\nEach line will be treated as a separate review.")
        if batch_text:
            texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
            if texts:
                df_input = pd.DataFrame({"text": texts})
                st.info(f"📋 Loaded {len(df_input)} reviews for analysis")
    else:
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"], key="batch_upload")
        if uploaded_file is not None:
            try:
                # Handle different file types
                if uploaded_file.type == "text/plain":
                    # For TXT files, read line by line
                    content = uploaded_file.read().decode("utf-8")
                    texts = [line.strip() for line in content.split('\n') if line.strip()]
                    df_input = pd.DataFrame({"text": texts})
                else:
                    # For CSV files
                    df_input = pd.read_csv(uploaded_file)
                
                # Check if 'text' column exists
                if 'text' not in df_input.columns:
                    st.error("❌ File must contain a 'text' column (for CSV) or one text per line (for TXT)")
                    df_input = None
                else:
                    st.info(f"📋 Loaded {len(df_input)} rows for analysis")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                df_input = None
                
    if df_input is not None and not df_input.empty:
        if st.button("🚀 Analyze Batch", use_container_width=True):
                    progress_bar = st.progress(0)
                    results_batch = []
                    
                    for idx, row in df_input.iterrows():
                        text = str(row['text']).strip()
                        
                        if text:
                            sentiment, confidence, neg, neu, pos = run_inference(text)
                            
                            results_batch.append({
                                "text": text,
                                "sentiment": sentiment,
                                "confidence": round(confidence, 4),
                                "positive_probability": round(pos, 4),
                                "neutral_probability": round(neu, 4),
                                "negative_probability": round(neg, 4),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        
                        progress_bar.progress((idx + 1) / len(df_input))
                    
                    st.session_state.batch_results = results_batch
                    st.success(f"✅ Analyzed {len(results_batch)} texts successfully!")
        

    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display batch results
    if "batch_results" in st.session_state and st.session_state.batch_results:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Batch Results")
        
        df_results = pd.DataFrame(st.session_state.batch_results)
        
        # Summary statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            positive_count = len(df_results[df_results['sentiment'] == 'Positive'])
            st.metric("🟢 Positive", positive_count)
        
        with col2:
            negative_count = len(df_results[df_results['sentiment'] == 'Negative'])
            st.metric("🔴 Negative", negative_count)
        
        with col3:
            neutral_count = len(df_results[df_results['sentiment'] == 'Neutral'])
            st.metric("🟡 Neutral", neutral_count)
        
        with col4:
            avg_confidence = df_results['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        
        with col5:
            total_texts = len(df_results)
            st.metric("Total Texts", total_texts)
        
        st.markdown("")
        
        # Results table
        st.dataframe(df_results, use_container_width=True)
        
        # Download results
        csv_results = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv_results,
            "batch_results.csv",
            "text/csv",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# ================= HISTORY =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📋 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    
    # Download history
    csv_history = df.to_csv(index=False)
    st.download_button(
        "📥 Download History (CSV)",
        csv_history,
        "prediction_history.csv",
        "text/csv",
        use_container_width=True
    )
else:
    st.info("No predictions yet. Start analyzing text to build history.")

st.markdown('</div>', unsafe_allow_html=True)