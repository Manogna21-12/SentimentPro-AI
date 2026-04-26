import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Sentiment Analysis | Professional AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("sentiment_model")
    return model, tokenizer

model, tokenizer = load_model()
labels = ['Negative', 'Positive']

# ==================== PROFESSIONAL STYLING ====================
st.markdown("""
<style>
    /* Color Scheme */
    :root {
        --primary: #3b2d20;
        --secondary: #7f5a38;
        --accent: #d4af37;
        --accent-light: #f0d177;
        --gold: #fbbf24;
        --success: #16a34a;
        --danger: #dc2626;
        --text: #f8f1e4;
        --text-secondary: #e6d8c0;
    }
    
    * {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #3b2d20 0%, #865d2f 100%);
        color: var(--text);
    }
    
    body, div, span, p, h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }
    
    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, #5d3f24 0%, #8f6d3d 100%);
        padding: 50px 40px;
        border-radius: 16px;
        margin-bottom: 30px;
        border: 2px solid #d4af37;
        box-shadow: 0 0 30px rgba(212, 175, 55, 0.18), 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12) 0%, rgba(255, 239, 180, 0.05) 100%);
        pointer-events: none;
    }
    
    .header-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #fbbf24, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 16px;
        color: #f0d177;
        margin-top: 12px;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Professional Cards - HIDDEN */
    .card {
        background: transparent;
        padding: 0;
        border-radius: 14px;
        box-shadow: none;
        border: none;
        margin-bottom: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: var(--text);
    }
    
    .card * {
        color: var(--text) !important;
    }
    
    .card:hover {
        box-shadow: none;
        border-color: transparent;
        transform: none;
    }
    
    /* Input Section */
    .input-card {
        background: linear-gradient(135deg, #5f442e 0%, #362617 100%);
        border: 2px solid rgba(219, 149, 59, 0.35);
        border-radius: 14px;
        padding: 32px;
        margin-bottom: 24px;
        color: var(--text);
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.15);
    }
    
    .input-card * {
        color: var(--text) !important;
    }
    
    /* Sentiment Badge */
    .sentiment-badge {
        display: inline-block;
        padding: 12px 28px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #16a34a, #65a30d);
        color: #ffffff;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: #ffffff;
    }
    
    /* Metric Cards - HIDDEN */
    .metric-card {
        background: transparent;
        padding: 0;
        border-radius: 12px;
        border: none;
        color: var(--text);
        backdrop-filter: none;
        box-shadow: none;
    }
    
    .metric-label {
        font-size: 12px;
        color: #e6d8c0 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #fbbf24 !important;
        margin-top: 12px;
        text-shadow: 0 4px 12px rgba(212, 175, 55, 0.3);
    }
    
    /* Progress Bar */
    .confidence-bar {
        background: rgba(219, 149, 59, 0.12);
        height: 10px;
        border-radius: 6px;
        overflow: hidden;
        margin-top: 16px;
        border: 1px solid rgba(219, 149, 59, 0.25);
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #d4af37, #fbbf24);
        height: 100%;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 0 16px rgba(212, 175, 55, 0.4);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #d4af37, #b48f32) !important;
        color: #3b2d20 !important;
        border: none !important;
        padding: 14px 36px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 24px rgba(212, 175, 55, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 32px rgba(212, 175, 55, 0.45) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Text Input */
    .stTextArea > div > div > textarea {
        border-radius: 10px !important;
        border: 2px solid rgba(219, 149, 59, 0.3) !important;
        background-color: rgba(63, 48, 33, 0.85) !important;
        color: var(--text) !important;
        font-size: 15px !important;
        padding: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #d4af37 !important;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.35) !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #c6ad8a !important;
    }
    
    /* Sidebar */
    .stSidebar {
        background: linear-gradient(135deg, #3b2d20 0%, #7f5a38 100%);
    }
    
    .stSidebar > div {
        background: transparent !important;
    }
    
    .sidebar-title {
        font-size: 14px;
        font-weight: 800;
        color: #d4af37 !important;
        margin-top: 24px;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 800;
        color: #d4af37 !important;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        color: #c6ad8a;
        font-size: 13px;
        border-top: 1px solid rgba(219, 149, 59, 0.25);
        margin-top: 50px;
    }
    
    .footer p {
        color: #c6ad8a !important;
    }
    
    /* Insights Box */
    .insights-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12) 0%, rgba(212, 175, 55, 0.05) 100%);
        padding: 18px;
        border-radius: 10px;
        border-left: 4px solid #fbbf24;
        margin-top: 12px;
        color: #f8f1e4 !important;
        backdrop-filter: blur(10px);
    }
    
    .insights-box * {
        color: #f0d177 !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.12) 0%, rgba(248, 113, 113, 0.05) 100%);
        padding: 18px;
        border-radius: 10px;
        border-left: 4px solid #dc2626;
        color: #fee2e2 !important;
        backdrop-filter: blur(10px);
    }
    
    .warning-box * {
        color: #fecaca !important;
    }
    
    /* Streamlit Elements */
    .stMarkdown {
        color: var(--text) !important;
    }
    
    .stSubheader {
        color: var(--text) !important;
    }
    
    .stCaption {
        color: #e6d8c0 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12) 0%, rgba(212, 175, 55, 0.05) 100%) !important;
        border-left: 4px solid #fbbf24 !important;
        color: var(--text) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.12) 0%, rgba(248, 113, 113, 0.05) 100%) !important;
        border-left: 4px solid #dc2626 !important;
        color: #fecaca !important;
    }
    
    /* Ensure all text is visible */
    div, p, span {
        color: var(--text) !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(219, 149, 59, 0.12);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #d4af37, #fbbf24);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #b48f32, #d4af37);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎯 Configuration")
    
    st.markdown("---")
    st.markdown("**⚙️ Model Information**")
    st.info("""
    • **Architecture**: DistilBERT Transformer
    • **Task**: Binary Classification
    • **Classes**: Positive | Negative
    • **Status**: ✅ Production Ready
    • **Framework**: Hugging Face
    """)
    
    st.markdown("---")
    st.markdown("**📖 How to Use**")
    st.markdown("""
    1️⃣ Enter text for analysis
    2️⃣ Click "Analyze"
    3️⃣ View sentiment prediction
    4️⃣ Check confidence score
    """)
    
    st.markdown("---")
    st.markdown("**🔒 About**")
    st.caption("Professional Sentiment Analysis Tool | Powered by Advanced AI")

# ==================== MAIN HEADER ====================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">✨ SENTIMENT ANALYSIS</h1>
    <p class="header-subtitle">Enterprise-Grade AI-Powered Text Classification | Real-Time Sentiment Detection</p>
</div>
""", unsafe_allow_html=True)

# ==================== INPUT SECTION ====================
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown("### 📝 Enter Text for Professional Analysis")
review = st.text_area(
    "Paste your review, feedback, or any text here:",
    height=140,
    placeholder="E.g., 'This product exceeded my expectations! Outstanding quality and excellent customer service.'",
    label_visibility="collapsed"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    analyze = st.button("🔍 ANALYZE", use_container_width=True)
with col2:
    clear_btn = st.button("🔄 CLEAR", use_container_width=True)
with col3:
    st.empty()
with col4:
    st.empty()

if clear_btn:
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ==================== RESULTS SECTION ====================
if analyze and review.strip():
    # Process input
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()
    
    sentiment = labels[pred]
    neg, pos = probs[0].item(), probs[1].item()
    
    # Result Badge
    badge_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"
    st.markdown(f"""
    <div style="text-align: center; margin: 30px 0;">
        <span class="sentiment-badge {badge_class}">
            ⭐ {sentiment.upper()} SENTIMENT
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Results Grid
    col1, col2, col3 = st.columns(3)
    
    # Left Column - Main Metrics
    with col1:
        st.markdown('<div class="metric-label">🎯 Sentiment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{sentiment}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-label">📊 Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{confidence:.0%}</div>', unsafe_allow_html=True)
        
        # Confidence bar
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence * 100}%"></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-label">📈 Statistics</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(review.split())}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 12px; color: #e6d8c0; margin-top: 8px;">Words Detected</p>', unsafe_allow_html=True)
    
    st.markdown("<div style='margin: 30px 0; height: 2px; background: linear-gradient(90deg, rgba(251, 191, 36, 0), rgba(251, 191, 36, 0.3), rgba(251, 191, 36, 0)); border-radius: 2px;'></div>", unsafe_allow_html=True)
    
    # Probability Distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">📊 CLASS PROBABILITY</h3>', unsafe_allow_html=True)
        
        # Negative
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #ffffff;">🔴 Negative</span>
                <span style="color: #cbd5e1; font-weight: 600;">{neg:.1%}</span>
            </div>
        <div style="background: rgba(251, 191, 36, 0.12); height: 10px; border-radius: 6px; overflow: hidden; border: 1px solid rgba(239, 68, 68, 0.2);">
                <div style="background: linear-gradient(90deg, #ef4444, #f97316); height: 100%; width: {neg*100}%; border-radius: 6px; box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Positive
        st.markdown(f"""
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #ffffff;">🟢 Positive</span>
                <span style="color: #cbd5e1; font-weight: 600;">{pos:.1%}</span>
            </div>
        <div style="background: rgba(251, 191, 36, 0.12); height: 10px; border-radius: 6px; overflow: hidden; border: 1px solid rgba(16, 185, 129, 0.2);">
                <div style="background: linear-gradient(90deg, #16a34a, #65a30d); height: 100%; width: {pos*100}%; border-radius: 6px; box-shadow: 0 0 12px rgba(101, 163, 13, 0.3);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="section-header">📈 DISTRIBUTION</h3>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        
        colors = ["#ef4444", "#10b981"]
        explode = (0.08, 0.08) if max(neg, pos) > 0.7 else (0, 0)
        
        wedges, texts, autotexts = ax.pie(
            [neg, pos],
            labels=["Negative", "Positive"],
            autopct="%1.1f%%",
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={'color': '#ffffff', 'fontsize': 11, 'weight': 'bold'},
            wedgeprops={'edgecolor': '#d4af37', 'linewidth': 2}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax.axis('equal')
        st.pyplot(fig, transparent=True, use_container_width=True)
    
    # Insights Section
    st.markdown("<div style='margin: 30px 0; height: 2px; background: linear-gradient(90deg, rgba(251, 191, 36, 0), rgba(251, 191, 36, 0.3), rgba(251, 191, 36, 0)); border-radius: 2px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">💡 INSIGHTS</h3>', unsafe_allow_html=True)
        
        if confidence >= 0.85:
            st.markdown('<div class="insights-box">✅ <b>HIGHLY CONFIDENT</b> - Clear and strong sentiment indicators detected.</div>', unsafe_allow_html=True)
        elif confidence >= 0.65:
            st.markdown('<div class="insights-box">ℹ️ <b>MODERATE CONFIDENCE</b> - Reasonable prediction with some mixed sentiment.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠️ <b>LOW CONFIDENCE</b> - Mixed or ambiguous sentiment detected in text.</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insights-box" style="margin-top: 12px;">
        📝 <b>TEXT STATISTICS:</b><br>
        • Total Words: {len(review.split())}<br>
        • Character Count: {len(review)}<br>
        • Average Word Length: {len(review)/len(review.split()):.1f} chars
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="section-header">🔧 MODEL INFO</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insights-box">
        <b>⏰ Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <b>🤖 Model Version:</b> DistilBERT v1.0<br>
        <b>⚙️ Framework:</b> Hugging Face Transformers<br>
        <b>✅ Status:</b> Analysis Complete
        </div>
        """, unsafe_allow_html=True)

elif analyze and not review.strip():
    st.error("⚠️ Please enter text to analyze before processing.")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p><strong>Professional Sentiment Analysis Platform</strong> | Enterprise-Grade AI Solutions</p>
    <p style="margin-top: 12px; font-size: 11px;">Powered by DistilBERT • Hugging Face Transformers • Streamlit</p>
    <p style="margin-top: 8px; font-size: 10px;">© 2026 Advanced AI Solutions. All rights reserved. | Enterprise Edition</p>
</div>
""", unsafe_allow_html=True)