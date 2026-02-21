import streamlit as st
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO


# =============================
# PAGE CONFIG (Responsive)
# =============================
st.set_page_config(page_title="MindShield AI", layout="centered")

# =============================
# CUSTOM CSS (Mobile + PC Friendly)
# =============================
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.main .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 1100px;
}

.big-title {
    font-size: 36px;
    font-weight: bold;
    color: #00F5FF;
    text-align: center;
}

.section-card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 20px;
    margin-top: 20px;
}

.highlight {
    background-color: #FF4B4B;
    padding: 2px 6px;
    border-radius: 6px;
    color: white;
}

.stButton>button {
    background-color: #00F5FF;
    color: black;
    font-weight: bold;
    width: 100%;
}

@media (max-width: 768px) {
    .big-title {
        font-size: 26px;
    }
}
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown("<div class='big-title'>🧠 AI-DETECTING PSYCHOLOGICAL MANIPULATION</div>", unsafe_allow_html=True)
st.write("Advanced text analyzer for psychological manipulation and meaning-aware fake news detection")

# =============================
# MODEL LOADING
# =============================
@st.cache_resource
def load_transformer_model():
    model_name = "Giyaseddin/distilbert-base-cased-finetuned-fake-and-real-news-dataset"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, transformer_model = load_transformer_model()

# =============================
# DETECTION RULES
# =============================
trigger_categories = {
    "Fear": ["danger", "threat", "risk", "destroy", "crisis", "attack"],
    "Urgency": ["now", "immediately", "urgent", "limited", "hurry", "act fast"],
    "Authority": ["expert", "official", "government", "scientists", "study", "research"],
    "Guilt": ["shame", "selfish", "responsible", "blame", "fault"]
}

propaganda_patterns = [
    "they don't want you to know",
    "wake up",
    "share before deleted",
    "mainstream media lies",
    "hidden truth"
]

# =============================
# TRANSFORMER FUNCTION
# =============================
def transformer_fake_real(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    fake_score = float(probs[0][0])
    real_score = float(probs[0][1])

    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(fake_score, real_score)

    return label, confidence

# =============================
# ANALYSIS FUNCTION
# =============================
def analyze_text(text):
    fake_label, fake_confidence = transformer_fake_real(text)

    trigger_counts = {}
    highlighted_text = text
    total_triggers = 0

    for cat, words in trigger_categories.items():
        count = sum(word in text.lower() for word in words)
        trigger_counts[cat] = count
        total_triggers += count

        for word in words:
            highlighted_text = re.sub(
                f"(?i){word}",
                f"<span class='highlight'>{word}</span>",
                highlighted_text
            )

    propaganda_detected = any(p in text.lower() for p in propaganda_patterns)
    sentiment = TextBlob(text).sentiment.polarity

    base = 5
    trigger_score = total_triggers * 10
    fake_boost = int(fake_label=="FAKE") * int(fake_confidence*20)
    sentiment_boost = 15 if sentiment < -0.5 else 0
    propaganda_boost = 20 if propaganda_detected else 0

    score = min(base + trigger_score + fake_boost + sentiment_boost + propaganda_boost, 100)

    if score > 70:
        risk = "🚨 High Psychological Manipulation"
    elif score > 40:
        risk = "⚠ Moderate Manipulation"
    else:
        risk = "✅ Low Manipulation"

    explanation = f"""
**Transformer Fake/Real:** {fake_label} (Confidence: {fake_confidence:.2f})  
Emotional triggers: {total_triggers}  
Propaganda detected: {propaganda_detected}  
Sentiment polarity: {sentiment:.2f}  
Manipulation score: {score}/100
"""

    counter_message = """
**Counter-message:**  
- Cross-check information with credible sources.  
- Emotional or urgent language does not guarantee truth.  
- Evaluate claims critically before sharing.
"""

    manipulative_words = [
        w for words in trigger_categories.values() for w in words if w in text.lower()
    ]

    return score, trigger_counts, highlighted_text, explanation, counter_message, risk, manipulative_words

# =============================
# PDF GENERATION
# =============================
def generate_pdf(text, score, trigger_counts, explanation, counter_message, risk):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    style = getSampleStyleSheet()

    elements.append(Paragraph("<b>MindShield AI Report</b>", style['Title']))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"Input Text: {text}", style['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"Manipulation Score: {score}/100", style['Normal']))
    elements.append(Paragraph(f"Risk Level: {risk}", style['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    for k,v in trigger_counts.items():
        elements.append(Paragraph(f"{k}: {v}", style['Normal']))

    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("AI Explanation:", style['Heading2']))
    elements.append(Paragraph(explanation, style['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Counter-message:", style['Heading2']))
    elements.append(Paragraph(counter_message, style['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =============================
# UI INPUT
# =============================
text_input = st.text_area("Paste the content here:", height=200)

if st.button("🔍 Analyze"):

    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        score, trigger_counts, highlighted_text, explanation, counter_message, risk, manipulative_words = analyze_text(text_input)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#FF4B4B'}},
            title={'text':"Manipulation Score"}
        ))
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

        # Bar Chart
        fig2, ax = plt.subplots()
        ax.bar(list(trigger_counts.keys()), list(trigger_counts.values()), color="#00F5FF")
        ax.set_title("Trigger Words Distribution")
        st.pyplot(fig2)

        # Explanation
        with st.expander("🤖 AI Explanation"):
            st.markdown(explanation)

        with st.expander("🛡 Counter-message"):
            st.markdown(counter_message)

        # Highlighted Text
        st.markdown("### 🔎 Highlighted Manipulative Words")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # Word List
        st.markdown("### 📝 Manipulative Words List")
        st.write(", ".join(set(manipulative_words)))

        # PDF Download
        pdf_buf = generate_pdf(text_input, score, trigger_counts, explanation, counter_message, risk)
        st.download_button("📄 Download PDF Report", pdf_buf, file_name="MindShield_Report.pdf", mime="application/pdf")

        st.markdown("</div>", unsafe_allow_html=True)
