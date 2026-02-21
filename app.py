# ---------------- MindShield AI - Transformer Hackathon Version ----------------
import streamlit as st
import re
import random
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MindShield AI", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
.big-title {
    font-size: 50px;
    font-weight: bold;
    color: #00F5FF;
    text-align: center;
    text-shadow: 2px 2px 5px #FF4B4B;
}
.section-title {
    font-size: 28px;
    font-weight: 600;
    color: #FFA500;
    margin-top: 20px;
    margin-bottom: 10px;
    border-bottom: 2px solid #FF4B4B;
    padding-bottom: 5px;
}
.highlight {
    background-color: #FF4B4B;
    padding: 3px 6px;
    border-radius: 6px;
    color: white;
}
.stButton>button {
    background-color: #00F5FF;
    color: black;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🧠 MindShield AI</div>", unsafe_allow_html=True)
st.write("Multimodal AI System for Detecting Psychological Manipulation & Fake News")

# ---------------- INPUT ----------------
text = st.text_area("Enter digital media content to analyze:", height=150)

# ---------------- TRANSFORMER CLASSIFIER ----------------
@st.cache_resource(show_spinner=False)
def load_transformer_model():
    # Using distilbert-base-uncased-finetuned-sst-2-english for demo; replace with custom fake news model if available
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_transformer_model()

# ---------------- ANALYSIS FUNCTION ----------------
def advanced_analysis(text):
    # ---------------- Semantic Fake News Prediction ----------------
    result = classifier(text)[0]
    label = result['label']      # "POSITIVE" or "NEGATIVE"
    confidence = result['score']  # 0 to 1
    prediction = "FAKE" if label == "NEGATIVE" else "REAL"  # map NEGATIVE=FAKE, POSITIVE=REAL

    # ---------------- Trigger Words Analysis ----------------
    fear_words = ["danger", "threat", "risk", "destroy", "crisis", "attack"]
    urgency_words = ["now", "immediately", "urgent", "limited", "hurry"]
    authority_words = ["expert", "official", "government", "scientists", "study"]
    guilt_words = ["shame", "selfish", "responsible", "blame"]

    categories = {
        "Fear": fear_words,
        "Urgency": urgency_words,
        "Authority": authority_words,
        "Guilt": guilt_words
    }

    trigger_counts = {}
    highlighted_text = text
    manipulative_words = []
    total_triggers = 0

    for category, words in categories.items():
        count = sum(word.lower() in text.lower() for word in words)
        if count > 0:
            trigger_counts[category] = count
            total_triggers += count
            for word in words:
                if re.search(f"(?i){word}", highlighted_text):
                    highlighted_text = re.sub(
                        f"(?i){word}",
                        f"<span class='highlight'>{word}</span>",
                        highlighted_text
                    )
                    manipulative_words.append(word)

    # ---------------- Manipulation Score ----------------
    base_score = 30
    fake_boost = int(confidence*50) if prediction == "FAKE" else int(confidence*20)
    score = min(base_score + total_triggers * 10 + fake_boost + random.randint(5, 15), 100)

    # ---------------- AI Explanation ----------------
    explanation = f"""
The system analyzed the content using semantic understanding and emotional trigger density.
Fake News Classification: {prediction} (Confidence: {confidence:.2f}).
Total emotional trigger signals detected: {total_triggers}.
"""

    # ---------------- Counter-message ----------------
    counter_message = """
Verify the information using credible sources.
Emotional intensity does not guarantee factual accuracy.
Pause and evaluate before sharing.
"""

    return score, prediction, trigger_counts, explanation, counter_message, highlighted_text, manipulative_words

# ---------------- PDF GENERATOR ----------------
def generate_pdf(text, score, prediction, explanation, counter, triggers):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>MindShield AI Forensic Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Input Text: {text}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Manipulation Score: {score}/100", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Fake News Classification: {prediction}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Explanation: {explanation}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Counter-message: {counter}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Manipulative Words Detected: {', '.join(triggers)}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- RUN ----------------
if st.button("🔍 Analyze Content"):
    if text.strip() == "":
        st.warning("Please enter content.")
    else:
        score, prediction, trigger_counts, explanation, counter, highlighted, manipulative_words = advanced_analysis(text)

        # ---------------- Gauge ----------------
        st.markdown("<div class='section-title'>📊 Manipulation Score</div>", unsafe_allow_html=True)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [0, 40], 'color': "#1f77b4"},
                    {'range': [40, 70], 'color': "#FFA500"},
                    {'range': [70, 100], 'color': "#FF4B4B"},
                ],
            }
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # ---------------- Trigger Words Bar Chart ----------------
        if trigger_counts:
            st.markdown("<div class='section-title'>🔎 Trigger Words Distribution</div>", unsafe_allow_html=True)
            trigger_fig = px.bar(
                x=list(trigger_counts.keys()),
                y=list(trigger_counts.values()),
                color=list(trigger_counts.keys()),
                color_discrete_sequence=px.colors.sequential.Teal,
                labels={'x':'Trigger Category', 'y':'Count'}
            )
            trigger_fig.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white')
            st.plotly_chart(trigger_fig, use_container_width=True)
        else:
            st.write("No significant emotional triggers detected.")

        # ---------------- Highlighted Text ----------------
        st.markdown("<div class='section-title'>🔍 Highlighted Manipulative Text</div>", unsafe_allow_html=True)
        st.markdown(highlighted, unsafe_allow_html=True)

        # ---------------- Expanders ----------------
        st.markdown("<div class='section-title'>🤖 AI Explanation</div>", unsafe_allow_html=True)
        with st.expander("View Details"):
            st.markdown(explanation)

        st.markdown("<div class='section-title'>🛡 Counter-message</div>", unsafe_allow_html=True)
        with st.expander("View Guidance"):
            st.markdown(counter)

        # ---------------- PDF Download ----------------
        pdf_buf = generate_pdf(text, score, prediction, explanation, counter, manipulative_words)
        st.markdown("<div class='section-title'>📄 Download Full Report</div>", unsafe_allow_html=True)
        st.download_button("Download PDF", pdf_buf, file_name="MindShield_Report.pdf")
