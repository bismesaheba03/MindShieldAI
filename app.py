import streamlit as st
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors as rl_colors
from io import BytesIO
import math

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MindShield AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:      #07090f;
  --surface: #0f1220;
  --card:    #141826;
  --border:  #1e2540;
  --accent:  #00e5ff;
  --accent2: #7c3aed;
  --danger:  #ff3c5c;
  --warn:    #f59e0b;
  --safe:    #10b981;
  --text:    #e2e8f0;
  --muted:   #64748b;
  --radius:  16px;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
  padding: 1.5rem 1rem 3rem;
  max-width: 1200px;
  margin: auto;
}

/* Hero */
.hero { text-align:center; padding:2.5rem 1rem 1.5rem; }
.hero-badge {
  display:inline-block;
  background:linear-gradient(135deg,var(--accent2),var(--accent));
  color:#fff;
  font-family:'Syne',sans-serif;
  font-size:0.72rem;
  letter-spacing:0.15em;
  text-transform:uppercase;
  padding:4px 14px;
  border-radius:999px;
  margin-bottom:1rem;
}
.hero-title {
  font-family:'Syne',sans-serif;
  font-size:clamp(1.8rem,5vw,3rem);
  font-weight:800;
  background:linear-gradient(135deg,#ffffff 0%,var(--accent) 60%,var(--accent2) 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  line-height:1.15;
  margin-bottom:0.6rem;
}
.hero-sub {
  color:var(--muted);
  font-size:clamp(0.85rem,2.5vw,1rem);
  max-width:600px;
  margin:0 auto 1.5rem;
  line-height:1.6;
}

/* Cards */
.ms-card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.4rem 1.6rem;
  margin-bottom:1.2rem;
  position:relative;
  overflow:hidden;
}
.ms-card::before {
  content:'';
  position:absolute;
  top:0;left:0;right:0;
  height:2px;
  background:linear-gradient(90deg,var(--accent2),var(--accent));
}
.card-title {
  font-family:'Syne',sans-serif;
  font-size:0.95rem;
  font-weight:700;
  letter-spacing:0.06em;
  text-transform:uppercase;
  color:var(--accent);
  margin-bottom:0.8rem;
}

/* Risk badges */
.risk-high   { background:#ff3c5c22; border:1px solid var(--danger); color:var(--danger);  border-radius:10px; padding:0.6rem 1rem; font-weight:700; text-align:center; font-size:1.1rem; font-family:'Syne',sans-serif; }
.risk-medium { background:#f59e0b22; border:1px solid var(--warn);   color:var(--warn);    border-radius:10px; padding:0.6rem 1rem; font-weight:700; text-align:center; font-size:1.1rem; font-family:'Syne',sans-serif; }
.risk-low    { background:#10b98122; border:1px solid var(--safe);   color:var(--safe);    border-radius:10px; padding:0.6rem 1rem; font-weight:700; text-align:center; font-size:1.1rem; font-family:'Syne',sans-serif; }

/* Highlight */
.highlight-word {
  background:linear-gradient(135deg,#ff3c5c55,#ff3c5c33);
  border:1px solid #ff3c5c66;
  border-radius:4px;
  padding:1px 4px;
  color:#ff8fa3;
  font-weight:600;
}

/* Advice */
.advice-item {
  display:flex;
  align-items:flex-start;
  gap:10px;
  padding:0.7rem 0;
  border-bottom:1px solid var(--border);
  font-size:0.92rem;
  line-height:1.5;
}
.advice-item:last-child { border-bottom:none; }
.advice-icon { font-size:1.1rem; flex-shrink:0; margin-top:1px; }

/* Counter */
.counter-block {
  background:#0d1f1a;
  border-left:3px solid var(--safe);
  border-radius:0 10px 10px 0;
  padding:0.7rem 1rem;
  margin:0.5rem 0;
  font-size:0.9rem;
  color:#a7f3d0;
  line-height:1.55;
}

/* Metric boxes */
.metric-box {
  background:#0f1220;
  border:1px solid var(--border);
  border-radius:12px;
  padding:0.8rem 1rem;
  text-align:center;
}
.metric-val {
  font-family:'Syne',sans-serif;
  font-size:1.5rem;
  font-weight:800;
  color:var(--accent);
}
.metric-label { font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }

/* Score bars */
.score-bar-bg { background:#1e2540; border-radius:999px; height:10px; margin:6px 0; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:999px; }

/* Extreme sentence */
.extreme-sent {
  border-left:3px solid #ff3c5c;
  padding:6px 12px;
  margin:6px 0;
  font-size:0.9rem;
  color:#fca5a5;
}

/* Inputs */
.stTextArea textarea {
  background:#0f1220 !important;
  border:1.5px solid var(--border) !important;
  border-radius:12px !important;
  color:var(--text) !important;
  font-family:'DM Sans',sans-serif !important;
  font-size:0.95rem !important;
}
.stTextArea textarea:focus {
  border-color:var(--accent) !important;
  box-shadow:0 0 0 2px #00e5ff22 !important;
}

/* Buttons */
.stButton > button {
  background:linear-gradient(135deg,var(--accent2),var(--accent)) !important;
  color:#fff !important;
  border:none !important;
  border-radius:12px !important;
  font-family:'Syne',sans-serif !important;
  font-weight:700 !important;
  font-size:0.95rem !important;
  letter-spacing:0.05em !important;
  padding:0.65rem 2rem !important;
  width:100% !important;
}
.stDownloadButton > button {
  background:#1e2540 !important;
  color:var(--accent) !important;
  border:1.5px solid var(--accent) !important;
  border-radius:12px !important;
  font-family:'Syne',sans-serif !important;
  font-weight:700 !important;
  width:100% !important;
}

.ms-divider { border:none; border-top:1px solid var(--border); margin:1rem 0; }

@media (max-width:640px) {
  .ms-card { padding:1rem; }
  .hero-title { font-size:1.7rem; }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MANIPULATION TECHNIQUE DEFINITIONS
# ──────────────────────────────────────────────
TECHNIQUES = {
    "Fear Appeal": {
        "keywords": [
            "danger", "threat", "risk", "destroy", "crisis", "attack",
            "catastrophe", "disaster", "death", "deadly", "fatal",
            "collapse", "chaos", "terror", "horrifying", "devastating",
            "lethal", "annihilate", "obliterate", "meltdown", "apocalypse",
        ],
        "weight": 3,
        "color": "#ff3c5c",
        "icon": "😨",
        "description": "Exploits fear to bypass rational thinking and push the audience toward a specific reaction.",
        "counter": [
            "Ask yourself: is this fear based on verified facts or just emotional framing?",
            "Fear-based claims often exaggerate probability. Look up base rates and real statistics.",
            "Search for data from peer-reviewed studies or official reports before accepting risk claims.",
        ],
    },
    "False Urgency": {
        "keywords": [
            "now", "immediately", "urgent", "limited", "hurry", "act fast",
            "last chance", "expires", "deadline", "before its too late",
            "don't wait", "time running out", "only hours left", "tonight only",
            "final warning", "breaking now",
        ],
        "weight": 3,
        "color": "#f59e0b",
        "icon": "⏰",
        "description": "Creates artificial time pressure so you act before you can think critically.",
        "counter": [
            "Pause. Genuine emergencies rarely require instant, uninformed decisions.",
            "If someone insists you cannot take time to verify, that itself is a red flag.",
            "Legitimate opportunities can withstand a 24-hour verification period.",
        ],
    },
    "Authority Bait": {
        "keywords": [
            "expert", "official", "government", "scientists say", "studies show",
            "research proves", "doctors say", "according to experts", "professor",
            "insider", "whistleblower", "top secret", "classified", "sources say",
        ],
        "weight": 2,
        "color": "#8b5cf6",
        "icon": "🎓",
        "description": "Vaguely invokes authority figures or 'studies' without citations to appear credible.",
        "counter": [
            "Demand the actual source: journal name, date, authors. Vague authority is no authority.",
            "Check if the cited institution actually made that claim on their official channels.",
            "Scientific consensus is built from many reproducible studies, not a single report.",
        ],
    },
    "Guilt & Shame": {
        "keywords": [
            "shame", "selfish", "responsible", "blame", "fault", "disgrace",
            "coward", "failure", "disappoint", "letting down", "your fault",
            "irresponsible", "negligent", "pathetic", "ignorant",
        ],
        "weight": 3,
        "color": "#ec4899",
        "icon": "😔",
        "description": "Induces guilt or shame to override your judgment and coerce compliance.",
        "counter": [
            "Healthy discourse does not require you to feel ashamed for asking questions.",
            "Responsibility framing should come with evidence, not emotional attack.",
            "If content makes you feel guilty without explaining why, question its motive.",
        ],
    },
    "Us vs. Them": {
        "keywords": [
            "they", "them", "elite", "globalists", "deep state", "enemy",
            "traitor", "puppet", "regime", "cabal", "establishment", "outsider",
            "our side", "their agenda", "against us", "real patriots",
            "true believers", "sheeple", "the masses",
        ],
        "weight": 4,
        "color": "#f97316",
        "icon": "⚔️",
        "description": "Divides people into opposing in-groups and out-groups to create tribalism.",
        "counter": [
            "Most complex issues cannot be reduced to two sides. Look for nuanced perspectives.",
            "Who benefits from you seeing a certain group as the enemy?",
            "In-group/out-group language is a known radicalization gateway — stay alert.",
        ],
    },
    "Conspiracy Framing": {
        "keywords": [
            "wake up", "hidden agenda", "secret", "cover up", "they dont want you to know",
            "suppressed", "silenced", "censored", "banned", "forbidden",
            "share before deleted", "mainstream media lies", "open your eyes",
            "the truth is", "what theyre hiding", "deep cover",
        ],
        "weight": 5,
        "color": "#14b8a6",
        "icon": "🕵️",
        "description": "Uses conspiracy language to make unverified claims seem suppressed and therefore credible.",
        "counter": [
            "If evidence for a claim is only found outside mainstream institutions, question why.",
            "The 'hidden truth' framing is self-sealing: any denial becomes part of the conspiracy.",
            "Real whistleblowing involves verifiable documents, not just social media posts.",
        ],
    },
    "False Certainty": {
        "keywords": [
            "always", "never", "everyone knows", "nobody believes", "obviously",
            "clearly", "undeniably", "absolute truth", "100%", "guaranteed",
            "no doubt", "without question", "fact is", "proven fact",
        ],
        "weight": 2,
        "color": "#06b6d4",
        "icon": "🔒",
        "description": "Presents opinions or interpretations as absolute, indisputable facts.",
        "counter": [
            "Absolute language ('always', 'never') rarely holds up in complex real-world situations.",
            "Ask for the evidence behind 'obvious' claims — obvious things still require proof.",
            "Strong certainty on contested topics is a signal to slow down and investigate.",
        ],
    },
    "Emotional Overload": {
        "keywords": [
            "heartbreaking", "outrageous", "disgusting", "shocking", "unbelievable",
            "horrifying", "devastating", "enraging", "infuriating", "appalling",
            "sickening", "monstrous", "vile", "evil", "demonic",
        ],
        "weight": 2,
        "color": "#a855f7",
        "icon": "🌪️",
        "description": "Floods content with intense emotional language to overwhelm critical thinking.",
        "counter": [
            "Strong emotions are a signal to slow down, not speed up your response.",
            "Ask: is this content giving me facts, or is it primarily trying to make me feel something?",
            "Emotional intensity in a message is not evidence of the truth of its claims.",
        ],
    },
    "Social Proof Manipulation": {
        "keywords": [
            "everyone is doing", "millions agree", "people are waking up",
            "the world knows", "spreading fast", "going viral", "nobody is talking about",
            "masses are rising", "the public agrees", "popular opinion",
        ],
        "weight": 2,
        "color": "#22d3ee",
        "icon": "👥",
        "description": "Fabricates or exaggerates consensus to pressure conformity.",
        "counter": [
            "Popularity is not the same as accuracy. Misinformation spreads fast too.",
            "Verify 'viral' claims independently before accepting their social proof.",
            "Ask what evidence underlies the claim of widespread agreement.",
        ],
    },
    "Black & White Thinking": {
        "keywords": [
            "either youre with us", "or youre against us", "no middle ground",
            "only two options", "there is no choice", "you must choose",
            "wake up or remain asleep", "truth or lies", "freedom or slavery",
        ],
        "weight": 3,
        "color": "#6366f1",
        "icon": "🎭",
        "description": "Presents false dilemmas to prevent nuanced thinking.",
        "counter": [
            "Binary framing is almost always an oversimplification of a complex issue.",
            "Most political, social, or scientific issues exist on a spectrum.",
            "Reject the premise: look for the third, fourth, or fifth option.",
        ],
    },
}

PROPAGANDA_PHRASES = [
    "they don't want you to know",
    "wake up sheeple",
    "share before deleted",
    "mainstream media lies",
    "hidden truth",
    "open your eyes",
    "the real truth",
    "what they're not telling you",
    "do your own research",
    "question everything you've been told",
    "deep state agenda",
    "new world order",
    "follow the money",
    "the elite don't want this",
    "shadow government",
]

LOGICAL_FALLACIES = {
    "Ad Hominem": [
        "stupid", "idiot", "moron", "liar", "hypocrite", "corrupt puppet",
        "paid shill", "brainwashed", "gullible fool",
    ],
    "Slippery Slope": [
        "will lead to", "next thing you know", "soon they will",
        "it starts with", "before long", "the beginning of the end",
    ],
    "Bandwagon": [
        "join the movement", "be part of history", "don't be left behind",
        "the smart ones already know", "those who are awake",
    ],
    "Straw Man": [
        "so you're saying", "you must believe", "people like you think",
        "that means you support", "by that logic you want",
    ],
}


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "Giyaseddin/distilbert-base-cased-finetuned-fake-and-real-news-dataset"
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSequenceClassification.from_pretrained(model_name)
    mod.eval()
    return tok, mod


# ──────────────────────────────────────────────
# TRANSFORMER PREDICTION
# ──────────────────────────────────────────────
def transformer_predict(text, tokenizer, model):
    """Chunk-aware fake/real detection. Returns (label, confidence, fake_p, real_p)."""
    words = text.split()
    chunk_size = 400
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    if not chunks:
        chunks = [text]

    fake_probs, real_probs = [], []
    for chunk in chunks:
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        fake_probs.append(float(probs[0]))
        real_probs.append(float(probs[1]))

    fake_p = float(np.mean(fake_probs))
    real_p = float(np.mean(real_probs))
    label = "FAKE" if fake_p > real_p else "REAL"
    return label, max(fake_p, real_p), fake_p, real_p


# ──────────────────────────────────────────────
# CORE ANALYSIS
# ──────────────────────────────────────────────
def analyze_text(text, tokenizer, model):
    text_lower = text.lower()

    # Transformer
    fake_label, fake_confidence, fake_p, real_p = transformer_predict(text, tokenizer, model)

    # Technique detection
    technique_hits = {}
    all_trigger_words = set()
    for tech, info in TECHNIQUES.items():
        matched = [
            kw for kw in info["keywords"]
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
        ]
        if matched:
            technique_hits[tech] = matched
            all_trigger_words.update(matched)

    # Propaganda
    propaganda_hits = [p for p in PROPAGANDA_PHRASES if p in text_lower]

    # Logical fallacies
    fallacy_hits = {}
    for fname, fwords in LOGICAL_FALLACIES.items():
        m = [w for w in fwords if w in text_lower]
        if m:
            fallacy_hits[fname] = m

    # Sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Sentence-level extremism
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
    extreme_sentences = [
        s for s in sentences
        if abs(TextBlob(s).sentiment.polarity) > 0.6 and TextBlob(s).sentiment.subjectivity > 0.6
    ]

    # Typography signals
    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0
    )
    exclaim_density = text.count("!") / max(len(sentences), 1)

    # ── SCORING (weighted multi-signal) ──
    tech_raw = sum(TECHNIQUES[t]["weight"] * len(kws) for t, kws in technique_hits.items())
    tech_score = min(40, tech_raw * 5)
    fake_model_score = fake_p * 25
    sentiment_score = abs(sentiment) * 10 if abs(sentiment) > 0.3 else 0
    subjectivity_score = subjectivity * 8 if subjectivity > 0.5 else 0
    propaganda_score = min(15, len(propaganda_hits) * 5)
    fallacy_score = min(10, len(fallacy_hits) * 3)
    caps_score = min(5, caps_ratio * 20)
    exclaim_score = min(5, exclaim_density * 5)

    score = math.floor(
        5 + tech_score + fake_model_score
        + sentiment_score + subjectivity_score
        + propaganda_score + fallacy_score
        + caps_score + exclaim_score
    )
    score = max(0, min(100, score))

    # Risk level
    if score >= 70:
        risk, risk_class, risk_color = "🚨 HIGH MANIPULATION RISK", "risk-high", "#ff3c5c"
    elif score >= 40:
        risk, risk_class, risk_color = "⚠️ MODERATE MANIPULATION RISK", "risk-medium", "#f59e0b"
    else:
        risk, risk_class, risk_color = "✅ LOW MANIPULATION RISK", "risk-low", "#10b981"

    # ── Dynamic Explanation ──
    parts = []
    if fake_label == "FAKE" and fake_confidence > 0.65:
        parts.append(
            f"**Fake News Model:** The transformer model classified this content as likely **fabricated or misleading** "
            f"(confidence: {fake_confidence:.0%}). Language patterns and narrative structure deviate significantly from authentic reporting."
        )
    elif fake_label == "FAKE":
        parts.append(
            f"**Fake News Model:** Content was flagged as **potentially misleading** (confidence: {fake_confidence:.0%}). "
            f"Several linguistic markers suggest distorted framing, though further verification is recommended."
        )
    else:
        parts.append(
            f"**Fake News Model:** Content appears broadly **authentic in structure** (confidence: {fake_confidence:.0%}). "
            f"However, factual accuracy should always be independently verified regardless of this score."
        )

    if technique_hits:
        top = sorted(technique_hits.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        tech_str = ", ".join(f"**{t}** ({len(k)} triggers)" for t, k in top)
        parts.append(
            f"**Primary Manipulation Techniques:** {tech_str}. "
            f"These exploit cognitive biases to influence beliefs without presenting rational evidence."
        )

    if propaganda_hits:
        parts.append(
            f"**Propaganda Markers Found:** {len(propaganda_hits)} known phrase(s) detected. "
            f"These phrases are engineered to prime distrust in credible institutions and make alternative narratives feel revelatory."
        )

    if fallacy_hits:
        parts.append(
            f"**Logical Fallacies:** {', '.join(f'**{f}**' for f in fallacy_hits)}. "
            f"These substitute emotional persuasion for logical argument."
        )

    if sentiment < -0.4:
        parts.append(
            f"**Strongly Negative Sentiment (polarity: {sentiment:.2f}):** "
            f"Persistently negative framing narrows perceived options and increases psychological stress in the reader."
        )
    elif sentiment > 0.4 and score > 40:
        parts.append(
            f"**Suspiciously Positive Sentiment ({sentiment:.2f}):** "
            f"Unrealistically upbeat framing at high manipulation scores can signal false promises or persuasion tactics."
        )

    if subjectivity > 0.65:
        parts.append(
            f"**High Subjectivity ({subjectivity:.0%}):** Most of this content is opinion and personal interpretation "
            f"rather than verifiable fact, yet it may be presented as objective truth."
        )

    if caps_ratio > 0.25:
        parts.append(
            f"**Aggressive Typography ({caps_ratio:.0%} uppercase):** Heavy capitalisation is a visual shouting technique "
            f"that bypasses calm reading and triggers emotional arousal."
        )

    if not parts:
        parts.append("No significant manipulation signals detected. Content appears informational in intent. Always verify claims independently.")

    explanation = "\n\n".join(parts)

    # ── Dynamic Counter-messages ──
    raw_counters = []
    for tech in technique_hits:
        raw_counters.extend(TECHNIQUES[tech]["counter"])
    seen, unique_counters = set(), []
    for c in raw_counters:
        if c not in seen:
            seen.add(c)
            unique_counters.append(c)
    unique_counters = unique_counters[:6]
    if not unique_counters:
        unique_counters = [
            "Cross-reference any factual claims with multiple independent credible sources.",
            "Consider the source's motivation: who benefits if you believe this?",
            "Share only after verification — even well-intentioned misinformation causes real harm.",
        ]

    # ── Advice ──
    advice = []
    if score >= 70:
        advice.append(("🔴", "Do NOT share this content without rigorous verification from at least 3 independent credentialed sources."))
        advice.append(("🔴", "Consider why this content provokes such a strong emotional reaction — that reaction may itself be the manipulation."))
    elif score >= 40:
        advice.append(("🟡", "Approach with caution. Verify key claims with primary sources before acting on or sharing this content."))
    if fake_p > 0.6:
        advice.append(("🔴", "The AI model detected likely fabrication. Look for the original event or study referenced — it may not exist or may be misrepresented."))
    elif fake_p > 0.4:
        advice.append(("🟡", "Possible content distortion detected. Check original sources rather than relying on this version."))
    if "Conspiracy Framing" in technique_hits:
        advice.append(("🔴", "Conspiracy framing is present. Real suppressed information comes with verifiable documents. Ask for primary evidence."))
    if "Us vs. Them" in technique_hits:
        advice.append(("🟡", "Tribal division tactics detected. Seek perspectives from multiple sides before forming strong opinions."))
    if "Fear Appeal" in technique_hits:
        advice.append(("🟡", "Fear is being used as a persuasion tool. Look up the actual statistical likelihood of the described threat."))
    if "False Urgency" in technique_hits:
        advice.append(("🟡", "Artificial urgency discourages verification. Real decisions rarely need to be made in minutes."))
    if sentiment < -0.5:
        advice.append(("🟡", "Strongly negative content can distort your threat perception. Take a break before reacting."))
    advice.append(("🟢", "Use fact-checking sites: Snopes, PolitiFact, FactCheck.org, AFP Fact Check, or AP Fact Check."))
    advice.append(("🟢", "Check for bylines, publication dates, and institutional affiliations when evaluating any piece of content."))
    advice.append(("🟢", "Ask: 'How would I react if the subject were someone I support?' — consistency reveals fair vs. biased reporting."))

    # ── Highlighted text ──
    highlighted = text
    for word in sorted(all_trigger_words, key=len, reverse=True):
        pattern = re.compile(r"\b(" + re.escape(word) + r")\b", re.IGNORECASE)
        highlighted = pattern.sub(r"<span class='highlight-word'>\1</span>", highlighted)

    return {
        "score": score,
        "risk": risk,
        "risk_class": risk_class,
        "risk_color": risk_color,
        "fake_label": fake_label,
        "fake_confidence": fake_confidence,
        "fake_p": fake_p,
        "real_p": real_p,
        "technique_hits": technique_hits,
        "propaganda_hits": propaganda_hits,
        "fallacy_hits": fallacy_hits,
        "sentiment": sentiment,
        "subjectivity": subjectivity,
        "extreme_sentences": extreme_sentences,
        "caps_ratio": caps_ratio,
        "explanation": explanation,
        "counter_messages": unique_counters,
        "advice": advice,
        "highlighted_text": highlighted,
        "all_trigger_words": sorted(all_trigger_words),
    }


# ──────────────────────────────────────────────
# PDF GENERATION
# ──────────────────────────────────────────────
def generate_pdf(text, result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T2", parent=styles["Title"], fontSize=20, spaceAfter=6,
        textColor=rl_colors.HexColor("#00e5ff"),
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13,
        spaceBefore=14, spaceAfter=4, textColor=rl_colors.HexColor("#7c3aed"),
    )
    body = ParagraphStyle("B2", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=4)

    els = []
    els.append(Paragraph("MindShield AI — Manipulation Analysis Report", title_style))
    els.append(Spacer(1, 0.15 * inch))
    els.append(Paragraph(f"Manipulation Score: {result['score']}/100", h2_style))
    els.append(Paragraph(f"Risk Level: {result['risk']}", body))
    els.append(Paragraph(f"Fake News Model: {result['fake_label']} ({result['fake_confidence']:.0%} confidence)", body))
    els.append(Paragraph(f"Sentiment: {result['sentiment']:.2f} | Subjectivity: {result['subjectivity']:.0%}", body))
    els.append(Spacer(1, 0.1 * inch))

    els.append(Paragraph("Analysed Text", h2_style))
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    els.append(Paragraph(safe[:1500] + ("..." if len(text) > 1500 else ""), body))
    els.append(Spacer(1, 0.1 * inch))

    els.append(Paragraph("Manipulation Techniques Detected", h2_style))
    if result["technique_hits"]:
        for tech, kws in result["technique_hits"].items():
            els.append(Paragraph(f"<b>{tech}:</b> {', '.join(kws)}", body))
    else:
        els.append(Paragraph("None detected.", body))
    els.append(Spacer(1, 0.1 * inch))

    if result["propaganda_hits"]:
        els.append(Paragraph("Propaganda Phrases", h2_style))
        for p in result["propaganda_hits"]:
            els.append(Paragraph(f"• {p}", body))
        els.append(Spacer(1, 0.1 * inch))

    els.append(Paragraph("AI Explanation", h2_style))
    clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result["explanation"])
    for line in clean.split("\n\n"):
        if line.strip():
            els.append(Paragraph(line.strip(), body))
            els.append(Spacer(1, 0.05 * inch))

    els.append(Paragraph("Counter-Messages", h2_style))
    for c in result["counter_messages"]:
        els.append(Paragraph(f"• {c}", body))
    els.append(Spacer(1, 0.1 * inch))

    els.append(Paragraph("Personalised Advice", h2_style))
    for icon, adv in result["advice"]:
        els.append(Paragraph(f"{icon} {adv}", body))

    doc.build(els)
    buffer.seek(0)
    return buffer


# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────
def make_gauge(score, risk_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 28, "color": "#e2e8f0", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1e2540",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": risk_color, "thickness": 0.25},
            "bgcolor": "#0f1220",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "#10b98115"},
                {"range": [40, 70],  "color": "#f59e0b15"},
                {"range": [70, 100], "color": "#ff3c5c15"},
            ],
            "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.8, "value": score},
        },
        title={"text": "Manipulation Score", "font": {"size": 14, "color": "#64748b", "family": "Syne"}},
    ))
    fig.update_layout(
        height=260, margin={"t": 30, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="#141826", font_color="#e2e8f0",
    )
    return fig


def make_technique_bar(technique_hits):
    names = list(technique_hits.keys())
    counts = [len(v) for v in technique_hits.values()]
    colors_list = [TECHNIQUES[n]["color"] for n in names]
    fig = go.Figure(go.Bar(
        x=counts, y=names, orientation="h",
        marker={"color": colors_list, "opacity": 0.85},
        text=counts, textposition="outside",
        textfont={"color": "#e2e8f0", "size": 11},
    ))
    fig.update_layout(
        height=max(200, len(names) * 44),
        margin={"t": 10, "b": 10, "l": 10, "r": 40},
        paper_bgcolor="#141826", plot_bgcolor="#141826",
        xaxis={"showgrid": True, "gridcolor": "#1e2540",
               "tickfont": {"color": "#64748b"},
               "title": {"text": "Keyword count", "font": {"color": "#64748b", "size": 11}}},
        yaxis={"tickfont": {"color": "#e2e8f0", "size": 11}, "automargin": True},
    )
    return fig


def make_radar(result):
    cats = [
        "Fear Appeal", "False Urgency", "Conspiracy Framing",
        "Us vs. Them", "Emotional Overload", "False Certainty",
    ]
    vals = [
        min(10, len(result["technique_hits"].get(c, [])) * 2)
        for c in cats
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="#7c3aed22",
        line={"color": "#00e5ff", "width": 2},
        marker={"color": "#00e5ff", "size": 5},
    ))
    fig.update_layout(
        polar={
            "bgcolor": "#0f1220",
            "radialaxis": {"visible": True, "range": [0, 10],
                           "gridcolor": "#1e2540", "tickfont": {"color": "#64748b", "size": 9}},
            "angularaxis": {"tickfont": {"color": "#c4cad8", "size": 10}, "gridcolor": "#1e2540"},
        },
        height=300, margin={"t": 20, "b": 20, "l": 20, "r": 20},
        paper_bgcolor="#141826", font_color="#e2e8f0",
    )
    return fig


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Powered by Transformer AI + Multi-Signal Analysis</div>
    <div class="hero-title">🧠 MindShield AI</div>
    <div class="hero-sub">
        Uncover hidden psychological manipulation, fake news patterns, cognitive traps,
        and propaganda techniques in any piece of text — with full AI-powered explanation.
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading AI model — first run may take 30–60 seconds…"):
    try:
        tokenizer, transformer_model = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        model_loaded = False

# Input
st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📋 Content to Analyse</div>", unsafe_allow_html=True)
text_input = st.text_area(
    label="",
    placeholder="Paste a news article, social media post, email, advertisement, or any text here…",
    height=220,
    label_visibility="collapsed",
)
analyze_btn = st.button("🔍 Analyse for Manipulation", disabled=not model_loaded)
st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text to analyse.")
    else:
        with st.spinner("Running deep multi-signal analysis…"):
            result = analyze_text(text_input, tokenizer, transformer_model)

        # Risk banner
        st.markdown(f"<div class='{result['risk_class']}'>{result['risk']}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, str(result["score"]), "Manip. Score"),
            (c2, str(len(result["technique_hits"])), "Techniques"),
            (c3, f"{result['fake_confidence']:.0%}", f"{result['fake_label']} Conf."),
            (c4, f"{result['subjectivity']:.0%}", "Subjectivity"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{val}</div>"
                    f"<div class='metric-label'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        gcol, rcol = st.columns(2)
        with gcol:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📊 Manipulation Gauge</div>", unsafe_allow_html=True)
            st.plotly_chart(make_gauge(result["score"], result["risk_color"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with rcol:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🕸️ Technique Radar</div>", unsafe_allow_html=True)
            st.plotly_chart(make_radar(result),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Technique breakdown bar
        if result["technique_hits"]:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🎯 Manipulation Technique Breakdown</div>", unsafe_allow_html=True)
            st.plotly_chart(make_technique_bar(result["technique_hits"]),
                            use_container_width=True, config={"displayModeBar": False})
            for tech, kws in result["technique_hits"].items():
                info = TECHNIQUES[tech]
                st.markdown(
                    f"**{info['icon']} {tech}** — {info['description']}  \n"
                    f"*Detected keywords:* `{', '.join(kws)}`"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Fake news model detail
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🤖 Fake News Model Results</div>", unsafe_allow_html=True)
        fn1, fn2 = st.columns(2)
        with fn1:
            st.markdown(
                f"<div style='text-align:center;padding:0.5rem;'>"
                f"<div style='font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;'>FAKE Probability</div>"
                f"<div style='font-size:2rem;font-weight:800;color:#ff3c5c;'>{result['fake_p']:.0%}</div>"
                f"</div>", unsafe_allow_html=True)
        with fn2:
            st.markdown(
                f"<div style='text-align:center;padding:0.5rem;'>"
                f"<div style='font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;'>REAL Probability</div>"
                f"<div style='font-size:2rem;font-weight:800;color:#10b981;'>{result['real_p']:.0%}</div>"
                f"</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='score-bar-bg'><div class='score-bar-fill' style='width:{result['fake_p']*100:.1f}%;background:#ff3c5c;'></div></div>"
            f"<div class='score-bar-bg'><div class='score-bar-fill' style='width:{result['real_p']*100:.1f}%;background:#10b981;'></div></div>",
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # AI Explanation
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🧠 AI Explanation</div>", unsafe_allow_html=True)
        with st.expander("View Full Analysis", expanded=True):
            st.markdown(result["explanation"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Counter-messages
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🛡️ Counter-Messages for This Specific Content</div>", unsafe_allow_html=True)
        for c in result["counter_messages"]:
            st.markdown(f"<div class='counter-block'>💬 {c}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Advice
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>💡 Personalised Advice</div>", unsafe_allow_html=True)
        for icon, adv_text in result["advice"]:
            st.markdown(
                f"<div class='advice-item'><div class='advice-icon'>{icon}</div><div>{adv_text}</div></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Highlighted text
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔎 Highlighted Trigger Words</div>", unsafe_allow_html=True)
        with st.expander("View Annotated Text", expanded=False):
            st.markdown(
                f"<div style='line-height:1.9;font-size:0.95rem;white-space:pre-wrap;'>{result['highlighted_text']}</div>",
                unsafe_allow_html=True)
        if result["all_trigger_words"]:
            st.markdown("**All flagged words:** " + " ".join(f"`{w}`" for w in result["all_trigger_words"]))
        st.markdown("</div>", unsafe_allow_html=True)

        # Propaganda & fallacies
        if result["propaganda_hits"] or result["fallacy_hits"]:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📢 Propaganda Phrases & Logical Fallacies</div>", unsafe_allow_html=True)
            if result["propaganda_hits"]:
                st.markdown("**Propaganda Phrases Detected:**")
                for p in result["propaganda_hits"]:
                    st.markdown(f"- `{p}`")
            if result["fallacy_hits"]:
                st.markdown("**Logical Fallacies Detected:**")
                for f, kws in result["fallacy_hits"].items():
                    st.markdown(f"- **{f}:** `{', '.join(kws)}`")
            st.markdown("</div>", unsafe_allow_html=True)

        # Extreme sentences
        if result["extreme_sentences"]:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>⚡ Most Emotionally Extreme Sentences</div>", unsafe_allow_html=True)
            with st.expander("View extreme sentences", expanded=False):
                for s in result["extreme_sentences"][:5]:
                    st.markdown(
                        f"<div class='extreme-sent'>{s}</div>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # PDF download
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_buf = generate_pdf(text_input, result)
        st.download_button(
            "📄 Download Full PDF Report",
            pdf_buf,
            file_name="MindShield_Report.pdf",
            mime="application/pdf",
        )

        st.markdown("<hr class='ms-divider'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:#64748b;font-size:0.8rem;'>"
            "MindShield AI • For educational use only • Always verify with primary sources"
            "</div>",
            unsafe_allow_html=True)
