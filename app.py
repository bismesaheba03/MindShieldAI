import re
import math
import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO
from textblob import TextBlob
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors as rc

st.set_page_config(page_title="MindShield AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #07090f;
    --surface: #0f1220;
    --card: #141826;
    --border: #1e2540;
    --accent: #00e5ff;
    --purple: #7c3aed;
    --red: #ff3c5c;
    --yellow: #f59e0b;
    --green: #10b981;
    --text: #e2e8f0;
    --muted: #64748b;
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

.hero {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--purple), var(--accent));
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 5vw, 2.9rem);
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 55%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero p {
    color: var(--muted);
    font-size: clamp(0.85rem, 2.5vw, 1rem);
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.6;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--purple), var(--accent));
}
.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
}

.risk-high {
    background: #ff3c5c18;
    border: 1px solid var(--red);
    color: var(--red);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    text-align: center;
    font-size: 1.05rem;
    font-family: 'Syne', sans-serif;
}
.risk-medium {
    background: #f59e0b18;
    border: 1px solid var(--yellow);
    color: var(--yellow);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    text-align: center;
    font-size: 1.05rem;
    font-family: 'Syne', sans-serif;
}
.risk-low {
    background: #10b98118;
    border: 1px solid var(--green);
    color: var(--green);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    text-align: center;
    font-size: 1.05rem;
    font-family: 'Syne', sans-serif;
}

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--accent);
}
.metric-lbl {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}

.flagged-word {
    background: linear-gradient(135deg, #ff3c5c44, #ff3c5c22);
    border: 1px solid #ff3c5c55;
    border-radius: 4px;
    padding: 1px 4px;
    color: #ff8fa3;
    font-weight: 600;
}

.counter-msg {
    background: #0d1f1a;
    border-left: 3px solid var(--green);
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1rem;
    margin: 0.45rem 0;
    font-size: 0.9rem;
    color: #a7f3d0;
    line-height: 1.55;
}

.advice-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
    line-height: 1.5;
}
.advice-row:last-child { border-bottom: none; }

.extreme-sent {
    border-left: 3px solid var(--red);
    padding: 5px 12px;
    margin: 5px 0;
    font-size: 0.9rem;
    color: #fca5a5;
}

.stTextArea textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px #00e5ff22 !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--purple), var(--accent)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
}
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    width: 100% !important;
}

hr.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

@media (max-width: 640px) {
    .card { padding: 1rem; }
    .hero h1 { font-size: 1.7rem; }
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# manipulation technique definitions
# ------------------------------------------------------------------
TECHNIQUES = {
    "Fear Appeal": {
        "keywords": [
            "danger", "threat", "risk", "destroy", "crisis", "attack",
            "catastrophe", "disaster", "death", "deadly", "fatal",
            "collapse", "chaos", "terror", "horrifying", "devastating",
            "lethal", "annihilate", "meltdown", "apocalypse",
        ],
        "weight": 3,
        "color": "#ff3c5c",
        "icon": "😨",
        "desc": "Exploits fear to bypass rational thinking and push toward a specific reaction.",
        "counters": [
            "Ask: is this fear based on verified facts or just emotional framing?",
            "Fear-based claims often exaggerate probability. Look up base rates and real statistics.",
            "Find a peer-reviewed study or official report before accepting any risk claims.",
        ],
    },
    "False Urgency": {
        "keywords": [
            "now", "immediately", "urgent", "limited", "hurry", "act fast",
            "last chance", "expires", "deadline", "before its too late",
            "dont wait", "time running out", "final warning", "breaking now",
        ],
        "weight": 3,
        "color": "#f59e0b",
        "icon": "⏰",
        "desc": "Creates artificial time pressure so you act before you can think critically.",
        "counters": [
            "Genuine emergencies rarely require instant, uninformed decisions. Pause.",
            "If someone insists you cannot take time to verify, that is itself a red flag.",
            "Legitimate offers can withstand a 24-hour verification window.",
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
        "desc": "Vaguely invokes authority figures or 'studies' without citations to appear credible.",
        "counters": [
            "Demand the actual source: journal name, date, authors. Vague authority is no authority.",
            "Check whether the cited institution actually made that claim on their official channels.",
            "Scientific consensus requires many reproducible studies, not a single unnamed report.",
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
        "desc": "Induces guilt or shame to override your judgment and coerce compliance.",
        "counters": [
            "Healthy discourse does not require you to feel ashamed for asking questions.",
            "Responsibility framing should come with evidence, not emotional attack.",
            "If content makes you feel guilty without explaining why, question its motive.",
        ],
    },
    "Us vs. Them": {
        "keywords": [
            "they", "them", "elite", "globalists", "deep state", "enemy",
            "traitor", "puppet", "regime", "cabal", "establishment",
            "their agenda", "against us", "real patriots", "sheeple",
        ],
        "weight": 4,
        "color": "#f97316",
        "icon": "⚔️",
        "desc": "Divides people into opposing in-groups and out-groups to create tribalism.",
        "counters": [
            "Most complex issues cannot be reduced to two opposing sides.",
            "Who benefits from you seeing a certain group as the enemy?",
            "In-group/out-group framing is a known radicalization gateway.",
        ],
    },
    "Conspiracy Framing": {
        "keywords": [
            "wake up", "hidden agenda", "secret", "cover up", "they dont want you to know",
            "suppressed", "silenced", "censored", "banned",
            "share before deleted", "mainstream media lies", "open your eyes",
            "the truth is", "what theyre hiding",
        ],
        "weight": 5,
        "color": "#14b8a6",
        "icon": "🕵️",
        "desc": "Uses conspiracy language to make unverified claims seem suppressed and credible.",
        "counters": [
            "If evidence only exists outside mainstream institutions, ask why peer review missed it.",
            "The 'hidden truth' frame is self-sealing: any denial just confirms the conspiracy.",
            "Real whistleblowing involves verifiable documents, not social media posts.",
        ],
    },
    "False Certainty": {
        "keywords": [
            "always", "never", "everyone knows", "nobody believes", "obviously",
            "clearly", "undeniably", "100%", "guaranteed", "no doubt",
            "without question", "fact is", "proven fact", "absolute truth",
        ],
        "weight": 2,
        "color": "#06b6d4",
        "icon": "🔒",
        "desc": "Presents opinions or interpretations as absolute, indisputable facts.",
        "counters": [
            "Absolute language ('always', 'never') rarely holds up in complex situations.",
            "Ask for evidence behind 'obvious' claims — obvious things still require proof.",
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
        "desc": "Floods content with intense emotional language to overwhelm critical thinking.",
        "counters": [
            "Strong emotions are a signal to slow down, not speed up your response.",
            "Ask: is this giving me facts, or mainly trying to make me feel something?",
            "Emotional intensity is not evidence that a claim is true.",
        ],
    },
    "Social Proof Manipulation": {
        "keywords": [
            "everyone is doing", "millions agree", "people are waking up",
            "spreading fast", "going viral", "nobody is talking about",
            "masses are rising", "popular opinion", "the world knows",
        ],
        "weight": 2,
        "color": "#22d3ee",
        "icon": "👥",
        "desc": "Fabricates or exaggerates consensus to pressure conformity.",
        "counters": [
            "Popularity is not the same as accuracy. Misinformation spreads fast too.",
            "Verify 'viral' claims independently before accepting their social proof.",
        ],
    },
    "Black & White Thinking": {
        "keywords": [
            "either youre with us", "or youre against us", "no middle ground",
            "only two options", "there is no choice", "you must choose",
            "truth or lies", "freedom or slavery",
        ],
        "weight": 3,
        "color": "#6366f1",
        "icon": "🎭",
        "desc": "Presents false dilemmas to prevent nuanced thinking.",
        "counters": [
            "Binary framing is almost always an oversimplification.",
            "Most political and social issues exist on a spectrum, not two sides.",
            "Reject the premise and look for the third, fourth, or fifth option.",
        ],
    },
}

PROPAGANDA = [
    "they don't want you to know",
    "wake up sheeple",
    "share before deleted",
    "mainstream media lies",
    "hidden truth",
    "open your eyes",
    "the real truth",
    "what they're not telling you",
    "do your own research",
    "deep state agenda",
    "new world order",
    "follow the money",
    "the elite don't want this",
    "shadow government",
]

FALLACIES = {
    "Ad Hominem": [
        "stupid", "idiot", "moron", "liar", "hypocrite",
        "paid shill", "brainwashed", "gullible fool",
    ],
    "Slippery Slope": [
        "will lead to", "next thing you know", "soon they will",
        "it starts with", "before long", "the beginning of the end",
    ],
    "Bandwagon": [
        "join the movement", "be part of history", "don't be left behind",
        "the smart ones already know",
    ],
    "Straw Man": [
        "so you're saying", "you must believe", "people like you think",
        "that means you support",
    ],
}


# ------------------------------------------------------------------
# model — cached, loads once
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    name = "Giyaseddin/distilbert-base-cased-finetuned-fake-and-real-news-dataset"
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSequenceClassification.from_pretrained(name)
    mod.eval()
    return tok, mod


# ------------------------------------------------------------------
# transformer prediction with chunk averaging
# ------------------------------------------------------------------
def predict_fake_real(text, tok, mod):
    words = text.split()
    # split into 400-word chunks so long texts aren't just truncated
    chunks = [" ".join(words[i:i+400]) for i in range(0, max(len(words), 1), 400)]

    fake_acc, real_acc = [], []
    for chunk in chunks:
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  padding=True, max_length=512)
        with torch.no_grad():
            logits = mod(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

        # read label order from the model itself instead of hardcoding
        id2label = mod.config.id2label
        fake_idx = next(
            (i for i, lbl in id2label.items() if "fake" in str(lbl).lower()), 0
        )
        real_idx = 1 - fake_idx

        fake_acc.append(probs[fake_idx])
        real_acc.append(probs[real_idx])

    fake_p = float(np.mean(fake_acc))
    real_p = float(np.mean(real_acc))
    label = "FAKE" if fake_p > real_p else "REAL"
    return label, max(fake_p, real_p), fake_p, real_p


# ------------------------------------------------------------------
# main analysis
# ------------------------------------------------------------------
def analyze(text, tok, mod):
    tl = text.lower()

    fake_label, confidence, fake_p, real_p = predict_fake_real(text, tok, mod)

    # keyword matching per technique
    hits = {}
    flagged = set()
    for name, info in TECHNIQUES.items():
        matched = [kw for kw in info["keywords"]
                   if re.search(r"\b" + re.escape(kw) + r"\b", tl)]
        if matched:
            hits[name] = matched
            flagged.update(matched)

    prop_hits = [p for p in PROPAGANDA if p in tl]

    fallacy_hits = {}
    for fname, fwords in FALLACIES.items():
        m = [w for w in fwords if w in tl]
        if m:
            fallacy_hits[fname] = m

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    sents = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
    extreme = [s for s in sents
               if abs(TextBlob(s).sentiment.polarity) > 0.6
               and TextBlob(s).sentiment.subjectivity > 0.6]

    alpha = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0
    exclaim_density = text.count("!") / max(len(sents), 1)

    # composite score
    tech_raw   = sum(TECHNIQUES[t]["weight"] * len(kws) for t, kws in hits.items())
    tech_score = min(40, tech_raw * 5)
    model_sc   = fake_p * 25
    sent_sc    = abs(sentiment) * 10 if abs(sentiment) > 0.3 else 0
    subj_sc    = subjectivity * 8 if subjectivity > 0.5 else 0
    prop_sc    = min(15, len(prop_hits) * 5)
    fall_sc    = min(10, len(fallacy_hits) * 3)
    caps_sc    = min(5, caps_ratio * 20)
    excl_sc    = min(5, exclaim_density * 5)

    score = math.floor(
        5 + tech_score + model_sc + sent_sc + subj_sc
        + prop_sc + fall_sc + caps_sc + excl_sc
    )
    score = max(0, min(100, score))

    if score >= 70:
        risk, risk_cls, risk_color = "🚨 HIGH MANIPULATION RISK", "risk-high", "#ff3c5c"
    elif score >= 40:
        risk, risk_cls, risk_color = "⚠️ MODERATE MANIPULATION RISK", "risk-medium", "#f59e0b"
    else:
        risk, risk_cls, risk_color = "✅ LOW MANIPULATION RISK", "risk-low", "#10b981"

    # build explanation dynamically from what was found
    parts = []

    if fake_label == "FAKE" and confidence > 0.65:
        parts.append(
            f"**Fake News Model:** Classified as likely **fabricated or misleading** "
            f"({confidence:.0%} confidence). Language patterns deviate significantly from authentic reporting."
        )
    elif fake_label == "FAKE":
        parts.append(
            f"**Fake News Model:** Flagged as **potentially misleading** ({confidence:.0%} confidence). "
            f"Some linguistic markers suggest distorted framing; independent verification recommended."
        )
    else:
        parts.append(
            f"**Fake News Model:** Broadly **authentic in structure** ({confidence:.0%} confidence). "
            f"Note: this checks writing patterns, not factual accuracy — always verify claims independently."
        )

    if hits:
        top = sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        tech_str = ", ".join(f"**{t}** ({len(k)} triggers)" for t, k in top)
        parts.append(
            f"**Primary Techniques Detected:** {tech_str}. "
            f"These exploit cognitive biases to influence beliefs without presenting rational evidence."
        )

    if prop_hits:
        parts.append(
            f"**Propaganda Phrases:** {len(prop_hits)} known phrase(s) found. "
            f"These prime distrust in credible institutions and make alternative narratives feel revelatory."
        )

    if fallacy_hits:
        parts.append(
            f"**Logical Fallacies:** {', '.join(f'**{f}**' for f in fallacy_hits)} — "
            f"substituting emotional persuasion for logical argument."
        )

    if sentiment < -0.4:
        parts.append(
            f"**Strongly Negative Tone (polarity: {sentiment:.2f}):** "
            f"Persistent negativity narrows perceived options and increases psychological stress on the reader."
        )
    elif sentiment > 0.4 and score > 40:
        parts.append(
            f"**Suspiciously Positive Tone ({sentiment:.2f}):** "
            f"Unrealistically upbeat framing at high manipulation scores can signal false promises."
        )

    if subjectivity > 0.65:
        parts.append(
            f"**High Subjectivity ({subjectivity:.0%}):** Most content is opinion and interpretation "
            f"rather than verifiable fact, though it may be presented as objective truth."
        )

    if caps_ratio > 0.25:
        parts.append(
            f"**Aggressive Typography ({caps_ratio:.0%} uppercase):** "
            f"Heavy capitalisation is a visual shouting technique that bypasses calm reading."
        )

    if not parts:
        parts.append(
            "No significant manipulation signals detected. "
            "Content appears informational in intent. Still verify any factual claims independently."
        )

    explanation = "\n\n".join(parts)

    # counter-messages pulled from matched techniques
    counters = []
    seen = set()
    for tech in hits:
        for c in TECHNIQUES[tech]["counters"]:
            if c not in seen:
                seen.add(c)
                counters.append(c)
    counters = counters[:6]
    if not counters:
        counters = [
            "Cross-reference any factual claims with multiple independent credible sources.",
            "Consider the source's motivation: who benefits if you believe this?",
            "Share only after verification — well-intentioned misinformation still causes harm.",
        ]

    advice = []
    if score >= 70:
        advice.append(("🔴", "Do NOT share without rigorous verification from at least 3 independent sources."))
        advice.append(("🔴", "Notice your emotional reaction — that reaction may itself be the manipulation."))
    elif score >= 40:
        advice.append(("🟡", "Approach with caution. Verify key claims with primary sources before sharing."))

    if fake_p > 0.6:
        advice.append(("🔴", "AI flagged likely fabrication. Look for the original event or study — it may not exist."))
    elif fake_p > 0.4:
        advice.append(("🟡", "Possible content distortion. Check the original source, not just this version."))

    if "Conspiracy Framing" in hits:
        advice.append(("🔴", "Real suppressed information comes with verifiable documents. Ask for primary evidence."))
    if "Us vs. Them" in hits:
        advice.append(("🟡", "Tribal division tactics present. Seek perspectives from multiple sides."))
    if "Fear Appeal" in hits:
        advice.append(("🟡", "Fear is being used as a tool. Look up the actual statistical likelihood of the threat."))
    if "False Urgency" in hits:
        advice.append(("🟡", "Artificial urgency discourages verification. Real decisions rarely need minutes."))
    if sentiment < -0.5:
        advice.append(("🟡", "Strongly negative content can distort threat perception. Take a break before reacting."))

    advice.append(("🟢", "Use fact-checkers: Snopes, PolitiFact, FactCheck.org, AFP Fact Check, AP Fact Check."))
    advice.append(("🟢", "Check for bylines, publication dates, and institutional affiliations."))
    advice.append(("🟢", "Ask: 'How would I react if the subject were someone I support?' — consistency reveals bias."))

    # highlight flagged words in original text
    highlighted = text
    for word in sorted(flagged, key=len, reverse=True):
        pat = re.compile(r"\b(" + re.escape(word) + r")\b", re.IGNORECASE)
        highlighted = pat.sub(r"<span class='flagged-word'>\1</span>", highlighted)

    return {
        "score": score,
        "risk": risk,
        "risk_cls": risk_cls,
        "risk_color": risk_color,
        "fake_label": fake_label,
        "confidence": confidence,
        "fake_p": fake_p,
        "real_p": real_p,
        "hits": hits,
        "prop_hits": prop_hits,
        "fallacy_hits": fallacy_hits,
        "sentiment": sentiment,
        "subjectivity": subjectivity,
        "extreme": extreme,
        "explanation": explanation,
        "counters": counters,
        "advice": advice,
        "highlighted": highlighted,
        "flagged_words": sorted(flagged),
    }


# ------------------------------------------------------------------
# chart helpers
# ------------------------------------------------------------------
def gauge_chart(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 28, "color": "#e2e8f0", "family": "Syne"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickfont": {"color": "#64748b", "size": 10},
                "tickcolor": "#1e2540",
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0f1220",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],   "color": "#10b98112"},
                {"range": [40, 70],  "color": "#f59e0b12"},
                {"range": [70, 100], "color": "#ff3c5c12"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": score},
        },
        title={"text": "Manipulation Score",
               "font": {"size": 13, "color": "#64748b", "family": "Syne"}},
    ))
    fig.update_layout(
        height=260,
        margin={"t": 30, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="#141826",
        font_color="#e2e8f0",
    )
    return fig


def radar_chart(hits):
    cats = [
        "Fear Appeal", "False Urgency", "Conspiracy Framing",
        "Us vs. Them", "Emotional Overload", "False Certainty",
    ]
    vals = [min(10, len(hits.get(c, [])) * 2) for c in cats]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="#7c3aed22",
        line={"color": "#00e5ff", "width": 2},
        marker={"color": "#00e5ff", "size": 5},
    ))
    fig.update_layout(
        polar={
            "bgcolor": "#0f1220",
            "radialaxis": {"visible": True, "range": [0, 10],
                           "gridcolor": "#1e2540",
                           "tickfont": {"color": "#64748b", "size": 9}},
            "angularaxis": {"tickfont": {"color": "#c4cad8", "size": 10},
                            "gridcolor": "#1e2540"},
        },
        height=300,
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        paper_bgcolor="#141826",
        font_color="#e2e8f0",
    )
    return fig


def bar_chart(hits):
    names = list(hits.keys())
    counts = [len(v) for v in hits.values()]
    clrs = [TECHNIQUES[n]["color"] for n in names]
    fig = go.Figure(go.Bar(
        x=counts, y=names, orientation="h",
        marker={"color": clrs, "opacity": 0.85},
        text=counts, textposition="outside",
        textfont={"color": "#e2e8f0", "size": 11},
    ))
    fig.update_layout(
        height=max(200, len(names) * 44),
        margin={"t": 10, "b": 10, "l": 10, "r": 40},
        paper_bgcolor="#141826",
        plot_bgcolor="#141826",
        xaxis={
            "showgrid": True, "gridcolor": "#1e2540",
            "tickfont": {"color": "#64748b"},
            "title": {"text": "Keyword count", "font": {"color": "#64748b", "size": 11}},
        },
        yaxis={"tickfont": {"color": "#e2e8f0", "size": 11}, "automargin": True},
    )
    return fig


# ------------------------------------------------------------------
# PDF report
# ------------------------------------------------------------------
def make_pdf(text, r):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=20, spaceAfter=6,
                              textColor=rc.HexColor("#00e5ff"))
    h2_s    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                              spaceBefore=14, spaceAfter=4, textColor=rc.HexColor("#7c3aed"))
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=4)

    els = []
    els.append(Paragraph("MindShield AI — Manipulation Analysis Report", title_s))
    els.append(Spacer(1, 0.15*inch))
    els.append(Paragraph(f"Manipulation Score: {r['score']}/100  |  {r['risk']}", h2_s))
    els.append(Paragraph(f"Fake News Model: {r['fake_label']} ({r['confidence']:.0%} confidence)", body_s))
    els.append(Paragraph(f"Sentiment: {r['sentiment']:.2f}  |  Subjectivity: {r['subjectivity']:.0%}", body_s))
    els.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_s))
    els.append(Spacer(1, 0.1*inch))

    els.append(Paragraph("Analysed Text", h2_s))
    safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
    els.append(Paragraph(safe_text[:1500] + ("..." if len(text) > 1500 else ""), body_s))
    els.append(Spacer(1, 0.1*inch))

    els.append(Paragraph("Manipulation Techniques Detected", h2_s))
    if r["hits"]:
        for tech, kws in r["hits"].items():
            els.append(Paragraph(f"<b>{tech}:</b> {', '.join(kws)}", body_s))
    else:
        els.append(Paragraph("None detected.", body_s))
    els.append(Spacer(1, 0.1*inch))

    if r["prop_hits"]:
        els.append(Paragraph("Propaganda Phrases", h2_s))
        for p in r["prop_hits"]:
            els.append(Paragraph(f"• {p}", body_s))
        els.append(Spacer(1, 0.1*inch))

    els.append(Paragraph("AI Explanation", h2_s))
    clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", r["explanation"])
    for line in clean.split("\n\n"):
        if line.strip():
            els.append(Paragraph(line.strip(), body_s))
            els.append(Spacer(1, 0.05*inch))

    els.append(Paragraph("Counter-Messages", h2_s))
    for c in r["counters"]:
        els.append(Paragraph(f"• {c}", body_s))
    els.append(Spacer(1, 0.1*inch))

    els.append(Paragraph("Advice", h2_s))
    for icon, txt in r["advice"]:
        els.append(Paragraph(f"{icon} {txt}", body_s))

    doc.build(els)
    buf.seek(0)
    return buf


# ------------------------------------------------------------------
# page
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">Multi-Signal Transformer Analysis</div>
    <h1>🧠 MindShield AI</h1>
    <p>Uncover hidden psychological manipulation, fake news patterns,
    cognitive traps, and propaganda techniques in any piece of text.</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading AI model — first run may take 30–60 seconds…"):
    try:
        tokenizer, model = load_model()
        model_ready = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model_ready = False

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-label'>📋 Content to Analyse</div>", unsafe_allow_html=True)
user_text = st.text_area(
    label="",
    placeholder="Paste a news article, social media post, email, advertisement, or any text here…",
    height=220,
    label_visibility="collapsed",
)
run_btn = st.button("🔍 Analyse for Manipulation", disabled=not model_ready)
st.markdown("</div>", unsafe_allow_html=True)


if run_btn:
    if not user_text.strip():
        st.warning("Please enter some text to analyse.")
    else:
        with st.spinner("Running deep multi-signal analysis…"):
            r = analyze(user_text, tokenizer, model)

        st.markdown(f"<div class='{r['risk_cls']}'>{r['risk']}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, str(r["score"]),           "Manip. Score"),
            (c2, str(len(r["hits"])),        "Techniques"),
            (c3, f"{r['confidence']:.0%}",   f"{r['fake_label']} Conf."),
            (c4, f"{r['subjectivity']:.0%}", "Subjectivity"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-val'>{val}</div>"
                    f"<div class='metric-lbl'>{lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>📊 Manipulation Gauge</div>", unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(r["score"], r["risk_color"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>🕸️ Technique Radar</div>", unsafe_allow_html=True)
            st.plotly_chart(radar_chart(r["hits"]),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        if r["hits"]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>🎯 Technique Breakdown</div>", unsafe_allow_html=True)
            st.plotly_chart(bar_chart(r["hits"]),
                            use_container_width=True, config={"displayModeBar": False})
            for tech, kws in r["hits"].items():
                info = TECHNIQUES[tech]
                st.markdown(
                    f"**{info['icon']} {tech}** — {info['desc']}  \n"
                    f"*Detected:* `{', '.join(kws)}`"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>🤖 Fake News Model</div>", unsafe_allow_html=True)
        fa, re_ = st.columns(2)
        with fa:
            st.markdown(
                f"<div style='text-align:center;padding:0.5rem;'>"
                f"<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;'>FAKE Probability</div>"
                f"<div style='font-size:1.9rem;font-weight:800;color:#ff3c5c;'>{r['fake_p']:.0%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with re_:
            st.markdown(
                f"<div style='text-align:center;padding:0.5rem;'>"
                f"<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;'>REAL Probability</div>"
                f"<div style='font-size:1.9rem;font-weight:800;color:#10b981;'>{r['real_p']:.0%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='background:#1e2540;border-radius:999px;height:10px;margin:6px 0;overflow:hidden;'>"
            f"<div style='height:100%;width:{r['fake_p']*100:.1f}%;background:#ff3c5c;border-radius:999px;'></div></div>"
            f"<div style='background:#1e2540;border-radius:999px;height:10px;margin:6px 0;overflow:hidden;'>"
            f"<div style='height:100%;width:{r['real_p']*100:.1f}%;background:#10b981;border-radius:999px;'></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>🧠 AI Explanation</div>", unsafe_allow_html=True)
        with st.expander("View full analysis", expanded=True):
            st.markdown(r["explanation"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>🛡️ Counter-Messages for This Content</div>", unsafe_allow_html=True)
        for c in r["counters"]:
            st.markdown(f"<div class='counter-msg'>💬 {c}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>💡 Personalised Advice</div>", unsafe_allow_html=True)
        for icon, txt in r["advice"]:
            st.markdown(
                f"<div class='advice-row'>"
                f"<span style='font-size:1.1rem;flex-shrink:0'>{icon}</span>"
                f"<span>{txt}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>🔎 Highlighted Trigger Words</div>", unsafe_allow_html=True)
        with st.expander("View annotated text", expanded=False):
            st.markdown(
                f"<div style='line-height:1.9;font-size:0.95rem;white-space:pre-wrap;'>{r['highlighted']}</div>",
                unsafe_allow_html=True,
            )
        if r["flagged_words"]:
            st.markdown("**All flagged words:** " + "  ".join(f"`{w}`" for w in r["flagged_words"]))
        st.markdown("</div>", unsafe_allow_html=True)

        if r["prop_hits"] or r["fallacy_hits"]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>📢 Propaganda & Logical Fallacies</div>", unsafe_allow_html=True)
            if r["prop_hits"]:
                st.markdown("**Propaganda Phrases Detected:**")
                for p in r["prop_hits"]:
                    st.markdown(f"- `{p}`")
            if r["fallacy_hits"]:
                st.markdown("**Logical Fallacies Detected:**")
                for fname, fkws in r["fallacy_hits"].items():
                    st.markdown(f"- **{fname}:** `{', '.join(fkws)}`")
            st.markdown("</div>", unsafe_allow_html=True)

        if r["extreme"]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-label'>⚡ Most Emotionally Extreme Sentences</div>", unsafe_allow_html=True)
            with st.expander("View extreme sentences", expanded=False):
                for s in r["extreme"][:5]:
                    st.markdown(f"<div class='extreme-sent'>{s}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        pdf = make_pdf(user_text, r)
        st.download_button(
            "📄 Download Full PDF Report",
            pdf,
            file_name=f"MindShield_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;color:#64748b;font-size:0.8rem;'>"
            "MindShield AI · For educational use only · Always verify with primary sources"
            "</div>",
            unsafe_allow_html=True,
        )
