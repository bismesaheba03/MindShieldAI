import re
import math
import torch
import math
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract
import cv2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="MindShield AI", layout="centered")


# =============================
# CUSTOM CSS
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
# TRIGGER WORD CATEGORIES
# =============================
trigger_categories = {
    "Fear": [
        "danger", "threat", "risk", "destroy", "crisis", "attack",
        "catastrophe", "disaster", "death", "deadly", "fatal", "collapse",
        "chaos", "terror", "horrifying", "devastating", "lethal", "apocalypse",
        "extinction", "meltdown", "annihilate", "wipeout", "obliterate",
    ],
    "Urgency": [
        "now", "immediately", "urgent", "limited", "hurry", "act fast",
        "last chance", "expires", "deadline", "before its too late",
        "dont wait", "time running out", "final warning", "breaking now",
        "tonight only", "seconds left",
    ],
    "Authority": [
        "expert", "official", "government", "scientists", "study", "research",
        "doctors say", "according to experts", "professor", "insider",
        "whistleblower", "top secret", "classified", "sources say",
        "studies show", "research proves",
    ],
    "Guilt": [
        "shame", "selfish", "responsible", "blame", "fault", "disgrace",
        "coward", "failure", "disappoint", "letting down", "your fault",
        "irresponsible", "negligent", "pathetic", "ignorant",
    ],
    "Us vs Them": [
        "they", "them", "elite", "globalists", "deep state", "enemy",
        "traitor", "puppet", "regime", "cabal", "establishment",
        "against us", "real patriots", "sheeple", "the masses",
    ],
    "Conspiracy": [
        "wake up", "hidden agenda", "cover up", "suppressed", "silenced",
        "censored", "banned", "share before deleted", "mainstream media lies",
        "open your eyes", "the truth is", "what theyre hiding",
        "they dont want you to know", "big pharma", "false flag",
    ],
    "False Certainty": [
        "always", "never", "everyone knows", "obviously", "clearly",
        "undeniably", "guaranteed", "no doubt", "without question",
        "proven fact", "absolute truth", "100 percent",
    ],
    "Emotional Overload": [
        "heartbreaking", "outrageous", "disgusting", "shocking", "unbelievable",
        "horrifying", "enraging", "infuriating", "appalling", "sickening",
        "monstrous", "vile", "evil", "demonic", "devastating",
    ],
}

CATEGORY_WEIGHTS = {
    "Fear": 3,
    "Urgency": 3,
    "Authority": 2,
    "Guilt": 3,
    "Us vs Them": 4,
    "Conspiracy": 5,
    "False Certainty": 2,
    "Emotional Overload": 2,
}

propaganda_patterns = [
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
    "shadow government",
]

logical_fallacies = {
    "Ad Hominem":     ["stupid", "idiot", "moron", "liar", "hypocrite", "paid shill", "brainwashed"],
    "Slippery Slope": ["will lead to", "next thing you know", "soon they will", "it starts with", "beginning of the end"],
    "Bandwagon":      ["join the movement", "be part of history", "dont be left behind", "the smart ones already know"],
    "Straw Man":      ["so youre saying", "you must believe", "people like you think", "that means you support"],
}


# =============================
# TRANSFORMER MODEL
# =============================
@st.cache_resource
def load_transformer_model():
    model_name = "Giyaseddin/distilbert-base-cased-finetuned-fake-and-real-news-dataset"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, transformer_model = load_transformer_model()


# =============================
# EASYOCR — loaded once and cached
# =============================
@st.cache_resource
def load_easyocr_reader():
    """Load EasyOCR with full error protection. Returns None if unavailable."""
    try:
        import easyocr
        # model_storage_directory prevents repeated downloads between reruns
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False,
            download_enabled=True,
        )
        return reader
    except Exception as e:
        return None


# =============================
# IMPROVED OCR — multi-engine, multi-strategy
# =============================

def _pil_to_cv2(pil_img):
    """Convert PIL RGB image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(cv2_img):
    """Convert OpenCV BGR array to PIL grayscale."""
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray)


def _score_text(text: str) -> float:
    """Score OCR output by real 3+ letter word count minus noise."""
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z]{3,}", text)
    junk = len(re.findall(r"[^A-Za-z0-9\s'\".,!?()\-]", text))
    return len(words) * 5 - junk * 2


def _clean_ocr(text: str) -> str:
    """Remove noisy lines; keep lines that are mostly alphabetic."""
    good = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        alpha = sum(1 for c in line if c.isalpha())
        if alpha >= 3 and alpha / max(len(line), 1) >= 0.45:
            line = re.sub(r"^[^A-Za-z'\"]+", "", line).strip()
            if line:
                good.append(line)
    out = " ".join(good)
    return re.sub(r"\s+", " ", out).strip()


def _tesseract_on_pil(pil_gray, psm: int = 11) -> str:
    """Run Tesseract on a PIL grayscale image."""
    config = f"--psm {psm} --oem 3"
    try:
        return pytesseract.image_to_string(pil_gray, config=config).strip()
    except Exception:
        return ""


def _preprocess_variants(pil_rgb: Image.Image):
    if not CV2_AVAILABLE:
        return []
    """
    Generate a list of preprocessed PIL-grayscale images from an RGB PIL image.
    Returns list of (label, pil_gray_image) tuples.
    """
    variants = []
    bgr = _pil_to_cv2(pil_rgb)

    # 1. Upscale 3x — best single improvement for Tesseract
    w, h = pil_rgb.size
    big_pil = pil_rgb.resize((w * 3, h * 3), Image.LANCZOS)
    big_bgr = _pil_to_cv2(big_pil)
    big_gray = cv2.cvtColor(big_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Standard grayscale, upscaled
    variants.append(("gray_3x", Image.fromarray(big_gray)))

    # 3. Adaptive thresholding (great for uneven lighting)
    adaptive = cv2.adaptiveThreshold(
        big_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    variants.append(("adaptive_thresh", Image.fromarray(adaptive)))

    # 4. Adaptive thresh inverted (dark text on light background)
    variants.append(("adaptive_thresh_inv", Image.fromarray(cv2.bitwise_not(adaptive))))

    # 5. Otsu binarization
    _, otsu = cv2.threshold(big_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", Image.fromarray(otsu)))
    variants.append(("otsu_inv", Image.fromarray(cv2.bitwise_not(otsu))))

    # 6. CLAHE (local contrast enhancement) + Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(big_gray)
    _, clahe_thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", Image.fromarray(clahe_thresh)))

    # 7. Morphological dilation to connect broken letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(clahe_thresh, kernel, iterations=1)
    variants.append(("dilated", Image.fromarray(dilated)))

    # 8. White/bright text isolation (for white text on dark backgrounds)
    arr = np.array(big_pil, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (R + G + B) / 3.0
    for thr in [150, 170, 190, 210]:
        mask = (brightness > thr).astype(np.uint8) * 255
        variants.append((f"bright_{thr}", Image.fromarray(mask)))

    # 9. Yellow text isolation (meme yellow/orange text)
    yellow = ((R > 150) & (G > 120) & (B < 110)).astype(np.uint8) * 255
    variants.append(("yellow_text", Image.fromarray(yellow)))

    # 10. White + Yellow combined
    combined = np.clip(mask.astype(np.uint16) + yellow.astype(np.uint16), 0, 255).astype(np.uint8)
    variants.append(("white_yellow", Image.fromarray(combined)))

    # 11. Sharpened + contrast enhanced
    pil_gray_big = Image.fromarray(big_gray)
    sharp = pil_gray_big.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    sharp = ImageEnhance.Contrast(sharp).enhance(2.5)
    variants.append(("sharp_contrast", sharp))

    # 12. Inverted for dark-on-light
    inv = ImageOps.invert(Image.fromarray(big_gray))
    inv = ImageEnhance.Contrast(inv).enhance(2.0)
    variants.append(("inverted", inv))

    # 13. Denoised with bilateral filter
    denoised = cv2.bilateralFilter(big_gray, 9, 75, 75)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu", Image.fromarray(denoised_thresh)))

    return variants


def _run_tesseract_all(variants):
    """Run Tesseract on all variants with multiple PSM modes."""
    results = []
    for label, pil_gray in variants:
        for psm in [6, 11, 3]:
            try:
                text = _tesseract_on_pil(pil_gray, psm=psm)
                if text:
                    results.append(text)
            except Exception:
                pass
    return results


def _run_easyocr(pil_rgb: Image.Image, reader) -> list:
    """Run EasyOCR on the original and upscaled image."""
    results = []
    try:
        np_img = np.array(pil_rgb.convert("RGB"))
        raw = reader.readtext(np_img, detail=0, paragraph=True)
        results.append(" ".join(raw))

        # Also try upscaled
        w, h = pil_rgb.size
        big = pil_rgb.resize((w * 2, h * 2), Image.LANCZOS)
        np_big = np.array(big.convert("RGB"))
        raw_big = reader.readtext(np_big, detail=0, paragraph=True)
        results.append(" ".join(raw_big))
    except Exception:
        pass
    return results


def extract_text_from_image(pil_image: Image.Image) -> str:
    """
    Robust multi-engine OCR:
      1. Generates 15+ preprocessing variants
      2. Runs Tesseract (PSM 3/6/11) on all variants
      3. Runs EasyOCR on original + upscaled (if available)
      4. Picks the result with the highest real-word score
      5. Cleans and returns the best output
    """
    pil_rgb = pil_image.convert("RGB")

    all_texts = []

    # --- Basic Tesseract fallback (always runs, even without cv2) ---
    try:
        gray_basic = pil_rgb.convert("L")
        w, h = gray_basic.size
        big_gray = gray_basic.resize((w * 3, h * 3), Image.LANCZOS)
        big_contrast = ImageEnhance.Contrast(big_gray).enhance(2.5)
        for psm in [6, 11, 3]:
            t = _tesseract_on_pil(big_gray, psm=psm)
            if t:
                all_texts.append(t)
            t2 = _tesseract_on_pil(big_contrast, psm=psm)
            if t2:
                all_texts.append(t2)
        inv = ImageOps.invert(big_gray)
        t3 = _tesseract_on_pil(inv, psm=11)
        if t3:
            all_texts.append(t3)
    except Exception:
        pass

    # --- Advanced cv2 preprocessing (runs only if cv2 available) ---
    try:
        if CV2_AVAILABLE:
            variants = _preprocess_variants(pil_rgb)
            tess_results = _run_tesseract_all(variants)
            all_texts.extend(tess_results)
    except Exception as e:
        pass

    # --- EasyOCR engine ---
    try:
        reader = load_easyocr_reader()
        if reader is not None:
            easy_results = _run_easyocr(pil_rgb, reader)
            all_texts.extend(easy_results)
    except Exception:
        pass

    if not all_texts:
        return ""

    best = max(all_texts, key=_score_text)
    cleaned = _clean_ocr(best)

    # If cleaned result is very short, try the raw best without strict cleaning
    if len(cleaned.split()) < 5:
        raw_cleaned = re.sub(r"\s+", " ", best).strip()
        if _score_text(raw_cleaned) > _score_text(cleaned):
            return raw_cleaned

    return cleaned


# =============================
# TRANSFORMER FAKE/REAL DETECTION
# =============================
def transformer_fake_real(text):
    words = text.split()
    chunks = [" ".join(words[i:i+400]) for i in range(0, max(len(words), 1), 400)]

    fake_scores, real_scores = [], []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True,
                           padding=True, max_length=512)
        with torch.no_grad():
            logits = transformer_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

        id2label = transformer_model.config.id2label
        fake_idx = next((i for i, v in id2label.items() if "fake" in str(v).lower()), 0)
        real_idx = 1 - fake_idx

        fake_scores.append(probs[fake_idx])
        real_scores.append(probs[real_idx])

    fake_p = float(np.mean(fake_scores))
    real_p = float(np.mean(real_scores))
    label = "FAKE" if fake_p > real_p else "REAL"
    confidence = max(fake_p, real_p)
    return label, confidence, fake_p, real_p


# =============================
# CORE ANALYSIS
# =============================
def analyze_text(text):
    text_lower = text.lower()

    fake_label, fake_confidence, fake_p, real_p = transformer_fake_real(text)

    trigger_counts = {}
    all_matched_words = set()
    total_triggers = 0
    weighted_trigger_total = 0

    for cat, words in trigger_categories.items():
        matched = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
        trigger_counts[cat] = len(matched)
        all_matched_words.update(matched)
        total_triggers += len(matched)
        weighted_trigger_total += len(matched) * CATEGORY_WEIGHTS.get(cat, 2)

    propaganda_hits = [p for p in propaganda_patterns if p in text_lower]
    propaganda_detected = len(propaganda_hits) > 0

    fallacy_hits = {}
    for fname, fwords in logical_fallacies.items():
        m = [w for w in fwords if w in text_lower]
        if m:
            fallacy_hits[fname] = m

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
    exclaim_density = text.count("!") / max(len(sentences), 1)

    extreme_sentences = []
    for s in sentences:
        b = TextBlob(s)
        if abs(b.sentiment.polarity) > 0.6 and b.sentiment.subjectivity > 0.6:
            extreme_sentences.append(s)

    tech_score      = min(35, weighted_trigger_total * 4)
    model_score     = fake_p * 25
    sentiment_sc    = abs(sentiment) * 10 if abs(sentiment) > 0.3 else 0
    subjectivity_sc = subjectivity * 8 if subjectivity > 0.5 else 0
    propaganda_sc   = min(15, len(propaganda_hits) * 5)
    fallacy_sc      = min(10, len(fallacy_hits) * 3)
    caps_sc         = min(5, caps_ratio * 20)
    exclaim_sc      = min(5, exclaim_density * 5)

    score = math.floor(
        5 + tech_score + model_score + sentiment_sc + subjectivity_sc
        + propaganda_sc + fallacy_sc + caps_sc + exclaim_sc
    )
    score = max(0, min(100, score))

    if score >= 70:
        risk = "🚨 High Psychological Manipulation"
    elif score >= 40:
        risk = "⚠ Moderate Manipulation"
    else:
        risk = "✅ Low Manipulation"

    parts = []

    if fake_label == "FAKE" and fake_confidence > 0.65:
        parts.append(
            f"**Fake News Model:** The transformer classified this as likely **fabricated or misleading** "
            f"({fake_confidence:.0%} confidence). Writing patterns deviate significantly from authentic reporting."
        )
    elif fake_label == "FAKE":
        parts.append(
            f"**Fake News Model:** Content flagged as **potentially misleading** "
            f"({fake_confidence:.0%} confidence). Some markers suggest distorted framing — "
            f"independent verification recommended."
        )
    else:
        parts.append(
            f"**Fake News Model:** Content appears broadly **authentic in structure** "
            f"({fake_confidence:.0%} confidence). This checks writing patterns only — "
            f"factual accuracy still needs independent verification."
        )

    active_cats = [(c, trigger_counts[c]) for c in trigger_counts if trigger_counts[c] > 0]
    if active_cats:
        top = sorted(active_cats, key=lambda x: x[1] * CATEGORY_WEIGHTS.get(x[0], 2), reverse=True)[:3]
        top_str = ", ".join(f"**{c}** ({n} triggers)" for c, n in top)
        parts.append(
            f"**Manipulation Techniques:** {top_str}. "
            f"These exploit cognitive biases to influence beliefs without rational evidence."
        )

    if propaganda_hits:
        joined = ", ".join(f'"{p}"' for p in propaganda_hits[:3])
        parts.append(
            f"**Propaganda Phrases:** {joined}. "
            f"These prime distrust in credible institutions and make alternative narratives feel like suppressed revelations."
        )

    if fallacy_hits:
        f_names = ", ".join(f"**{f}**" for f in fallacy_hits)
        parts.append(
            f"**Logical Fallacies:** {f_names}. "
            f"These substitute emotional pressure for actual logical argument."
        )

    if sentiment < -0.4:
        parts.append(
            f"**Strongly Negative Sentiment (polarity: {sentiment:.2f}):** "
            f"Persistent negativity narrows perceived options and raises psychological stress in the reader."
        )
    elif sentiment > 0.4 and score > 40:
        parts.append(
            f"**Suspiciously Positive Tone ({sentiment:.2f}):** "
            f"Unrealistically upbeat framing at high manipulation scores can signal false promises."
        )

    if subjectivity > 0.65:
        parts.append(
            f"**High Subjectivity ({subjectivity:.0%}):** Most content is opinion and interpretation, "
            f"yet may be presented as objective fact."
        )

    if caps_ratio > 0.25:
        parts.append(
            f"**Aggressive Typography ({caps_ratio:.0%} uppercase):** "
            f"Heavy capitalisation is a visual shouting technique that bypasses calm reading."
        )

    if extreme_sentences:
        example = extreme_sentences[0]
        snippet = (example[:120] + "...") if len(example) > 120 else example
        parts.append(f"**{len(extreme_sentences)} emotionally extreme sentence(s) found.** Example: \"{snippet}\"")

    if not parts:
        parts.append(
            "No significant manipulation signals detected. "
            "Content appears informational in intent. Always verify claims independently."
        )

    explanation = "\n\n".join(parts)
    explanation += (
        f"\n\n**Score breakdown:** keywords {math.floor(tech_score)}/35 · "
        f"model {math.floor(model_score)}/25 · sentiment {math.floor(sentiment_sc)}/10 · "
        f"propaganda {math.floor(propaganda_sc)}/15 · fallacies {math.floor(fallacy_sc)}/10 · "
        f"typography {math.floor(caps_sc + exclaim_sc)}/10"
    )

    counter_lookup = {
        "Fear": [
            "Fear-based claims often exaggerate probability. Look up base rates and actual statistics before reacting.",
            "Ask: is this fear based on verified facts or just emotional framing?",
        ],
        "Urgency": [
            "Genuine emergencies rarely require instant, uninformed decisions. Take time to verify.",
            "Any source insisting you cannot pause to check is itself a red flag.",
        ],
        "Authority": [
            "Demand the actual source: journal name, date, authors. Vague authority is no authority.",
            "Check whether the cited institution actually made that claim on their official channels.",
        ],
        "Guilt": [
            "Healthy discourse does not require you to feel ashamed for asking questions.",
            "Responsibility claims should come with evidence, not emotional attack.",
        ],
        "Us vs Them": [
            "Most complex issues cannot be reduced to two opposing sides. Look for nuanced perspectives.",
            "Who benefits from you seeing a certain group as the enemy?",
        ],
        "Conspiracy": [
            "Real suppressed information comes with verifiable documents, not just social media posts.",
            "The 'hidden truth' frame is self-sealing — any denial just confirms the conspiracy.",
        ],
        "False Certainty": [
            "Absolute language ('always', 'never') rarely holds up in complex real-world situations.",
            "Ask for evidence behind 'obvious' claims — obvious things still require proof.",
        ],
        "Emotional Overload": [
            "Strong emotions are a signal to slow down, not speed up your response.",
            "Ask: is this giving you facts, or mainly trying to make you feel something?",
        ],
    }

    raw_counters = []
    seen = set()
    for cat in trigger_counts:
        if trigger_counts[cat] > 0 and cat in counter_lookup:
            for c in counter_lookup[cat]:
                if c not in seen:
                    seen.add(c)
                    raw_counters.append(c)

    if propaganda_detected:
        raw_counters.append(
            "Phrases like 'share before deleted' or 'they don't want you to know' are manipulation hooks "
            "with no verifiable basis. Treat them as immediate red flags."
        )

    if not raw_counters:
        raw_counters = [
            "Cross-check information with at least 2–3 independent credible sources.",
            "Emotional or urgent language does not guarantee truth.",
            "Evaluate claims critically before sharing.",
        ]

    counter_message = "**Counter-messages for this specific content:**\n\n"
    counter_message += "\n\n".join(f"- {c}" for c in raw_counters[:6])

    advice_lines = []
    if score >= 70:
        advice_lines.append("🔴 **Do NOT share** this content without verifying through at least 3 independent credible sources.")
        advice_lines.append("🔴 Notice your emotional reaction — the reaction itself may be the manipulation.")
    elif score >= 40:
        advice_lines.append("🟡 **Approach with caution.** Verify key claims with primary sources before acting or sharing.")

    if fake_p > 0.6:
        advice_lines.append("🔴 The AI flagged likely fabrication. Try to find the original event or study mentioned — it may not exist.")
    elif fake_p > 0.4:
        advice_lines.append("🟡 Possible content distortion. Check the original source rather than relying on this version.")

    if "Conspiracy" in trigger_counts and trigger_counts["Conspiracy"] > 0:
        advice_lines.append("🔴 Conspiracy framing detected. Real whistleblowing involves verifiable documents — ask for primary evidence.")
    if "Us vs Them" in trigger_counts and trigger_counts["Us vs Them"] > 0:
        advice_lines.append("🟡 Tribal division tactics present. Seek perspectives from multiple sides before forming strong opinions.")
    if "Fear" in trigger_counts and trigger_counts["Fear"] > 0:
        advice_lines.append("🟡 Fear is being used as a persuasion tool. Look up the actual statistical likelihood of the described threat.")
    if "Urgency" in trigger_counts and trigger_counts["Urgency"] > 0:
        advice_lines.append("🟡 Artificial urgency discourages verification. Real decisions rarely need to be made in minutes.")

    advice_lines.append("🟢 Use fact-checkers: Snopes, PolitiFact, FactCheck.org, AFP Fact Check, AP Fact Check.")
    advice_lines.append("🟢 Check for bylines, publication dates, and institutional affiliations.")
    advice_lines.append("🟢 Ask: 'How would I react if the subject were someone I support?' — consistency reveals bias.")

    advice_block = "\n\n".join(advice_lines)

    highlighted_text = text
    for word in sorted(all_matched_words, key=len, reverse=True):
        highlighted_text = re.sub(
            r"\b(" + re.escape(word) + r")\b",
            r"<span class='highlight'>\1</span>",
            highlighted_text,
            flags=re.IGNORECASE,
        )

    manipulative_words = sorted(all_matched_words)

    return (
        score, trigger_counts, highlighted_text,
        explanation, counter_message, risk,
        manipulative_words, advice_block,
        propaganda_hits, fallacy_hits, extreme_sentences,
    )


# =============================
# PDF GENERATION
# =============================
def generate_pdf(text, score, trigger_counts, explanation, counter_message, risk, advice_block, propaganda_hits, fallacy_hits):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    style = getSampleStyleSheet()

    elements.append(Paragraph("<b>MindShield AI Report</b>", style["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Manipulation Score: {score}/100", style["Normal"]))
    elements.append(Paragraph(f"Risk Level: {risk}", style["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Trigger Word Counts:", style["Heading2"]))
    for k, v in trigger_counts.items():
        if v > 0:
            elements.append(Paragraph(f"  {k}: {v}", style["Normal"]))

    if propaganda_hits:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph("Propaganda Phrases Detected:", style["Heading2"]))
        for p in propaganda_hits:
            elements.append(Paragraph(f"  - {p}", style["Normal"]))

    if fallacy_hits:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph("Logical Fallacies Detected:", style["Heading2"]))
        for f, kws in fallacy_hits.items():
            elements.append(Paragraph(f"  {f}: {', '.join(kws)}", style["Normal"]))

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("AI Explanation:", style["Heading2"]))
    clean_exp = re.sub(r"\*\*(.+?)\*\*", r"\1", explanation)
    for line in clean_exp.split("\n\n"):
        if line.strip():
            elements.append(Paragraph(line.strip(), style["Normal"]))
            elements.append(Spacer(1, 0.08 * inch))

    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("Counter-Messages:", style["Heading2"]))
    clean_cm = re.sub(r"\*\*(.+?)\*\*", r"\1", counter_message)
    for line in clean_cm.split("\n\n"):
        if line.strip():
            elements.append(Paragraph(line.strip(), style["Normal"]))
            elements.append(Spacer(1, 0.06 * inch))

    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("Advice:", style["Heading2"]))
    clean_adv = re.sub(r"\*\*(.+?)\*\*", r"\1", advice_block)
    clean_adv = re.sub(r"[🔴🟡🟢]", "", clean_adv)
    for line in clean_adv.split("\n\n"):
        if line.strip():
            elements.append(Paragraph(line.strip(), style["Normal"]))
            elements.append(Spacer(1, 0.06 * inch))

    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Analysed Text:", style["Heading2"]))
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    elements.append(Paragraph(safe[:2000] + ("..." if len(text) > 2000 else ""), style["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# =============================
# UI — IMAGE UPLOAD
# =============================
st.markdown("#### 🖼️ Upload an Image (optional)")
st.caption("Upload a screenshot, meme, news photo, or any image with text — we'll extract and analyse it.")

uploaded_image = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if uploaded_image is not None:
    pil_img = Image.open(uploaded_image)
    st.image(pil_img, caption="Uploaded Image", width=600)

    if st.button("🔠 Extract Text from Image"):
        with st.spinner("Reading text from image (multi-engine OCR)..."):
            extracted = extract_text_from_image(pil_img)
        if extracted:
            st.session_state["extracted_text"] = extracted
            st.success(f"Extracted {len(extracted.split())} words from image.")
        else:
            st.warning("Could not find readable text in this image. Try a clearer or higher-resolution photo.")

    if st.session_state.get("extracted_text"):
        with st.expander("📄 Extracted Text (auto-filled below)", expanded=True):
            st.text(st.session_state["extracted_text"])

st.markdown("---")
st.markdown("#### 📝 Or Paste Text Directly")

prefill = st.session_state.get("extracted_text", "")
text_input = st.text_area("Paste the content here:", value=prefill, height=200)

if st.button("🔍 Analyze"):

    if text_input.strip() == "":
        st.warning("Please enter some text or upload an image first.")
    else:
        with st.spinner("Running analysis..."):
            (
                score, trigger_counts, highlighted_text,
                explanation, counter_message, risk,
                manipulative_words, advice_block,
                propaganda_hits, fallacy_hits, extreme_sentences,
            ) = analyze_text(text_input)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        gauge_color = "#FF4B4B" if score >= 70 else ("#F59E0B" if score >= 40 else "#10B981")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40],   "color": "#1C1F26"},
                    {"range": [40, 70],  "color": "#1C1F26"},
                    {"range": [70, 100], "color": "#1C1F26"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 4},
                    "thickness": 0.85,
                    "value": score,
                },
            },
            title={"text": "Manipulation Score"},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=280)
        st.plotly_chart(fig, width="stretch", config={"responsive": True})

        st.markdown(f"### {risk}")

        fig2, ax = plt.subplots(figsize=(8, 3))
        fig2.patch.set_alpha(0)
        ax.set_facecolor("#1C1F26")
        cats = list(trigger_counts.keys())
        vals = list(trigger_counts.values())
        bar_colors = ["#FF4B4B" if v >= 3 else "#00F5FF" if v >= 1 else "#2D3142" for v in vals]
        ax.barh(cats, vals, color=bar_colors)
        ax.set_xlabel("Keyword count", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D3142")
        ax.set_title("Trigger Words by Category", color="white")
        st.pyplot(fig2)

        with st.expander("🤖 AI Explanation"):
            st.markdown(explanation)

        with st.expander("🛡 Counter-Messages"):
            st.markdown(counter_message)

        with st.expander("💡 Advice"):
            st.markdown(advice_block)

        if propaganda_hits or fallacy_hits:
            with st.expander("📢 Propaganda Phrases & Logical Fallacies"):
                if propaganda_hits:
                    st.markdown("**Propaganda phrases detected:**")
                    for p in propaganda_hits:
                        st.markdown(f"- `{p}`")
                if fallacy_hits:
                    st.markdown("**Logical fallacies detected:**")
                    for fname, fkws in fallacy_hits.items():
                        st.markdown(f"- **{fname}:** `{', '.join(fkws)}`")

        if extreme_sentences:
            with st.expander("⚡ Most Emotionally Extreme Sentences"):
                for s in extreme_sentences[:5]:
                    st.markdown(
                        f"<div style='border-left:3px solid #FF4B4B;padding:4px 12px;"
                        f"color:#fca5a5;font-size:0.9rem;margin:4px 0'>{s}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("### 🔎 Highlighted Manipulative Words")
        st.markdown(
            f"<div style='line-height:1.9;font-size:0.95rem;white-space:pre-wrap;'>{highlighted_text}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 📝 Manipulative Words Found")
        if manipulative_words:
            st.write(", ".join(manipulative_words))
        else:
            st.write("None detected.")

        pdf_buf = generate_pdf(
            text_input, score, trigger_counts,
            explanation, counter_message, risk,
            advice_block, propaganda_hits, fallacy_hits,
        )
        st.download_button(
            "📄 Download PDF Report", pdf_buf,
            file_name="MindShield_Report.pdf",
            mime="application/pdf",
        )

        st.markdown("</div>", unsafe_allow_html=True)
