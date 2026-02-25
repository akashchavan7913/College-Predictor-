"""
╔══════════════════════════════════════════════════════════════╗
║       ENGINEERING COLLEGE PREDICTOR - AI/ML PROJECT          ║
║   Exams: JEE | State CETs | COMEDK | Other Engineering Exams ║
║                   Powered by Gemini AI                        ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
    pip install streamlit google-generativeai pandas plotly scikit-learn

RUN:
    streamlit run college_predictor.py
"""

# ─────────────────────────── IMPORTS ───────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()  # Loads GEMINI_API_KEY from .env automatically

import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ─────────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="Engineering College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CUSTOM CSS ────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Sora:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f093fb, #f5576c, #fda085);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        text-align: center;
        color: #aaa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .exam-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
    }

    .college-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.6rem 0;
    }

    .chance-high   { color: #00e676; font-weight: 700; }
    .chance-medium { color: #ffea00; font-weight: 700; }
    .chance-low    { color: #ff5252; font-weight: 700; }

    .metric-box {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stButton>button {
        background: linear-gradient(135deg, #f093fb, #f5576c) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.4) !important;
    }

    .sidebar-header {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        color: #f093fb;
        font-size: 1.1rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label {
        color: #ddd !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── CONSTANTS ─────────────────────────

EXAM_CONFIG = {
    "JEE Main": {
        "score_label": "JEE Main Score (out of 300)",
        "rank_label": "JEE Main Rank (CRL)",
        "percentile_label": "JEE Main Percentile",
        "score_max": 300,
        "rank_max": 1200000,
        "categories": ["General", "OBC-NCL", "SC", "ST", "EWS", "PwD"],
        "quota": ["Home State", "All India", "Management", "NRI"],
        "states": ["All India"],
        "description": "National exam for NITs, IIITs, GFTIs",
    },
    "JEE Advanced": {
        "score_label": "JEE Advanced Score (out of 360)",
        "score_max": 360,
        "rank_label": "JEE Advanced Rank (CRL)",
        "rank_max": 50000,
        "percentile_label": "JEE Advanced Percentile",
        "categories": ["General", "OBC-NCL", "SC", "ST", "EWS", "PwD"],
        "quota": ["Common", "Preparatory"],
        "states": ["All India"],
        "description": "Exam for IITs – India's premier engineering institutions",
    },
    "MHT-CET": {
        "score_label": "MHT-CET Score (out of 200)",
        "score_max": 200,
        "rank_label": "MHT-CET Percentile",
        "rank_max": 100,
        "percentile_label": "MHT-CET Percentile",
        "categories": ["OPEN", "OBC", "SC", "ST", "VJ/DT", "NT1", "NT2", "NT3", "SBC", "EWS"],
        "quota": ["Home University", "Other Than Home University", "State Level"],
        "states": ["Maharashtra"],
        "description": "Maharashtra state engineering entrance exam",
    },
    "COMEDK": {
        "score_label": "COMEDK Score (out of 180)",
        "score_max": 180,
        "rank_label": "COMEDK Rank",
        "rank_max": 100000,
        "percentile_label": "COMEDK Percentile",
        "categories": ["General", "SC", "ST", "OBC"],
        "quota": ["General Merit", "Management", "NRI"],
        "states": ["Karnataka"],
        "description": "Karnataka private engineering college entrance",
    },
    "KCET": {
        "score_label": "KCET Score (out of 120)",
        "score_max": 120,
        "rank_label": "KCET Rank",
        "rank_max": 200000,
        "percentile_label": "KCET Percentile",
        "categories": ["GM", "SC", "ST", "OBC", "Cat-1", "2A", "2B", "3A", "3B"],
        "quota": ["Horanadu Kannadiga", "Gadinadu Kannadiga", "General Merit"],
        "states": ["Karnataka"],
        "description": "Karnataka state CET for government engineering colleges",
    },
    "WBJEE": {
        "score_label": "WBJEE Score (out of 200)",
        "score_max": 200,
        "rank_label": "WBJEE Rank",
        "rank_max": 100000,
        "percentile_label": "WBJEE Percentile",
        "categories": ["UR", "OBC-A", "OBC-B", "SC", "ST"],
        "quota": ["State", "All India (30%)"],
        "states": ["West Bengal"],
        "description": "West Bengal Joint Entrance Examination",
    },
    "KEAM": {
        "score_label": "KEAM Score (out of 960)",
        "score_max": 960,
        "rank_label": "KEAM Rank",
        "rank_max": 100000,
        "percentile_label": "KEAM Percentile",
        "categories": ["General", "SEBC", "SC", "ST"],
        "quota": ["General", "Management", "NRI"],
        "states": ["Kerala"],
        "description": "Kerala Engineering Architecture Medical entrance",
    },
    "VITEEE": {
        "score_label": "VITEEE Score (out of 125)",
        "score_max": 125,
        "rank_label": "VITEEE Rank",
        "rank_max": 200000,
        "percentile_label": "VITEEE Percentile",
        "categories": ["General", "SC", "ST"],
        "quota": ["General", "NRI", "Management"],
        "states": ["Tamil Nadu"],
        "description": "VIT University Engineering Entrance Exam",
    },
    "BITSAT": {
        "score_label": "BITSAT Score (out of 450)",
        "score_max": 450,
        "rank_label": "BITSAT Score Rank",
        "rank_max": 100000,
        "percentile_label": "BITSAT Percentile",
        "categories": ["General"],
        "quota": ["General", "Industry Integrated"],
        "states": ["All India"],
        "description": "BITS Pilani, Goa, Hyderabad campuses",
    },
    "UPCET/AKTU": {
        "score_label": "UPCET Score",
        "score_max": 600,
        "rank_label": "UPCET Rank",
        "rank_max": 500000,
        "percentile_label": "UPCET Percentile",
        "categories": ["UR", "OBC", "SC", "ST"],
        "quota": ["State", "Home District"],
        "states": ["Uttar Pradesh"],
        "description": "Uttar Pradesh Combined Entrance Test",
    },
}

BRANCHES = [
    "Computer Science & Engineering (CSE)",
    "CSE - Artificial Intelligence & ML",
    "CSE - Data Science",
    "CSE - Cyber Security",
    "Information Technology (IT)",
    "Electronics & Communication Engineering (ECE)",
    "Electrical Engineering (EE)",
    "Mechanical Engineering (ME)",
    "Civil Engineering",
    "Chemical Engineering",
    "Biotechnology Engineering",
    "Aerospace Engineering",
    "Production Engineering",
    "Instrumentation Engineering",
    "Mining Engineering",
    "Naval Architecture",
    "Any (Show All Branches)",
]

# ─────────────────────────── GEMINI SETUP ──────────────────────

def configure_gemini(api_key: str):
    genai.configure(api_key="AIzaSyC-6OqXApfO0Ugrb10HdLDRfgtYuLKL0Sw")
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def build_prompt(data: dict) -> str:
    return f"""
You are an expert Indian engineering college admission counselor with deep knowledge of all entrance exams, cutoffs, and counseling processes.

A student needs college predictions based on their entrance exam score. Return a JSON response ONLY.

STUDENT PROFILE:
- Exam: {data['exam']}
- Score/Percentile: {data['score']}
- Rank: {data['rank']}
- Category: {data['category']}
- Home State: {data['home_state']}
- Preferred Branch: {data['branch']}
- Gender: {data.get('gender', 'Not specified')}
- 12th Percentage: {data.get('board_pct', 'Not provided')}%
- Preferred Cities/States: {data.get('pref_location', 'Anywhere in India')}
- Budget (Annual Fees): ₹{data.get('budget', 'No preference')} Lakhs

RETURN STRICT JSON FORMAT:
{{
  "summary": "2-3 line personalized summary of the student's profile and chances",
  "overall_chance": "High/Medium/Low",
  "colleges": [
    {{
      "rank": 1,
      "college_name": "Full College Name",
      "location": "City, State",
      "branch": "Branch Name",
      "type": "IIT/NIT/IIIT/Government/Private/Deemed",
      "cutoff_score": "Expected cutoff for this category",
      "student_score": "{data['score']}",
      "admission_chance": "High (>80%)/Medium (40-80%)/Low (<40%)",
      "annual_fees": "₹X Lakhs",
      "nirf_rank": "NIRF Ranking (if applicable)",
      "placement_avg": "Average package in LPA",
      "placement_highest": "Highest package in LPA",
      "key_highlights": ["highlight1", "highlight2", "highlight3"],
      "counseling_round": "Expected admission round (Round 1/2/3/Spot)",
      "quota_applicable": "Category/Quota applicable"
    }}
  ],
  "strategy_tips": [
    "Tip 1 for counseling strategy",
    "Tip 2",
    "Tip 3",
    "Tip 4"
  ],
  "cutoff_trend": {{
    "2022": "score/rank",
    "2023": "score/rank",
    "2024": "score/rank",
    "trend": "Rising/Stable/Falling"
  }},
  "alternative_exams": ["Exam1 if applicable", "Exam2"],
  "important_dates": "Key counseling dates and deadlines"
}}

Generate EXACTLY 8-12 college predictions ranging from Safe → Moderate → Ambitious choices.
Ensure realistic, accurate cutoff data for {data['exam']} {data['category']} category.
Return ONLY the JSON, no markdown, no extra text.
"""


def parse_gemini_response(response_text: str) -> dict:
    """Extract JSON from Gemini response."""
    # Try direct parse
    try:
        return json.loads(response_text)
    except Exception:
        pass
    # Extract JSON block
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ─────────────────────────── ML SCORING ────────────────────────

def ml_score_colleges(colleges: list, student_score: float, budget_lakhs: float) -> list:
    """
    Apply a simple ML-style scoring on top of Gemini predictions
    using weighted feature normalization (MinMaxScaler).
    """
    if not colleges:
        return colleges

    records = []
    for c in colleges:
        try:
            cutoff_raw = re.sub(r'[^\d.]', '', str(c.get("cutoff_score", "0"))) or "0"
            cutoff = float(cutoff_raw)
        except Exception:
            cutoff = 0

        try:
            fees_raw = re.sub(r'[^\d.]', '', str(c.get("annual_fees", "0"))) or "0"
            fees = float(fees_raw)
        except Exception:
            fees = 10

        try:
            pkg_raw = re.sub(r'[^\d.]', '', str(c.get("placement_avg", "0"))) or "0"
            pkg = float(pkg_raw)
        except Exception:
            pkg = 0

        chance_map = {"High (>80%)": 3, "Medium (40-80%)": 2, "Low (<40%)": 1}
        chance_score = chance_map.get(c.get("admission_chance", "Low (<40%)"), 1)

        records.append({"cutoff": cutoff, "fees": fees, "pkg": pkg, "chance": chance_score})

    df = pd.DataFrame(records)

    # Normalize
    scaler = MinMaxScaler()
    try:
        df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    except Exception:
        df_norm = df.copy()

    # Weighted composite score
    weights = {"chance": 0.4, "pkg": 0.35, "fees": -0.15, "cutoff": 0.1}
    df_norm["composite"] = sum(df_norm[col] * w for col, w in weights.items())

    for i, c in enumerate(colleges):
        c["ml_score"] = round(df_norm["composite"].iloc[i] * 100, 1)

    # Sort by ml_score descending
    colleges.sort(key=lambda x: x.get("ml_score", 0), reverse=True)
    for i, c in enumerate(colleges):
        c["rank"] = i + 1
    return colleges


# ─────────────────────────── UI COMPONENTS ─────────────────────

def render_hero():
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem;'>
        <span class='exam-badge'>JEE Main</span>
        <span class='exam-badge'>JEE Advanced</span>
        <span class='exam-badge'>MHT-CET</span>
        <span class='exam-badge'>COMEDK</span>
        <span class='exam-badge'>KCET</span>
        <span class='exam-badge'>WBJEE</span>
        <span class='exam-badge'>BITSAT</span>
        <span class='exam-badge'>VITEEE</span>
        <span class='exam-badge'>KEAM</span>
        <span class='exam-badge'>UPCET</span>
    </div>
    <div class='hero-title'>🎓 Engineering College Predictor</div>
    <div class='hero-sub'>AI-powered college predictions using Gemini · All major entrance exams covered</div>
    """, unsafe_allow_html=True)


def render_college_card(c: dict, idx: int):
    chance = c.get("admission_chance", "")
    if "High" in chance:
        badge = "<span class='chance-high'>✅ High Chance</span>"
    elif "Medium" in chance:
        badge = "<span class='chance-medium'>⚡ Medium Chance</span>"
    else:
        badge = "<span class='chance-low'>⚠️ Low Chance</span>"

    highlights = "".join(f"<li style='color:#ccc;font-size:0.82rem'>{h}</li>"
                         for h in c.get("key_highlights", []))

    ml_score = c.get("ml_score", "—")
    ml_bar = f"""
    <div style='margin-top:0.5rem'>
        <span style='color:#aaa;font-size:0.75rem'>ML Fit Score: {ml_score}/100</span>
        <div style='background:#333;border-radius:6px;height:6px;margin-top:3px'>
            <div style='background:linear-gradient(90deg,#f093fb,#f5576c);
                        width:{ml_score}%;height:6px;border-radius:6px'></div>
        </div>
    </div>
    """ if ml_score != "—" else ""

    st.markdown(f"""
    <div class='college-card'>
        <div style='display:flex;justify-content:space-between;align-items:start'>
            <div>
                <span style='color:#f093fb;font-weight:700;font-size:0.85rem'>#{c.get('rank','?')}</span>
                <span style='color:white;font-weight:700;font-size:1.05rem;margin-left:8px'>{c.get('college_name','')}</span>
                <span style='background:rgba(255,255,255,0.1);color:#ddd;padding:2px 8px;
                             border-radius:10px;font-size:0.72rem;margin-left:8px'>{c.get('type','')}</span>
            </div>
            <div>{badge}</div>
        </div>
        <div style='color:#aaa;font-size:0.85rem;margin-top:4px'>
            📍 {c.get('location','')} &nbsp;|&nbsp;
            🎓 {c.get('branch','')} &nbsp;|&nbsp;
            🏆 NIRF: {c.get('nirf_rank','—')}
        </div>
        <div style='display:flex;gap:1.5rem;margin-top:0.8rem;flex-wrap:wrap'>
            <div><span style='color:#aaa;font-size:0.75rem'>Cutoff</span><br>
                 <span style='color:#f5576c;font-weight:600'>{c.get('cutoff_score','—')}</span></div>
            <div><span style='color:#aaa;font-size:0.75rem'>Annual Fees</span><br>
                 <span style='color:#fda085;font-weight:600'>{c.get('annual_fees','—')}</span></div>
            <div><span style='color:#aaa;font-size:0.75rem'>Avg Package</span><br>
                 <span style='color:#00e676;font-weight:600'>{c.get('placement_avg','—')} LPA</span></div>
            <div><span style='color:#aaa;font-size:0.75rem'>Highest Pkg</span><br>
                 <span style='color:#64ffda;font-weight:600'>{c.get('placement_highest','—')} LPA</span></div>
            <div><span style='color:#aaa;font-size:0.75rem'>Round</span><br>
                 <span style='color:#e0e0e0;font-weight:600'>{c.get('counseling_round','—')}</span></div>
        </div>
        <ul style='margin-top:0.6rem;padding-left:1.2rem'>{highlights}</ul>
        {ml_bar}
    </div>
    """, unsafe_allow_html=True)


def render_analytics(colleges: list):
    if not colleges:
        return

    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    chance_counts = {"High (>80%)": 0, "Medium (40-80%)": 0, "Low (<40%)": 0}
    for c in colleges:
        ch = c.get("admission_chance", "Low (<40%)")
        if ch in chance_counts:
            chance_counts[ch] += 1

    with col1:
        fig_pie = px.pie(
            names=list(chance_counts.keys()),
            values=list(chance_counts.values()),
            title="Admission Chance Distribution",
            color_discrete_sequence=["#00e676", "#ffea00", "#ff5252"],
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        type_counts = {}
        for c in colleges:
            t = c.get("type", "Other")
            type_counts[t] = type_counts.get(t, 0) + 1
        fig_bar = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            title="Colleges by Type",
            color_discrete_sequence=["#f093fb"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col3:
        pkgs, names = [], []
        for c in colleges:
            try:
                pkg = float(re.sub(r'[^\d.]', '', str(c.get("placement_avg", "0"))) or 0)
                if pkg > 0:
                    pkgs.append(pkg)
                    names.append(c.get("college_name", "")[:20])
            except Exception:
                pass
        if pkgs:
            fig_pkg = px.bar(
                x=pkgs, y=names, orientation="h",
                title="Avg Placement Package (LPA)",
                color=pkgs,
                color_continuous_scale="Plasma",
            )
            fig_pkg.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False,
            )
            st.plotly_chart(fig_pkg, use_container_width=True)

    # ML Score scatter
    ml_data = [(c.get("college_name", "")[:25], c.get("ml_score", 0),
                c.get("admission_chance", "")) for c in colleges if c.get("ml_score")]
    if ml_data:
        df_ml = pd.DataFrame(ml_data, columns=["College", "ML Score", "Chance"])
        fig_scatter = px.scatter(
            df_ml, x="College", y="ML Score", color="Chance",
            title="ML Fit Score per College",
            color_discrete_map={"High (>80%)": "#00e676", "Medium (40-80%)": "#ffea00", "Low (<40%)": "#ff5252"},
            size="ML Score",
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────── SIDEBAR ───────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        # Load from .env first, fallback to sidebar input
        env_key = os.getenv("api_key", "")
        if env_key:
            api_key = env_key
            st.success("✅ API Key loaded from .env")
        else:
            st.markdown("<div class='sidebar-header'>🔑 Gemini API Key</div>", unsafe_allow_html=True)
            api_key = st.text_input(
                "Paste your Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Get free key at https://aistudio.google.com/app/apikey"
            )
            st.caption("💡 Or add GEMINI_API_KEY=your_key in .env file")
        st.markdown("---")
        st.markdown("<div class='sidebar-header'>📝 Student Profile</div>", unsafe_allow_html=True)

        exam = st.selectbox("Select Entrance Exam", list(EXAM_CONFIG.keys()))
        cfg = EXAM_CONFIG[exam]

        st.caption(f"ℹ️ {cfg['description']}")

        score = st.number_input(
            cfg["score_label"],
            min_value=0.0,
            max_value=float(cfg["score_max"]),
            value=float(cfg["score_max"]) * 0.5,
            step=0.5,
        )

        rank = st.number_input(
            cfg["rank_label"],
            min_value=1,
            max_value=int(cfg["rank_max"]),
            value=int(cfg["rank_max"]) * 10 // 100,
            step=1,
        )

        category = st.selectbox("Category / Caste", cfg["categories"])
        gender = st.selectbox("Gender", ["Male", "Female", "Transgender"])

        home_state = st.text_input("Home State", placeholder="e.g. Maharashtra")
        board_pct = st.slider("12th Board Percentage", 50, 100, 80)

        branch = st.selectbox("Preferred Branch", BRANCHES)
        pref_location = st.text_input("Preferred Location (optional)",
                                      placeholder="e.g. Pune, Bangalore")
        budget = st.slider("Max Annual Fees (₹ Lakhs)", 0, 25, 8)

        st.markdown("---")

        return {
            "api_key": api_key,
            "exam": exam,
            "score": score,
            "rank": rank,
            "category": category,
            "gender": gender,
            "home_state": home_state,
            "board_pct": board_pct,
            "branch": branch,
            "pref_location": pref_location,
            "budget": budget,
        }


# ─────────────────────────── MAIN APP ──────────────────────────

def main():
    render_hero()

    # Sidebar inputs
    inputs = render_sidebar()

    with st.sidebar:
        predict_btn = st.button("🚀 Predict My Colleges", use_container_width=True)

    # ── Summary metrics row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-box'>
            <div style='color:#aaa;font-size:0.75rem'>Selected Exam</div>
            <div style='color:#f093fb;font-weight:700;font-size:1.1rem'>{inputs['exam']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-box'>
            <div style='color:#aaa;font-size:0.75rem'>Your Score</div>
            <div style='color:#fda085;font-weight:700;font-size:1.1rem'>{inputs['score']}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-box'>
            <div style='color:#aaa;font-size:0.75rem'>Your Rank</div>
            <div style='color:#f5576c;font-weight:700;font-size:1.1rem'>{inputs['rank']:,}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-box'>
            <div style='color:#aaa;font-size:0.75rem'>Category</div>
            <div style='color:#64ffda;font-weight:700;font-size:1.1rem'>{inputs['category']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction logic ──
    if predict_btn:
        if not inputs["api_key"]:
            st.error("⚠️ Gemini API key not found. Please add GEMINI_API_KEY=your_key in your .env file and restart the app.")
            st.info("Get a free key at https://aistudio.google.com/app/apikey")
            return
        if not inputs["home_state"]:
            st.warning("⚠️ Please enter your home state for accurate predictions.")
            return

        with st.spinner("🤖 Gemini AI is analyzing your profile and predicting colleges..."):
            try:
                model = configure_gemini(inputs["api_key"])
                prompt = build_prompt(inputs)
                response = model.generate_content(prompt)
                result = parse_gemini_response(response.text)

                if not result:
                    st.error("❌ Could not parse Gemini response. Try again.")
                    with st.expander("Raw Gemini Response"):
                        st.text(response.text)
                    return

                # Apply ML scoring
                colleges = ml_score_colleges(
                    result.get("colleges", []),
                    float(inputs["score"]),
                    float(inputs["budget"])
                )

                # ── Summary banner ──
                overall = result.get("overall_chance", "—")
                color_map = {"High": "#00e676", "Medium": "#ffea00", "Low": "#ff5252"}
                oc = color_map.get(overall, "#fff")
                st.markdown(f"""
                <div class='card'>
                    <div style='font-size:0.85rem;color:#aaa;margin-bottom:4px'>AI Analysis Summary</div>
                    <div style='color:white;font-size:1rem'>{result.get('summary','')}</div>
                    <div style='margin-top:0.8rem'>
                        Overall Admission Chances: <strong style='color:{oc}'>{overall}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Tabs ──
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["🏛️ College Predictions", "📊 Analytics", "💡 Strategy Tips", "📅 Dates & Alternatives"]
                )

                with tab1:
                    st.markdown(f"### 🏛️ {len(colleges)} College Predictions for {inputs['exam']}")
                    for i, c in enumerate(colleges):
                        render_college_card(c, i)

                with tab2:
                    render_analytics(colleges)

                with tab3:
                    st.markdown("### 💡 Counseling Strategy Tips")
                    tips = result.get("strategy_tips", [])
                    for i, tip in enumerate(tips, 1):
                        st.markdown(f"""
                        <div class='card' style='border-left:3px solid #f093fb'>
                            <span style='color:#f093fb;font-weight:700'>Tip {i}:</span>
                            <span style='color:#ddd;margin-left:8px'>{tip}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Cutoff trend
                    trend = result.get("cutoff_trend", {})
                    if trend:
                        st.markdown("### 📈 Cutoff Trend (Last 3 Years)")
                        years = [k for k in trend if k.isdigit()]
                        vals = [trend[y] for y in years]
                        st.markdown(f"""
                        <div class='card'>
                            {"  →  ".join(f"<strong>{y}</strong>: {v}" for y, v in zip(years, vals))}
                            <br><span style='color:#aaa;font-size:0.8rem'>Trend: {trend.get('trend','—')}</span>
                        </div>
                        """, unsafe_allow_html=True)

                with tab4:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("### 🗓️ Important Dates")
                        st.markdown(f"""<div class='card' style='color:#ddd'>
                            {result.get('important_dates','No date info available.')}
                        </div>""", unsafe_allow_html=True)
                    with col_b:
                        st.markdown("### 🔄 Alternative Exams You Can Apply For")
                        alts = result.get("alternative_exams", [])
                        for a in alts:
                            st.markdown(f"<div class='card' style='color:#fda085'>🎯 {a}</div>",
                                        unsafe_allow_html=True)

                # ── Export to CSV ──
                st.markdown("---")
                df_export = pd.DataFrame(colleges)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"college_predictions_{inputs['exam'].replace(' ','_')}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                    st.warning("⏳ Free tier quota exceeded. Options:")
                    st.markdown("""
                    - **Wait a few minutes** and try again (per-minute limit)
                    - **Wait until tomorrow** if daily limit is hit
                    - **Upgrade** your Google AI Studio plan for higher limits
                    - Visit [ai.dev/rate-limit](https://ai.dev/rate-limit) to check your usage
                    """)
                elif "400" in err_str or "API_KEY" in err_str.upper() or "invalid" in err_str.lower():
                    st.error("🔑 Invalid API key. Please check your .env file.")
                    st.info("💡 Get a valid free key at https://aistudio.google.com/app/apikey")

    else:
        # ── Landing instructions ──
        st.markdown("""
        <div class='card' style='text-align:center;padding:3rem'>
            <div style='font-size:4rem'>🤖</div>
            <div style='color:white;font-size:1.3rem;font-weight:700;margin:1rem 0'>How It Works</div>
            <div style='color:#aaa;max-width:600px;margin:0 auto;line-height:1.8'>
                1️⃣ Enter your <strong style='color:#f093fb'>Gemini API key</strong> in the sidebar<br>
                2️⃣ Select your <strong style='color:#fda085'>entrance exam</strong> (JEE / CET / COMEDK / etc.)<br>
                3️⃣ Fill in your <strong style='color:#64ffda'>score, rank, category & preferences</strong><br>
                4️⃣ Click <strong style='color:#f5576c'>🚀 Predict My Colleges</strong><br><br>
                Get AI-powered predictions with <strong>ML scoring</strong>, placement data,
                counseling strategy, and cutoff trends for <strong>8-12 colleges</strong> tailored to you.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class='card'>
                <div style='font-size:2rem'>🎯</div>
                <div style='color:#f093fb;font-weight:700;margin:0.5rem 0'>10 Exams Covered</div>
                <div style='color:#aaa;font-size:0.85rem'>JEE, MHT-CET, COMEDK, KCET, WBJEE, KEAM, VITEEE, BITSAT, UPCET & more</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class='card'>
                <div style='font-size:2rem'>🤖</div>
                <div style='color:#f093fb;font-weight:700;margin:0.5rem 0'>Gemini AI + ML Scoring</div>
                <div style='color:#aaa;font-size:0.85rem'>Gemini predictions enhanced with scikit-learn weighted scoring for better ranking</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class='card'>
                <div style='font-size:2rem'>📊</div>
                <div style='color:#f093fb;font-weight:700;margin:0.5rem 0'>Visual Analytics</div>
                <div style='color:#aaa;font-size:0.85rem'>Placement data, cutoff trends, chance distribution charts & CSV export</div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()