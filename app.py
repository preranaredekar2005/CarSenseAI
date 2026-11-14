"""
CarSense AI | Premium decision-intelligence dashboard
Run with: streamlit run app.py

The interface presents the implemented recommendation pipeline, model evaluation,
cost analysis, feedback loop and limitations for an academic demonstration.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent import (
    build_assets, run_agent, parse_query, default_profile, compute_costs,
    priority_scores, PRIORITIES, CRIT_LABELS
)
from data import BODY_TYPES, FUEL_TYPES, CARS_DF
from evaluate import run_evaluation

st.set_page_config(
    page_title="CarSense AI | Decision Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Design Tokens & Coordinated Palette
# ---------------------------------------------------------------------
PALETTE = {
    "navy": "#071426",
    "surface": "#0C1E35",
    "surface_elevated": "#112947",
    "border": "#1E3A5F",
    "border_strong": "#2E5382",
    "text": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "muted": "#94A3B8",
    "electric_blue": "#38BDF8",
    "teal": "#2DD4BF",
    "purple": "#A78BFA",
    "gold": "#FBBF24",
    "rose": "#FB7185",
}

CHART_COLORS = [
    "#38BDF8",  # Electric Blue
    "#2DD4BF",  # Teal
    "#A78BFA",  # Purple
    "#FBBF24",  # Gold
    "#FB7185",  # Coral / Rose
    "#60A5FA",  # Sky Indigo
    "#34D399",  # Emerald
    "#F472B6",  # Pink
]

# ---------------------------------------------------------------------
# Custom CSS Styling (Embedded inside Python for Streamlit)
# ---------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --navy: {PALETTE['navy']};
  --surface: {PALETTE['surface']};
  --surface-elevated: {PALETTE['surface_elevated']};
  --border: {PALETTE['border']};
  --border-strong: {PALETTE['border_strong']};
  --text: {PALETTE['text']};
  --text-secondary: {PALETTE['text_secondary']};
  --muted: {PALETTE['muted']};
  --blue: {PALETTE['electric_blue']};
  --teal: {PALETTE['teal']};
  --purple: {PALETTE['purple']};
  --gold: {PALETTE['gold']};
}}

html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}}

.stApp {{
  background-color: var(--navy);
  background-image:
    radial-gradient(circle at 12% 0%, rgba(56, 189, 248, 0.10) 0%, transparent 42%),
    radial-gradient(circle at 88% 8%, rgba(45, 212, 191, 0.07) 0%, transparent 38%),
    linear-gradient(180deg, #071426 0%, #050F1E 100%);
  color: var(--text);
}}

[data-testid="stHeader"] {{
  background: rgba(7, 20, 38, 0.88);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0A1B30 0%, #071426 100%);
  border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] * {{
  color: var(--text);
}}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {{
  color: var(--muted) !important;
}}

/* Typography & Section Headings */
h1, h2, h3, h4 {{
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
  letter-spacing: -0.025em;
}}
h1 {{
  font-size: 2.35rem !important;
  font-weight: 700 !important;
  margin-bottom: 0.25rem !important;
}}
h2 {{
  font-size: 1.5rem !important;
  font-weight: 600 !important;
  padding-bottom: 0.45rem;
  border-bottom: 1px solid var(--border);
  margin-top: 1.25rem !important;
  margin-bottom: 0.85rem !important;
}}
h3 {{
  font-size: 1.18rem !important;
  font-weight: 600 !important;
}}

p, .stMarkdown, [data-testid="stText"] {{
  color: var(--text-secondary);
  line-height: 1.6;
}}
[data-testid="stCaptionContainer"], .stCaption {{
  color: var(--muted) !important;
  font-size: 0.86rem !important;
}}

/* Form Controls, Labels, Inputs & Dropdowns */
label, [data-testid="stWidgetLabel"] p {{
  color: var(--text) !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
}}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {{
  background-color: var(--surface-elevated) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 10px !important;
}}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
  color: var(--muted) !important;
}}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.22) !important;
}}

/* Selectbox & Multiselect Dropdowns */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
  background-color: var(--surface-elevated) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 10px !important;
}}
.stSelectbox div[data-baseweb="select"] span,
.stMultiSelect div[data-baseweb="select"] span {{
  color: var(--text) !important;
}}
div[data-baseweb="popover"],
ul[data-baseweb="menu"] {{
  background-color: var(--surface-elevated) !important;
  border: 1px solid var(--border-strong) !important;
}}
ul[data-baseweb="menu"] li {{
  color: var(--text) !important;
}}
ul[data-baseweb="menu"] li:hover {{
  background-color: rgba(56, 189, 248, 0.16) !important;
}}
span[data-baseweb="tag"] {{
  background-color: rgba(56, 189, 248, 0.18) !important;
  border: 1px solid rgba(56, 189, 248, 0.45) !important;
  color: var(--text) !important;
}}

/* Buttons */
.stButton > button {{
  background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
  color: #04101E !important;
  border: 1px solid rgba(56, 189, 248, 0.5);
  border-radius: 10px;
  font-weight: 700;
  padding: 0.55rem 1rem;
  transition: all 0.2s ease;
}}
.stButton > button:hover {{
  background: linear-gradient(135deg, #38BDF8 0%, #2DD4BF 100%);
  color: #04101E !important;
  border-color: var(--teal);
  box-shadow: 0 4px 18px rgba(56, 189, 248, 0.30);
  transform: translateY(-1px);
}}

/* Metric Cards */
[data-testid="stMetric"] {{
  background: linear-gradient(155deg, var(--surface-elevated) 0%, var(--surface) 100%);
  border: 1px solid var(--border-strong);
  border-top: 3px solid var(--blue);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 26px rgba(2, 8, 20, 0.35);
}}
[data-testid="stMetricLabel"] p {{
  color: var(--muted) !important;
  font-size: 0.82rem !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
  color: var(--text) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.65rem !important;
  font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{
  color: var(--teal) !important;
  font-weight: 500 !important;
}}

/* Containers & Borders */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: linear-gradient(160deg, rgba(17, 41, 71, 0.92), rgba(12, 30, 53, 0.96));
  border: 1px solid var(--border-strong);
  border-radius: 14px;
  padding: 4px;
}}

/* Navigation Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 6px;
  border-radius: 12px;
  margin-bottom: 1rem;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  color: var(--muted);
  border-radius: 8px;
  padding: 9px 15px;
  height: auto;
  font-weight: 600;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: var(--text);
  background: rgba(56, 189, 248, 0.08);
}}
.stTabs [aria-selected="true"] {{
  background: rgba(56, 189, 248, 0.16) !important;
  color: var(--blue) !important;
  border: 1px solid rgba(56, 189, 248, 0.35) !important;
}}

/* Tables & Dataframes */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border-strong);
  border-radius: 12px;
  overflow: hidden;
  background: var(--surface);
}}

/* Alerts & Expanders */
div[data-testid="stAlert"] {{
  background: var(--surface-elevated);
  border: 1px solid var(--border-strong);
  border-left: 4px solid var(--blue);
  color: var(--text);
  border-radius: 10px;
}}
div[data-testid="stExpander"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
}}
div[data-testid="stExpander"] summary p {{
  color: var(--text) !important;
  font-weight: 600 !important;
}}

hr {{
  border-color: var(--border);
  margin: 1.2rem 0;
}}

/* Custom Utility Classes */
.cs-eyebrow {{
  color: var(--blue);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 0.25rem;
}}
.cs-subtitle {{
  color: var(--text-secondary);
  font-size: 1.05rem;
  margin-top: 0;
}}
.cs-section-tag {{
  display: inline-block;
  color: var(--teal);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 2px;
}}
.cs-card {{
  background: linear-gradient(155deg, var(--surface-elevated) 0%, var(--surface) 100%);
  border: 1px solid var(--border-strong);
  border-radius: 16px;
  padding: 20px 22px;
  margin: 0.4rem 0;
  box-shadow: 0 10px 28px rgba(2, 8, 20, 0.32);
}}
.cs-card-hero {{
  border-top: 3px solid var(--gold);
}}
.cs-pill {{
  display: inline-block;
  padding: 4px 11px;
  border-radius: 999px;
  background: rgba(45, 212, 191, 0.14);
  border: 1px solid rgba(45, 212, 191, 0.35);
  color: var(--teal);
  font-size: 0.76rem;
  font-weight: 600;
  margin-right: 6px;
  margin-bottom: 4px;
}}
.cs-pill-gold {{
  background: rgba(251, 191, 36, 0.15);
  border: 1px solid rgba(251, 191, 36, 0.45);
  color: var(--gold);
}}
.cs-pill-blue {{
  background: rgba(56, 189, 248, 0.15);
  border: 1px solid rgba(56, 189, 248, 0.4);
  color: var(--blue);
}}
.cs-muted {{
  color: var(--muted);
  font-size: 0.86rem;
}}
.cs-number {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--text);
}}
.cs-rank-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 12px;
  border-radius: 8px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  background: rgba(56, 189, 248, 0.15);
  border: 1px solid rgba(56, 189, 248, 0.4);
  color: var(--blue);
}}
.cs-rank-badge-top {{
  background: rgba(251, 191, 36, 0.18);
  border: 1px solid rgba(251, 191, 36, 0.55);
  color: var(--gold);
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Plotly Styling Helper (Prevents duplicate xaxis/yaxis keyword errors)
# ---------------------------------------------------------------------
def style_fig(fig, **kwargs):
    base = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12, 30, 53, 0.45)",
        font=dict(family="DM Sans", color=PALETTE["text"], size=13),
        title_font=dict(family="Space Grotesk", color=PALETTE["text"], size=16),
        margin=dict(l=24, r=24, t=58, b=32),
        legend=dict(
            bgcolor="rgba(12, 30, 53, 0.85)",
            bordercolor=PALETTE["border_strong"],
            borderwidth=1,
            font=dict(color=PALETTE["text"], size=12),
        ),
    )
    base.update(kwargs)
    fig.update_layout(**base)
    fig.update_xaxes(
        gridcolor="rgba(46, 83, 130, 0.35)",
        zerolinecolor=PALETTE["border_strong"],
        tickfont=dict(color=PALETTE["text_secondary"], size=12),
        title_font=dict(color=PALETTE["text"], size=13),
    )
    fig.update_yaxes(
        gridcolor="rgba(46, 83, 130, 0.35)",
        zerolinecolor=PALETTE["border_strong"],
        tickfont=dict(color=PALETTE["text_secondary"], size=12),
        title_font=dict(color=PALETTE["text"], size=13),
    )
    return fig


@st.cache_resource(show_spinner="Building the recommendation, resale and peer models...")
def get_assets():
    return build_assets()


@st.cache_data(show_spinner="Evaluating recommendation methods across 8 buyer personas...")
def get_evaluation():
    return run_evaluation(get_assets())


assets = get_assets()

# ---------------------------------------------------------------------
# State and controls
# ---------------------------------------------------------------------
for key, val in {
    "budget": 12.0, "seats": 5, "km": 12000, "usage": "Mixed",
    "fuels": list(FUEL_TYPES), "bodies": [], "min_ncap": 0,
    "home": True, "years": 5, "liked": [], "disliked": [],
    "nlq": "",
}.items():
    st.session_state.setdefault(key, val)
for p in PRIORITIES:
    st.session_state.setdefault("pri_" + p, 5)


def apply_query():
    prof = parse_query(st.session_state["nlq"])
    st.session_state.update({
        "budget": prof["budget_lakh"], "seats": prof["seats"],
        "km": prof["km_per_year"], "usage": prof["usage"],
        "fuels": prof["fuels"], "bodies": prof["bodies"],
        "home": prof["home_charging"],
    })
    for p in PRIORITIES:
        st.session_state["pri_" + p] = int(prof["priorities"][p])
    st.session_state["liked"], st.session_state["disliked"] = [], []


def like(i):
    if i not in st.session_state["liked"]:
        st.session_state["liked"].append(i)
    if i in st.session_state["disliked"]:
        st.session_state["disliked"].remove(i)


def dislike(i):
    if i not in st.session_state["disliked"]:
        st.session_state["disliked"].append(i)
    if i in st.session_state["liked"]:
        st.session_state["liked"].remove(i)


def reset_feedback():
    st.session_state["liked"], st.session_state["disliked"] = [], []


with st.sidebar:
    st.markdown("<div class='cs-eyebrow'>Decision Intelligence Suite</div>", unsafe_allow_html=True)
    st.title("🚘 CarSense AI")
    st.caption("Autonomous car-buying decision agent")
    st.markdown("---")
    st.markdown("<span class='cs-section-tag'>Step 01</span><br>**Describe your needs**", unsafe_allow_html=True)
    st.text_area("Natural-language request", key="nlq", height=95,
                 placeholder="Family of 5, budget 12 lakh, 15k km/year, safe SUV, good resale")
    st.button("✦ Understand my request", on_click=apply_query, use_container_width=True)
    st.markdown("---")
    st.markdown("<span class='cs-section-tag'>Step 02</span><br>**Buyer constraints**", unsafe_allow_html=True)
    st.number_input("On-road budget (₹ lakh)", 3.0, 60.0, step=0.5, key="budget")
    st.selectbox("Minimum seats", [4, 5, 7], key="seats")
    st.slider("Annual distance (km)", 3000, 40000, step=1000, key="km")
    st.selectbox("Driving pattern", ["City", "Mixed", "Highway"], key="usage")
    st.slider("Ownership horizon (years)", 3, 8, key="years")
    st.multiselect("Allowed fuel types", FUEL_TYPES, key="fuels")
    st.multiselect("Body styles (none = any)", BODY_TYPES, key="bodies")
    st.selectbox("Minimum safety stars", [0, 3, 4, 5], key="min_ncap")
    st.checkbox("Home EV charging available", key="home")
    st.markdown("---")
    st.markdown("<span class='cs-section-tag'>Step 03</span><br>**Preference priorities**", unsafe_allow_html=True)
    for p in PRIORITIES:
        st.slider(p, 1, 10, key="pri_" + p)
    st.markdown("---")
    st.markdown("<span class='cs-section-tag'>Step 04</span><br>**Hybrid model weights**", unsafe_allow_html=True)
    wt = st.slider("TOPSIS weight", 0.0, 1.0, 0.60, 0.05)
    wc = st.slider("Content similarity weight", 0.0, 1.0, 0.20, 0.05)
    wp = st.slider("Peer / SVD weight", 0.0, 1.0, 0.20, 0.05)
    st.caption(f"Current weights total: {wt+wc+wp:.2f}. Normalized automatically during scoring.")
    if st.button("Reset feedback", use_container_width=True):
        reset_feedback()

if not st.session_state["fuels"]:
    st.session_state["fuels"] = list(FUEL_TYPES)

profile = {
    "budget_lakh": st.session_state["budget"], "seats": st.session_state["seats"],
    "km_per_year": st.session_state["km"], "usage": st.session_state["usage"],
    "years": st.session_state["years"], "fuels": st.session_state["fuels"],
    "bodies": st.session_state["bodies"], "min_ncap": st.session_state["min_ncap"],
    "home_charging": st.session_state["home"],
    "priorities": {p: float(st.session_state["pri_" + p]) for p in PRIORITIES},
}
mw = (wt, wc, wp) if (wt + wc + wp) > 0 else (0.6, 0.2, 0.2)
result = run_agent(
    profile, assets, liked=st.session_state["liked"],
    disliked=st.session_state["disliked"], method_weights=mw
)
ranked = result["ranked"]
if len(ranked) == 0:
    st.error("No feasible cars were found. Increase the budget or relax a hard constraint.")
    st.stop()

# ---------------------------------------------------------------------
# Header / top-level overview
# ---------------------------------------------------------------------
st.markdown("<div class='cs-eyebrow'>Autonomous Decision Support • India Car Market</div>", unsafe_allow_html=True)
st.title("CarSense AI")
st.markdown("<div class='cs-subtitle'>From buyer intent to explainable, cost-aware vehicle recommendations.</div>", unsafe_allow_html=True)
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

top = ranked.iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Feasible shortlist", f"{result['n_candidates']} / {len(CARS_DF)}", "After hard-constraint filtering")
k2.metric("Top-ranked vehicle", top["name"], f"{top['score']*100:.1f} / 100 relative score")
k3.metric(f"{profile['years']}-year ownership cost", f"₹{top['tco']/1e5:.2f} L", f"₹{top['tco_per_km']:.2f} per km")
k4.metric("Ranking robustness", f"{top['stability']*100:.0f}%", "Top-3 under ±20% weight perturbations")

if result["relaxed"]:
    st.warning("The agent automatically re-planned some constraints: " + "; ".join(result["relaxed"]))
st.caption("Academic prototype • 61 illustrative catalogue entries • outputs are estimates, not live dealer quotations.")

tabs = st.tabs([
    "✦ Executive overview", "🏆 Recommendations", "🧠 How the agent works",
    "📊 Algorithms & evaluation", "💰 Ownership economics", "⚖️ Compare vehicles"
])

# ---------------------------------------------------------------------
# 1 Executive overview
# ---------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader("Your decision at a glance")
        st.markdown(
            f"""<div class='cs-card cs-card-hero'>
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
              <span class='cs-pill cs-pill-gold'>★ #1 Top Recommendation</span>
              <span style='color:{PALETTE["gold"]};font-family:Space Grotesk;font-weight:700;font-size:1.05rem'>{top['score']*100:.1f} / 100 Match Score</span>
            </div>
            <div style='font-family:Space Grotesk;font-size:1.95rem;font-weight:700;color:{PALETTE["text"]};margin:0.45rem 0 0.55rem 0'>{top['name']}</div>
            <div>
              <span class='cs-pill'>{top['fuel']}</span>
              <span class='cs-pill'>{top['body']}</span>
              <span class='cs-pill'>{int(top['seats'])} seats</span>
              <span class='cs-pill cs-pill-blue'>{int(top['ncap'])}★ NCAP</span>
            </div>
            <p style='color:{PALETTE["text_secondary"]};margin-top:16px;font-size:0.98rem'>{top['why']}</p>
            <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:18px;padding-top:14px;border-top:1px solid {PALETTE["border"]}'>
              <div><div class='cs-muted'>On-road estimate</div><div class='cs-number'>₹{top['onroad_lakh']:.2f} L</div></div>
              <div><div class='cs-muted'>Estimated resale</div><div class='cs-number' style='color:{PALETTE["teal"]}'>{top['resale_pct']:.1f}%</div></div>
              <div><div class='cs-muted'>{profile['years']}-yr total cost</div><div class='cs-number' style='color:{PALETTE["gold"]}'>₹{top['tco']/1e5:.2f} L</div></div>
            </div></div>""", unsafe_allow_html=True
        )
        st.subheader("Top five recommendation scores")
        top5 = ranked.head(5).sort_values("score")
        bar_colors = [PALETTE["electric_blue"]] * (len(top5) - 1) + [PALETTE["gold"]]
        fig = px.bar(
            top5, x="score", y="name", orientation="h",
            text=top5["score"].map(lambda x: f"{x*100:.1f}"),
            labels={"score": "Relative shortlist score", "name": "Vehicle"}
        )
        fig.update_traces(marker_color=bar_colors, textposition="outside", textfont=dict(color=PALETTE["text"], size=12))
        style_fig(fig, xaxis=dict(range=[0, 1.15], title="Relative score (0–1 scale, #1 highlighted in gold)"), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Buyer profile snapshot")
        snapshot = [
            ("Budget", f"₹{profile['budget_lakh']:.1f} lakh"),
            ("Minimum seating", f"{profile['seats']} seats"),
            ("Annual driving", f"{profile['km_per_year']:,} km"),
            ("Usage pattern", profile["usage"]),
            ("Ownership horizon", f"{profile['years']} years"),
            ("Fuel preference", ", ".join(profile["fuels"])),
            ("Body style", ", ".join(profile["bodies"]) if profile["bodies"] else "Any"),
            ("Home charging", "Available" if profile["home_charging"] else "Not available"),
        ]
        rows_html = "".join(
            f"<div style='display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid {PALETTE['border']};padding:10px 4px'>"
            f"<span class='cs-muted'>{label}</span>"
            f"<b style='color:{PALETTE['text']};text-align:right'>{value}</b></div>"
            for label, value in snapshot
        )
        st.markdown(f"<div class='cs-card' style='padding:12px 18px'>{rows_html}</div>", unsafe_allow_html=True)

        st.subheader("Preference priorities")
        pri_df = pd.DataFrame({
            "Priority": list(profile["priorities"].keys()),
            "Importance": list(profile["priorities"].values())
        }).sort_values("Importance")
        fig = px.bar(pri_df, x="Importance", y="Priority", orientation="h", range_x=[0, 11], text="Importance")
        fig.update_traces(marker_color=PALETTE["teal"], textposition="outside", textfont=dict(color=PALETTE["text"]))
        style_fig(fig, height=310, xaxis_title="Importance weight (1–10)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.info("Use the left control panel to adjust buyer constraints or weights. All recommendation, cost, and comparison views update automatically.")

# ---------------------------------------------------------------------
# 2 Recommendations
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Ranked shortlist")
    st.caption("Hard constraints determine eligibility; the hybrid model ranks eligible vehicles. Feedback re-ranks the shortlist in real time.")
    for i, row in ranked.head(8).iterrows():
        is_top = int(row["rank"]) == 1
        badge_cls = "cs-rank-badge cs-rank-badge-top" if is_top else "cs-rank-badge"
        score_color = PALETTE["gold"] if is_top else PALETTE["electric_blue"]
        fb_badge = ""
        if i in st.session_state["liked"]:
            fb_badge = "<span class='cs-pill' style='margin-top:6px'>👍 Liked</span>"
        elif i in st.session_state["disliked"]:
            fb_badge = f"<span class='cs-pill' style='background:rgba(251,113,133,0.15);border-color:rgba(251,113,133,0.45);color:{PALETTE['rose']};margin-top:6px'>👎 Disliked</span>"

        with st.container(border=True):
            c1, c2, c3 = st.columns([1.25, 2.55, 1.05])
            with c1:
                st.markdown(
                    f"<span class='{badge_cls}'>#{int(row['rank'])} {'TOP PICK' if is_top else ''}</span>"
                    f"<div style='font-family:Space Grotesk;font-size:1.22rem;font-weight:700;color:{PALETTE['text']};margin-top:8px'>{row['name']}</div>"
                    f"<div class='cs-muted'>{row['brand']}</div>{fb_badge}",
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"<div style='margin-bottom:6px'>"
                    f"<span class='cs-pill'>{row['fuel']}</span>"
                    f"<span class='cs-pill'>{row['body']}</span>"
                    f"<span class='cs-pill cs-pill-blue'>{int(row['seats'])} seats</span>"
                    f"<span class='cs-pill cs-pill-gold'>{int(row['ncap'])}★ NCAP</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"<p style='color:{PALETTE['text_secondary']};margin:6px 0 10px 0'>{row['why']}</p>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='display:flex;gap:18px;flex-wrap:wrap;font-size:0.86rem;color:{PALETTE['muted']}'>"
                    f"<span>On-road: <b style='color:{PALETTE['text']}'>₹{row['onroad_lakh']:.2f} L</b></span>"
                    f"<span>{profile['years']}-yr TCO: <b style='color:{PALETTE['text']}'>₹{row['tco']/1e5:.2f} L</b></span>"
                    f"<span>Running: <b style='color:{PALETTE['text']}'>₹{row['cost_km']:.2f}/km</b></span>"
                    f"<span>Resale: <b style='color:{PALETTE['teal']}'>{row['resale_pct']:.1f}%</b></span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f"<div style='text-align:right;margin-bottom:8px'>"
                    f"<div class='cs-muted' style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em'>Match Score</div>"
                    f"<div style='font-family:Space Grotesk;font-size:1.75rem;font-weight:700;color:{score_color}'>{row['score']*100:.1f}<span style='font-size:0.95rem;color:{PALETTE['muted']}'>/100</span></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                b1, b2 = st.columns(2)
                b1.button("👍", key=f"like_{i}", on_click=like, args=(i,), help="Increase preference for this car", use_container_width=True)
                b2.button("👎", key=f"dislike_{i}", on_click=dislike, args=(i,), help="Reduce preference for this car", use_container_width=True)

    full = ranked[["rank", "name", "brand", "fuel", "body", "onroad_lakh", "tco", "cost_km", "ncap", "resale_pct", "score", "stability"]].copy()
    full["tco"] = full["tco"] / 1e5
    full["score"] = full["score"] * 100
    full["stability"] = full["stability"] * 100
    full.columns = ["Rank", "Vehicle", "Brand", "Fuel", "Body type", "On-road (₹L)", f"{profile['years']}-yr TCO (₹L)", "₹/km", "Safety ★", "Resale %", "Relative score", "Stability %"]
    st.subheader("Complete eligible shortlist")
    st.dataframe(full.round(2), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# 3 Agent architecture and reasoning
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("End-to-end decision pipeline")
    steps_meta = [
        ("01 · PERCEIVE", "Natural-language request + filters", PALETTE["electric_blue"]),
        ("02 · ESTIMATE", "TCO + Random Forest resale", PALETTE["teal"]),
        ("03 · FILTER", "Budget, seats, fuel, safety", PALETTE["purple"]),
        ("04 · SCORE", "TOPSIS + content + peer", PALETTE["gold"]),
        ("05 · DECIDE", "Rank and select top options", PALETTE["electric_blue"]),
        ("06 · EXPLAIN", "Strengths, trade-offs, trace", PALETTE["teal"]),
        ("07 · LEARN", "Like / dislike preference update", PALETTE["purple"]),
        ("08 · RE-PLAN", "Relax selected constraints if needed", PALETTE["gold"]),
    ]
    cards_html = "".join(
        f"<div style='background:{PALETTE['surface_elevated']};border:1px solid {PALETTE['border_strong']};border-top:3px solid {accent};padding:14px;border-radius:12px'>"
        f"<b style='color:{accent};font-family:Space Grotesk;font-size:0.92rem'>{title}</b><br>"
        f"<span style='color:{PALETTE['text_secondary']};font-size:0.84rem'>{desc}</span></div>"
        for title, desc, accent in steps_meta
    )
    st.markdown(
        f"<div class='cs-card'><div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;text-align:center'>{cards_html}</div></div>",
        unsafe_allow_html=True
    )

    st.subheader("What happened for this buyer?")
    for step in result["trace"]:
        label, _, detail = step.partition(": ")
        with st.container(border=True):
            st.markdown(f"<span style='color:{PALETTE['electric_blue']};font-weight:700'>{label}</span>", unsafe_allow_html=True)
            st.write(detail)

    col_pie, col_crit = st.columns(2, gap="large")
    with col_pie:
        st.subheader("Hybrid score composition")
        weights_total = sum(mw)
        wdf = pd.DataFrame({
            "Component": ["TOPSIS / multi-criteria", "Content-based similarity", "Peer-based collaborative (SVD)"],
            "Configured weight": [mw[0] / weights_total, mw[1] / weights_total, mw[2] / weights_total]
        })
        fig = px.pie(
            wdf, names="Component", values="Configured weight", hole=0.58,
            color_discrete_sequence=[PALETTE["electric_blue"], PALETTE["teal"], PALETTE["purple"]]
        )
        fig.update_traces(textposition="inside", textinfo="percent", textfont=dict(color="#071426", size=13, family="Space Grotesk"))
        style_fig(fig, height=380, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_crit:
        st.subheader("TOPSIS criteria weighting")
        wcrit = pd.DataFrame({"Criterion": CRIT_LABELS, "Weight": result["weights"]}).sort_values("Weight")
        fig = px.bar(wcrit, x="Weight", y="Criterion", orientation="h", text=wcrit["Weight"].map(lambda x: f"{x:.2f}"))
        fig.update_traces(marker_color=PALETTE["teal"], textposition="outside", textfont=dict(color=PALETTE["text"]))
        style_fig(fig, height=380, xaxis=dict(range=[0, max(wcrit["Weight"]) * 1.2], title="Final criterion weight"), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("The agent blends 75% buyer-priority weights with 25% entropy-derived data weights so rankings reflect both stated intent and catalogue differentiation.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**Natural-language understanding**")
            st.write("Offline, rule-based parser extracts common budget, seats, distance, fuel, body style, usage and priority phrases. Follow-up inputs can retain the previous buyer profile in the CLI chatbot.")
    with c2:
        with st.container(border=True):
            st.markdown("**Feedback loop**")
            st.write("Liked and disliked cars adjust preference priorities and feed the next ranking. The dashboard stores feedback in the current Streamlit session.")
    st.info("This is a rule-based conversational decision agent. It does not currently call an LLM API or use RAG.")

# ---------------------------------------------------------------------
# 4 Algorithms and model evaluation
# ---------------------------------------------------------------------
with tabs[3]:
    ev = get_evaluation()
    st.subheader("Algorithm comparison laboratory")
    st.markdown("Compares simple baselines, individual recommendation components (ablations), the complete hybrid, and the hybrid after simulated feedback.")
    methods = ev["summary"].reset_index().rename(columns={"index": "Method"})
    if "method" in methods.columns:
        methods = methods.rename(columns={"method": "Method"})

    metric_options = ["Precision@5", "NDCG@5", "GradedRel@5", "HitRate@5", "MRR"]
    chosen_metric = st.selectbox("Select evaluation metric", metric_options, index=0)
    sort_df = methods.sort_values(chosen_metric, ascending=True)

    algo_colors = []
    for m_name in sort_df["Method"]:
        lower = str(m_name).lower()
        if "feedback" in lower:
            algo_colors.append(PALETTE["gold"])
        elif "hybrid" in lower:
            algo_colors.append(PALETTE["purple"])
        elif any(k in lower for k in ["topsis", "content", "svd", "peer"]):
            algo_colors.append(PALETTE["electric_blue"])
        else:
            algo_colors.append(PALETTE["teal"])

    fig = px.bar(
        sort_df, x=chosen_metric, y="Method", orientation="h", text=chosen_metric,
        range_x=[0, 1.12]
    )
    fig.update_traces(
        marker_color=algo_colors,
        texttemplate="%{x:.3f}",
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=12)
    )
    style_fig(
        fig,
        height=510,
        showlegend=False,
        title=f"Method Comparison by {chosen_metric} (Purple = Hybrid, Gold = Hybrid + Feedback)",
        xaxis_title=f"{chosen_metric} (higher is better)",
        yaxis_title=""
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(methods.round(3), use_container_width=True, hide_index=True)

    st.subheader("What the recommendation metrics mean")
    metric_cols = st.columns(5)
    metric_help = [
        ("Precision@5", "Share of top-five recommendations that satisfy every expert-defined must-have.", PALETTE["electric_blue"]),
        ("NDCG@5", "Rewards relevant cars appearing higher in the ranked list.", PALETTE["teal"]),
        ("GradedRel@5", "Average fraction of must-have conditions satisfied by top-five cars.", PALETTE["purple"]),
        ("HitRate@5", "Whether at least one fully relevant car appears in the top five.", PALETTE["gold"]),
        ("MRR", "Rewards how early the first fully relevant car appears.", PALETTE["electric_blue"]),
    ]
    for col, (name, desc, accent) in zip(metric_cols, metric_help):
        with col:
            st.markdown(
                f"<div class='cs-card' style='border-top:3px solid {accent};height:100%'>"
                f"<b style='color:{accent};font-family:Space Grotesk'>{name}</b>"
                f"<p class='cs-muted' style='margin-top:6px'>{desc}</p></div>",
                unsafe_allow_html=True
            )

    st.subheader("Resale-value model · Random Forest Regressor")
    m = ev["resale_metrics"]
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{m['MAE (pct points)']:.2f}", "percentage points")
    b.metric("RMSE", f"{m['RMSE (pct points)']:.2f}", "percentage points")
    c.metric("R²", f"{m['R2']:.3f}")
    d.metric("Held-out test rows", f"{int(m['test rows (unseen models)']):,}", f"Train rows: {int(m['train rows']):,}")

    actual, predicted = ev["test_pairs"]
    x, y = st.columns(2, gap="large")
    with x:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=predicted, mode="markers", name="Predicted vs Actual",
            marker=dict(color=PALETTE["purple"], size=8, opacity=0.75, line=dict(width=1, color=PALETTE["text"]))
        ))
        lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal Fit (y = x)",
            line=dict(color=PALETTE["gold"], width=2.5, dash="dash")
        ))
        style_fig(
            fig,
            title="Actual vs predicted resale retention",
            xaxis_title="Actual resale retention (%)",
            yaxis_title="Predicted resale retention (%)",
            legend=dict(x=0.04, y=0.96, bgcolor="rgba(12, 30, 53, 0.85)", bordercolor=PALETTE["border_strong"], borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    with y:
        imp = ev["importances"].head(10).sort_values()
        fig = px.bar(
            x=imp.values, y=imp.index, orientation="h",
            text=[f"{v:.3f}" for v in imp.values],
            labels={"x": "Feature importance", "y": "Input feature"}
        )
        fig.update_traces(marker_color=PALETTE["purple"], textposition="outside", textfont=dict(color=PALETTE["text"]))
        style_fig(
            fig,
            title="Random Forest feature importance (Top 10)",
            xaxis=dict(range=[0, max(imp.values) * 1.18], title="Relative importance"),
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Resale model inputs:** ex-showroom price, brand score, safety rating, vehicle age, annual kilometres (in thousands), fuel-type indicators and body-type indicators.")
    st.subheader("Persona-level robustness")
    stability = ev["stability"].copy()
    stability["Top-1 robustness"] = (stability["top1_stability"] * 100).round(1)
    st.dataframe(stability[["persona", "fully_relevant_cars", "Top-1 robustness"]], use_container_width=True, hide_index=True)
    st.caption("Evaluation protocol: 8 expert-defined buyer personas. Must-have rules form the relevance ground truth and are not passed to the recommender. Baselines include random-in-budget, cheapest-first, mileage-first and brand-popularity. Results are offline academic evaluation, not a live-market benchmark.")
    with st.expander("Read the model limitations before presenting"):
        st.write("The resale model is trained on synthetic resale targets and tested with a group split by car model. Peer ratings are also synthetic. The catalogue contains 61 illustrative car models, while persona ground truth is expert-defined. Strong test metrics therefore validate the implementation pipeline on generated data; they do not establish real-world resale accuracy or production recommendation quality.")

# ---------------------------------------------------------------------
# 5 TCO and what-if economics
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Total cost of ownership over {profile['years']} years")
    st.caption("TCO = depreciation + fuel/charging + servicing + insurance. Values are estimates driven by the buyer's annual distance and ownership horizon.")
    t5 = ranked.head(5)
    cb = pd.DataFrame({
        "Vehicle": t5["name"],
        "Depreciation": t5["depreciation"] / 1e5,
        "Fuel / charging": t5["fuel_cost"] / 1e5,
        "Servicing": t5["maint_cost"] / 1e5,
        "Insurance": t5["ins_cost"] / 1e5
    })
    fig = px.bar(
        cb.melt(id_vars="Vehicle", var_name="Cost component", value_name="₹ lakh"),
        x="Vehicle", y="₹ lakh", color="Cost component", barmode="stack",
        color_discrete_sequence=[PALETTE["electric_blue"], PALETTE["teal"], PALETTE["purple"], PALETTE["gold"]]
    )
    style_fig(fig, title="Ownership-cost composition for the top five")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensitivity to annual driving distance")
    kms = list(range(3000, 40001, 3000))
    curves = {}
    for km in kms:
        cost = compute_costs({**profile, "km_per_year": km}, assets["model"])
        for idx in ranked.head(5).index:
            curves.setdefault(cost.loc[idx, "name"], []).append(cost.loc[idx, "tco"] / 1e5)
    wf = pd.DataFrame(curves, index=kms).rename_axis("Annual km").reset_index().melt(id_vars="Annual km", var_name="Vehicle", value_name="TCO (₹ lakh)")
    fig = px.line(wf, x="Annual km", y="TCO (₹ lakh)", color="Vehicle", markers=True, color_discrete_sequence=CHART_COLORS)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    style_fig(fig, title="How annual usage changes total ownership cost")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("EV vs petrol / diesel break-even explorer")
    ice_names = CARS_DF.loc[CARS_DF["fuel"] != "Electric", "name"].tolist()
    ev_names = CARS_DF.loc[CARS_DF["fuel"] == "Electric", "name"].tolist()
    ca, cbcol = st.columns(2, gap="large")
    ice = ca.selectbox("Compare conventional vehicle", ice_names, index=ice_names.index("Tata Nexon") if "Tata Nexon" in ice_names else 0)
    ev_name = cbcol.selectbox("Compare electric vehicle", ev_names, index=ev_names.index("Tata Nexon EV") if "Tata Nexon EV" in ev_names else 0)
    grid = list(range(2000, 40001, 2000))
    ii = CARS_DF.index[CARS_DF["name"] == ice][0]
    ei = CARS_DF.index[CARS_DF["name"] == ev_name][0]
    ice_tco, ev_tco = [], []
    for km in grid:
        cc = compute_costs({**profile, "km_per_year": km}, assets["model"])
        ice_tco.append(cc.loc[ii, "tco"] / 1e5)
        ev_tco.append(cc.loc[ei, "tco"] / 1e5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=ice_tco, name=ice, mode="lines+markers", line=dict(color=PALETTE["gold"], width=3)))
    fig.add_trace(go.Scatter(x=grid, y=ev_tco, name=ev_name, mode="lines+markers", line=dict(color=PALETTE["teal"], width=3)))
    style_fig(
        fig,
        title="Estimated ownership cost by annual distance",
        xaxis_title="Annual kilometres",
        yaxis_title=f"{profile['years']}-year TCO (₹ lakh)"
    )
    st.plotly_chart(fig, use_container_width=True)
    diff = np.array(ev_tco) - np.array(ice_tco)
    cross = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(cross):
        j = cross[0]
        be = grid[j] + (grid[j + 1] - grid[j]) * (0 - diff[j]) / (diff[j + 1] - diff[j])
        st.success(f"Under the current assumptions, the estimated cost curves cross near {be:,.0f} km/year.")
    else:
        st.info("No cost-curve crossing was detected in the 2,000–40,000 km/year range under the current assumptions.")
    st.warning("Illustrative assumptions only: fuel prices, real-world mileage, service and insurance are modelled. Local taxes, variant, financing, charging access and dealer quotes can materially change actual costs.")

# ---------------------------------------------------------------------
# 6 Vehicle comparison
# ---------------------------------------------------------------------
with tabs[5]:
    st.subheader("Side-by-side vehicle comparison")
    selected = st.multiselect(
        "Select 2–4 cars from the eligible shortlist",
        ranked["name"].tolist(),
        default=ranked["name"].head(min(3, len(ranked))).tolist(),
        max_selections=4
    )
    if len(selected) >= 2:
        sel = ranked[ranked["name"].isin(selected)]
        pscore = priority_scores(ranked).loc[sel.index]
        fig = go.Figure()
        for c_idx, idx in enumerate(sel.index):
            vals = pscore.loc[idx].tolist()
            color = CHART_COLORS[c_idx % len(CHART_COLORS)]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=PRIORITIES + [PRIORITIES[0]],
                fill="toself",
                name=sel.loc[idx, "name"],
                line=dict(color=color, width=2.5),
                opacity=0.8
            ))
        style_fig(
            fig,
            polar=dict(
                bgcolor="rgba(12, 30, 53, 0.55)",
                radialaxis=dict(range=[0, 1], gridcolor="rgba(46, 83, 130, 0.5)", tickfont=dict(color=PALETTE["muted"])),
                angularaxis=dict(gridcolor="rgba(46, 83, 130, 0.5)", tickfont=dict(color=PALETTE["text"], size=12))
            ),
            title="Relative strengths across buyer priorities"
        )
        st.plotly_chart(fig, use_container_width=True)
        compare = sel.set_index("name")[["fuel", "body", "onroad_lakh", "tco", "cost_km", "bhp", "boot_l", "clearance", "ncap", "resale_pct"]].copy()
        compare["tco"] = compare["tco"] / 1e5
        compare = compare.T
        compare.index = ["Fuel", "Body type", "On-road (₹ lakh)", f"{profile['years']}-yr TCO (₹ lakh)", "Running cost (₹/km)", "Power (bhp)", "Boot (L)", "Ground clearance (mm)", "Safety stars", "Resale estimate (%)"]
        st.dataframe(compare, use_container_width=True)
    else:
        st.info("Select at least two vehicles to display a meaningful comparison.")

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<div style='display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap'>"
    f"<span class='cs-muted'><b style='color:{PALETTE['text']}'>CAR SENSE AI</b> · Explainable decision-support prototype</span>"
    f"<span class='cs-muted'>Synthetic resale labels & peer feedback · Illustrative catalogue · Not financial or purchase advice</span>"
    f"</div>",
    unsafe_allow_html=True
)