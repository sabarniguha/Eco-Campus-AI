import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# 1. CONFIGURATION AND CONSTANTS
# ==========================================================================

st.set_page_config(
    page_title="EcoCampus AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "EcoCampus AI"
APP_TAGLINE = "Intelligent Campus Sustainability Assistant"
AUTHOR = "Sabarni Guha"
AUTHOR_DETAIL = "B.Tech CSE (AI & ML), IEM-UEM"
PROGRAM = "1M1B AI for Sustainability Virtual Internship"

BUILDINGS = ["CSE Block", "Library", "Hostel A", "Administrative Block", "Science Block"]
DAYS = 30
RANDOM_SEED = 42

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sustainability_knowledge.txt")

ELEC_ANOMALY_THRESHOLD = 1.25   # flag when reading > 125% of building baseline
WATER_ANOMALY_THRESHOLD = 1.30  # flag when reading > 130% of building baseline

# Palette used across CSS and Plotly so the product feels like one thing.
INK = "#10231C"
SAGE = "#2F6D5B"
SAGE_SOFT = "#7FA99B"
WATER = "#2B6A8F"
ALERT = "#B8742A"
PAPER = "#F2F5F1"
LINE = "#DCE4DD"

BUILDING_COLORS = {
    "CSE Block": "#2F6D5B",
    "Library": "#5E8C7D",
    "Hostel A": "#2B6A8F",
    "Administrative Block": "#8AA79B",
    "Science Block": "#3F7F6D",
}

# ==========================================================================
# 2. STYLING
# ==========================================================================


def inject_styles() -> None:
    """Single CSS block for the whole app so pages stay visually consistent."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PAPER}; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

        h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}

        /* Product header */
        .eco-header {{
            background: linear-gradient(135deg, {INK} 0%, {SAGE} 100%);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.1rem;
            color: #FFFFFF;
        }}
        .eco-header .eco-title {{ font-size: 1.65rem; font-weight: 700; margin: 0; }}
        .eco-header .eco-sub {{ font-size: 0.98rem; opacity: 0.88; margin: 0.15rem 0 0.75rem 0; }}
        .eco-chip {{
            display: inline-block; padding: 0.22rem 0.62rem; margin: 0 0.35rem 0.35rem 0;
            border-radius: 999px; font-size: 0.74rem; font-weight: 600;
            background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
        }}
        .eco-chip-demo {{ background: {ALERT}; border-color: {ALERT}; }}

        /* Cards */
        .eco-card {{
            background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px;
            padding: 1rem 1.1rem; margin-bottom: 0.85rem;
        }}
        .eco-card h4 {{ margin: 0 0 0.4rem 0; font-size: 1rem; }}
        .eco-card p {{ margin: 0; color: #3C4A44; font-size: 0.92rem; line-height: 1.55; }}

        .eco-metric {{
            background: #FFFFFF; border: 1px solid {LINE}; border-left: 4px solid {SAGE};
            border-radius: 12px; padding: 0.85rem 1rem; height: 100%;
        }}
        .eco-metric .label {{ font-size: 0.78rem; color: #5B6B64; font-weight: 600; }}
        .eco-metric .value {{ font-size: 1.55rem; font-weight: 700; color: {INK}; line-height: 1.25; }}
        .eco-metric .delta {{ font-size: 0.78rem; color: #5B6B64; }}
        .eco-metric.alert {{ border-left-color: {ALERT}; }}
        .eco-metric.water {{ border-left-color: {WATER}; }}

        .eco-insight {{
            background: #FFFFFF; border: 1px solid {LINE}; border-left: 4px solid {SAGE_SOFT};
            border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.55rem;
            font-size: 0.92rem; color: #24352E;
        }}
        .eco-insight.flag {{ border-left-color: {ALERT}; }}

        .eco-status {{
            display: inline-block; padding: 0.35rem 0.85rem; border-radius: 8px;
            font-weight: 700; font-size: 0.92rem;
        }}
        .status-normal {{ background: #E4F0EA; color: {SAGE}; border: 1px solid {SAGE_SOFT}; }}
        .status-attention {{ background: #FBF0E2; color: {ALERT}; border: 1px solid #E5C69C; }}
        .status-investigate {{ background: #F7E4DE; color: #9B3B23; border: 1px solid #E0B4A6; }}

        .eco-calc {{
            background: #FFFFFF; border: 1px dashed {SAGE_SOFT}; border-radius: 10px;
            padding: 0.85rem 1rem; font-family: ui-monospace, "SF Mono", Menlo, monospace;
            font-size: 0.84rem; color: #24352E; line-height: 1.75;
        }}

        section[data-testid="stSidebar"] {{ background: {INK}; }}
        section[data-testid="stSidebar"] * {{ color: #E8EFEA; }}
        section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.93rem; }}

        div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
        .stButton button {{ border-radius: 9px; border: 1px solid {LINE}; font-weight: 600; }}
        .stDownloadButton button {{ border-radius: 9px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def product_header() -> None:
    st.markdown(
        f"""
        <div class="eco-header">
            <p class="eco-title">🌱 {APP_NAME}</p>
            <p class="eco-sub">{APP_TAGLINE} — AI, sustainability analytics and explainable insights</p>
            <span class="eco-chip eco-chip-demo">DEMO MODE · SYNTHETIC DATA</span>
            <span class="eco-chip">SDG 7</span>
            <span class="eco-chip">SDG 11</span>
            <span class="eco-chip">SDG 12</span>
            <span class="eco-chip">SDG 13</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "", variant: str = "") -> str:
    return f"""
    <div class="eco-metric {variant}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="delta">{delta}</div>
    </div>
    """


def info_card(title: str, body: str) -> str:
    return f'<div class="eco-card"><h4>{title}</h4><p>{body}</p></div>'


def style_fig(fig: go.Figure, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=INK, size=13),
        title=dict(font=dict(size=16)),
        margin=dict(l=10, r=10, t=55, b=10),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=y_title,
        xaxis_title=x_title,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


# ==========================================================================
# 3. SYNTHETIC DATA GENERATION
# ==========================================================================

BASELINE = {
    "CSE Block":           {"elec": 480, "water": 14, "occ": 78, "temp": 29},
    "Library":             {"elec": 260, "water": 8,  "occ": 65, "temp": 27},
    "Hostel A":            {"elec": 620, "water": 32, "occ": 90, "temp": 28},
    "Administrative Block": {"elec": 210, "water": 6,  "occ": 55, "temp": 27},
    "Science Block":       {"elec": 540, "water": 18, "occ": 70, "temp": 29},
}

# Deliberate anomalies: (building, day offset from start, electricity x, water x)
INJECTED_ANOMALIES = [
    ("CSE Block", 12, 1.55, None),
    ("Hostel A", 20, None, 1.60),
    ("Science Block", 25, 1.40, 1.50),
]


@st.cache_data
def generate_synthetic_data(seed: int = RANDOM_SEED, days: int = DAYS) -> pd.DataFrame:
    """Reproducible synthetic campus consumption data with planted anomalies."""
    rng = np.random.default_rng(seed)
    start_date = datetime.today().date() - timedelta(days=days - 1)

    rows = []
    for building in BUILDINGS:
        b = BASELINE[building]
        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)
            weekend_factor = 0.75 if date.weekday() >= 5 else 1.0
            # Hostels stay occupied at weekends, academic blocks empty out.
            if building == "Hostel A":
                weekend_factor = 1.0 if date.weekday() >= 5 else 0.95

            elec = b["elec"] * weekend_factor * rng.normal(1.0, 0.06)
            water = b["water"] * weekend_factor * rng.normal(1.0, 0.08)
            occ = float(np.clip(b["occ"] * weekend_factor * rng.normal(1.0, 0.05), 10, 100))
            temp = b["temp"] + rng.normal(0, 1.2)

            rows.append(
                {
                    "date": date,
                    "building": building,
                    "electricity_kwh": round(max(elec, 10.0), 1),
                    "water_kl": round(max(water, 1.0), 1),
                    "occupancy_pct": round(occ, 1),
                    "temperature_c": round(temp, 1),
                }
            )

    df = pd.DataFrame(rows)

    for building, offset, elec_mult, water_mult in INJECTED_ANOMALIES:
        target_date = start_date + timedelta(days=offset)
        mask = (df["building"] == building) & (df["date"] == target_date)
        if elec_mult:
            df.loc[mask, "electricity_kwh"] = (df.loc[mask, "electricity_kwh"] * elec_mult).round(1)
        if water_mult:
            df.loc[mask, "water_kl"] = (df.loc[mask, "water_kl"] * water_mult).round(1)

    return df


# ==========================================================================
# 4. ANOMALY DETECTION (transparent statistical thresholds)
# ==========================================================================


@st.cache_data
def compute_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each daily reading with its own building's period baseline."""
    if df.empty:
        return df

    out = df.copy()
    elec_avg = out.groupby("building")["electricity_kwh"].transform("mean")
    water_avg = out.groupby("building")["water_kl"].transform("mean")

    out["elec_building_avg"] = elec_avg.round(1)
    out["water_building_avg"] = water_avg.round(1)
    out["elec_pct_diff"] = ((out["electricity_kwh"] - elec_avg) / elec_avg * 100).round(1)
    out["water_pct_diff"] = ((out["water_kl"] - water_avg) / water_avg * 100).round(1)
    out["elec_anomaly"] = out["electricity_kwh"] > ELEC_ANOMALY_THRESHOLD * elec_avg
    out["water_anomaly"] = out["water_kl"] > WATER_ANOMALY_THRESHOLD * water_avg
    out["is_anomaly"] = out["elec_anomaly"] | out["water_anomaly"]
    return out


def baseline_for(df: pd.DataFrame, building: str) -> dict:
    subset = df[df["building"] == building]
    if subset.empty:
        return {"elec": 0.0, "water": 0.0}
    return {
        "elec": float(subset["electricity_kwh"].mean()),
        "water": float(subset["water_kl"].mean()),
    }


def evaluate_reading(elec: float, water: float, elec_avg: float, water_avg: float) -> dict:
    """Shared rule engine used by both the dashboard and the analysis form."""
    elec_pct = ((elec - elec_avg) / elec_avg * 100) if elec_avg else 0.0
    water_pct = ((water - water_avg) / water_avg * 100) if water_avg else 0.0
    elec_flag = elec_avg > 0 and elec > ELEC_ANOMALY_THRESHOLD * elec_avg
    water_flag = water_avg > 0 and water > WATER_ANOMALY_THRESHOLD * water_avg

    if elec_flag and water_flag:
        status, css = "Investigation recommended", "status-investigate"
    elif elec_flag or water_flag:
        status, css = "Attention required", "status-attention"
    else:
        status, css = "Normal range", "status-normal"

    return {
        "elec_pct": elec_pct,
        "water_pct": water_pct,
        "elec_flag": elec_flag,
        "water_flag": water_flag,
        "status": status,
        "status_css": css,
    }


def trend_vs_previous_week(df: pd.DataFrame, column: str) -> str:
    """Compare the last 7 days with the 7 days before them."""
    if df.empty:
        return ""
    daily = df.groupby("date")[column].sum().sort_index()
    if len(daily) < 14:
        return ""
    recent = daily.iloc[-7:].mean()
    prior = daily.iloc[-14:-7].mean()
    if prior == 0:
        return ""
    change = (recent - prior) / prior * 100
    arrow = "▲" if change > 0 else "▼"
    return f"{arrow} {abs(change):.1f}% vs previous 7 days"


def build_key_insights(df: pd.DataFrame) -> list:
    """Insights generated from the dataset only — nothing hard-coded."""
    insights = []
    if df.empty:
        return insights

    anomalies = df[df["is_anomaly"]]
    if anomalies.empty:
        insights.append(("ok", "No readings exceeded the current anomaly thresholds in this period."))
    else:
        for building, group in anomalies.groupby("building"):
            elec_hits = group[group["elec_anomaly"]]
            water_hits = group[group["water_anomaly"]]
            if not elec_hits.empty:
                worst = elec_hits.loc[elec_hits["elec_pct_diff"].idxmax()]
                insights.append(
                    (
                        "flag",
                        f"{building} recorded an electricity reading {worst['elec_pct_diff']:.1f}% above its "
                        f"{worst['elec_building_avg']:.0f} kWh/day baseline on {worst['date']}.",
                    )
                )
            if not water_hits.empty:
                worst = water_hits.loc[water_hits["water_pct_diff"].idxmax()]
                insights.append(
                    (
                        "flag",
                        f"{building} shows an unusual water pattern: {worst['water_kl']:.1f} kL on "
                        f"{worst['date']}, {worst['water_pct_diff']:.1f}% above its baseline.",
                    )
                )

    top_elec = df.groupby("building")["electricity_kwh"].mean().idxmax()
    top_elec_val = df.groupby("building")["electricity_kwh"].mean().max()
    insights.append(
        ("ok", f"{top_elec} is the largest electricity consumer at {top_elec_val:.0f} kWh/day on average.")
    )

    top_water = df.groupby("building")["water_kl"].mean().idxmax()
    top_water_val = df.groupby("building")["water_kl"].mean().max()
    insights.append(
        ("ok", f"{top_water} uses the most water at {top_water_val:.1f} kL/day on average.")
    )

    trend = trend_vs_previous_week(df, "electricity_kwh")
    if trend:
        insights.append(("ok", f"Campus electricity trend: {trend}."))

    return insights


# ==========================================================================
# 5. KNOWLEDGE BASE AND LIGHTWEIGHT RAG RETRIEVAL
# ==========================================================================

# Fallback knowledge used only if data/sustainability_knowledge.txt is missing,
# so the assistant never breaks during a demo.
FALLBACK_KB = [
    {
        "title": "LED lighting and lighting controls",
        "keywords": ["light", "lighting", "led", "bulb", "electricity", "energy"],
        "content": "Replacing older tube lights and bulbs with LED fittings typically lowers lighting load, "
                   "and occupancy sensors in corridors, washrooms and seminar halls stop lights running in empty rooms.",
    },
    {
        "title": "HVAC and cooling optimisation",
        "keywords": ["hvac", "air conditioning", "ac", "cooling", "temperature", "electricity"],
        "content": "Set air-conditioning to a fixed comfort setpoint, schedule cooling around real class timetables, "
                   "clean filters on a routine, and keep doors and windows shut in cooled rooms.",
    },
    {
        "title": "Computer lab and equipment shutdown",
        "keywords": ["computer", "lab", "pc", "equipment", "sleep", "shutdown", "idle", "electricity"],
        "content": "In computer labs, enable sleep and display-off policies, switch off monitors and projectors between "
                   "sessions, use smart power strips for peripherals, and assign a lab-closing shutdown checklist.",
    },
    {
        "title": "Peak-hour monitoring",
        "keywords": ["peak", "load", "demand", "monitoring", "spike", "increase", "sudden", "electricity"],
        "content": "Sudden electricity increases usually come from extended operating hours, extra cooling load on hot days, "
                   "equipment left running overnight, or a faulty appliance. Sub-metering and peak-hour logs narrow this down quickly.",
    },
    {
        "title": "Leak detection and water monitoring",
        "keywords": ["water", "leak", "tap", "pipe", "plumbing", "wastage", "waste water"],
        "content": "Overnight meter readings that do not drop to near zero are a strong leak signal. Routine checks of taps, "
                   "flush tanks and underground lines catch losses that are invisible during the day.",
    },
    {
        "title": "Low-flow fixtures and responsible water use",
        "keywords": ["water", "fixture", "flow", "aerator", "hostel", "conservation", "rainwater"],
        "content": "Aerators, dual-flush cisterns and push taps cut water use per person without changing behaviour. "
                   "Rainwater harvesting and treated water reuse for gardens reduce fresh-water demand further.",
    },
    {
        "title": "Waste reduction and segregation",
        "keywords": ["waste", "recycle", "reuse", "reduce", "segregation", "compost", "plastic"],
        "content": "Segregate dry, wet and e-waste at the point of collection, compost canteen waste on site, "
                   "and cut single-use plastic through refill stations and reusable crockery.",
    },
    {
        "title": "Emissions and renewable energy",
        "keywords": ["climate", "carbon", "emission", "renewable", "solar", "transport", "footprint"],
        "content": "Rooftop solar, efficiency retrofits and shared or electric transport lower campus emissions. "
                   "A yearly consumption baseline makes progress measurable rather than anecdotal.",
    },
    {
        "title": "Student, faculty and administrative involvement",
        "keywords": ["student", "faculty", "staff", "awareness", "campus", "administration", "contribute", "participation", "general"],
        "content": "Students can run audits, report leaks and faults, and lead awareness drives; faculty can embed "
                   "sustainability in coursework; administration can fund retrofits and publish consumption data openly.",
    },
    {
        "title": "Human oversight of AI recommendations",
        "keywords": ["ai", "oversight", "decision", "recommendation", "responsible", "general"],
        "content": "Automated flags point to where a person should look. A facilities team should verify each flag on site "
                   "before any equipment, schedule or budget decision is taken.",
    },
]


@st.cache_data
def load_knowledge_base(path: str) -> list:
    """
    Parse the knowledge-base file. Expected block format:

        TITLE: ...
        KEYWORDS: a, b, c
        CONTENT: ...
        ---

    Falls back to the built-in knowledge base if the file is missing or unreadable.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return []

    entries = []
    for block in raw.split("---"):
        block = block.strip()
        if not block:
            continue
        kw_match = re.search(r"KEYWORDS:\s*(.*)", block)
        content_match = re.search(r"CONTENT:\s*(.*)", block, re.DOTALL)
        title_match = re.search(r"TITLE:\s*(.*)", block)
        if kw_match and content_match:
            entries.append(
                {
                    "title": title_match.group(1).strip() if title_match else "Sustainability guidance",
                    "keywords": [k.strip().lower() for k in kw_match.group(1).split(",") if k.strip()],
                    "content": content_match.group(1).strip(),
                }
            )
    return entries


def get_knowledge_base() -> tuple:
    """Returns (entries, source_label, file_found)."""
    entries = load_knowledge_base(KB_PATH)
    if entries:
        return entries, "data/sustainability_knowledge.txt", True
    return FALLBACK_KB, "built-in knowledge base", False


def retrieve_relevant_knowledge(query: str, kb: list, top_n: int = 2) -> list:
    """Keyword scoring retrieval — deliberately simple and inspectable."""
    if not kb or not query:
        return []

    query_lower = query.lower()
    query_tokens = set(re.findall(r"[a-z]+", query_lower))

    scored = []
    for entry in kb:
        score = 0
        matched = []
        for kw in entry["keywords"]:
            if kw in query_lower:
                score += 2
                matched.append(kw)
            elif set(kw.split()) & query_tokens:
                score += 1
                matched.append(kw)
        if score > 0:
            scored.append((score, matched, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [{"score": s, "matched": m, **e} for s, m, e in scored[:top_n]]

    if not results:
        for entry in kb:
            if "general" in entry["keywords"]:
                results = [{"score": 0, "matched": [], **entry}]
                break
    return results


def generate_local_response(query: str, retrieved: list) -> str:
    """
    Prototype AI recommendation engine: local, rule-based, template driven.
    This is not a large language model and does not generate novel text.
    """
    if not retrieved:
        return (
            "No matching guidance was found in the local knowledge base. "
            "Try asking about electricity, cooling, computer labs, water leaks, waste, "
            "renewable energy, or how students and faculty can take part."
        )

    lines = [f"Here is the guidance that best matches your question ({len(retrieved)} entries retrieved):", ""]
    for entry in retrieved:
        lines.append(f"**{entry['title']}**")
        lines.append(entry["content"])
        lines.append("")
    lines.append("Use this as a starting point for a site check rather than a final answer.")
    return "\n".join(lines)


# ==========================================================================
# 6. REPORT GENERATION
# ==========================================================================


def generate_report(df: pd.DataFrame, insights: list) -> str:
    total_elec = df["electricity_kwh"].sum()
    total_water = df["water_kl"].sum()
    avg_daily_elec = df.groupby("date")["electricity_kwh"].sum().mean()
    anomalies = df[df["is_anomaly"]].sort_values("date")

    building_table = ["| Building | Avg electricity (kWh/day) | Avg water (kL/day) | Avg occupancy (%) |",
                      "| --- | --- | --- | --- |"]
    grouped = df.groupby("building")[["electricity_kwh", "water_kl", "occupancy_pct"]].mean().round(1)
    for building, row in grouped.iterrows():
        building_table.append(
            f"| {building} | {row['electricity_kwh']} | {row['water_kl']} | {row['occupancy_pct']} |"
        )

    anomaly_lines = []
    for _, row in anomalies.iterrows():
        parts = []
        if row["elec_anomaly"]:
            parts.append(
                f"electricity {row['electricity_kwh']:.1f} kWh vs baseline "
                f"{row['elec_building_avg']:.1f} kWh ({row['elec_pct_diff']:+.1f}%)"
            )
        if row["water_anomaly"]:
            parts.append(
                f"water {row['water_kl']:.1f} kL vs baseline "
                f"{row['water_building_avg']:.1f} kL ({row['water_pct_diff']:+.1f}%)"
            )
        anomaly_lines.append(f"- {row['date']} — {row['building']}: " + "; ".join(parts))

    insight_lines = [f"- {text}" for _, text in insights]

    elec_pct_rule = int((ELEC_ANOMALY_THRESHOLD - 1) * 100)
    water_pct_rule = int((WATER_ANOMALY_THRESHOLD - 1) * 100)

    report = f"""# {APP_NAME} — Sustainability Report

Generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}
Dataset: synthetic demonstration data — {DAYS} days, {len(BUILDINGS)} buildings, electricity and water

## Executive Summary

This report covers {DAYS} days of electricity and water consumption across {len(BUILDINGS)} campus
buildings. Campus electricity totalled {total_elec:,.0f} kWh, averaging {avg_daily_elec:,.0f} kWh per day.
Water consumption totalled {total_water:,.0f} kL. The prototype flagged {len(anomalies)} reading(s)
that exceeded the published anomaly thresholds and are recommended for review by facilities staff.

All figures come from synthetic demonstration data generated inside the application with a fixed
random seed. They do not represent any real campus.

## Resource Overview

- Total electricity ({DAYS} days): {total_elec:,.0f} kWh
- Average daily electricity: {avg_daily_elec:,.0f} kWh/day
- Total water ({DAYS} days): {total_water:,.0f} kL
- Buildings covered: {', '.join(BUILDINGS)}

## Building Performance

{chr(10).join(building_table)}

## Detected Anomalies

Readings flagged: {len(anomalies)}

{chr(10).join(anomaly_lines) if anomaly_lines else "No readings exceeded the current thresholds."}

Detection rule: a reading is flagged when electricity exceeds {elec_pct_rule}% above that building's
period baseline, or water exceeds {water_pct_rule}% above its baseline. These are statistical flags
that indicate where to look. They are not confirmed causes.

## Key Sustainability Insights

{chr(10).join(insight_lines) if insight_lines else "- No insights available for this dataset."}

## Recommended Actions

- Inspect flagged buildings for idle equipment, HVAC scheduling gaps and possible leaks.
- Add sub-metering or more frequent readings where flags repeat.
- Share consumption trends with facilities staff and student sustainability groups.
- Re-run this analysis as new readings arrive and review whether thresholds still fit.

## SDG Alignment

- SDG 7 — Affordable and Clean Energy (primary)
- SDG 11 — Sustainable Cities and Communities
- SDG 12 — Responsible Consumption and Production
- SDG 13 — Climate Action

## Responsible AI

- No personally identifiable student information is collected or required.
- Anomaly detection uses disclosed statistical thresholds rather than an opaque model.
- No building or group is assumed responsible for waste without evidence.
- Recommendations support human decisions; the system controls no campus infrastructure.
- External AI services are optional; a local recommendation engine runs by default.

## Limitations

- The dataset is synthetic and exists for demonstration only.
- Thresholds are simple statistical rules, not validated machine-learning models.
- A building baseline that includes an unusual day is slightly raised by that day.
- Real deployment would need validated metering data and facilities review.
- This is a working prototype, not a production monitoring deployment.

---
{APP_NAME} — {PROGRAM}
Author: {AUTHOR}, {AUTHOR_DETAIL}
"""
    return report


def markdown_to_text(md: str) -> str:
    """Plain-text version of the report for the TXT download."""
    text = re.sub(r"^#{1,6}\s*", "", md, flags=re.MULTILINE)
    text = text.replace("**", "").replace("|", " ")
    text = re.sub(r"^\s*-{3,}\s*$", "-" * 60, text, flags=re.MULTILINE)
    return text


# ==========================================================================
# 7. DATA BOOTSTRAP
# ==========================================================================

inject_styles()

try:
    raw_df = generate_synthetic_data()
    data_df = compute_anomalies(raw_df)
except Exception:
    raw_df = pd.DataFrame()
    data_df = pd.DataFrame()

kb, kb_source, kb_file_found = get_knowledge_base()
DATA_READY = not data_df.empty

# ==========================================================================
# 8. SIDEBAR
# ==========================================================================

st.sidebar.markdown(f"### 🌱 {APP_NAME}")
st.sidebar.caption(APP_TAGLINE)

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "AI Sustainability Assistant",
        "Resource Analysis",
        "Sustainability Report",
        "Responsible AI",
        "About Project",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset status**")
if DATA_READY:
    st.sidebar.success("Synthetic demonstration data")
    st.sidebar.caption(f"{DAYS} days · {len(BUILDINGS)} buildings · electricity + water")
    st.sidebar.caption(f"Seed {RANDOM_SEED} — reproducible")
else:
    st.sidebar.error("No dataset available")

st.sidebar.markdown("**Knowledge base**")
if kb_file_found:
    st.sidebar.caption(f"{len(kb)} entries · {kb_source}")
else:
    st.sidebar.warning("File not found — using built-in entries")

st.sidebar.markdown("---")
st.sidebar.caption("SDG 7 · SDG 11 · SDG 12 · SDG 13")
st.sidebar.caption(PROGRAM)

product_header()

if not DATA_READY:
    st.error("No sustainability data is currently available for analysis. Restart the application to regenerate the demo dataset.")
    st.stop()

# ==========================================================================
# 9. PAGE — DASHBOARD
# ==========================================================================

if page == "Dashboard":
    st.subheader("Campus consumption overview")
    st.caption("What is happening, what is unusual, and what to investigate next.")

    total_elec = data_df["electricity_kwh"].sum()
    avg_daily_elec = data_df.groupby("date")["electricity_kwh"].sum().mean()
    total_water = data_df["water_kl"].sum()
    anomalies = data_df[data_df["is_anomaly"]]
    anomaly_count = int(len(anomalies))

    elec_trend = trend_vs_previous_week(data_df, "electricity_kwh")
    water_trend = trend_vs_previous_week(data_df, "water_kl")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total electricity", f"{total_elec:,.0f} kWh", f"{DAYS}-day period"), unsafe_allow_html=True)
    c2.markdown(metric_card("Average daily electricity", f"{avg_daily_elec:,.0f} kWh", elec_trend or "campus-wide"), unsafe_allow_html=True)
    c3.markdown(metric_card("Total water", f"{total_water:,.0f} kL", water_trend or f"{DAYS}-day period", "water"), unsafe_allow_html=True)
    c4.markdown(
        metric_card("Readings flagged", f"{anomaly_count}", "above threshold", "alert" if anomaly_count else ""),
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --- Daily campus electricity with 7-day rolling average ---
    daily = data_df.groupby("date")["electricity_kwh"].sum().reset_index()
    daily["rolling_7d"] = daily["electricity_kwh"].rolling(7, min_periods=1).mean()

    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["electricity_kwh"], mode="lines+markers", name="Daily total",
            line=dict(color=SAGE, width=2.5), marker=dict(size=6),
            hovertemplate="%{x}<br>%{y:.0f} kWh<extra></extra>",
        )
    )
    fig_daily.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["rolling_7d"], mode="lines", name="7-day average",
            line=dict(color=SAGE_SOFT, width=2, dash="dot"),
            hovertemplate="%{x}<br>%{y:.0f} kWh<extra></extra>",
        )
    )
    elec_anoms = data_df[data_df["elec_anomaly"]]
    if not elec_anoms.empty:
        spike_days = elec_anoms.groupby("date")["building"].apply(lambda s: ", ".join(s)).reset_index()
        spike_days = spike_days.merge(daily[["date", "electricity_kwh"]], on="date", how="left")
        fig_daily.add_trace(
            go.Scatter(
                x=spike_days["date"], y=spike_days["electricity_kwh"], mode="markers",
                name="Flagged day", marker=dict(color=ALERT, size=13, symbol="diamond"),
                customdata=spike_days["building"],
                hovertemplate="%{x}<br>%{y:.0f} kWh<br>Flagged: %{customdata}<extra></extra>",
            )
        )
    fig_daily.update_layout(title="Daily campus electricity consumption")
    st.plotly_chart(style_fig(fig_daily, "Electricity (kWh)", "Date"), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        avg_elec = data_df.groupby("building")["electricity_kwh"].mean().reset_index()
        fig_elec = px.bar(
            avg_elec, x="building", y="electricity_kwh", color="building",
            color_discrete_map=BUILDING_COLORS, title="Average electricity by building",
        )
        fig_elec.add_hline(
            y=avg_elec["electricity_kwh"].mean(), line_dash="dot", line_color=ALERT,
            annotation_text="Campus average", annotation_position="top left",
        )
        fig_elec.update_traces(hovertemplate="%{x}<br>%{y:.0f} kWh/day<extra></extra>")
        fig_elec.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_elec, "Average kWh/day", ""), use_container_width=True)

    with col_b:
        avg_water = data_df.groupby("building")["water_kl"].mean().reset_index()
        fig_water = px.bar(
            avg_water, x="building", y="water_kl", color="building",
            color_discrete_map=BUILDING_COLORS, title="Average water by building",
        )
        fig_water.add_hline(
            y=avg_water["water_kl"].mean(), line_dash="dot", line_color=ALERT,
            annotation_text="Campus average", annotation_position="top left",
        )
        fig_water.update_traces(hovertemplate="%{x}<br>%{y:.1f} kL/day<extra></extra>")
        fig_water.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig_water, "Average kL/day", ""), use_container_width=True)

    # --- Building drill-down ---
    st.markdown("#### Building detail")
    st.caption("Compare one building against its own baseline and see which readings crossed the threshold.")
    sel_col, metric_col = st.columns([2, 1])
    selected_building = sel_col.selectbox("Building", BUILDINGS, key="dash_building")
    metric_choice = metric_col.selectbox("Measure", ["Electricity (kWh)", "Water (kL)"], key="dash_metric")

    is_elec = metric_choice.startswith("Electricity")
    value_col = "electricity_kwh" if is_elec else "water_kl"
    flag_col = "elec_anomaly" if is_elec else "water_anomaly"
    avg_col = "elec_building_avg" if is_elec else "water_building_avg"
    threshold = ELEC_ANOMALY_THRESHOLD if is_elec else WATER_ANOMALY_THRESHOLD
    unit = "kWh" if is_elec else "kL"

    bdf = data_df[data_df["building"] == selected_building].sort_values("date")
    baseline_val = float(bdf[avg_col].iloc[0])

    fig_b = go.Figure()
    fig_b.add_trace(
        go.Scatter(
            x=bdf["date"], y=bdf[value_col], mode="lines+markers", name=f"Daily {unit}",
            line=dict(color=SAGE if is_elec else WATER, width=2.5), marker=dict(size=6),
            hovertemplate="%{x}<br>%{y:.1f} " + unit + "<extra></extra>",
        )
    )
    fig_b.add_hline(y=baseline_val, line_dash="dash", line_color=SAGE_SOFT,
                    annotation_text=f"Baseline {baseline_val:.1f} {unit}", annotation_position="top left")
    fig_b.add_hline(y=baseline_val * threshold, line_dash="dot", line_color=ALERT,
                    annotation_text=f"Threshold {baseline_val * threshold:.1f} {unit}", annotation_position="bottom left")
    flagged = bdf[bdf[flag_col]]
    if not flagged.empty:
        fig_b.add_trace(
            go.Scatter(
                x=flagged["date"], y=flagged[value_col], mode="markers", name="Flagged reading",
                marker=dict(color=ALERT, size=14, symbol="diamond"),
                hovertemplate="%{x}<br>%{y:.1f} " + unit + " — flagged<extra></extra>",
            )
        )
    fig_b.update_layout(title=f"{selected_building} — {metric_choice} against its baseline")
    st.plotly_chart(style_fig(fig_b, metric_choice, "Date"), use_container_width=True)

    # --- Key insights ---
    st.markdown("#### Key insights")
    insights = build_key_insights(data_df)
    if not insights:
        st.info("No insights could be generated from the current dataset.")
    for kind, text in insights:
        css = "eco-insight flag" if kind == "flag" else "eco-insight"
        st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

    # --- Anomaly table ---
    st.markdown("#### Flagged readings")
    if anomalies.empty:
        st.success("No unusual consumption patterns were detected using the current thresholds.")
    else:
        table = anomalies[[
            "date", "building", "electricity_kwh", "elec_building_avg", "elec_pct_diff",
            "water_kl", "water_building_avg", "water_pct_diff",
        ]].sort_values("date").rename(columns={
            "date": "Date", "building": "Building",
            "electricity_kwh": "Electricity (kWh)", "elec_building_avg": "Elec baseline",
            "elec_pct_diff": "Elec % vs baseline", "water_kl": "Water (kL)",
            "water_building_avg": "Water baseline", "water_pct_diff": "Water % vs baseline",
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            f"Rule: electricity flagged above +{int((ELEC_ANOMALY_THRESHOLD-1)*100)}% of the building baseline; "
            f"water flagged above +{int((WATER_ANOMALY_THRESHOLD-1)*100)}%. Flags mark where to look, not what caused it."
        )

    with st.expander("View the full synthetic dataset"):
        st.dataframe(data_df, use_container_width=True, hide_index=True)

# ==========================================================================
# 10. PAGE — AI SUSTAINABILITY ASSISTANT
# ==========================================================================

elif page == "AI Sustainability Assistant":
    st.subheader("AI Sustainability Assistant")
    st.caption("Prototype AI recommendation engine — local keyword retrieval over a sustainability knowledge base.")

    st.warning(
        "This assistant is a local, rule-based prototype engine, not a large language model. "
        "It retrieves stored guidance and formats it. No API key and no internet connection are required.",
        icon="ℹ️",
    )

    if not kb_file_found:
        st.info("The knowledge-base file was not found, so built-in entries are being used. Recommendations remain available.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    sample_qs = [
        "How can our computer lab reduce electricity consumption?",
        "How can our campus reduce water wastage?",
        "Why might electricity consumption suddenly increase?",
        "How can students contribute to campus sustainability?",
    ]

    st.markdown("**Suggested questions**")
    cols = st.columns(2)
    clicked_q = None
    for i, q in enumerate(sample_qs):
        if cols[i % 2].button(q, use_container_width=True, key=f"sample_{i}"):
            clicked_q = q

    st.markdown("---")

    if not st.session_state.chat_history:
        st.markdown(
            info_card(
                "Start here",
                "Ask about energy, water, waste, climate action or campus participation. "
                "Every answer shows which knowledge entries were retrieved and how the response was built.",
            ),
            unsafe_allow_html=True,
        )

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"], avatar="🌱" if entry["role"] == "assistant" else "🧑"):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("sources"):
                with st.expander(f"Retrieved knowledge — {len(entry['sources'])} entries"):
                    for src in entry["sources"]:
                        matched = ", ".join(src.get("matched", [])) or "general match"
                        st.markdown(f"**{src['title']}**  \nMatched terms: `{matched}`  \n{src['content']}")
                with st.expander("How was this answer generated?"):
                    st.markdown(
                        "1. Your question is lower-cased and split into terms.\n"
                        "2. Each knowledge entry is scored on keyword overlap (exact phrase = 2 points, word overlap = 1).\n"
                        "3. The top-scoring entries are retrieved.\n"
                        "4. A rule-based template combines them into the response above.\n\n"
                        "`Question → keyword retrieval → relevant knowledge → recommendation`\n\n"
                        "No text is generated by a language model."
                    )

    typed_q = st.chat_input("Ask a sustainability question")
    final_query = clicked_q or typed_q

    if final_query:
        retrieved = retrieve_relevant_knowledge(final_query, kb)
        response = generate_local_response(final_query, retrieved)
        st.session_state.chat_history.append({"role": "user", "content": final_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response, "sources": retrieved})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================================================
# 11. PAGE — RESOURCE ANALYSIS
# ==========================================================================

elif page == "Resource Analysis":
    st.subheader("Resource Analysis")
    st.caption("Enter a current reading for a building and compare it against that building's demo baseline.")

    with st.form("resource_form"):
        col1, col2 = st.columns(2)
        with col1:
            building = st.selectbox("Building", BUILDINGS, help="Baselines are calculated per building.")
            elec_input = st.number_input("Electricity usage (kWh/day)", min_value=0.0, value=620.0, step=10.0)
            water_input = st.number_input("Water usage (kL/day)", min_value=0.0, value=15.0, step=0.5)
        with col2:
            occupancy_input = st.slider("Occupancy (%)", 0, 100, 80)
            operating_hours = st.slider("Operating hours per day", 0, 24, 12)
            temperature_input = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=32.0, step=0.5)
        submitted = st.form_submit_button("Run analysis", type="primary")

    if not submitted:
        st.markdown(
            info_card(
                "No analysis yet",
                "Submit a reading to see the status, the threshold calculation behind it, "
                "possible factors to investigate, and recommended actions.",
            ),
            unsafe_allow_html=True,
        )
    else:
        base = baseline_for(data_df, building)
        result = evaluate_reading(elec_input, water_input, base["elec"], base["water"])

        st.markdown("---")
        st.markdown(f"#### {building}")
        st.markdown(
            f'<span class="eco-status {result["status_css"]}">Status: {result["status"]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.markdown("**Observed data**")
        o1, o2, o3 = st.columns(3)
        o1.markdown(
            metric_card("Electricity submitted", f"{elec_input:,.0f} kWh",
                        f"baseline {base['elec']:,.0f} kWh · {result['elec_pct']:+.1f}%",
                        "alert" if result["elec_flag"] else ""),
            unsafe_allow_html=True,
        )
        o2.markdown(
            metric_card("Water submitted", f"{water_input:,.1f} kL",
                        f"baseline {base['water']:,.1f} kL · {result['water_pct']:+.1f}%",
                        "alert" if result["water_flag"] else "water"),
            unsafe_allow_html=True,
        )
        o3.markdown(
            metric_card("Operating context", f"{occupancy_input}% occupancy",
                        f"{operating_hours} h/day · {temperature_input:.1f} °C"),
            unsafe_allow_html=True,
        )

        st.markdown("**How this was calculated**")
        st.markdown(
            f"""
            <div class="eco-calc">
            Electricity baseline ({building}): {base['elec']:.1f} kWh/day<br>
            Current reading: {elec_input:.1f} kWh/day<br>
            Difference: {result['elec_pct']:+.1f}% &nbsp;|&nbsp; Threshold: +{int((ELEC_ANOMALY_THRESHOLD-1)*100)}%
            &nbsp;→&nbsp; {"FLAGGED" if result['elec_flag'] else "within range"}<br><br>
            Water baseline ({building}): {base['water']:.1f} kL/day<br>
            Current reading: {water_input:.1f} kL/day<br>
            Difference: {result['water_pct']:+.1f}% &nbsp;|&nbsp; Threshold: +{int((WATER_ANOMALY_THRESHOLD-1)*100)}%
            &nbsp;→&nbsp; {"FLAGGED" if result['water_flag'] else "within range"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["elec_flag"]:
            st.warning(f"Electricity consumption is {result['elec_pct']:.1f}% above the recent building average.")
        if result["water_flag"]:
            st.warning(f"Water consumption is {result['water_pct']:.1f}% above the recent building average.")
        if not result["elec_flag"] and not result["water_flag"]:
            st.success("Both readings are within the normal range for this building. Continue routine monitoring.")

        if result["elec_flag"] or result["water_flag"]:
            st.markdown("**Possible factors to investigate**")
            factors = []
            if occupancy_input > 85:
                factors.append("Occupancy is unusually high for this building, which raises expected load.")
            if operating_hours > 12:
                factors.append("Operating hours extend beyond a typical schedule.")
            if temperature_input > 31:
                factors.append("High ambient temperature increases cooling and HVAC load.")
            if result["elec_flag"]:
                factors.append("Equipment may have been left running, or an appliance may be faulty or ageing.")
            if result["water_flag"]:
                factors.append("A leak, a running tank or a faulty fixture could explain the water reading.")
            if not factors:
                factors.append("No operational cause is visible in the submitted inputs — a physical inspection is the next step.")
            for f in factors:
                st.markdown(f"- {f}")
            st.caption("These are possible factors to investigate, not confirmed causes.")

            st.markdown("**Recommended actions**")
            for action in [
                "Check for idle or malfunctioning equipment in the building.",
                "Review HVAC and lighting schedules against actual occupancy.",
                "Log peak-usage hours for the next few days.",
                "Escalate to facilities if the pattern repeats beyond a single day.",
            ]:
                st.markdown(f"- {action}")

            query_text = f"{building} electricity equipment {'leak water' if result['water_flag'] else ''} sudden increase"
            retrieved = retrieve_relevant_knowledge(query_text, kb, top_n=2)
            if retrieved:
                with st.expander(f"Related guidance from the knowledge base ({len(retrieved)} entries)"):
                    for entry in retrieved:
                        st.markdown(f"**{entry['title']}** — {entry['content']}")
            else:
                st.caption("The sustainability knowledge base returned no matching guidance for this case.")

# ==========================================================================
# 12. PAGE — SUSTAINABILITY REPORT
# ==========================================================================

elif page == "Sustainability Report":
    st.subheader("Sustainability Report")
    st.caption("A dated report built from the current demo dataset, ready to download and share.")

    if st.button("Generate sustainability report", type="primary"):
        try:
            st.session_state["generated_report"] = generate_report(data_df, build_key_insights(data_df))
        except Exception:
            st.session_state.pop("generated_report", None)
            st.error("The report could not be generated from the current dataset. Try regenerating the demo data.")

    if "generated_report" not in st.session_state:
        st.markdown(
            info_card(
                "No report generated yet",
                "Select the button above. The report pulls its figures, anomalies and insights directly "
                "from the dataset shown on the dashboard.",
            ),
            unsafe_allow_html=True,
        )
    else:
        report_md = st.session_state["generated_report"]
        st.success("Report generated. Preview below.")

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download Markdown", data=report_md,
            file_name="ecocampus_sustainability_report.md", mime="text/markdown",
            use_container_width=True,
        )
        d2.download_button(
            "Download plain text", data=markdown_to_text(report_md),
            file_name="ecocampus_sustainability_report.txt", mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown(report_md)

# ==========================================================================
# 13. PAGE — RESPONSIBLE AI
# ==========================================================================

elif page == "Responsible AI":
    st.subheader("Responsible AI")
    st.caption("How this prototype handles privacy, transparency, fairness, oversight and failure.")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(info_card(
            "🔒 Privacy",
            "No personally identifiable student or staff information is collected or required. "
            "All analysis runs on building-level consumption readings.",
        ), unsafe_allow_html=True)
        st.markdown(info_card(
            "🔍 Transparency",
            "Anomaly detection compares each reading with its own building baseline using fixed, published "
            "percentage thresholds. The calculation is shown on screen for every flag.",
        ), unsafe_allow_html=True)
        st.markdown(info_card(
            "⚖️ Fairness",
            "No building, department or group is assumed responsible for waste without evidence. "
            "A flag marks where to look, not who is at fault.",
        ), unsafe_allow_html=True)
        st.markdown(info_card(
            "🧑‍💼 Human oversight",
            "Recommendations support human decisions. The system controls no HVAC, lighting or water "
            "infrastructure, and takes no automated action.",
        ), unsafe_allow_html=True)
    with r2:
        st.markdown(info_card(
            "📊 Data quality",
            "The dataset is synthetic, generated in-app with a fixed seed so results are reproducible. "
            "Real deployment would need validated, consented campus metering data.",
        ), unsafe_allow_html=True)
        st.markdown(info_card(
            "🔁 Reliability",
            "External AI services are optional. A local rule-based recommendation engine runs by default, "
            "so the application works with no internet connection and no API key.",
        ), unsafe_allow_html=True)
        st.markdown(info_card(
            "📌 Limitations",
            "Thresholds are simple statistics, not validated models. A baseline that contains an unusual day "
            "is slightly raised by it, which can mask smaller deviations.",
        ), unsafe_allow_html=True)
        st.markdown(info_card(
            "🗣️ Honest labelling",
            "The assistant is described as a rule-based prototype engine throughout. No claim is made that a "
            "large language model or external provider is in use.",
        ), unsafe_allow_html=True)

    st.markdown("#### AI risk controls")
    risk_df = pd.DataFrame(
        [
            ["Incorrect recommendation", "Human review before any action is taken"],
            ["Synthetic data mistaken for real", "Demo-mode badge and dataset labels on every page"],
            ["False anomaly interpretation", "Thresholds and the full calculation shown on screen"],
            ["Privacy risk", "No personal data collected or stored"],
            ["Over-reliance on AI", "Decision-support framing; no automated control"],
            ["Model or API failure", "Local rule-based fallback runs by default"],
            ["Missing knowledge base", "Built-in entries load automatically"],
        ],
        columns=["Risk", "Control"],
    )
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

# ==========================================================================
# 14. PAGE — ABOUT PROJECT
# ==========================================================================

elif page == "About Project":
    st.subheader("About the project")

    st.markdown("#### Problem statement")
    st.write(
        "Educational institutions consume electricity, water and other resources every day, but that data is "
        "not always turned into understandable, actionable sustainability insights."
    )

    st.markdown("#### Solution")
    st.write(
        "EcoCampus AI combines data analytics, transparent anomaly detection, lightweight knowledge retrieval "
        "and AI-supported recommendations so campus stakeholders can make better-informed sustainability decisions. "
        "It is a production-quality prototype, not a production deployment."
    )

    st.markdown("#### How it fits together")
    st.code(
        "Streamlit UI\n"
        "   ↓\n"
        "Synthetic resource data\n"
        "   ↓\n"
        "Data analysis  →  Anomaly detection\n"
        "   ↓\n"
        "Knowledge retrieval  →  Recommendation engine\n"
        "   ↓\n"
        "Sustainability insights  →  Human decision maker",
        language="text",
    )

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("#### SDG alignment")
        st.markdown(
            "- **SDG 7** — Affordable and Clean Energy (primary)\n"
            "- **SDG 11** — Sustainable Cities and Communities\n"
            "- **SDG 12** — Responsible Consumption and Production\n"
            "- **SDG 13** — Climate Action"
        )
        st.markdown("#### Target users")
        st.markdown(
            "- Campus administrators\n- Facilities and maintenance teams\n"
            "- Sustainability coordinators\n- Faculty and student sustainability groups"
        )
    with a2:
        st.markdown("#### Future scope")
        st.markdown(
            "- IoT sensor integration\n- Real-time electricity monitoring\n- Water sensors\n"
            "- Carbon footprint estimation\n- Renewable energy monitoring\n- Optional IBM Granite integration\n"
            "- Advanced predictive models\n- Multi-campus analytics"
        )
        st.markdown("#### Expected impact")
        st.markdown(
            "- Potentially improve resource awareness across campus stakeholders\n"
            "- Support data-driven investigation of unusual consumption\n"
            "- Make sustainability reporting repeatable rather than manual"
        )
        st.caption("No verified environmental savings are claimed. This prototype runs on synthetic data.")

    st.markdown("---")
    st.markdown(f"**Author:** {AUTHOR}  \n{AUTHOR_DETAIL}  \n{PROGRAM}")
