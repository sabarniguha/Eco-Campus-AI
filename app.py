"""
EcoCampus AI: Intelligent Campus Sustainability Assistant
1M1B AI for Sustainability Virtual Internship

A working prototype using synthetic demonstration data.
No external API key is required. All AI-style features run locally.
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoCampus AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUILDINGS = ["CSE Block", "Library", "Hostel A", "Administrative Block", "Science Block"]
KB_PATH = os.path.join(os.path.dirname(__file__), "data", "sustainability_knowledge.txt")

ELEC_ANOMALY_THRESHOLD = 1.25   # 25% above average
WATER_ANOMALY_THRESHOLD = 1.30  # 30% above average

# --------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION
# --------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(seed: int = 42, days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Baseline daily profile per building (elec kWh/day, water kL/day)
    baseline = {
        "CSE Block":            {"elec": 480, "water": 14, "occ": 78, "temp": 29},
        "Library":               {"elec": 260, "water": 8,  "occ": 65, "temp": 27},
        "Hostel A":              {"elec": 620, "water": 32, "occ": 90, "temp": 28},
        "Administrative Block":  {"elec": 210, "water": 6,  "occ": 55, "temp": 27},
        "Science Block":         {"elec": 540, "water": 18, "occ": 70, "temp": 29},
    }

    start_date = datetime.today().date() - timedelta(days=days - 1)
    rows = []
    for building in BUILDINGS:
        b = baseline[building]
        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)
            weekday = date.weekday()
            weekend_factor = 0.75 if weekday >= 5 else 1.0

            elec = b["elec"] * weekend_factor * rng.normal(1.0, 0.06)
            water = b["water"] * weekend_factor * rng.normal(1.0, 0.08)
            occ = np.clip(b["occ"] * weekend_factor * rng.normal(1.0, 0.05), 10, 100)
            temp = b["temp"] + rng.normal(0, 1.2)

            rows.append({
                "date": date,
                "building": building,
                "electricity_kwh": round(max(elec, 10), 1),
                "water_kl": round(max(water, 1), 1),
                "occupancy_pct": round(occ, 1),
                "temperature_c": round(temp, 1),
            })

    df = pd.DataFrame(rows)

    # ---- Inject deliberate anomalies (so anomaly detection has something to find) ----
    def inject(building, day_offset, elec_mult=None, water_mult=None):
        target_date = start_date + timedelta(days=day_offset)
        mask = (df["building"] == building) & (df["date"] == target_date)
        if elec_mult:
            df.loc[mask, "electricity_kwh"] = (df.loc[mask, "electricity_kwh"] * elec_mult).round(1)
        if water_mult:
            df.loc[mask, "water_kl"] = (df.loc[mask, "water_kl"] * water_mult).round(1)

    inject("CSE Block", 12, elec_mult=1.55)          # unexplained electricity spike
    inject("Hostel A", 20, water_mult=1.60)          # likely leak
    inject("Science Block", 25, elec_mult=1.40, water_mult=1.35)  # combined spike

    return df


@st.cache_data
def compute_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    elec_avg = df.groupby("building")["electricity_kwh"].transform("mean")
    water_avg = df.groupby("building")["water_kl"].transform("mean")

    df["elec_building_avg"] = elec_avg.round(1)
    df["water_building_avg"] = water_avg.round(1)
    df["elec_pct_diff"] = ((df["electricity_kwh"] - elec_avg) / elec_avg * 100).round(1)
    df["water_pct_diff"] = ((df["water_kl"] - water_avg) / water_avg * 100).round(1)
    df["elec_anomaly"] = df["electricity_kwh"] > ELEC_ANOMALY_THRESHOLD * elec_avg
    df["water_anomaly"] = df["water_kl"] > WATER_ANOMALY_THRESHOLD * water_avg
    df["is_anomaly"] = df["elec_anomaly"] | df["water_anomaly"]
    return df


# --------------------------------------------------------------------------
# KNOWLEDGE BASE + LIGHTWEIGHT RAG RETRIEVAL
# --------------------------------------------------------------------------
@st.cache_data
def load_knowledge_base(path: str):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = raw.split("---")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        kw_match = re.search(r"KEYWORDS:\s*(.*)", block)
        content_match = re.search(r"CONTENT:\s*(.*)", block, re.DOTALL)
        if kw_match and content_match:
            keywords = [k.strip().lower() for k in kw_match.group(1).split(",")]
            content = content_match.group(1).strip()
            entries.append({"keywords": keywords, "content": content})
    return entries


def retrieve_relevant_knowledge(query: str, kb: list, top_n: int = 2):
    query_lower = query.lower()
    query_tokens = set(re.findall(r"[a-z]+", query_lower))

    scored = []
    for entry in kb:
        score = 0
        for kw in entry["keywords"]:
            if kw in query_lower:
                score += 2
            elif set(kw.split()) & query_tokens:
                score += 1
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [entry for score, entry in scored[:top_n]]

    if not top:
        # fallback to a general entry if nothing matched
        for entry in kb:
            if "general" in entry["keywords"]:
                top = [entry]
                break
    return top


def generate_local_response(query: str, retrieved: list) -> str:
    """
    Prototype AI recommendation engine (local, rule-based).
    This is NOT an actual LLM — it combines retrieved knowledge-base
    context with simple templated phrasing.
    """
    if not retrieved:
        return (
            "I could not find specific guidance for that question in the local "
            "knowledge base. Try asking about electricity, water, waste, climate, "
            "or student/faculty involvement in campus sustainability."
        )

    intro = "Based on the campus sustainability knowledge base, here is guidance relevant to your question:\n\n"
    body = "\n\n".join(f"• {entry['content']}" for entry in retrieved)
    outro = (
        "\n\n_This response was generated by a local, rule-based prototype "
        "AI recommendation engine using keyword retrieval — not a live LLM. "
        "Treat it as a starting point for human review._"
    )
    return intro + body + outro


# --------------------------------------------------------------------------
# SHARED DATA (generated once)
# --------------------------------------------------------------------------
raw_df = generate_synthetic_data()
data_df = compute_anomalies(raw_df)
kb = load_knowledge_base(KB_PATH)

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.title("🌱 EcoCampus AI")
st.sidebar.caption("Intelligent Campus Sustainability Assistant")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dashboard",
        "💬 AI Sustainability Assistant",
        "🔍 Resource Analysis",
        "📄 Sustainability Report",
        "🛡️ Responsible AI",
        "ℹ️ About Project",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("SDG 7 · SDG 11 · SDG 12 · SDG 13")
st.sidebar.caption("1M1B AI for Sustainability Virtual Internship")

# --------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Campus Sustainability Dashboard")
    st.info("**Demo data — synthetic data used for demonstration.**")

    total_elec = data_df["electricity_kwh"].sum()
    avg_daily_elec = data_df.groupby("date")["electricity_kwh"].sum().mean()
    total_water = data_df["water_kl"].sum()
    anomaly_count = int(data_df["is_anomaly"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Total Electricity (30 days)", f"{total_elec:,.0f} kWh")
    c2.metric("📈 Avg Daily Electricity", f"{avg_daily_elec:,.0f} kWh/day")
    c3.metric("💧 Total Water (30 days)", f"{total_water:,.0f} kL")
    c4.metric("⚠️ Anomalies Detected", anomaly_count)

    st.markdown("---")

    daily_elec = data_df.groupby("date")["electricity_kwh"].sum().reset_index()
    fig1 = px.line(daily_elec, x="date", y="electricity_kwh",
                    title="Daily Total Electricity Consumption (All Buildings)",
                    markers=True)
    fig1.update_layout(yaxis_title="Electricity (kWh)", xaxis_title="Date")
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        avg_by_building_elec = data_df.groupby("building")["electricity_kwh"].mean().reset_index()
        fig2 = px.bar(avg_by_building_elec, x="building", y="electricity_kwh",
                       title="Average Electricity Consumption by Building",
                       color="building")
        fig2.update_layout(yaxis_title="Avg kWh/day", xaxis_title="", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        avg_by_building_water = data_df.groupby("building")["water_kl"].mean().reset_index()
        fig3 = px.bar(avg_by_building_water, x="building", y="water_kl",
                       title="Average Water Consumption by Building",
                       color="building")
        fig3.update_layout(yaxis_title="Avg kL/day", xaxis_title="", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ Detected Anomalies")
    anomalies = data_df[data_df["is_anomaly"]].copy()
    if anomalies.empty:
        st.success("No anomalies detected in the current dataset.")
    else:
        display_cols = ["date", "building", "electricity_kwh", "elec_pct_diff",
                         "water_kl", "water_pct_diff"]
        anomalies_display = anomalies[display_cols].rename(columns={
            "elec_pct_diff": "elec_%_vs_avg",
            "water_pct_diff": "water_%_vs_avg",
        }).sort_values("date")
        st.dataframe(anomalies_display, use_container_width=True, hide_index=True)
        st.caption(
            f"Flagging rule: electricity > {int((ELEC_ANOMALY_THRESHOLD-1)*100)}% above the "
            f"building's 30-day average, or water > {int((WATER_ANOMALY_THRESHOLD-1)*100)}% above average."
        )

    with st.expander("View raw synthetic dataset"):
        st.dataframe(data_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# PAGE 2: AI SUSTAINABILITY ASSISTANT
# --------------------------------------------------------------------------
elif page == "💬 AI Sustainability Assistant":
    st.title("💬 AI Sustainability Assistant")
    st.caption("Prototype AI recommendation engine — local, rule-based retrieval. No API key required.")

    with st.expander("ℹ️ How this works"):
        st.write(
            "This assistant uses a lightweight RAG-style (Retrieval-Augmented Generation) workflow:\n\n"
            "1. Your question is matched against a local sustainability knowledge base using keyword retrieval.\n"
            "2. The most relevant knowledge entries are retrieved.\n"
            "3. A local, rule-based prototype AI recommendation engine combines that context into a response.\n\n"
            "This is **not** a live large language model. It is a transparent, explainable fallback "
            "that works without any external API key."
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("**Try asking:**")
    sample_qs = [
        "How can our computer lab reduce electricity consumption?",
        "How can our campus reduce water wastage?",
        "Why might electricity consumption suddenly increase?",
        "How can students contribute to campus sustainability?",
    ]
    cols = st.columns(len(sample_qs))
    clicked_q = None
    for col, q in zip(cols, sample_qs):
        if col.button(q, use_container_width=True):
            clicked_q = q

    user_query = st.chat_input("Ask a sustainability question...")
    final_query = clicked_q or user_query

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    if final_query:
        st.session_state.chat_history.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        retrieved = retrieve_relevant_knowledge(final_query, kb)
        response = generate_local_response(final_query, retrieved)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# --------------------------------------------------------------------------
# PAGE 3: RESOURCE ANALYSIS
# --------------------------------------------------------------------------
elif page == "🔍 Resource Analysis":
    st.title("🔍 Resource Analysis")
    st.caption("Submit current readings for a building to check them against historical patterns.")

    with st.form("resource_form"):
        col1, col2 = st.columns(2)
        with col1:
            building = st.selectbox("Building", BUILDINGS)
            elec_input = st.number_input("Electricity usage (kWh/day)", min_value=0.0, value=400.0, step=10.0)
            water_input = st.number_input("Water usage (kL/day)", min_value=0.0, value=15.0, step=1.0)
        with col2:
            occupancy_input = st.slider("Occupancy (%)", 0, 100, 75)
            operating_hours = st.slider("Operating hours/day", 0, 24, 10)
            temperature_input = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=29.0, step=0.5)

        submitted = st.form_submit_button("Analyze")

    if submitted:
        building_data = data_df[data_df["building"] == building]
        elec_avg = building_data["electricity_kwh"].mean()
        water_avg = building_data["water_kl"].mean()

        elec_pct = (elec_input - elec_avg) / elec_avg * 100
        water_pct = (water_input - water_avg) / water_avg * 100

        elec_flag = elec_input > ELEC_ANOMALY_THRESHOLD * elec_avg
        water_flag = water_input > WATER_ANOMALY_THRESHOLD * water_avg

        st.markdown("---")
        st.subheader(f"Analysis for {building}")

        m1, m2 = st.columns(2)
        m1.metric("Electricity vs. building average", f"{elec_pct:+.1f}%",
                   delta=f"{elec_input:.0f} kWh vs avg {elec_avg:.0f} kWh")
        m2.metric("Water vs. building average", f"{water_pct:+.1f}%",
                   delta=f"{water_input:.0f} kL vs avg {water_avg:.0f} kL")

        if elec_flag:
            st.warning(f"⚡ Electricity consumption is {elec_pct:.0f}% above the recent building average.")
        else:
            st.success("⚡ Electricity consumption is within the normal range.")

        if water_flag:
            st.warning(f"💧 Water consumption is {water_pct:.0f}% above the recent building average.")
        else:
            st.success("💧 Water consumption is within the normal range.")

        if elec_flag or water_flag:
            st.markdown("### Possible factors to investigate")
            factors = []
            if occupancy_input > 85:
                factors.append("Unusually high occupancy for this building")
            if operating_hours > 12:
                factors.append("Extended operating hours beyond typical schedule")
            if temperature_input > 31:
                factors.append("High temperature likely increasing HVAC/cooling load")
            if elec_flag:
                factors.append("Possible equipment left running or inefficient/aging equipment")
            if water_flag:
                factors.append("Possible leak or fixture malfunction")
            if not factors:
                factors.append("No obvious operational cause from the inputs provided — recommend a physical inspection")
            for f in factors:
                st.write(f"- {f}")

            st.caption("Note: these are possible factors to investigate, not confirmed causes.")

            st.markdown("### Recommended actions")
            base_actions = [
                "Check for idle or malfunctioning equipment",
                "Review HVAC operating schedules against actual occupancy",
                "Monitor peak-usage hours for this building over the next few days",
                "Investigate further if the pattern persists beyond a single day",
            ]
            for a in base_actions:
                st.write(f"- {a}")

            # RAG-style contextual guidance
            query_text = f"{building} electricity water {'anomaly' if elec_flag else ''} {'leak' if water_flag else ''}"
            retrieved = retrieve_relevant_knowledge(query_text, kb, top_n=2)
            if retrieved:
                with st.expander("📚 Related guidance from knowledge base"):
                    for entry in retrieved:
                        st.write(f"• {entry['content']}")
        else:
            st.info("No anomaly flagged. Continue routine monitoring.")

# --------------------------------------------------------------------------
# PAGE 4: SUSTAINABILITY REPORT
# --------------------------------------------------------------------------
elif page == "📄 Sustainability Report":
    st.title("📄 Sustainability Report")
    st.caption("Generate a summary report based on the current synthetic dataset.")

    if st.button("Generate Sustainability Report", type="primary"):
        total_elec = data_df["electricity_kwh"].sum()
        total_water = data_df["water_kl"].sum()
        anomalies = data_df[data_df["is_anomaly"]].sort_values("date")
        anomaly_count = len(anomalies)

        anomaly_lines = []
        for _, row in anomalies.iterrows():
            parts = []
            if row["elec_anomaly"]:
                parts.append(f"electricity {row['elec_pct_diff']:+.1f}% vs average")
            if row["water_anomaly"]:
                parts.append(f"water {row['water_pct_diff']:+.1f}% vs average")
            anomaly_lines.append(f"- {row['date']} — {row['building']}: " + "; ".join(parts))

        report = f"""# EcoCampus AI — Sustainability Report

*Generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}*
*Data: synthetic demonstration data (30-day period)*

## Overview

This report summarizes electricity and water consumption patterns across
{len(BUILDINGS)} campus buildings, based on a working prototype developed for the
1M1B AI for Sustainability Virtual Internship. All figures are derived from
synthetic demonstration data, not live campus meters.

## Resource Consumption

- Total electricity consumption (30 days): {total_elec:,.0f} kWh
- Total water consumption (30 days): {total_water:,.0f} kL
- Buildings monitored: {', '.join(BUILDINGS)}

### Average consumption by building

{data_df.groupby('building')[['electricity_kwh','water_kl']].mean().round(1).to_string()}

## Detected Anomalies

Total anomalies flagged: {anomaly_count}

{chr(10).join(anomaly_lines) if anomaly_lines else "No anomalies detected in this dataset."}

Flagging rule: electricity flagged when more than {int((ELEC_ANOMALY_THRESHOLD-1)*100)}% above
a building's 30-day average; water flagged when more than {int((WATER_ANOMALY_THRESHOLD-1)*100)}%
above average. These are statistical flags, not confirmed causes.

## Recommended Actions

- Investigate flagged buildings for idle equipment, HVAC scheduling issues, or possible leaks.
- Introduce sub-metering or more frequent readings for buildings with recurring flags.
- Share consumption trends with facility staff and student sustainability groups.
- Re-run this analysis regularly as new readings become available.

## SDG Alignment

- SDG 7: Affordable and Clean Energy (primary)
- SDG 11: Sustainable Cities and Communities
- SDG 12: Responsible Consumption and Production
- SDG 13: Climate Action

## Responsible AI

- No personally identifiable student information is used or required.
- Anomaly detection uses transparent statistical thresholds, not opaque models.
- The system does not assume any building or group is responsible for waste without evidence.
- Recommendations support human decision-making; the system does not control infrastructure.
- External AI services are optional; a local fallback recommendation engine is used by default.

## Limitations

- Data used in this report is synthetic and generated for demonstration purposes only.
- Anomaly thresholds are simple statistical rules, not machine-learning-validated models.
- Real deployment would require validated campus metering data and facility review.
- This is a working prototype, not a production monitoring system.

---
*EcoCampus AI — 1M1B AI for Sustainability Virtual Internship*
*Author: Sabarni Guha, B.Tech CSE (AI & ML), IEM-UEM*
"""
        st.session_state["generated_report"] = report
        st.success("Report generated below.")

    if "generated_report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["generated_report"])
        st.download_button(
            "⬇️ Download Report (Markdown)",
            data=st.session_state["generated_report"],
            file_name="ecocampus_sustainability_report.md",
            mime="text/markdown",
        )

# --------------------------------------------------------------------------
# PAGE 5: RESPONSIBLE AI
# --------------------------------------------------------------------------
elif page == "🛡️ Responsible AI":
    st.title("🛡️ Responsible AI")
    st.caption("How this prototype handles privacy, transparency, fairness, and oversight.")

    with st.expander("🔒 Privacy", expanded=True):
        st.write("The prototype does not require personally identifiable student information. "
                 "All analysis is performed on building-level resource data, not individual records.")

    with st.expander("🔍 Transparency"):
        st.write("Anomaly detection uses understandable statistical thresholds — electricity and water "
                 "usage are compared to historical building averages using fixed, disclosed percentage "
                 "cutoffs, not a black-box model.")

    with st.expander("⚖️ Fairness"):
        st.write("The system does not assume that a particular group, building, or individual is "
                 "responsible for waste without sufficient evidence. Flags indicate where to look, "
                 "not who is at fault.")

    with st.expander("🧑‍💼 Human Oversight"):
        st.write("AI-generated recommendations are intended to support human decision-making. The "
                 "system does not automatically control campus infrastructure such as HVAC, lighting, "
                 "or water systems.")

    with st.expander("📊 Data Limitations"):
        st.write("This project uses synthetic demonstration data generated with a fixed random seed. "
                 "Real deployment would require validated, consented campus metering data.")

    with st.expander("🔁 Reliability"):
        st.write("External AI services (such as an LLM API) are optional and not required for this "
                 "prototype to function. The application includes a local, rule-based fallback so it "
                 "works reliably even with no internet connection or API key.")

# --------------------------------------------------------------------------
# PAGE 6: ABOUT PROJECT
# --------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About EcoCampus AI")

    st.markdown("## Problem Statement")
    st.write(
        "Educational institutions consume electricity, water, and other resources every day, but "
        "resource data is not always converted into understandable and actionable sustainability insights."
    )

    st.markdown("## Solution")
    st.write(
        "EcoCampus AI uses data analytics, anomaly detection, lightweight knowledge retrieval, and "
        "AI-supported recommendations to help campus stakeholders make more informed sustainability decisions."
    )

    st.markdown("## SDGs")
    st.write("SDG 7 · SDG 11 · SDG 12 · SDG 13")

    st.markdown("## Future Scope")
    st.write(
        "- IoT sensor integration\n"
        "- Real-time electricity monitoring\n"
        "- Water sensors\n"
        "- Carbon footprint estimation\n"
        "- Renewable energy monitoring\n"
        "- IBM Granite integration\n"
        "- Advanced predictive models\n"
        "- Multi-campus analytics"
    )

    st.markdown("---")
    st.markdown("**Author:** Sabarni Guha")
    st.markdown("B.Tech CSE (AI & ML), IEM-UEM")
    st.markdown("1M1B AI for Sustainability Virtual Internship")
