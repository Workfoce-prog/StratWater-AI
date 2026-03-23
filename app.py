
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="StratWater AI", page_icon="💧", layout="wide")

LANG = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

TEXT = {
    "English": {
        "title": "💧 StratWater AI Dashboard",
        "subtitle": "Smart water infrastructure monitoring for West Africa pilots",
        "upload": "Upload replacement village CSV (optional)",
        "village": "Select village",
        "overview": "Overview",
        "usage": "Water Usage",
        "ai": "AI Scores",
        "maintenance": "Maintenance",
        "financials": "Financials",
        "map": "Map View",
        "data": "Data Explorer",
        "population": "Population served",
        "uptime": "Uptime",
        "revenue": "Annual revenue",
        "water": "Average daily water",
        "status": "System status",
        "healthy": "Healthy",
        "watch": "Watch",
        "critical": "Critical",
        "download": "Download current village table as CSV",
        "notes": "Illustrative demo data for a Mali-first West Africa pilot."
    },
    "Français": {
        "title": "💧 Tableau de bord StratWater AI",
        "subtitle": "Suivi intelligent des infrastructures hydrauliques pour les pilotes en Afrique de l’Ouest",
        "upload": "Télécharger un CSV de remplacement (optionnel)",
        "village": "Sélectionner un village",
        "overview": "Vue d’ensemble",
        "usage": "Consommation d’eau",
        "ai": "Scores IA",
        "maintenance": "Maintenance",
        "financials": "Finances",
        "map": "Carte",
        "data": "Explorateur de données",
        "population": "Population desservie",
        "uptime": "Disponibilité",
        "revenue": "Revenu annuel",
        "water": "Volume quotidien moyen",
        "status": "État du système",
        "healthy": "Bon",
        "watch": "Surveillance",
        "critical": "Critique",
        "download": "Télécharger le tableau CSV",
        "notes": "Données de démonstration illustratives pour un pilote Mali + Afrique de l’Ouest."
    }
}
T = TEXT[LANG]

@st.cache_data
def load_data():
    villages = pd.read_csv(DATA_DIR / "villages.csv")
    daily = pd.read_csv(DATA_DIR / "daily_usage.csv", parse_dates=["date"])
    maint = pd.read_csv(DATA_DIR / "maintenance_log.csv", parse_dates=["event_date"])
    return villages, daily, maint

def compute_scores(df):
    out = df.copy()
    out["Reliability Score"] = (
        100
        - np.where(out["uptime_pct"] < 95, 12, 0)
        - np.where(out["uptime_pct"] < 90, 12, 0)
        - np.where(out["failures_ytd"] > 2, 8, 0)
        - np.where(out["tank_fill_pct"] < 35, 10, 0)
    ).clip(lower=20)
    out["Maintenance Risk Score"] = (
        20
        + (100 - out["uptime_pct"]) * 1.1
        + out["failures_ytd"] * 8
        + np.where(out["pump_load_pct"] > 85, 10, 0)
    ).clip(upper=95)
    out["Water Stress Index"] = (
        (out["avg_daily_liters"] / (out["population_served"] * 20)) * 100
    ).clip(upper=160)
    out["Sustainability Score"] = (
        (out["collection_rate"] * 60)
        + np.where(out["annual_revenue_usd"] > out["annual_opex_usd"], 25, 10)
        + np.where(out["operator_trained"] == 1, 15, 0)
    ).clip(upper=100)
    return out

def status_from_scores(rel, risk):
    if rel >= 85 and risk <= 45:
        return T["healthy"]
    if rel >= 70:
        return T["watch"]
    return T["critical"]

villages, daily_usage, maintenance = load_data()
villages = compute_scores(villages)

st.title(T["title"])
st.caption(T["subtitle"])

uploaded = st.sidebar.file_uploader(T["upload"], type=["csv"])
if uploaded is not None:
    try:
        villages = pd.read_csv(uploaded)
        villages = compute_scores(villages)
        st.sidebar.success("Custom CSV loaded")
    except Exception as e:
        st.sidebar.error(f"Could not load file: {e}")

selected = st.sidebar.selectbox(T["village"], villages["village"])
row = villages.loc[villages["village"] == selected].iloc[0]
filtered_usage = daily_usage[daily_usage["village"] == selected].sort_values("date")
filtered_maint = maintenance[maintenance["village"] == selected].sort_values("event_date", ascending=False)

k1, k2, k3, k4 = st.columns(4)
k1.metric(T["population"], f"{int(row['population_served']):,}")
k2.metric(T["uptime"], f"{row['uptime_pct']:.1f}%")
k3.metric(T["revenue"], f"${row['annual_revenue_usd']:,.0f}")
k4.metric(T["water"], f"{row['avg_daily_liters']:,.0f} L")

tabs = st.tabs([T["overview"], T["usage"], T["ai"], T["maintenance"], T["financials"], T["map"], T["data"]])

with tabs[0]:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        score_df = pd.DataFrame({
            "Metric": ["Reliability", "Risk", "Stress", "Sustainability"],
            "Score": [row["Reliability Score"], row["Maintenance Risk Score"], row["Water Stress Index"], row["Sustainability Score"]]
        })
        fig = px.bar(score_df, x="Score", y="Metric", orientation="h", range_x=[0, 100])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader(T["status"])
        st.markdown(f"### {status_from_scores(row['Reliability Score'], row['Maintenance Risk Score'])}")
        st.write(f"- Households: **{int(row['households'])}**")
        st.write(f"- Tank fill: **{row['tank_fill_pct']:.0f}%**")
        st.write(f"- Pump load: **{row['pump_load_pct']:.0f}%**")
        st.write(f"- Collection rate: **{row['collection_rate']:.0%}**")
        st.info(T["notes"])

with tabs[1]:
    usage_fig = px.line(filtered_usage, x="date", y="liters", title=f"{selected} | Daily water use")
    usage_fig.update_layout(height=380)
    st.plotly_chart(usage_fig, use_container_width=True)
    st.dataframe(filtered_usage.tail(10), use_container_width=True)

with tabs[2]:
    left, right = st.columns(2)
    peer = villages[["village", "Reliability Score", "Maintenance Risk Score", "Sustainability Score"]].sort_values("Reliability Score", ascending=False)
    with left:
        fig = px.scatter(
            villages,
            x="Reliability Score",
            y="Maintenance Risk Score",
            size="population_served",
            hover_name="village",
            title="Reliability vs maintenance risk"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(peer, use_container_width=True)

with tabs[3]:
    st.subheader("Recent work orders")
    st.dataframe(filtered_maint, use_container_width=True)
    alert_count = (villages["Maintenance Risk Score"] > 55).sum()
    st.warning(f"{alert_count} village(s) currently exceed the maintenance-risk threshold of 55.")

with tabs[4]:
    fin = villages[["village", "annual_revenue_usd", "annual_opex_usd", "capex_usd"]].copy()
    fin["EBITDA"] = fin["annual_revenue_usd"] - fin["annual_opex_usd"]
    top1, top2 = st.columns(2)
    with top1:
        fig = px.bar(fin, x="village", y=["annual_revenue_usd", "annual_opex_usd"], barmode="group", title="Revenue vs opex")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with top2:
        fig = px.bar(fin, x="village", y="EBITDA", title="EBITDA by village")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fin, use_container_width=True)

with tabs[5]:
    fig = px.scatter_geo(
        villages,
        lat="lat",
        lon="lon",
        hover_name="village",
        size="population_served",
        color="Reliability Score",
        projection="natural earth",
        scope="africa",
        title="Pilot sites"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tabs[6]:
    st.dataframe(villages, use_container_width=True)
    st.download_button(
        label=T["download"],
        data=villages.to_csv(index=False).encode("utf-8"),
        file_name="stratwater_villages.csv",
        mime="text/csv"
    )
