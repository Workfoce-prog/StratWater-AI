from pathlib import Path
from io import BytesIO
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="StratWater AI", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# ---------- Helpers ----------
def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True):
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    if required:
        raise KeyError(f"Missing one of these columns: {candidates}")
    return None


def reliability_score(uptime: float, failures: float) -> int:
    score = 100
    if pd.notna(uptime) and uptime < 90:
        score -= 20
    if pd.notna(failures) and failures > 3:
        score -= 25
    return max(int(score), 0)


def maintenance_risk_score(failures: float, downtime_hours: float = 0) -> int:
    score = 20
    if pd.notna(failures):
        score += min(int(failures * 15), 60)
    if pd.notna(downtime_hours):
        score += min(int(downtime_hours / 2), 20)
    return min(score, 100)


def water_stress_index(population: float, avg_daily_usage: float) -> int:
    if pd.isna(population) or population <= 0 or pd.isna(avg_daily_usage):
        return 0
    liters_per_person = avg_daily_usage / population
    if liters_per_person >= 30:
        return 20
    if liters_per_person >= 20:
        return 45
    if liters_per_person >= 15:
        return 65
    return 85


def seasonal_dryness_score(avg_rainfall_mm: float, dry_season_months: float, temperature_c: float = 0) -> int:
    score = 20

    if pd.notna(avg_rainfall_mm):
        if avg_rainfall_mm < 400:
            score += 35
        elif avg_rainfall_mm < 700:
            score += 20
        elif avg_rainfall_mm < 1000:
            score += 10

    if pd.notna(dry_season_months):
        if dry_season_months >= 8:
            score += 30
        elif dry_season_months >= 6:
            score += 20
        elif dry_season_months >= 4:
            score += 10

    if pd.notna(temperature_c):
        if temperature_c >= 38:
            score += 15
        elif temperature_c >= 33:
            score += 8

    return min(int(score), 100)


def sustainability_score(monthly_revenue: float, monthly_cost: float) -> int:
    if pd.isna(monthly_revenue) or pd.isna(monthly_cost) or monthly_cost <= 0:
        return 0
    ratio = monthly_revenue / monthly_cost
    if ratio >= 1.5:
        return 90
    if ratio >= 1.2:
        return 75
    if ratio >= 1.0:
        return 60
    if ratio >= 0.8:
        return 40
    return 20


def system_status(score: int) -> str:
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Watch"
    return "Critical"


def rag_label(score: int, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if score >= 80:
            return "Excellent"
        if score >= 60:
            return "Green"
        if score >= 40:
            return "Amber"
        return "Red"
    else:
        if score <= 30:
            return "Green"
        if score <= 60:
            return "Amber"
        return "Red"


def rag_badge(score: int, higher_is_better: bool = True) -> str:
    label = rag_label(score, higher_is_better)
    emoji = {
        "Excellent": "🔵",
        "Green": "🟢",
        "Amber": "🟠",
        "Red": "🔴",
    }[label]
    return f"{emoji} {label}"


def interpret_reliability(score: int) -> tuple[str, str]:
    if score >= 80:
        return (
            "The water system is operating reliably with limited disruption.",
            "Maintain current preventive maintenance schedule, continue uptime monitoring, and document best practices for replication."
        )
    if score >= 60:
        return (
            "The system is moderately reliable, but periodic performance issues are emerging.",
            "Increase maintenance frequency, inspect pump components, and track downtime causes before failures worsen."
        )
    return (
        "The system has low reliability and frequent interruptions are likely affecting service delivery.",
        "Urgent technical review is needed. Inspect the pump, power system, and piping. Consider component replacement or system redesign."
    )


def interpret_maintenance_risk(score: int) -> tuple[str, str]:
    if score <= 30:
        return (
            "Current maintenance risk is low.",
            "Continue routine inspections and preserve a spare-parts inventory for quick response."
        )
    if score <= 60:
        return (
            "Maintenance risk is moderate and requires proactive attention.",
            "Schedule preventive servicing, review downtime patterns, and assign a local maintenance focal point."
        )
    return (
        "Maintenance risk is high and the system may be approaching failure.",
        "Prioritize immediate inspection, allocate repair funds, and prepare contingency plans to avoid service interruption."
    )


def interpret_water_stress(score: int) -> tuple[str, str]:
    if score <= 30:
        return (
            "Current water supply appears sufficient relative to community demand.",
            "Maintain monitoring and preserve current operating capacity."
        )
    if score <= 60:
        return (
            "Demand is beginning to pressure available water supply.",
            "Monitor seasonal demand, educate users on conservation, and assess whether storage expansion is needed."
        )
    return (
        "Water demand is likely exceeding or approaching the limits of current supply capacity.",
        "Evaluate options for added storage, additional boreholes, or distribution redesign to prevent shortages."
    )


def interpret_seasonal_dryness(score: int) -> tuple[str, str]:
    if score <= 30:
        return (
            "Seasonal weather conditions appear favorable, with relatively low dryness pressure on water supply.",
            "Maintain current monitoring and continue observing seasonal rainfall patterns."
        )
    if score <= 60:
        return (
            "Moderate seasonal dryness may place periodic pressure on supply, especially during dry months.",
            "Prepare for seasonal demand surges, monitor storage levels, and strengthen drought contingency planning."
        )
    return (
        "High seasonal dryness risk suggests strong exposure to drought-related supply stress.",
        "Increase storage capacity, strengthen dry-season planning, and assess backup supply options or additional boreholes."
    )


def interpret_sustainability(score: int) -> tuple[str, str]:
    if score >= 80:
        return (
            "The system appears financially sustainable under current conditions.",
            "Maintain the current fee model, keep transparent records, and assess readiness for expansion."
        )
    if score >= 60:
        return (
            "The system is near sustainability but remains vulnerable to cost or revenue shocks.",
            "Strengthen collection practices, reduce avoidable costs, and build a reserve fund."
        )
    return (
        "The current financial model is not strong enough to sustain long-term operations.",
        "Review pricing, collections, subsidies, and partnership options. External support may be needed in the short term."
    )


def overall_narrative(row: pd.Series) -> str:
    reliability = int(row["Reliability Score"])
    risk = int(row["Maintenance Risk"])
    stress = int(row["Water Stress"])
    dryness = int(row["Seasonal Dryness Score"])
    sustainability = int(row["Sustainability Score"])

    if reliability < 60 or risk > 60:
        return (
            f"{row['Village']} is currently a high-priority site. Reliability is weak and/or maintenance risk is elevated, "
            "suggesting a near-term operational threat. Immediate technical assessment and repair planning are recommended."
        )
    if stress > 60:
        return (
            f"{row['Village']} is showing signs of supply pressure. The current system may be insufficient for demand, "
            "especially during peak or seasonal stress periods. Capacity expansion should be assessed."
        )
    if dryness > 60:
        return (
            f"{row['Village']} faces elevated seasonal dryness pressure. Even if the current system is functioning, "
            "dry-season conditions may reduce resilience and increase future water stress. Storage and drought planning should be strengthened."
        )
    if sustainability < 60:
        return (
            f"{row['Village']} is operationally functional, but the financial model is fragile. "
            "Improved collections, cost control, or supplemental financing are recommended."
        )
    return (
        f"{row['Village']} is performing well across technical, climate, and financial indicators. "
        "This site may be suitable as a model for replication and scale."
    )


def overall_recommendation(row: pd.Series) -> str:
    reliability = int(row["Reliability Score"])
    risk = int(row["Maintenance Risk"])
    stress = int(row["Water Stress"])
    dryness = int(row["Seasonal Dryness Score"])
    sustainability = int(row["Sustainability Score"])

    if reliability < 60 or risk > 60:
        return "Immediate technical intervention required."
    if stress > 60:
        return "Capacity expansion or storage improvement recommended."
    if dryness > 60:
        return "Dry-season resilience planning and additional storage are recommended."
    if sustainability < 60:
        return "Financial model review recommended."
    return "System is stable and ready for scale or replication."


def build_pdf_report(row: pd.Series) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    line_gap = 18

    def write_line(text: str, font="Helvetica", size=11, gap=line_gap):
        nonlocal y
        pdf.setFont(font, size)
        pdf.drawString(50, y, str(text)[:110])
        y -= gap

    pdf.setTitle(f"StratWater_AI_Report_{row['Village']}")

    write_line("StratWater AI - Village Water Performance Report", "Helvetica-Bold", 16, 24)
    write_line(f"Village: {row['Village']}", "Helvetica-Bold", 12)
    write_line(f"Population Served: {int(row['Population']):,}")
    write_line(f"Uptime (%): {float(row['Uptime']):.1f}")
    write_line(f"Average Daily Usage (L): {float(row['Avg Daily Usage']):,.0f}")
    write_line(f"Failures: {int(row['Failures'])}")
    write_line(f"Maintenance Events: {int(row['Maintenance_Events'])}")
    write_line(f"Downtime Hours: {float(row['Total_Downtime_Hours']):.1f}")
    write_line(f"Average Rainfall (mm): {float(row['avg_rainfall_mm']):,.0f}")
    write_line(f"Dry Season Months: {float(row['dry_season_months']):.0f}")
    write_line(f"Average Temperature (C): {float(row['temperature_c']):.1f}")
    write_line("")

    write_line("AI Scores", "Helvetica-Bold", 13)
    write_line(f"Reliability Score: {int(row['Reliability Score'])}/100")
    write_line(f"Maintenance Risk: {int(row['Maintenance Risk'])}/100")
    write_line(f"Water Stress: {int(row['Water Stress'])}/100")
    write_line(f"Seasonal Dryness Score: {int(row['Seasonal Dryness Score'])}/100")
    write_line(f"Sustainability Score: {int(row['Sustainability Score'])}/100")
    write_line("")

    rel_text, rel_rec = interpret_reliability(int(row["Reliability Score"]))
    risk_text, risk_rec = interpret_maintenance_risk(int(row["Maintenance Risk"]))
    stress_text, stress_rec = interpret_water_stress(int(row["Water Stress"]))
    dry_text, dry_rec = interpret_seasonal_dryness(int(row["Seasonal Dryness Score"]))
    sust_text, sust_rec = interpret_sustainability(int(row["Sustainability Score"]))

    write_line("Interpretation & Recommendations", "Helvetica-Bold", 13)
    write_line(f"Reliability: {rel_text}")
    write_line(f"Recommendation: {rel_rec}")
    write_line("")
    write_line(f"Maintenance Risk: {risk_text}")
    write_line(f"Recommendation: {risk_rec}")
    write_line("")
    write_line(f"Water Stress: {stress_text}")
    write_line(f"Recommendation: {stress_rec}")
    write_line("")
    write_line(f"Seasonal Dryness: {dry_text}")
    write_line(f"Recommendation: {dry_rec}")
    write_line("")
    write_line(f"Sustainability: {sust_text}")
    write_line(f"Recommendation: {sust_rec}")
    write_line("")

    write_line("Overall Narrative", "Helvetica-Bold", 13)
    write_line(overall_narrative(row))
    write_line(f"Overall Recommendation: {overall_recommendation(row)}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------- Data Loading ----------
@st.cache_data
def load_data():
    files = {
        "villages": BASE_DIR / "villages.csv",
        "daily": BASE_DIR / "daily_usage.csv",
        "maintenance": BASE_DIR / "maintenance_log.csv",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        st.error("Missing required files:")
        for item in missing:
            st.write(f"- {item}")
        st.info("Put the CSV files in the repo root next to app.py.")
        st.stop()

    villages = pd.read_csv(files["villages"])
    daily_usage = pd.read_csv(files["daily"])
    maintenance = pd.read_csv(files["maintenance"])

    village_col = find_column(villages, ["Village", "village", "Village Name", "name"])
    villages = villages.rename(columns={village_col: "Village"})

    pop_col = find_column(villages, ["Population", "population", "People Served"], required=False)
    uptime_col = find_column(villages, ["Uptime", "uptime", "Uptime (%)", "uptime_pct"], required=False)
    fail_col = find_column(villages, ["Failures", "failures", "Failure Count"], required=False)
    revenue_col = find_column(villages, ["Monthly Revenue", "monthly_revenue", "Revenue"], required=False)
    cost_col = find_column(villages, ["Monthly Cost", "monthly_cost", "Cost"], required=False)
    lat_col = find_column(villages, ["lat", "latitude", "Latitude"], required=False)
    lon_col = find_column(villages, ["lon", "lng", "longitude", "Longitude"], required=False)
    rain_col = find_column(villages, ["avg_rainfall_mm", "Average Rainfall", "rainfall_mm", "Rainfall"], required=False)
    dry_col = find_column(villages, ["dry_season_months", "Dry Season Months", "dry_months"], required=False)
    temp_col = find_column(villages, ["temperature_c", "Temperature", "temp_c"], required=False)

    if pop_col and pop_col != "Population":
        villages = villages.rename(columns={pop_col: "Population"})
    if uptime_col and uptime_col != "Uptime":
        villages = villages.rename(columns={uptime_col: "Uptime"})
    if fail_col and fail_col != "Failures":
        villages = villages.rename(columns={fail_col: "Failures"})
    if revenue_col and revenue_col != "Monthly Revenue":
        villages = villages.rename(columns={revenue_col: "Monthly Revenue"})
    if cost_col and cost_col != "Monthly Cost":
        villages = villages.rename(columns={cost_col: "Monthly Cost"})
    if lat_col and lat_col != "Latitude":
        villages = villages.rename(columns={lat_col: "Latitude"})
    if lon_col and lon_col != "Longitude":
        villages = villages.rename(columns={lon_col: "Longitude"})
    if rain_col and rain_col != "avg_rainfall_mm":
        villages = villages.rename(columns={rain_col: "avg_rainfall_mm"})
    if dry_col and dry_col != "dry_season_months":
        villages = villages.rename(columns={dry_col: "dry_season_months"})
    if temp_col and temp_col != "temperature_c":
        villages = villages.rename(columns={temp_col: "temperature_c"})

    for col in [
        "Population", "Uptime", "Failures", "Monthly Revenue", "Monthly Cost",
        "Latitude", "Longitude", "avg_rainfall_mm", "dry_season_months", "temperature_c"
    ]:
        if col not in villages.columns:
            villages[col] = pd.NA

    daily_village_col = find_column(daily_usage, ["Village", "village", "Village Name", "name"])
    daily_date_col = find_column(daily_usage, ["date", "Date"])
    usage_col = find_column(daily_usage, ["Usage", "usage", "Daily Usage", "Water Usage", "liters", "Liters"])

    daily_usage = daily_usage.rename(
        columns={
            daily_village_col: "Village",
            daily_date_col: "date",
            usage_col: "Usage",
        }
    )
    daily_usage["date"] = pd.to_datetime(daily_usage["date"], errors="coerce")
    daily_usage["Usage"] = pd.to_numeric(daily_usage["Usage"], errors="coerce")

    maint_village_col = find_column(maintenance, ["Village", "village", "Village Name", "name"])
    maint_date_col = find_column(maintenance, ["event_date", "Event Date", "date", "Date"])
    issue_col = find_column(maintenance, ["issue", "Issue", "Event", "Description"], required=False)
    downtime_col = find_column(maintenance, ["downtime_hours", "Downtime Hours", "downtime"], required=False)
    status_col = find_column(maintenance, ["status", "Status"], required=False)

    rename_map = {
        maint_village_col: "Village",
        maint_date_col: "event_date",
    }
    if issue_col:
        rename_map[issue_col] = "issue"
    if downtime_col:
        rename_map[downtime_col] = "downtime_hours"
    if status_col:
        rename_map[status_col] = "status"

    maintenance = maintenance.rename(columns=rename_map)
    maintenance["event_date"] = pd.to_datetime(maintenance["event_date"], errors="coerce")

    if "issue" not in maintenance.columns:
        maintenance["issue"] = "Maintenance event"
    if "downtime_hours" not in maintenance.columns:
        maintenance["downtime_hours"] = 0
    if "status" not in maintenance.columns:
        maintenance["status"] = "Open"

    maintenance["downtime_hours"] = pd.to_numeric(maintenance["downtime_hours"], errors="coerce").fillna(0)

    avg_usage = (
        daily_usage.groupby("Village", as_index=False)["Usage"]
        .mean()
        .rename(columns={"Usage": "Avg Daily Usage"})
    )

    maint_summary = (
        maintenance.groupby("Village", as_index=False)
        .agg(
            Maintenance_Events=("issue", "count"),
            Total_Downtime_Hours=("downtime_hours", "sum"),
        )
    )

    villages = villages.merge(avg_usage, on="Village", how="left")
    villages = villages.merge(maint_summary, on="Village", how="left")

    villages["Avg Daily Usage"] = pd.to_numeric(villages["Avg Daily Usage"], errors="coerce").fillna(0)
    villages["Maintenance_Events"] = pd.to_numeric(villages["Maintenance_Events"], errors="coerce").fillna(0)
    villages["Total_Downtime_Hours"] = pd.to_numeric(villages["Total_Downtime_Hours"], errors="coerce").fillna(0)
    villages["Uptime"] = pd.to_numeric(villages["Uptime"], errors="coerce").fillna(0)
    villages["Failures"] = pd.to_numeric(villages["Failures"], errors="coerce").fillna(0)
    villages["Population"] = pd.to_numeric(villages["Population"], errors="coerce").fillna(0)
    villages["Monthly Revenue"] = pd.to_numeric(villages["Monthly Revenue"], errors="coerce").fillna(0)
    villages["Monthly Cost"] = pd.to_numeric(villages["Monthly Cost"], errors="coerce").fillna(0)
    villages["avg_rainfall_mm"] = pd.to_numeric(villages["avg_rainfall_mm"], errors="coerce").fillna(0)
    villages["dry_season_months"] = pd.to_numeric(villages["dry_season_months"], errors="coerce").fillna(0)
    villages["temperature_c"] = pd.to_numeric(villages["temperature_c"], errors="coerce").fillna(0)

    villages["Reliability Score"] = villages.apply(
        lambda row: reliability_score(row["Uptime"], row["Failures"]), axis=1
    )
    villages["Maintenance Risk"] = villages.apply(
        lambda row: maintenance_risk_score(row["Failures"], row["Total_Downtime_Hours"]), axis=1
    )
    villages["Water Stress"] = villages.apply(
        lambda row: water_stress_index(row["Population"], row["Avg Daily Usage"]), axis=1
    )
    villages["Seasonal Dryness Score"] = villages.apply(
        lambda row: seasonal_dryness_score(
            row["avg_rainfall_mm"],
            row["dry_season_months"],
            row["temperature_c"]
        ),
        axis=1
    )
    villages["Sustainability Score"] = villages.apply(
        lambda row: sustainability_score(row["Monthly Revenue"], row["Monthly Cost"]), axis=1
    )
    villages["System Status"] = villages["Reliability Score"].apply(system_status)

    return villages, daily_usage, maintenance


villages, daily_usage, maintenance = load_data()

# ---------- Sidebar ----------
st.sidebar.title("StratWater AI")
selected_village = st.sidebar.selectbox("Select Village", sorted(villages["Village"].dropna().unique()))

village_df = villages[villages["Village"] == selected_village].copy()
village_row = village_df.iloc[0]

village_daily = daily_usage[daily_usage["Village"] == selected_village].sort_values("date")
village_maintenance = maintenance[maintenance["Village"] == selected_village].sort_values("event_date", ascending=False)

# ---------- Header ----------
st.title("💧 StratWater AI Dashboard")
st.caption("Smart Water. Sustainable Communities.")

# ---------- KPI Row ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Population Served", f"{int(village_row['Population']):,}")
k2.metric("Uptime (%)", f"{float(village_row['Uptime']):.1f}")
k3.metric("Avg Daily Usage (L)", f"{float(village_row['Avg Daily Usage']):,.0f}")
k4.metric("System Status", village_row["System Status"])

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Overview", "Water Usage", "AI Scores", "Maintenance", "Financials", "Map", "Weather & Seasonality", "Insights & Recommendations"]
)

with tab1:
    st.subheader(f"Village Overview: {selected_village}")

    c1, c2 = st.columns(2)

    with c1:
        overview = pd.DataFrame(
            {
                "Metric": [
                    "Population",
                    "Uptime (%)",
                    "Failures",
                    "Avg Daily Usage (L)",
                    "Maintenance Events",
                    "Downtime Hours",
                ],
                "Value": [
                    int(village_row["Population"]),
                    round(float(village_row["Uptime"]), 1),
                    int(village_row["Failures"]),
                    round(float(village_row["Avg Daily Usage"]), 0),
                    int(village_row["Maintenance_Events"]),
                    round(float(village_row["Total_Downtime_Hours"]), 1),
                ],
            }
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

    with c2:
        comparison_cols = [
            "Village",
            "Reliability Score",
            "Maintenance Risk",
            "Water Stress",
            "Seasonal Dryness Score",
            "Sustainability Score",
            "System Status",
        ]
        st.write("Portfolio Snapshot")
        st.dataframe(villages[comparison_cols], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Water Usage Trend")

    if village_daily.empty:
        st.warning("No daily usage data available for this village.")
    else:
        usage_chart = village_daily.set_index("date")[["Usage"]]
        st.line_chart(usage_chart, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Max Daily Usage", f"{village_daily['Usage'].max():,.0f} L")
        c2.metric("Min Daily Usage", f"{village_daily['Usage'].min():,.0f} L")
        c3.metric("Average Daily Usage", f"{village_daily['Usage'].mean():,.0f} L")

        st.dataframe(village_daily, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("AI Scores")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Reliability", int(village_row["Reliability Score"]))
    s2.metric("Maintenance Risk", int(village_row["Maintenance Risk"]))
    s3.metric("Water Stress", int(village_row["Water Stress"]))
    s4.metric("Seasonal Dryness", int(village_row["Seasonal Dryness Score"]))
    s5.metric("Sustainability", int(village_row["Sustainability Score"]))

    st.write(f"Reliability: {rag_badge(int(village_row['Reliability Score']), True)}")
    st.progress(int(village_row["Reliability Score"]))

    st.write(f"Maintenance Risk: {rag_badge(int(village_row['Maintenance Risk']), False)}")
    st.progress(int(village_row["Maintenance Risk"]))

    st.write(f"Water Stress: {rag_badge(int(village_row['Water Stress']), False)}")
    st.progress(int(village_row["Water Stress"]))

    st.write(f"Seasonal Dryness: {rag_badge(int(village_row['Seasonal Dryness Score']), False)}")
    st.progress(int(village_row["Seasonal Dryness Score"]))

    st.write(f"Sustainability: {rag_badge(int(village_row['Sustainability Score']), True)}")
    st.progress(int(village_row["Sustainability Score"]))

    st.write("All Villages Score Comparison")
    score_df = villages.set_index("Village")[
        ["Reliability Score", "Maintenance Risk", "Water Stress", "Seasonal Dryness Score", "Sustainability Score"]
    ]
    st.bar_chart(score_df, use_container_width=True)

    st.markdown("### Score Legend")
    legend_df = pd.DataFrame(
        {
            "Score Type": [
                "Reliability Score",
                "Maintenance Risk",
                "Water Stress Index",
                "Seasonal Dryness Score",
                "Sustainability Score",
            ],
            "Meaning": [
                "Higher is better",
                "Lower is better",
                "Lower is better",
                "Lower is better",
                "Higher is better",
            ],
            "Red": [
                "0-39 = frequent breakdowns / weak service",
                "61-100 = high failure risk",
                "61-100 = high pressure on supply",
                "61-100 = severe dry-season / climate pressure",
                "0-39 = financially weak",
            ],
            "Amber": [
                "40-59 = unstable",
                "31-60 = moderate risk",
                "31-60 = moderate stress",
                "31-60 = moderate seasonal dryness",
                "40-59 = near break-even risk",
            ],
            "Green": [
                "60-79 = stable / acceptable",
                "0-30 = manageable risk",
                "0-30 = supply generally adequate",
                "0-30 = manageable dryness pressure",
                "60-79 = sustainable",
            ],
            "Excellent": [
                "80-100 = strong performance",
                "N/A",
                "N/A",
                "N/A",
                "80-100 = highly sustainable",
            ],
        }
    )
    st.dataframe(legend_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Maintenance Log")

    if village_maintenance.empty:
        st.info("No maintenance records available for this village.")
    else:
        open_count = (village_maintenance["status"].astype(str).str.lower() == "open").sum()
        closed_count = (village_maintenance["status"].astype(str).str.lower() == "closed").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Events", len(village_maintenance))
        c2.metric("Open", int(open_count))
        c3.metric("Closed", int(closed_count))

        st.dataframe(village_maintenance, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Financials")

    monthly_revenue = float(village_row["Monthly Revenue"])
    monthly_cost = float(village_row["Monthly Cost"])
    annual_revenue = monthly_revenue * 12
    annual_cost = monthly_cost * 12
    annual_surplus = annual_revenue - annual_cost

    f1, f2, f3 = st.columns(3)
    f1.metric("Monthly Revenue", f"${monthly_revenue:,.0f}")
    f2.metric("Monthly Cost", f"${monthly_cost:,.0f}")
    f3.metric("Annual Surplus", f"${annual_surplus:,.0f}")

    finance_df = pd.DataFrame(
        {
            "Metric": ["Annual Revenue", "Annual Cost", "Annual Surplus"],
            "Amount": [annual_revenue, annual_cost, annual_surplus],
        }
    ).set_index("Metric")

    st.bar_chart(finance_df, use_container_width=True)

with tab6:
    st.subheader("Village Map")

    map_df = villages[["Village", "Latitude", "Longitude"]].copy()
    map_df["Latitude"] = pd.to_numeric(map_df["Latitude"], errors="coerce")
    map_df["Longitude"] = pd.to_numeric(map_df["Longitude"], errors="coerce")
    map_df = map_df.dropna(subset=["Latitude", "Longitude"])

    if map_df.empty:
        st.info("No latitude/longitude data available.")
    else:
        st.map(map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}), use_container_width=True)
        st.dataframe(map_df, use_container_width=True, hide_index=True)

with tab7:
    st.subheader("Weather & Seasonality")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rainfall (mm)", f"{float(village_row['avg_rainfall_mm']):,.0f}")
    c2.metric("Dry Season (months)", f"{float(village_row['dry_season_months']):.0f}")
    c3.metric("Average Temperature (°C)", f"{float(village_row['temperature_c']):.1f}")

    st.write(f"Seasonal Dryness Score: {rag_badge(int(village_row['Seasonal Dryness Score']), False)}")
    st.progress(int(village_row["Seasonal Dryness Score"]))

    weather_df = pd.DataFrame(
        {
            "Metric": ["Average Rainfall (mm)", "Dry Season Months", "Average Temperature (°C)", "Seasonal Dryness Score"],
            "Value": [
                float(village_row["avg_rainfall_mm"]),
                float(village_row["dry_season_months"]),
                float(village_row["temperature_c"]),
                int(village_row["Seasonal Dryness Score"]),
            ],
        }
    )
    st.dataframe(weather_df, use_container_width=True, hide_index=True)

with tab8:
    st.subheader("AI Insights, Meaning, and Recommendations")

    reliability = int(village_row["Reliability Score"])
    risk = int(village_row["Maintenance Risk"])
    stress = int(village_row["Water Stress"])
    dryness = int(village_row["Seasonal Dryness Score"])
    sustainability = int(village_row["Sustainability Score"])

    rel_text, rel_rec = interpret_reliability(reliability)
    risk_text, risk_rec = interpret_maintenance_risk(risk)
    stress_text, stress_rec = interpret_water_stress(stress)
    dry_text, dry_rec = interpret_seasonal_dryness(dryness)
    sust_text, sust_rec = interpret_sustainability(sustainability)

    st.markdown("### 💧 Water Reliability Score")
    st.write("**Meaning:** Measures how consistently the system delivers water with limited breakdowns and downtime.")
    st.write(f"**Current Score:** {reliability}/100 — {rag_badge(reliability, True)}")
    st.write(f"**Interpretation:** {rel_text}")
    st.write(f"**Recommendation:** {rel_rec}")
    st.markdown("---")

    st.markdown("### 🔧 Maintenance Risk Score")
    st.write("**Meaning:** Estimates the likelihood that the system may require repair or may fail if not serviced.")
    st.write(f"**Current Score:** {risk}/100 — {rag_badge(risk, False)}")
    st.write(f"**Interpretation:** {risk_text}")
    st.write(f"**Recommendation:** {risk_rec}")
    st.markdown("---")

    st.markdown("### 🌍 Water Stress Index")
    st.write("**Meaning:** Assesses whether current water supply is sufficient relative to community demand.")
    st.write(f"**Current Score:** {stress}/100 — {rag_badge(stress, False)}")
    st.write(f"**Interpretation:** {stress_text}")
    st.write(f"**Recommendation:** {stress_rec}")
    st.markdown("---")

    st.markdown("### ☀️ Seasonal Dryness Score")
    st.write("**Meaning:** Measures drought and weather-related pressure on the water system based on rainfall, dry-season duration, and heat conditions.")
    st.write(f"**Current Score:** {dryness}/100 — {rag_badge(dryness, False)}")
    st.write(f"**Interpretation:** {dry_text}")
    st.write(f"**Recommendation:** {dry_rec}")
    st.markdown("---")

    st.markdown("### 💰 Financial Sustainability Score")
    st.write("**Meaning:** Measures whether current revenue is strong enough to support operating and maintenance costs over time.")
    st.write(f"**Current Score:** {sustainability}/100 — {rag_badge(sustainability, True)}")
    st.write(f"**Interpretation:** {sust_text}")
    st.write(f"**Recommendation:** {sust_rec}")
    st.markdown("---")

    st.markdown("## 🤖 AI-Generated Narrative Summary")
    st.info(overall_narrative(village_row))

    st.markdown("## 🚀 Overall Strategic Recommendation")
    overall = overall_recommendation(village_row)
    if "Immediate" in overall:
        st.error(overall)
    elif "Capacity" in overall or "Financial" in overall or "Dry-season" in overall:
        st.warning(overall)
    else:
        st.success(overall)

    st.markdown("## 📄 Export Report")
    pdf_bytes = build_pdf_report(village_row)
    st.download_button(
        label="Download Village Performance Report (PDF)",
        data=pdf_bytes,
        file_name=f"StratWater_AI_Report_{selected_village.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )
