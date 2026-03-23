from pathlib import Path
import pandas as pd
import streamlit as st

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


def status_label(score: int) -> str:
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Watch"
    return "Critical"


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

    # Normalize village table
    village_col = find_column(villages, ["Village", "village", "Village Name", "name"])
    villages = villages.rename(columns={village_col: "Village"})

    pop_col = find_column(villages, ["Population", "population", "People Served"], required=False)
    uptime_col = find_column(villages, ["Uptime", "uptime", "Uptime (%)", "uptime_pct"], required=False)
    fail_col = find_column(villages, ["Failures", "failures", "Failure Count"], required=False)
    revenue_col = find_column(villages, ["Monthly Revenue", "monthly_revenue", "Revenue"], required=False)
    cost_col = find_column(villages, ["Monthly Cost", "monthly_cost", "Cost"], required=False)
    lat_col = find_column(villages, ["lat", "latitude", "Latitude"], required=False)
    lon_col = find_column(villages, ["lon", "lng", "longitude", "Longitude"], required=False)

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

    for col in ["Population", "Uptime", "Failures", "Monthly Revenue", "Monthly Cost", "Latitude", "Longitude"]:
        if col not in villages.columns:
            villages[col] = pd.NA

    # Normalize daily usage table
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

    # Normalize maintenance table
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

    # Derived metrics
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

    villages["Reliability Score"] = villages.apply(
        lambda row: reliability_score(row["Uptime"], row["Failures"]), axis=1
    )
    villages["Maintenance Risk"] = villages.apply(
        lambda row: maintenance_risk_score(row["Failures"], row["Total_Downtime_Hours"]), axis=1
    )
    villages["Water Stress"] = villages.apply(
        lambda row: water_stress_index(row["Population"], row["Avg Daily Usage"]), axis=1
    )
    villages["Sustainability Score"] = villages.apply(
        lambda row: sustainability_score(row["Monthly Revenue"], row["Monthly Cost"]), axis=1
    )
    villages["System Status"] = villages["Reliability Score"].apply(status_label)

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Water Usage", "AI Scores", "Maintenance", "Financials", "Map"]
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

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Reliability", int(village_row["Reliability Score"]))
    s2.metric("Maintenance Risk", int(village_row["Maintenance Risk"]))
    s3.metric("Water Stress", int(village_row["Water Stress"]))
    s4.metric("Sustainability", int(village_row["Sustainability Score"]))

    st.write("Reliability Score")
    st.progress(int(village_row["Reliability Score"]))

    st.write("Maintenance Risk")
    st.progress(int(village_row["Maintenance Risk"]))

    st.write("Water Stress")
    st.progress(int(village_row["Water Stress"]))

    st.write("Sustainability Score")
    st.progress(int(village_row["Sustainability Score"]))

    st.write("All Villages Score Comparison")
    score_df = villages.set_index("Village")[
        ["Reliability Score", "Maintenance Risk", "Water Stress", "Sustainability Score"]
    ]
    st.bar_chart(score_df, use_container_width=True)

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
