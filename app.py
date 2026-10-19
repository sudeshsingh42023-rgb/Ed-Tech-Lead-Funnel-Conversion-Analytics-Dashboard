"""Streamlit dashboard. Run: python generate_data.py && python analysis.py && streamlit run app.py"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ed-Tech Funnel Analytics", layout="wide")
st.title("Ed-Tech Lead Funnel & Conversion Analytics")
st.caption("Simulated dataset (5,000 leads) - replace data/leads.csv with real CRM exports.")

df = pd.read_csv("outputs/leads_scored.csv", parse_dates=["created_date"])

# Sidebar filters
st.sidebar.header("Filters")
channels = st.sidebar.multiselect("Channel", sorted(df.channel.unique()), default=sorted(df.channel.unique()))
courses = st.sidebar.multiselect("Course", sorted(df.course_interest.unique()), default=sorted(df.course_interest.unique()))
tiers = st.sidebar.multiselect("College tier", sorted(df.college_tier.unique()), default=sorted(df.college_tier.unique()))
f = df[df.channel.isin(channels) & df.course_interest.isin(courses) & df.college_tier.isin(tiers)]
if f.empty:
    st.warning("No leads match these filters."); st.stop()

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Leads", f"{len(f):,}")
k2.metric("Enrolled", f"{int(f.enrolled.sum()):,}", f"{f.enrolled.mean():.1%} conversion")
k3.metric("Cost / enrolment", f"INR {f.cost_inr.sum() / max(f.enrolled.sum(), 1):,.0f}")
k4.metric("Revenue / spend", f"{f.revenue_inr.sum() / f.cost_inr.sum():.1f}x")

# Funnel
st.subheader("Funnel")
fun = pd.DataFrame({"Stage": ["Leads", "Contacted", "Demo attended", "Enrolled"],
                    "Count": [len(f), f.contacted.sum(), f.demo_attended.sum(), f.enrolled.sum()]}).set_index("Stage")
st.bar_chart(fun)

# Channel table
st.subheader("Channel performance")
g = f.groupby("channel")
ch = pd.DataFrame({"Leads": g.size(), "Enrolled": g.enrolled.sum(), "Conversion %": (g.enrolled.mean() * 100).round(1),
                   "Spend (INR)": g.cost_inr.sum().round(0), "Revenue (INR)": g.revenue_inr.sum()})
ch["Cost / enrolment (INR)"] = (ch["Spend (INR)"] / ch.Enrolled.replace(0, float("nan"))).round(0)
ch["ROI (x)"] = (ch["Revenue (INR)"] / ch["Spend (INR)"]).round(1)
st.dataframe(ch.sort_values("Conversion %", ascending=False), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Speed-to-lead")
    b = pd.cut(f.first_response_hrs, [0, 2, 6, 12, 24, 1e9], labels=["<2h", "2-6h", "6-12h", "12-24h", ">24h"])
    st.bar_chart((f.groupby(b, observed=True).enrolled.mean() * 100).rename("Enrolment %"))
with c2:
    st.subheader("Lead-score bands (A = highest)")
    st.bar_chart((f.groupby("score_band", observed=True).enrolled.mean() * 100).rename("Enrolment %"))

# Call list
st.subheader("Priority call list (top-scored leads not yet enrolled)")
cols = ["lead_id", "channel", "course_interest", "city", "college_tier", "touchpoints", "lead_score"]
st.dataframe(f[f.enrolled == 0].sort_values("lead_score", ascending=False)[cols].head(50), use_container_width=True)
