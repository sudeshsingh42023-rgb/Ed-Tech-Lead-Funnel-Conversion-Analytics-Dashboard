"""
Generate a SIMULATED ed-tech lead dataset (5,000 leads).
Replace with real/exported CRM data or a public marketing-funnel dataset when available.
All numbers produced from this file are simulated and must be described as such.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

# Channel profile: share of leads, cost per lead (INR), base intent
CHANNELS = {
    "Campus Drive":   dict(share=0.20, cpl=140, intent=0.55),
    "Instagram Ads":  dict(share=0.25, cpl=95,  intent=0.30),
    "WhatsApp Groups":dict(share=0.20, cpl=40,  intent=0.38),
    "Referral":       dict(share=0.10, cpl=60,  intent=0.75),
    "Webinar":        dict(share=0.15, cpl=170, intent=0.62),
    "Cold Outreach":  dict(share=0.10, cpl=55,  intent=0.22),
}
COURSES = ["AI & ML", "Full Stack", "Data Analytics", "Cloud", "VLSI", "Core Engineering"]
COURSE_P = [0.26, 0.24, 0.22, 0.12, 0.06, 0.10]
CITIES = ["Bengaluru", "Hyderabad", "Pune", "Delhi NCR", "Chennai", "Kolkata", "Tier-3 / Other"]
CITY_P = [0.17, 0.12, 0.10, 0.15, 0.09, 0.08, 0.29]
COURSE_PRICE = 2949  # incl. GST, from the public website

names = list(CHANNELS)
shares = [CHANNELS[c]["share"] for c in names]
channel = rng.choice(names, size=N, p=shares)

df = pd.DataFrame({
    "lead_id": np.arange(1, N + 1),
    "channel": channel,
    "course_interest": rng.choice(COURSES, size=N, p=COURSE_P),
    "city": rng.choice(CITIES, size=N, p=CITY_P),
    "college_tier": rng.choice(["Tier-1", "Tier-2", "Tier-3"], size=N, p=[0.15, 0.40, 0.45]),
    "year_of_study": rng.choice([1, 2, 3, 4], size=N, p=[0.12, 0.28, 0.35, 0.25]),
})
start = pd.Timestamp("2026-07-01")
df["created_date"] = start + pd.to_timedelta(rng.integers(0, 90, N), unit="D")
df["first_response_hrs"] = np.round(rng.gamma(2.0, 6.0, N), 1)   # speed-to-lead
df["touchpoints"] = rng.poisson(2.2, N) + 1
df["attended_webinar"] = (rng.random(N) < np.where(df.channel == "Webinar", 0.9, 0.12)).astype(int)
df["cost_inr"] = [round(rng.normal(CHANNELS[c]["cpl"], CHANNELS[c]["cpl"] * 0.2), 2) for c in df.channel]

# Latent intent -> stage progression (hidden ground truth used for simulation only)
base = df.channel.map(lambda c: CHANNELS[c]["intent"]).values
z = (np.log(base / (1 - base))
     - 0.035 * df.first_response_hrs.values
     + 0.22 * (df.touchpoints.values - 2)
     + 0.6 * df.attended_webinar.values
     + 0.25 * (df.year_of_study.values >= 3)
     - 0.3 * (df.college_tier.values == "Tier-3")
     + rng.normal(0, 0.6, N))
p_intent = 1 / (1 + np.exp(-z))

contacted = rng.random(N) < 0.88
demo = contacted & (rng.random(N) < 0.25 + 0.55 * p_intent)
enrolled = demo & (rng.random(N) < 0.10 + 0.55 * p_intent)

df["contacted"], df["demo_attended"], df["enrolled"] = contacted.astype(int), demo.astype(int), enrolled.astype(int)
df["days_to_close"] = np.where(enrolled, np.clip(rng.gamma(2.5, 3.0, N).round(), 1, 45), np.nan)
df["revenue_inr"] = np.where(enrolled, COURSE_PRICE, 0)
df.to_csv("data/leads.csv", index=False)
print(df.shape, "enrolled:", int(df.enrolled.sum()), f"({df.enrolled.mean():.1%})")
