"""Funnel, channel-efficiency and lead-scoring analysis. Run: python analysis.py"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/leads.csv", parse_dates=["created_date"])

# 1. Funnel -------------------------------------------------------------
funnel = pd.DataFrame({
    "stage": ["Leads", "Contacted", "Demo attended", "Enrolled"],
    "count": [len(df), df.contacted.sum(), df.demo_attended.sum(), df.enrolled.sum()],
})
funnel["pct_of_leads"] = (funnel["count"] / len(df) * 100).round(1)
funnel["stage_conversion_pct"] = (funnel["count"] / funnel["count"].shift(1) * 100).round(1)

# 2. Channel efficiency ---------------------------------------------------
g = df.groupby("channel")
ch = pd.DataFrame({
    "leads": g.size(),
    "contacted_pct": g.contacted.mean() * 100,
    "demo_pct": g.demo_attended.mean() * 100,
    "enrolled": g.enrolled.sum(),
    "conversion_pct": g.enrolled.mean() * 100,
    "spend_inr": g.cost_inr.sum(),
    "revenue_inr": g.revenue_inr.sum(),
    "median_days_to_close": g.days_to_close.median(),
})
ch["cost_per_enrolment_inr"] = ch.spend_inr / ch.enrolled
ch["roi_x"] = ch.revenue_inr / ch.spend_inr
ch = ch.round(1).sort_values("conversion_pct", ascending=False)

# 3. Speed-to-lead --------------------------------------------------------
df["response_bucket"] = pd.cut(df.first_response_hrs, [0, 2, 6, 12, 24, 1e9],
                               labels=["<2h", "2-6h", "6-12h", "12-24h", ">24h"])
speed = (df.groupby("response_bucket", observed=True).enrolled.mean() * 100).round(1)

# 4. Lead scoring (logistic regression, no leakage: only features known at lead creation)
cat = ["channel", "course_interest", "city", "college_tier"]
num = ["year_of_study", "first_response_hrs", "touchpoints", "attended_webinar"]
X, y = df[cat + num], df.enrolled
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                         ("n", StandardScaler(), num)])
model = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
model.fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
df["lead_score"] = (model.predict_proba(X)[:, 1] * 100).round(1)
df["score_band"] = pd.qcut(df.lead_score.rank(method="first"), 4, labels=["D", "C", "B", "A"])
band = (df.groupby("score_band", observed=True).enrolled.mean() * 100).round(1)

# Top-quartile lift: if reps only call the top 25% scored leads
top = df[df.score_band == "A"]
lift = top.enrolled.mean() / df.enrolled.mean()
capture = top.enrolled.sum() / df.enrolled.sum() * 100

names = model.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(model.named_steps["lr"].coef_[0], index=names).sort_values()

# Save tables ---------------------------------------------------------------
funnel.to_csv("outputs/funnel.csv", index=False)
ch.to_csv("outputs/channel_performance.csv")
df.drop(columns=["response_bucket"]).to_csv("outputs/leads_scored.csv", index=False)
metrics = dict(overall_conversion_pct=round(df.enrolled.mean() * 100, 1), model_auc=round(auc, 3),
               top_quartile_conversion_pct=float(band["A"]), bottom_quartile_conversion_pct=float(band["D"]),
               top_quartile_lift_x=round(lift, 2), top_quartile_captures_pct_of_enrolments=round(capture, 1),
               best_channel_by_conversion=ch.index[0], best_channel_by_roi=ch.roi_x.idxmax(),
               worst_channel_by_roi=ch.roi_x.idxmin())
json.dump(metrics, open("outputs/metrics.json", "w"), indent=2)

# Charts --------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].barh(funnel.stage[::-1], funnel["count"][::-1], color="#2b6cb0")
for i, v in enumerate(funnel["count"][::-1]): ax[0].text(v, i, f" {v}", va="center")
ax[0].set_title("Sales funnel")
ch.conversion_pct.sort_values().plot.barh(ax=ax[1], color="#2f855a"); ax[1].set_title("Lead -> enrolment % by channel")
ch.roi_x.sort_values().plot.barh(ax=ax[2], color="#c05621"); ax[2].set_title("Revenue / spend (x) by channel")
plt.tight_layout(); plt.savefig("outputs/funnel_channel.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
speed.plot.bar(ax=ax[0], color="#6b46c1", rot=0); ax[0].set_title("Enrolment % by first-response time"); ax[0].set_ylabel("%")
band.plot.bar(ax=ax[1], color="#2b6cb0", rot=0); ax[1].set_title("Enrolment % by lead-score band (A = top 25%)")
plt.tight_layout(); plt.savefig("outputs/speed_and_score.png", dpi=140); plt.close()

print(funnel.to_string(index=False)); print(); print(ch.to_string()); print()
print("Speed-to-lead conversion %:\n", speed.to_string()); print()
print(json.dumps(metrics, indent=2)); print("\nTop positive drivers:\n", coefs.tail(5).round(2).to_string())
