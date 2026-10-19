# Ed-Tech Lead Funnel & Conversion Analytics

End-to-end sales-funnel analytics for an online skill-training / ed-tech business: funnel drop-off, channel efficiency,
speed-to-lead impact, and a lead-scoring model that ranks who a sales rep should call first.

> **Data note:** `data/leads.csv` is a **simulated** 5,000-lead dataset created by `generate_data.py` (seed 42), with channel
> mix, costs and course price (INR 2,949 incl. GST) modelled on a typical Indian ed-tech funnel. Findings reflect the
> simulation's assumptions, so describe them as simulated on your resume, or swap in real/exported CRM data or a public
> marketing-funnel dataset (e.g. from Kaggle) and re-run.

## Run
```bash
pip install -r requirements.txt
python generate_data.py     # -> data/leads.csv
python analysis.py          # -> outputs/ (tables, charts, metrics.json, leads_scored.csv)
streamlit run app.py        # interactive dashboard
```

## What it does
1. **Funnel**: Lead -> Contacted -> Demo -> Enrolled, with stage-to-stage conversion.
2. **Channel efficiency**: conversion %, spend, cost per enrolment and revenue/spend for 6 channels.
3. **Speed-to-lead**: enrolment rate by first-response time bucket.
4. **Lead scoring**: logistic regression using only features known at lead creation (no leakage), evaluated on a held-out
   25% split (ROC-AUC), with leads grouped into score bands A-D.
5. **Dashboard**: filters by channel/course/college tier, KPIs, and a priority call list.

## Results from the simulated run (seed 42)
| Metric | Value |
|---|---|
| Overall lead -> enrolment | 16.6% |
| Best channel by conversion / ROI | Referral (27.9%, 13.8x) |
| Worst channel by ROI | Instagram Ads (3.6x) |
| Model ROC-AUC (held-out) | 0.667 |
| Top-quartile (band A) conversion vs bottom (band D) | 27.0% vs 6.4% |
| Share of all enrolments captured by calling only top 25% scored leads | 40.8% |
| Enrolment rate, response <6h vs >24h | ~18-19% vs 10.8% |

## Limitations (worth saying in an interview)
- Spend counts lead-acquisition cost only (no sales-rep salary or demo cost), so ROI is overstated.
- Simulated data: relationships such as "faster response -> higher conversion" were built into the generator, so treat
  them as illustrations of the method, not discoveries.
- Model is a baseline; next steps would be gradient boosting, calibration, and an A/B test of score-based call prioritisation.
