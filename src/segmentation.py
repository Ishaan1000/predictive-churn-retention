"""
segmentation.py
===============
Turns out-of-fold churn probabilities into three action tiers.

  Tier 1 - Critical : p >= 0.60                      -> proactive save call within 7 days
  Tier 2 - High     : 0.40 <= p < 0.60               -> targeted offer by channel
  Tier 3 - Watch    : 0.25 <= p < 0.40               -> automated nurture, monitor monthly
  Not at risk       : p < 0.25                       -> business as usual

Inside each tier customers are ordered by annual revenue at risk
(p x average monthly bill x 12), so a fixed-size retention team works the
most valuable saves first.

Author: Ishaan Mankotia
"""

import os
import sqlite3

import numpy as np
import pandas as pd

TIERS = [("Tier 1 - Critical", 0.60, 1.01), ("Tier 2 - High", 0.40, 0.60),
         ("Tier 3 - Watch", 0.25, 0.40), ("Not at risk", 0.0, 0.25)]

PLAYBOOK = {
    "Tier 1 - Critical": "Retention-desk call within 7 days; fix any open ticket first, then offer a "
                         "12-month contract with a loyalty discount (up to 15%) or a free plan upgrade.",
    "Tier 2 - High": "Targeted offer through the customer's own channel (app/SMS): contract migration "
                     "incentive for month-to-month users, bill-shock reversal for customers whose bill rose.",
    "Tier 3 - Watch": "Automated nurture: usage tips, data top-up bonus, CSAT survey after every ticket; "
                      "re-score monthly and escalate on any new unresolved ticket.",
    "Not at risk": "Business as usual; upsell eligible.",
}


def assign_tiers(df, proba):
    s = pd.DataFrame({"customer_id": df.customer_id.values, "churn_probability": proba.round(4)})
    s["annual_revenue_at_risk"] = (proba * df.avg_monthly_bill.values * 12).round(0)
    s["priority_tier"] = "Not at risk"
    s["tier_rank"] = 4
    for rank, (name, lo, hi) in enumerate(TIERS, start=1):
        m = (s.churn_probability >= lo) & (s.churn_probability < hi)
        s.loc[m, ["priority_tier", "tier_rank"]] = [name, rank]
    s = s.sort_values(["tier_rank", "annual_revenue_at_risk"], ascending=[True, False])
    s["action"] = s.priority_tier.map(PLAYBOOK)
    return s.reset_index(drop=True)


def write_scores(scores, db, sql_dir):
    con = sqlite3.connect(db)
    scores.to_sql("churn_scores", con, index=False, if_exists="replace")
    with open(os.path.join(sql_dir, "02_tier_summary.sql"), encoding="utf-8") as f:
        con.executescript(f.read())
    summary = pd.read_sql("SELECT * FROM tier_summary", con)
    con.close()
    return summary


def capture_curve(y, proba):
    """Share of all churners caught when contacting the top x% of customers by score."""
    order = np.argsort(-proba)
    caught = np.cumsum(y[order]) / y.sum()
    pct = np.arange(1, len(y) + 1) / len(y)
    return pct, caught
