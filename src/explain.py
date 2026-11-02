"""
explain.py
==========
SHAP explanations for the selected tree model.

  * global driver ranking  = mean |SHAP| per feature on the hold-out set
  * direction              = sign of the correlation between feature value and SHAP value
  * per-customer reasons   = top 3 positive SHAP contributions (used for tier playbooks)

Author: Ishaan Mankotia
"""

import numpy as np
import pandas as pd
import shap

LABELS = {
    "usage_drop_rate": "Usage drop rate (recent vs early 3 months)",
    "contract_type": "Contract type",
    "tenure_months": "Tenure (months)",
    "unresolved_tickets": "Unresolved support tickets",
    "support_calls_recent": "Support calls, last 3 months",
    "support_calls_6m": "Support calls, 6 months",
    "bill_change_pct": "Bill change (month 5-6 vs 1-4)",
    "payment_method": "Payment method",
    "months_usage_declined": "Months with falling usage",
    "last_month_vs_avg": "Last month usage vs 6-month average",
    "avg_dropped_call_pct": "Dropped-call rate",
    "paperless_billing": "Paperless billing",
    "lifetime_revenue": "Lifetime revenue",
    "has_device_emi": "Device EMI on account",
    "n_lines": "Number of lines",
    "billing_tickets": "Billing tickets",
    "network_tickets": "Network tickets",
    "avg_monthly_bill": "Average monthly bill",
    "late_payments": "Late payments",
    "avg_data_gb": "Average data use (GB/month)",
    "avg_voice_minutes": "Average voice minutes",
}


def label(f):
    return LABELS.get(f, f.replace("_", " ").capitalize())


def shap_values(pipe, X):
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older RF API: one array per class
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xt


def driver_ranking(sv, Xt):
    imp = np.abs(sv).mean(axis=0)
    direction = []
    for j, c in enumerate(Xt.columns):
        x = Xt[c].astype(float).values
        r = np.corrcoef(x, sv[:, j])[0, 1] if x.std() > 0 else 0.0
        direction.append("higher -> more churn" if r > 0 else "higher -> less churn")
    out = pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": imp, "direction": direction})
    out["share_of_total"] = out.mean_abs_shap / out.mean_abs_shap.sum()
    out["label"] = out.feature.map(label)
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True).round(4)


def category_effects(sv, Xt, pipe, feature):
    """Mean SHAP per category level, for categorical features (ordinal-encoded in trees)."""
    enc = pipe.named_steps["prep"].named_transformers_["cat"]
    cats = list(enc.feature_names_in_)
    levels = enc.categories_[cats.index(feature)]
    j = list(Xt.columns).index(feature)
    codes = Xt[feature].astype(int).values
    return pd.Series({levels[k]: sv[codes == k, j].mean() for k in range(len(levels))}).sort_values()


def top_reasons(sv, Xt, k=3):
    cols = np.array([label(c) for c in Xt.columns])
    idx = np.argsort(-sv, axis=1)[:, :k]
    return [" | ".join(cols[i]) for i in idx]
