"""
run_all.py
==========
One command, every output:

    python src/run_all.py

  1. Generate raw tables, validate, load SQLite, SQL feature engineering   (pipeline.py)
  2. Train + compare Logistic Regression / Random Forest / Gradient Boosting (models.py)
  3. SHAP driver ranking on the best model                                  (explain.py)
  4. Out-of-fold scoring of every customer -> 3 priority tiers              (segmentation.py)
  5. Figures, Excel ROI model, executive summary, PDF brief                 (report.py)

Author: Ishaan Mankotia
"""

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import report as R
from explain import category_effects, driver_ranking, shap_values, top_reasons
from models import out_of_fold_scores, train_and_compare
from pipeline import DB, SQL, run_pipeline
from segmentation import assign_tiers, capture_curve, write_scores

OUT = R.OUT


def recommendations(drivers, contract_fx, summary, df):
    t = summary.set_index("priority_tier")
    m2m = df[df.contract_type == "Month-to-month"].churned_next_month.mean()
    two = df[df.contract_type == "Two year"].churned_next_month.mean()
    shock = df[df.bill_change_pct > 0.05]
    shock_rate, base = shock.churned_next_month.mean(), df.churned_next_month.mean()
    unres = df[df.unresolved_tickets >= 2].churned_next_month.mean()
    drop = df[df.usage_drop_rate > 0.3].churned_next_month.mean()
    note = (f"Month-to-month subscribers churn at {m2m:.0%} against {two:.0%} on two-year contracts. "
            f"Customers whose usage fell more than 30% churn at {drop:.0%}, those with 2+ unresolved "
            f"tickets at {unres:.0%}, and those hit by a bill increase of more than 5% at {shock_rate:.0%} "
            f"(base rate {base:.0%}).")
    actions = [
        f"Work Tier 1 first ({int(t.loc['Tier 1 - Critical'].customers):,} customers, "
        f"{t.loc['Tier 1 - Critical'].actual_churn_rate:.0%} actual churn): retention-desk call within 7 days, "
        "ordered by revenue at risk (call list in the Excel model).",
        "Treat a falling usage trend as the earliest warning: trigger an automated check-in when usage drops "
        "15%+ month on month, before the customer reaches the retention desk.",
        "Close the support loop: any customer with a second unresolved ticket goes to a senior agent and gets a "
        "callback after resolution. Unresolved tickets are a top-5 driver and are fully within our control.",
        "Migrate month-to-month customers to 12-month contracts with a loyalty discount or device bundle; "
        "contract type is the largest structural driver.",
        f"Review the month-5 price rise: {len(shock):,} affected customers churn at {shock_rate:.0%}. Offer a "
        "grandfathered rate to Tier 1-2 customers on Basic and Standard plans.",
        "Run every offer with a 10% randomised control group so the save rate is measured, not assumed, "
        "and re-score the base monthly.",
    ]
    return {"driver_note": note, "actions": actions}


def main():
    print("=" * 64 + "\n  Predictive Churn Analysis & Retention Modelling: full run\n" + "=" * 64)
    df = run_pipeline()

    print("\n[2] Model comparison")
    m = train_and_compare(df)
    results, best = m["results"], m["best"]
    results.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    print(f"  best model (by CV AUC): {best}")

    print("\n[3] SHAP explanations")
    tree_name = best if best != "Logistic Regression" else "Gradient Boosting"
    pipe = m["models"][tree_name]
    sv, Xt = shap_values(pipe, m["X_test"])
    drivers = driver_ranking(sv, Xt)
    drivers.to_csv(os.path.join(OUT, "shap_driver_ranking.csv"), index=False)
    contract_fx = category_effects(sv, Xt, pipe, "contract_type")
    for cat in ["contract_type", "payment_method", "plan", "region"]:
        fx = category_effects(sv, Xt, pipe, cat)
        drivers.loc[drivers.feature == cat, "direction"] = f"highest risk: {fx.idxmax()}, lowest: {fx.idxmin()}"
    drivers.to_csv(os.path.join(OUT, "shap_driver_ranking.csv"), index=False)
    for r in drivers.head(5).itertuples():
        print(f"  {r.label:<52} {r.share_of_total:.1%}")

    print("\n[4] Scoring and tiers")
    proba = out_of_fold_scores(df, best)
    scores = assign_tiers(df, proba)
    # top-3 SHAP reasons per customer (from the hold-out-fitted model, for the call list)
    sv_all, Xt_all = shap_values(pipe, m["X"])
    reasons = dict(zip(df.customer_id, top_reasons(sv_all, Xt_all)))
    scores["top_reasons"] = scores.customer_id.map(reasons)
    summary = write_scores(scores, DB, SQL)
    scores.to_csv(os.path.join(OUT, "customer_risk_scores.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "tier_summary.csv"), index=False)
    print(summary[["priority_tier", "customers", "avg_churn_probability", "actual_churn_rate",
                   "revenue_at_risk_lakh"]].to_string(index=False))

    print("\n[5] Reports")
    y = df.churned_next_month.values
    pct, caught = capture_curve(y, proba)
    at_risk = summary[summary.priority_tier != "Not at risk"]
    best_row = results.set_index("model").loc[best]
    k = {
        "customers": len(df), "churn_rate": float(y.mean()), "best_model": best,
        "best_auc": float(best_row.test_roc_auc), "best_cv_auc": float(best_row.cv_roc_auc_mean),
        "capture_top10": float(caught[int(len(y) * 0.1) - 1]), "capture_top20": float(caught[int(len(y) * 0.2) - 1]),
        "at_risk_customers": int(at_risk.customers.sum()),
        "at_risk_revenue_lakh": float(at_risk.revenue_at_risk_lakh.sum()),
        "annual_revenue_churned_lakh": float((df.avg_monthly_bill * 12 * y).sum() / 1e5),
        "n_features": int(df.shape[1] - 2),
        "top5_drivers": drivers.head(5).label.tolist(),
        "contract_shap": contract_fx.round(3).to_dict(),
    }
    k.update(R.roi_defaults(summary, scores))
    recs = recommendations(drivers, contract_fx, summary, df)

    R.fig_roc(m["rocs"], results)
    R.fig_model_compare(results)
    R.fig_shap_bar(drivers)
    R.fig_shap_beeswarm(sv, Xt)
    R.fig_tiers(summary)
    R.fig_capture(pct, caught)
    R.fig_driver_profiles(df)
    os.makedirs(os.path.join(OUT, "excel"), exist_ok=True)
    R.build_excel(results, drivers, summary, scores, os.path.join(OUT, "excel", "Churn_Retention_Model.xlsx"))
    with open(os.path.join(OUT, "executive_summary.md"), "w", encoding="utf-8") as f:
        f.write(R.executive_summary(k, results, drivers, summary, recs))
    R.build_pdf(os.path.join(OUT, "Churn_Executive_Brief.pdf"), k, results, drivers, summary,
                m["rocs"], pct, caught, recs)
    with open(os.path.join(OUT, "kpis.json"), "w", encoding="utf-8") as f:
        json.dump(k, f, indent=2, default=float)
    print("  wrote figures, Excel model, executive summary, PDF brief, kpis.json")
    print(json.dumps({x: k[x] for x in ["best_auc", "best_cv_auc", "capture_top20", "at_risk_customers",
                                         "at_risk_revenue_lakh", "roi_multiple"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
