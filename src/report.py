"""
report.py
=========
Figures, Excel retention-ROI model, executive summary (Markdown) and a
two-page PDF brief.

Author: Ishaan Mankotia
"""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_pdf import PdfPages

from explain import label
from segmentation import PLAYBOOK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures")

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
MODEL_COLORS = {"Logistic Regression": "#94a3b8", "Random Forest": "#60a5fa", "Gradient Boosting": "#1d4ed8"}
TIER_COLORS = {"Tier 1 - Critical": "#b91c1c", "Tier 2 - High": "#ea580c",
               "Tier 3 - Watch": "#ca8a04", "Not at risk": "#9ca3af"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlecolor": INK})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figures
def fig_roc(rocs, results, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 5))
    auc = results.set_index("model").test_roc_auc
    for name, (fpr, tpr, _) in rocs.items():
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2.2 if name == "Gradient Boosting" else 1.6,
                label=f"{name} (AUC {auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRID)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves, hold-out set (n=3,000)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=GRID, lw=0.6)
    if own:
        _save(fig, "01_roc_curves.png")


def fig_model_compare(results):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    r = results.sort_values("cv_roc_auc_mean")
    y = np.arange(len(r))
    ax.barh(y - 0.18, r.cv_roc_auc_mean, 0.34, xerr=r.cv_roc_auc_std, color=[MODEL_COLORS[m] for m in r.model],
            label="5-fold CV (train)", capsize=3)
    ax.barh(y + 0.18, r.test_roc_auc, 0.34, color=[MODEL_COLORS[m] for m in r.model], alpha=0.45, label="Hold-out test")
    for i, (cv, te) in enumerate(zip(r.cv_roc_auc_mean, r.test_roc_auc)):
        ax.text(cv + 0.004, i - 0.18, f"{cv:.3f}", va="center", fontsize=9, color=INK)
        ax.text(te + 0.004, i + 0.18, f"{te:.3f}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(y, r.model)
    ax.set_xlim(0.8, 0.92)
    ax.set(xlabel="ROC-AUC", title="Model comparison")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(fig, "02_model_comparison.png")


def fig_shap_bar(drivers, ax=None, n=12):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4.8))
    d = drivers.head(n).iloc[::-1]
    colors = ["#1d4ed8" if i >= n - 5 else "#93c5fd" for i in range(n)]
    ax.barh(d.label, d.mean_abs_shap, color=colors)
    for i, v in enumerate(d.mean_abs_shap):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title="Churn drivers: top 5 highlighted")
    if own:
        _save(fig, "03_shap_importance.png")


def fig_shap_beeswarm(sv, Xt):
    Xl = Xt.rename(columns=label)
    shap.summary_plot(sv, Xl, max_display=12, show=False, plot_size=(8, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary: how each feature moves churn risk", fontweight="bold")
    _save(fig, "04_shap_beeswarm.png")


def fig_tiers(summary, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 3.6))
    s = summary.copy()
    x = np.arange(len(s))
    bars = ax.bar(x, s.customers, color=[TIER_COLORS[t] for t in s.priority_tier])
    for b, c, rr, ch in zip(bars, s.customers, s.revenue_at_risk_lakh, s.actual_churn_rate):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c:,}\n₹{rr:,.0f}L at risk\n{ch:.0%} churned",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [t.replace(" - ", "\n") for t in s.priority_tier])
    ax.set_ylim(0, s.customers.max() * 1.35)
    ax.set(ylabel="Customers", title="Priority tiers (out-of-fold scores, all 15,000 customers)")
    if own:
        _save(fig, "05_priority_tiers.png")


def fig_capture(pct, caught, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(pct * 100, caught * 100, color="#1d4ed8", lw=2, label="Model ranking")
    ax.plot([0, 100], [0, 100], ls="--", color=MUTED, label="Random contact")
    for p in (10, 20, 30):
        c = caught[int(len(caught) * p / 100) - 1] * 100
        ax.scatter([p], [c], color="#1d4ed8", zorder=3)
        ax.annotate(f"top {p}% -> {c:.0f}% of churners", (p, c), xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set(xlabel="% of base contacted (highest risk first)", ylabel="% of churners reached",
           title="Cumulative gains")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=GRID, lw=0.6)
    if own:
        _save(fig, "06_cumulative_gains.png")


def fig_driver_profiles(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    d = df.copy()
    d["usage_band"] = pd.cut(d.usage_drop_rate, [-1, 0, 0.15, 0.3, 1],
                             labels=["Growing", "0-15%\ndrop", "15-30%\ndrop", ">30%\ndrop"])
    d["calls_band"] = d.support_calls_recent.clip(upper=4).astype(str).replace("4", "4+")
    d["tenure_band"] = pd.cut(d.tenure_months, [0, 6, 12, 24, 48, 72],
                              labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    for ax, col, title in zip(axes, ["usage_band", "contract_type", "calls_band"],
                              ["Usage drop rate", "Contract type", "Support calls, last 3 months"]):
        r = d.groupby(col, observed=True).churned_next_month.mean()
        ax.bar(r.index.astype(str), r.values * 100, color="#1d4ed8")
        ax.axhline(d.churned_next_month.mean() * 100, ls="--", color=MUTED, lw=1)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Actual churn rate (%)" if ax is axes[0] else "")
    fig.suptitle("Observed churn rate by driver (dashed = base rate)", fontweight="bold", y=1.03)
    _save(fig, "07_churn_by_driver.png")


# ---------------------------------------------------------------- excel
def build_excel(results, drivers, summary, scores, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.table import Table, TableStyleInfo

    wb = Workbook()
    hdr = PatternFill("solid", fgColor="1D4ED8")
    inp = PatternFill("solid", fgColor="FEF3C7")
    bold = Font(bold=True)

    ws = wb.active
    ws.title = "Retention_ROI"
    ws["A1"] = "Retention campaign ROI: edit the yellow cells"
    ws["A1"].font = Font(bold=True, size=13)
    heads = ["Tier", "Customers", "Avg churn prob.", "Avg monthly bill (₹)", "Contact rate",
             "Save rate (of churners reached)", "Offer cost per contact (₹)", "Customers saved",
             "Annual revenue saved (₹)", "Campaign cost (₹)", "Net benefit (₹)", "ROI"]
    for j, h in enumerate(heads, 1):
        c = ws.cell(3, j, h)
        c.font, c.fill, c.alignment = Font(bold=True, color="FFFFFF"), hdr, Alignment(wrap_text=True)
    defaults = {"Tier 1 - Critical": (0.90, 0.35, 900), "Tier 2 - High": (0.75, 0.25, 400),
                "Tier 3 - Watch": (0.60, 0.15, 80)}
    tiers = summary[summary.priority_tier != "Not at risk"]
    for i, row in enumerate(tiers.itertuples(), start=4):
        avg_p = scores.loc[scores.priority_tier == row.priority_tier, "churn_probability"].mean()
        contact, save, cost = defaults[row.priority_tier]
        vals = [row.priority_tier, row.customers, round(avg_p, 3), row.avg_monthly_bill, contact, save, cost]
        for j, v in enumerate(vals, 1):
            ws.cell(i, j, v)
        for j in (5, 6, 7):
            ws.cell(i, j).fill = inp
        ws.cell(i, 8, f"=ROUND(B{i}*E{i}*C{i}*F{i},0)")
        ws.cell(i, 9, f"=H{i}*D{i}*12")
        ws.cell(i, 10, f"=B{i}*E{i}*G{i}")
        ws.cell(i, 11, f"=I{i}-J{i}")
        ws.cell(i, 12, f"=IF(J{i}=0,0,K{i}/J{i})")
    last = 3 + len(tiers)
    t = last + 1
    ws.cell(t, 1, "Total").font = bold
    for col in "BHIJK":
        ws[f"{col}{t}"] = f"=SUM({col}4:{col}{last})"
        ws[f"{col}{t}"].font = bold
    ws[f"L{t}"] = f"=IF(J{t}=0,0,K{t}/J{t})"
    for r in range(4, t + 1):
        for col, fmt in zip("CEFDGIJKL", ["0.0%", "0%", "0%", "#,##0", "#,##0", "#,##0", "#,##0", "#,##0", "0.0x"]):
            ws[f"{col}{r}"].number_format = fmt
    ws.cell(t + 2, 1, "Customers saved = customers x contact rate x avg churn probability x save rate. "
                      "Revenue saved counts 12 months of the average bill.").font = Font(italic=True, color="6B7280")
    for col, w in zip("ABCDEFGHIJKL", [20, 11, 12, 13, 10, 14, 14, 12, 16, 15, 15, 8]):
        ws.column_dimensions[col].width = w

    def sheet(name, frame):
        s = wb.create_sheet(name)
        s.append(list(frame.columns))
        for r in frame.itertuples(index=False):
            s.append(list(r))
        ref = f"A1:{chr(64 + frame.shape[1])}{len(frame) + 1}"
        tb = Table(displayName=name, ref=ref)
        tb.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        s.add_table(tb)
        for i, c in enumerate(frame.columns):
            s.column_dimensions[chr(65 + i)].width = max(12, min(60, len(str(c)) + 4))

    sheet("Tier_Summary", summary)
    sheet("Model_Comparison", results)
    sheet("Churn_Drivers", drivers[["label", "feature", "mean_abs_shap", "share_of_total", "direction"]])
    sheet("Tier1_Call_List", scores[scores.tier_rank == 1].drop(columns=["tier_rank", "action"]).head(1000))
    wb.save(path)


# ---------------------------------------------------------------- narrative
def executive_summary(k, results, drivers, summary, recs):
    best = results.set_index("model").loc[k["best_model"]]
    top5 = drivers.head(5)
    t = summary.set_index("priority_tier")
    lines = [
        "# Executive summary: subscriber churn and retention priorities",
        "",
        f"*Base: {k['customers']:,} subscribers, 6-month behaviour window, churn measured in the following month. "
        "Synthetic data, see README.*",
        "",
        "## Headline",
        f"- **{k['churn_rate']:.1%}** of subscribers churned in the outcome month, "
        f"₹{k['annual_revenue_churned_lakh']:,.0f} lakh of annualised revenue.",
        f"- A {k['best_model']} model predicts churn with **{best.test_roc_auc:.1%} ROC-AUC** on held-out customers "
        f"(5-fold CV {best.cv_roc_auc_mean:.1%}). Contacting the riskiest 20% of the base reaches "
        f"**{k['capture_top20']:.0%} of churners**, against 20% for random outreach.",
        f"- **{k['at_risk_customers']:,} customers** fall into three priority tiers, carrying "
        f"**₹{k['at_risk_revenue_lakh']:,.0f} lakh** of annual revenue at risk.",
        "",
        "## What drives churn (SHAP, ranked)",
    ]
    for i, r in enumerate(top5.itertuples(), 1):
        lines.append(f"{i}. **{r.label}**: {r.share_of_total:.0%} of total model attribution ({r.direction}).")
    lines += ["", recs["driver_note"], "", "## Priority tiers", "",
              "| Tier | Customers | Avg predicted risk | Actual churn | Revenue at risk (₹ lakh/yr) | Action |",
              "|---|---|---|---|---|---|"]
    for tier in ["Tier 1 - Critical", "Tier 2 - High", "Tier 3 - Watch"]:
        r = t.loc[tier]
        lines.append(f"| {tier} | {int(r.customers):,} | {r.avg_churn_probability:.0%} | {r.actual_churn_rate:.0%} | "
                     f"{r.revenue_at_risk_lakh:,.1f} | {PLAYBOOK[tier]} |")
    lines += ["", "## Recommendations", ""]
    lines += [f"{i}. {r}" for i, r in enumerate(recs["actions"], 1)]
    lines += ["", "## Expected impact (default assumptions, editable in the Excel model)",
              f"- Working all three tiers at the default contact and save rates retains about "
              f"**{k['roi_saved']:,} customers** and **₹{k['roi_revenue_lakh']:,.0f} lakh** of annual revenue for "
              f"₹{k['roi_cost_lakh']:,.1f} lakh of offer cost (**{k['roi_multiple']:.1f}x** net return on cost).",
              "", "## Caveats",
              "- Data is synthetic. The pipeline, SQL and model code run unchanged on a real warehouse extract.",
              "- SHAP shows what the model relies on, not causation. Validate the top offers with a holdout-controlled "
              "campaign before scaling.",
              "- Save rates and offer costs in the ROI model are planning assumptions, not measured results."]
    return "\n".join(lines) + "\n"


def roi_defaults(summary, scores):
    d = {"Tier 1 - Critical": (0.90, 0.35, 900), "Tier 2 - High": (0.75, 0.25, 400),
         "Tier 3 - Watch": (0.60, 0.15, 80)}
    saved = revenue = cost = 0
    for tier, (contact, save, c) in d.items():
        r = summary.set_index("priority_tier").loc[tier]
        p = scores.loc[scores.priority_tier == tier, "churn_probability"].mean()
        s = round(r.customers * contact * p * save)
        saved += s
        revenue += s * r.avg_monthly_bill * 12
        cost += r.customers * contact * c
    return {"roi_saved": int(saved), "roi_revenue_lakh": revenue / 1e5, "roi_cost_lakh": cost / 1e5,
            "roi_multiple": (revenue - cost) / cost}


# ---------------------------------------------------------------- pdf
def build_pdf(path, k, results, drivers, summary, rocs, pct, caught, recs):
    with PdfPages(path) as pdf:
        # page 1
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.955, "Predictive Churn Analysis & Retention Modelling", fontsize=17, weight="bold", color=INK)
        fig.text(0.06, 0.935, f"Telecom subscriber base · {k['customers']:,} customers · Ishaan Mankotia",
                 fontsize=9.5, color=MUTED)
        kpis = [(f"{k['churn_rate']:.1%}", "churn rate"),
                (f"{k['best_auc']:.3f}", f"ROC-AUC ({k['best_model']})"),
                (f"{k['capture_top20']:.0%}", "churners in top 20% scored"),
                (f"₹{k['at_risk_revenue_lakh']:,.0f}L", "annual revenue at risk")]
        for i, (v, l) in enumerate(kpis):
            x = 0.06 + i * 0.225
            fig.text(x, 0.885, v, fontsize=19, weight="bold", color="#1d4ed8")
            fig.text(x, 0.868, l, fontsize=8.5, color=MUTED)

        fig.text(0.06, 0.835, "Model comparison (hold-out n = 3,000)", fontsize=11, weight="bold", color=INK)
        tbl = results[["model", "cv_roc_auc_mean", "test_roc_auc", "test_pr_auc", "precision", "recall", "f1"]]
        ax = fig.add_axes([0.06, 0.755, 0.88, 0.07]); ax.axis("off")
        t = ax.table(cellText=[[r[0]] + [f"{v:.3f}" for v in r[1:]] for r in tbl.values],
                     colLabels=["Model", "CV AUC", "Test AUC", "PR-AUC", "Precision", "Recall", "F1"],
                     loc="center", cellLoc="center", colWidths=[0.22] + [0.13] * 6)
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)

        ax = fig.add_axes([0.08, 0.47, 0.38, 0.25]); fig_roc(rocs, results, ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax = fig.add_axes([0.58, 0.47, 0.37, 0.25]); fig_capture(pct, caught, ax); ax.legend(fontsize=7, frameon=False, loc="lower right")

        fig.text(0.06, 0.42, "Top 5 churn drivers (SHAP)", fontsize=11, weight="bold", color=INK)
        y = 0.398
        for i, r in enumerate(drivers.head(5).itertuples(), 1):
            fig.text(0.07, y, f"{i}. {r.label}: {r.share_of_total:.0%} of attribution, {r.direction}", fontsize=9, color=INK)
            y -= 0.019
        fig.text(0.06, y - 0.015, "Recommendations", fontsize=11, weight="bold", color=INK)
        y -= 0.037
        for i, r in enumerate(recs["actions"], 1):
            for j, line in enumerate(textwrap.wrap(f"{i}. {r}", 108)):
                fig.text(0.07 if j == 0 else 0.085, y, line, fontsize=8.6, color=INK)
                y -= 0.016
            y -= 0.004
        pdf.savefig(fig); plt.close(fig)

        # page 2
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.955, "Drivers and priority tiers", fontsize=15, weight="bold", color=INK)
        ax = fig.add_axes([0.40, 0.60, 0.53, 0.32]); fig_shap_bar(drivers, ax)
        ax = fig.add_axes([0.10, 0.28, 0.83, 0.24]); fig_tiers(summary, ax)
        fig.text(0.06, 0.215, "Tier playbook", fontsize=11, weight="bold", color=INK)
        y = 0.192
        for tier in ["Tier 1 - Critical", "Tier 2 - High", "Tier 3 - Watch"]:
            for j, line in enumerate(textwrap.wrap(f"{tier}: {PLAYBOOK[tier]}", 110)):
                fig.text(0.07, y, line, fontsize=8.6, color=TIER_COLORS[tier] if j == 0 else INK)
                y -= 0.016
            y -= 0.006
        fig.text(0.06, 0.04, "Synthetic data (seed 42). Methodology: github.com/Ishaan1000/predictive-churn-retention",
                 fontsize=7.5, color=MUTED)
        pdf.savefig(fig); plt.close(fig)
