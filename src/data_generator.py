"""
data_generator.py
=================
Builds a synthetic telecom dataset (15,000 subscribers) as three raw tables,
shaped like a typical operator's warehouse extract:

  customers        one row per subscriber (plan, contract, tenure, billing)
  usage_monthly    6 months of usage per subscriber (90,000 rows)
  support_tickets  one row per support contact (last 6 months)

Churn (did the subscriber leave in the month after the observation window?)
is drawn from a latent logistic model of *behaviour*: falling usage, support
friction, contract type, tenure, bill shock, payment method. The model never
sees that formula. It only sees the raw tables, and has to recover the
signal through SQL feature engineering.

Seeded (42) so every run is identical.

Author: Ishaan Mankotia
"""

import numpy as np
import pandas as pd

N_CUSTOMERS = 15_000
N_MONTHS = 6
SEED = 42

PLANS = {            # plan: (base monthly charge INR, data allowance GB)
    "Basic": (299, 2),
    "Standard": (499, 6),
    "Premium": (799, 15),
    "Family": (1199, 30),
}
REGIONS = ["North", "South", "East", "West", "Central"]


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    n = N_CUSTOMERS
    ids = np.array([f"C{i:05d}" for i in range(1, n + 1)])

    # ---------------- customers ----------------
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.52, 0.27, 0.21])
    tenure = np.where(contract == "Month-to-month",
                      rng.gamma(1.6, 9, n), rng.gamma(3.2, 10, n)).clip(1, 72).round().astype(int)
    plan = rng.choice(list(PLANS), n, p=[0.30, 0.36, 0.22, 0.12])
    base_charge = np.array([PLANS[p][0] for p in plan])
    allowance = np.array([PLANS[p][1] for p in plan])
    payment = rng.choice(["Auto-debit", "Credit card", "UPI", "Cash / voucher"], n, p=[0.34, 0.22, 0.28, 0.16])
    paperless = rng.random(n) < np.where(payment == "Cash / voucher", 0.25, 0.70)
    age = rng.normal(38, 12, n).clip(18, 80).round().astype(int)
    region = rng.choice(REGIONS, n)
    n_lines = np.where(plan == "Family", rng.integers(2, 6, n), rng.choice([1, 1, 1, 2], n))
    has_streaming = rng.random(n) < np.where(np.isin(plan, ["Premium", "Family"]), 0.6, 0.2)
    has_device_emi = rng.random(n) < 0.18
    network_quality = rng.beta(6, 2, n)  # 0-1, share of sessions without drops in home cell

    # latent engagement trend: negative = disengaging subscriber
    disengage = rng.random(n) < 0.24
    trend = np.where(disengage, rng.uniform(-0.18, -0.04, n), rng.normal(0.005, 0.02, n))

    # ---------------- usage_monthly ----------------
    base_data = allowance * rng.uniform(0.4, 1.1, n)
    base_min = rng.gamma(4, 80, n)
    rows = []
    for m in range(1, N_MONTHS + 1):
        mult = np.clip(1 + trend * (m - 1) + rng.normal(0, 0.07, n), 0.05, None)
        data_gb = (base_data * mult).round(2)
        minutes = (base_min * mult).round()
        sms = rng.poisson(15 * mult)
        overage = np.clip(data_gb - allowance, 0, None) * 45
        bill = base_charge + (n_lines - 1) * 149 + has_streaming * 99 + has_device_emi * 650 + overage.round()
        late = rng.random(n) < np.where(payment == "Cash / voucher", 0.14, 0.04)
        rows.append(pd.DataFrame({
            "customer_id": ids, "month": m, "data_gb": data_gb, "voice_minutes": minutes,
            "sms_count": sms, "bill_amount": bill.round(2), "days_late": np.where(late, rng.integers(3, 30, n), 0),
            "dropped_call_pct": ((1 - network_quality) * 12 + rng.normal(0, 1.2, n)).clip(0).round(2),
        }))
    usage = pd.concat(rows, ignore_index=True)

    # bill shock: some customers get a price increase in month 5
    shock = rng.random(n) < 0.12
    usage.loc[usage["month"] >= 5, "bill_amount"] += np.tile(np.where(shock, base_charge * 0.18, 0), 2).round(2)

    # ---------------- support_tickets ----------------
    friction = 0.35 + 1.4 * (1 - network_quality) + 0.6 * disengage + 0.3 * shock
    n_tickets = rng.poisson(friction * 1.2)
    t_cust = np.repeat(ids, n_tickets)
    t_n = len(t_cust)
    cats = rng.choice(["Network", "Billing", "Plan change", "Device", "Other"], t_n, p=[0.34, 0.28, 0.14, 0.12, 0.12])
    unresolved_p = np.repeat(0.15 + 0.35 * disengage, n_tickets)
    tickets = pd.DataFrame({
        "ticket_id": [f"T{i:06d}" for i in range(1, t_n + 1)],
        "customer_id": t_cust,
        "month": rng.integers(1, N_MONTHS + 1, t_n),
        "category": cats,
        "resolved": ~(rng.random(t_n) < unresolved_p),
        "resolution_hours": rng.gamma(2, 14, t_n).round(1),
        "csat": rng.integers(1, 6, t_n),
    })

    # ---------------- churn label (hidden data-generating process) ----------------
    early = usage[usage.month <= 3].groupby("customer_id")["data_gb"].mean().reindex(ids).values
    late_u = usage[usage.month >= 4].groupby("customer_id")["data_gb"].mean().reindex(ids).values
    drop = np.clip((early - late_u) / np.maximum(early, 0.1), -1, 1)
    tickets_recent = tickets[tickets.month >= 4].groupby("customer_id").size().reindex(ids, fill_value=0).values
    unresolved = tickets[~tickets.resolved].groupby("customer_id").size().reindex(ids, fill_value=0).values

    m2m = contract == "Month-to-month"
    logit = (-2.9
             + 2.6 * drop
             + 1.6 * (drop > 0.25) * (1 + 0.8 * m2m)          # sharp disengagement, worse off-contract
             + 0.45 * tickets_recent
             + 0.6 * unresolved + 0.9 * (unresolved >= 2)
             + np.select([m2m, contract == "One year"], [1.3, 0.0], -1.4)
             - 0.6 * np.log1p(tenure) + 1.6
             + 1.0 * (tenure <= 6) * m2m                        # new, uncommitted subscribers
             + 1.1 * shock * (1 + 0.6 * np.isin(plan, ["Basic", "Standard"]))
             + 0.55 * (payment == "Cash / voucher")
             + 0.25 * (~paperless)
             - 1.1 * (network_quality - 0.75)
             - 0.45 * has_device_emi
             - 0.12 * (n_lines - 1)
             + rng.normal(0, 0.35, n))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    customers = pd.DataFrame({
        "customer_id": ids, "age": age, "region": region, "plan": plan, "contract_type": contract,
        "tenure_months": tenure, "payment_method": payment, "paperless_billing": paperless,
        "n_lines": n_lines, "has_streaming_addon": has_streaming, "has_device_emi": has_device_emi,
        "churned_next_month": churn.astype(int),
    })
    return customers, usage, tickets


if __name__ == "__main__":
    c, u, t = generate()
    print(c.shape, u.shape, t.shape, "churn rate:", round(c.churned_next_month.mean(), 3))
