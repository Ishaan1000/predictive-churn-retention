"""
pipeline.py
===========
Raw tables -> quality checks -> SQLite -> SQL feature engineering.

    generate()  ->  outputs/churn.db  [customers, usage_monthly, support_tickets]
                ->  sql/01_feature_engineering.sql  ->  customer_features

Author: Ishaan Mankotia
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

from data_generator import generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SQL = os.path.join(ROOT, "sql")
DB = os.path.join(OUT, "churn.db")


def check(cond, msg, log):
    log.append(("PASS" if cond else "FAIL", msg))
    if not cond:
        raise ValueError(f"Data quality check failed: {msg}")


def validate(customers, usage, tickets):
    log = []
    check(customers.customer_id.is_unique, "customer_id is unique", log)
    check(customers.notna().all().all(), "customers has no nulls", log)
    check(customers.tenure_months.between(1, 72).all(), "tenure_months within 1-72", log)
    check(set(customers.churned_next_month.unique()) <= {0, 1}, "churn label is binary", log)
    check(len(usage) == len(customers) * 6, "usage has exactly 6 months per customer", log)
    check(not usage.duplicated(["customer_id", "month"]).any(), "usage (customer_id, month) is unique", log)
    check((usage[["data_gb", "voice_minutes", "bill_amount"]] >= 0).all().all(), "usage and bills are non-negative", log)
    check(tickets.customer_id.isin(customers.customer_id).all(), "every ticket maps to a known customer", log)
    check(tickets.csat.between(1, 5).all(), "CSAT within 1-5", log)
    return log


def run_sql_file(con, name):
    with open(os.path.join(SQL, name), encoding="utf-8") as f:
        con.executescript(f.read())


def run_pipeline():
    os.makedirs(OUT, exist_ok=True)
    customers, usage, tickets = generate()
    log = validate(customers, usage, tickets)

    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    customers.to_sql("customers", con, index=False)
    usage.to_sql("usage_monthly", con, index=False)
    tickets.to_sql("support_tickets", con, index=False)
    con.execute("CREATE INDEX ix_usage ON usage_monthly(customer_id, month)")
    con.execute("CREATE INDEX ix_tickets ON support_tickets(customer_id)")
    run_sql_file(con, "01_feature_engineering.sql")

    feats = pd.read_sql("SELECT * FROM customer_features", con)
    check(len(feats) == len(customers), "customer_features has one row per customer", log)
    check(feats.drop(columns="customer_id").notna().all().all(), "customer_features has no nulls", log)

    con.execute("CREATE TABLE pipeline_runs (run_at TEXT, customers INT, usage_rows INT, tickets INT, churn_rate REAL)")
    con.execute("INSERT INTO pipeline_runs VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), len(customers), len(usage), len(tickets),
                 float(customers.churned_next_month.mean())))
    con.commit()
    con.close()

    with open(os.path.join(OUT, "data_quality_report.md"), "w", encoding="utf-8") as f:
        f.write("# Data quality report\n\n| Check | Result |\n|---|---|\n")
        f.writelines(f"| {m} | {s} |\n" for s, m in log)
        f.write(f"\nRows: customers {len(customers):,} · usage_monthly {len(usage):,} · "
                f"support_tickets {len(tickets):,} · customer_features {len(feats):,} "
                f"({feats.shape[1] - 2} features)\n")
    print(f"  pipeline: {len(customers):,} customers, {feats.shape[1] - 2} features, "
          f"churn rate {customers.churned_next_month.mean():.1%}, {len(log)} checks passed")
    return feats


def load_features():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM customer_features", con)
    con.close()
    return df


if __name__ == "__main__":
    run_pipeline()
