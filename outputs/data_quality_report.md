# Data quality report

| Check | Result |
|---|---|
| customer_id is unique | PASS |
| customers has no nulls | PASS |
| tenure_months within 1-72 | PASS |
| churn label is binary | PASS |
| usage has exactly 6 months per customer | PASS |
| usage (customer_id, month) is unique | PASS |
| usage and bills are non-negative | PASS |
| every ticket maps to a known customer | PASS |
| CSAT within 1-5 | PASS |
| customer_features has one row per customer | PASS |
| customer_features has no nulls | PASS |

Rows: customers 15,000 · usage_monthly 90,000 · support_tickets 15,864 · customer_features 15,000 (30 features)
