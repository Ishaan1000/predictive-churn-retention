# Methodology: churn prediction and retention prioritisation

## Pipeline

```
customers (15k)   usage_monthly (90k)   support_tickets (~16k)
        └──────────────┬──────────────────────┘
                       ▼
        11 data-quality checks -> SQLite (outputs/churn.db)
                       ▼
     sql/01_feature_engineering.sql -> customer_features (30 features)
                       ▼
   LR / RF / GB  -- 5-fold CV on 80% train, final metrics on 20% hold-out
                       ▼
    SHAP TreeExplainer (best tree model) -> driver ranking, per-customer reasons
                       ▼
   5-fold out-of-fold probabilities for all customers -> 3 priority tiers
                       ▼
        tier_summary SQL view -> figures, Excel ROI model, PDF, summary
```

## Observation design
- **Window:** months 1-6 of behaviour. **Outcome:** churn in month 7 (`churned_next_month`).
- "Early" = months 1-3, "recent" = months 4-6. Every feature uses only window data, so nothing leaks from the outcome month.

## Features (30)
| Group | Features |
|---|---|
| Profile (10) | age, region, plan, contract_type, tenure_months, payment_method, paperless_billing, n_lines, has_streaming_addon, has_device_emi |
| Usage level and trend (7) | avg_data_gb, avg_voice_minutes, avg_sms, **usage_drop_rate**, recent_vs_last_month, last_month_vs_avg, months_usage_declined (via `LAG`) |
| Billing (4) | avg_monthly_bill, bill_change_pct, late_payments, max_days_late |
| Network (1) | avg_dropped_call_pct |
| Support friction (7) | support_calls_6m, support_calls_recent, unresolved_tickets, billing_tickets, network_tickets, avg_resolution_hours, avg_csat |
| Value (1) | lifetime_revenue |

`usage_drop_rate = 0.5 × (data_early − data_recent) / data_early + 0.5 × (mins_early − mins_recent) / mins_early`. Data and voice move together, so one blended rate avoids splitting SHAP credit between two near-duplicate features.

## Models
| Model | Preprocessing | Settings |
|---|---|---|
| Logistic Regression | StandardScaler on numerics, one-hot (drop first) on 4 categoricals | L2, C = 1 |
| Random Forest | ordinal-encoded categoricals | 400 trees, min_samples_leaf 5, max_features sqrt |
| Gradient Boosting | ordinal-encoded categoricals | 300 trees, learning rate 0.05, depth 3, subsample 0.8 |

- The model is selected on mean 5-fold CV ROC-AUC on the training set. The hold-out set is used once.
- The threshold for precision, recall and F1 is chosen to maximise F1 on CV predictions from the training set.
- Tree models use ordinal encoding so that SHAP returns one value per original categorical feature.

## Explainability
- Global importance = mean |SHAP| over the 3,000 hold-out customers, reported as a share of the total.
- Direction: for numeric features, the sign of corr(feature value, SHAP value). For categoricals, the levels with the highest and lowest mean SHAP.
- Per-customer reasons are the three largest positive SHAP contributions, written to `customer_risk_scores.csv`.
- SHAP describes the model, not causation. Retention offers should be tested against a randomised control group.

## Tiers
| Tier | Rule | Playbook |
|---|---|---|
| 1: Critical | p ≥ 0.60 | Retention-desk call within 7 days; resolve open tickets; 12-month contract offer |
| 2: High | 0.40-0.60 | Channel-targeted contract migration or bill-shock offer |
| 3: Watch | 0.25-0.40 | Automated nurture, monthly re-score, escalate on a new unresolved ticket |

Probabilities are out-of-fold (5 folds over the whole base), so each customer is scored by a model trained without them. Within a tier, customers are sorted by `annual_revenue_at_risk = p × avg_monthly_bill × 12`.

## ROI model (Excel)
`saved = customers × contact rate × avg churn probability × save rate`. Revenue retained = saved × average bill × 12. Cost = customers contacted × offer cost per contact. The defaults (contact 90/75/60%, save 35/25/15%, cost ₹900/400/80) are planning assumptions to replace with measured campaign results.

## Limitations
- The data is synthetic, so the absolute numbers show the method and are not market estimates.
- There is a single snapshot and no time-based back-test. In production, train on earlier cohorts and validate on a later month.
- Tier cut-offs are fixed probability bands. With a known retention-team capacity, cut by capacity instead (e.g. the top N by revenue at risk).
