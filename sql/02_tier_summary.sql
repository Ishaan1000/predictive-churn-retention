-- =====================================================================
-- 02_tier_summary.sql
-- Business view of the scored base: one row per priority tier.
-- Needs the churn_scores table written by src/segmentation.py.
-- =====================================================================

DROP VIEW IF EXISTS tier_summary;

CREATE VIEW tier_summary AS
SELECT
    s.priority_tier,
    COUNT(*)                                          AS customers,
    ROUND(AVG(s.churn_probability), 3)                AS avg_churn_probability,
    ROUND(AVG(f.churned_next_month), 3)               AS actual_churn_rate,
    ROUND(AVG(f.avg_monthly_bill), 0)                 AS avg_monthly_bill,
    ROUND(SUM(s.annual_revenue_at_risk) / 1e5, 1)     AS revenue_at_risk_lakh,
    ROUND(AVG(f.tenure_months), 1)                    AS avg_tenure_months,
    ROUND(AVG(f.usage_drop_rate), 3)                  AS avg_usage_drop_rate,
    ROUND(AVG(f.support_calls_recent), 2)             AS avg_recent_support_calls,
    ROUND(AVG(CASE WHEN f.contract_type = 'Month-to-month' THEN 1.0 ELSE 0 END), 3)
                                                      AS share_month_to_month
FROM churn_scores s
JOIN customer_features f USING (customer_id)
GROUP BY s.priority_tier
ORDER BY MIN(s.tier_rank);
