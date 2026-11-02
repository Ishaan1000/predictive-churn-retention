-- =====================================================================
-- 01_feature_engineering.sql
-- Builds one model-ready row per subscriber from the three raw tables.
-- Observation window = months 1-6; "early" = months 1-3, "recent" = 4-6.
-- Run by src/pipeline.py against outputs/churn.db (SQLite).
-- =====================================================================

DROP TABLE IF EXISTS customer_features;

CREATE TABLE customer_features AS
WITH usage_lag AS (
    SELECT u.*,
           LAG(data_gb) OVER (PARTITION BY customer_id ORDER BY month) AS prev_data_gb
    FROM usage_monthly u
),
usage_agg AS (
    SELECT
        customer_id,
        AVG(CASE WHEN month <= 3 THEN data_gb END)        AS data_early,
        AVG(CASE WHEN month >= 4 THEN data_gb END)        AS data_recent,
        AVG(CASE WHEN month <= 3 THEN voice_minutes END)  AS mins_early,
        AVG(CASE WHEN month >= 4 THEN voice_minutes END)  AS mins_recent,
        AVG(data_gb)                                      AS avg_data_gb,
        AVG(voice_minutes)                                AS avg_voice_minutes,
        AVG(sms_count)                                    AS avg_sms,
        MAX(CASE WHEN month = 6 THEN data_gb END)         AS data_last_month,
        AVG(bill_amount)                                  AS avg_bill,
        AVG(CASE WHEN month <= 4 THEN bill_amount END)    AS bill_before,
        AVG(CASE WHEN month >= 5 THEN bill_amount END)    AS bill_after,
        SUM(CASE WHEN days_late > 0 THEN 1 ELSE 0 END)    AS late_payments,
        MAX(days_late)                                    AS max_days_late,
        AVG(dropped_call_pct)                             AS avg_dropped_call_pct,
        -- months in which usage fell vs the previous month
        SUM(CASE WHEN data_gb < prev_data_gb THEN 1 ELSE 0 END) AS months_usage_declined
    FROM usage_lag
    GROUP BY customer_id
),
ticket_agg AS (
    SELECT
        customer_id,
        COUNT(*)                                                  AS support_calls_6m,
        SUM(CASE WHEN month >= 4 THEN 1 ELSE 0 END)               AS support_calls_recent,
        SUM(CASE WHEN NOT resolved THEN 1 ELSE 0 END)             AS unresolved_tickets,
        SUM(CASE WHEN category = 'Billing' THEN 1 ELSE 0 END)     AS billing_tickets,
        SUM(CASE WHEN category = 'Network' THEN 1 ELSE 0 END)     AS network_tickets,
        AVG(resolution_hours)                                     AS avg_resolution_hours,
        AVG(csat)                                                 AS avg_csat
    FROM support_tickets
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    -- profile
    c.age, c.region, c.plan, c.contract_type, c.tenure_months, c.payment_method,
    c.paperless_billing, c.n_lines, c.has_streaming_addon, c.has_device_emi,
    -- usage level and trend
    ROUND(u.avg_data_gb, 3)                                              AS avg_data_gb,
    ROUND(u.avg_voice_minutes, 1)                                        AS avg_voice_minutes,
    ROUND(u.avg_sms, 2)                                                  AS avg_sms,
    -- blended data + voice decline, recent 3 months vs early 3 months (+ = falling)
    ROUND(0.5 * (u.data_early - u.data_recent) / MAX(u.data_early, 0.1)
        + 0.5 * (u.mins_early - u.mins_recent) / MAX(u.mins_early, 1), 4) AS usage_drop_rate,
    ROUND(u.data_recent / MAX(u.data_last_month, 0.1), 4)                AS recent_vs_last_month,
    ROUND(u.data_last_month / MAX(u.avg_data_gb, 0.1), 4)                AS last_month_vs_avg,
    u.months_usage_declined,
    -- billing
    ROUND(u.avg_bill, 2)                                                 AS avg_monthly_bill,
    ROUND((u.bill_after - u.bill_before) / u.bill_before, 4)             AS bill_change_pct,
    u.late_payments, u.max_days_late,
    -- network experience
    ROUND(u.avg_dropped_call_pct, 3)                                     AS avg_dropped_call_pct,
    -- support friction
    COALESCE(t.support_calls_6m, 0)                                      AS support_calls_6m,
    COALESCE(t.support_calls_recent, 0)                                  AS support_calls_recent,
    COALESCE(t.unresolved_tickets, 0)                                    AS unresolved_tickets,
    COALESCE(t.billing_tickets, 0)                                       AS billing_tickets,
    COALESCE(t.network_tickets, 0)                                       AS network_tickets,
    ROUND(COALESCE(t.avg_resolution_hours, 0), 1)                        AS avg_resolution_hours,
    ROUND(COALESCE(t.avg_csat, 5), 2)                                    AS avg_csat,
    -- value
    ROUND(u.avg_bill * c.tenure_months, 0)                               AS lifetime_revenue,
    -- target
    c.churned_next_month
FROM customers c
JOIN usage_agg u        ON u.customer_id = c.customer_id
LEFT JOIN ticket_agg t  ON t.customer_id = c.customer_id;

CREATE INDEX IF NOT EXISTS ix_features_customer ON customer_features(customer_id);
