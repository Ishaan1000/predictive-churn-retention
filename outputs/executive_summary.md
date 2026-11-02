# Executive summary: subscriber churn and retention priorities

*Base: 15,000 subscribers, 6-month behaviour window, churn measured in the following month. Synthetic data, see README.*

## Headline
- **31.9%** of subscribers churned in the outcome month, ₹462 lakh of annualised revenue.
- A Gradient Boosting model predicts churn with **88.7% ROC-AUC** on held-out customers (5-fold CV 89.4%). Contacting the riskiest 20% of the base reaches **53% of churners**, against 20% for random outreach.
- **6,353 customers** fall into three priority tiers, carrying **₹386 lakh** of annual revenue at risk.

## What drives churn (SHAP, ranked)
1. **Contract type**: 25% of total model attribution (highest risk: Month-to-month, lowest: Two year).
2. **Usage drop rate (recent vs early 3 months)**: 21% of total model attribution (higher -> more churn).
3. **Tenure (months)**: 10% of total model attribution (higher -> less churn).
4. **Unresolved support tickets**: 8% of total model attribution (higher -> more churn).
5. **Bill change (month 5-6 vs 1-4)**: 7% of total model attribution (higher -> more churn).

Month-to-month subscribers churn at 49% against 8% on two-year contracts. Customers whose usage fell more than 30% churn at 74%, those with 2+ unresolved tickets at 81%, and those hit by a bill increase of more than 5% at 45% (base rate 32%).

## Priority tiers

| Tier | Customers | Avg predicted risk | Actual churn | Revenue at risk (₹ lakh/yr) | Action |
|---|---|---|---|---|---|
| Tier 1 - Critical | 3,305 | 83% | 83% | 264.9 | Retention-desk call within 7 days; fix any open ticket first, then offer a 12-month contract with a loyalty discount (up to 15%) or a free plan upgrade. |
| Tier 2 - High | 1,604 | 50% | 50% | 75.9 | Targeted offer through the customer's own channel (app/SMS): contract migration incentive for month-to-month users, bill-shock reversal for customers whose bill rose. |
| Tier 3 - Watch | 1,444 | 32% | 34% | 45.1 | Automated nurture: usage tips, data top-up bonus, CSAT survey after every ticket; re-score monthly and escalate on any new unresolved ticket. |

## Recommendations

1. Work Tier 1 first (3,305 customers, 83% actual churn): retention-desk call within 7 days, ordered by revenue at risk (call list in the Excel model).
2. Treat a falling usage trend as the earliest warning: trigger an automated check-in when usage drops 15%+ month on month, before the customer reaches the retention desk.
3. Close the support loop: any customer with a second unresolved ticket goes to a senior agent and gets a callback after resolution. Unresolved tickets are a top-5 driver and are fully within our control.
4. Migrate month-to-month customers to 12-month contracts with a loyalty discount or device bundle; contract type is the largest structural driver.
5. Review the month-5 price rise: 2,068 affected customers churn at 45%. Offer a grandfathered rate to Tier 1-2 customers on Basic and Standard plans.
6. Run every offer with a 10% randomised control group so the save rate is measured, not assumed, and re-score the base monthly.

## Expected impact (default assumptions, editable in the Excel model)
- Working all three tiers at the default contact and save rates retains about **1,060 customers** and **₹102 lakh** of annual revenue for ₹32.3 lakh of offer cost (**2.2x** net return on cost).

## Caveats
- Data is synthetic. The pipeline, SQL and model code run unchanged on a real warehouse extract.
- SHAP shows what the model relies on, not causation. Validate the top offers with a holdout-controlled campaign before scaling.
- Save rates and offer costs in the ROI model are planning assumptions, not measured results.
