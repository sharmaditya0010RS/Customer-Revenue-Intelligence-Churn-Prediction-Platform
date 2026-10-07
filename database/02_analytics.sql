/* ============================================================
   CUSTOMER REVENUE INTELLIGENCE
   Advanced Analytical Queries

   Database : customer_intelligence
   Schema   : analytics
   ============================================================ */


/* ============================================================
   1. EXECUTIVE CUSTOMER KPIs
   ============================================================ */

SELECT
    COUNT(*) AS total_customers,

    COUNT(*) FILTER (
        WHERE churn_flag = 0
    ) AS active_customers,

    SUM(churn_flag) AS churned_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS total_monthly_revenue,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        AVG(total_charges),
        2
    ) AS avg_customer_lifetime_revenue,

    ROUND(
        AVG(tenure),
        2
    ) AS avg_tenure_months,

    ROUND(
        SUM(monthly_revenue_at_risk),
        2
    ) AS current_monthly_revenue_at_risk

FROM analytics.customers;



/* ============================================================
   2. CHURN BY CONTRACT
   ============================================================ */

SELECT
    contract,

    COUNT(*) AS total_customers,

    SUM(churn_flag) AS churned_customers,

    COUNT(*) - SUM(churn_flag)
        AS retained_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS monthly_revenue

FROM analytics.customers

GROUP BY contract

ORDER BY churn_rate_pct DESC;



/* ============================================================
   3. CHURN BY TENURE GROUP
   ============================================================ */

SELECT
    tenure_group,

    COUNT(*) AS total_customers,

    SUM(churn_flag)
        AS churned_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge

FROM analytics.customers

GROUP BY tenure_group

ORDER BY
    MIN(tenure);



/* ============================================================
   4. INTERNET SERVICE PERFORMANCE
   ============================================================ */

SELECT
    internet_service,

    COUNT(*) AS customers,

    SUM(churn_flag)
        AS churned_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS monthly_revenue

FROM analytics.customers

GROUP BY internet_service

ORDER BY churn_rate_pct DESC;



/* ============================================================
   5. PAYMENT METHOD ANALYSIS
   ============================================================ */

SELECT
    payment_method,

    COUNT(*) AS customers,

    SUM(churn_flag)
        AS churned_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS monthly_revenue

FROM analytics.customers

GROUP BY payment_method

ORDER BY churn_rate_pct DESC;



/* ============================================================
   6. SERVICE ADOPTION ANALYSIS
   ============================================================ */

SELECT
    service_count,

    COUNT(*) AS customers,

    SUM(churn_flag)
        AS churned_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        AVG(total_charges),
        2
    ) AS avg_total_charges

FROM analytics.customers

GROUP BY service_count

ORDER BY service_count;



/* ============================================================
   7. CUSTOMER REVENUE SEGMENTATION
   CTE + NTILE WINDOW FUNCTION
   ============================================================ */

WITH revenue_segments AS (

    SELECT
        customer_id,
        monthly_charges,
        total_charges,
        tenure,
        churn_flag,

        NTILE(4) OVER (
            ORDER BY monthly_charges
        ) AS revenue_quartile

    FROM analytics.customers
)

SELECT
    revenue_quartile,

    COUNT(*) AS customers,

    ROUND(
        MIN(monthly_charges),
        2
    ) AS min_monthly_charge,

    ROUND(
        MAX(monthly_charges),
        2
    ) AS max_monthly_charge,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct

FROM revenue_segments

GROUP BY revenue_quartile

ORDER BY revenue_quartile;



/* ============================================================
   8. CONTRACT PERFORMANCE RANKING
   Window functions
   ============================================================ */

WITH contract_metrics AS (

    SELECT
        contract,

        COUNT(*) AS customers,

        SUM(monthly_charges)
            AS monthly_revenue,

        AVG(churn_flag::numeric)
            AS churn_rate

    FROM analytics.customers

    GROUP BY contract
)

SELECT
    contract,
    customers,

    ROUND(
        monthly_revenue,
        2
    ) AS monthly_revenue,

    ROUND(
        churn_rate * 100,
        2
    ) AS churn_rate_pct,

    RANK() OVER (
        ORDER BY monthly_revenue DESC
    ) AS revenue_rank,

    RANK() OVER (
        ORDER BY churn_rate DESC
    ) AS churn_risk_rank

FROM contract_metrics;



/* ============================================================
   9. HIGH-VALUE CHURNED CUSTOMERS
   ============================================================ */

SELECT
    customer_id,
    contract,
    tenure,
    internet_service,
    payment_method,
    service_count,
    monthly_charges,
    total_charges

FROM analytics.customers

WHERE churn_flag = 1

ORDER BY monthly_charges DESC

LIMIT 25;



/* ============================================================
   10. REVENUE CONTRIBUTION
   Window aggregation
   ============================================================ */

WITH customer_revenue AS (

    SELECT
        customer_id,
        contract,
        monthly_charges,

        SUM(monthly_charges) OVER ()
            AS company_monthly_revenue

    FROM analytics.customers
)

SELECT
    customer_id,
    contract,
    monthly_charges,

    ROUND(
        monthly_charges
        / company_monthly_revenue
        * 100,
        4
    ) AS revenue_contribution_pct

FROM customer_revenue

ORDER BY monthly_charges DESC

LIMIT 25;



/* ============================================================
   11. CUSTOMER SEGMENTATION
   ============================================================ */

WITH customer_segments AS (

    SELECT
        customer_id,
        tenure,
        monthly_charges,
        total_charges,
        contract,
        churn_flag,

        CASE

            WHEN tenure <= 12
                AND monthly_charges >= 70
            THEN 'New High Value'

            WHEN tenure > 48
                AND monthly_charges >= 70
            THEN 'Loyal High Value'

            WHEN tenure <= 12
            THEN 'New Customer'

            WHEN tenure > 48
            THEN 'Loyal Customer'

            ELSE 'Established Customer'

        END AS customer_segment

    FROM analytics.customers
)

SELECT
    customer_segment,

    COUNT(*) AS customers,

    SUM(churn_flag)
        AS churned_customers,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS monthly_revenue,

    ROUND(
        AVG(total_charges),
        2
    ) AS avg_lifetime_revenue

FROM customer_segments

GROUP BY customer_segment

ORDER BY monthly_revenue DESC;



/* ============================================================
   12. REVENUE AT RISK BY BUSINESS SEGMENT
   ============================================================ */

SELECT
    contract,

    COUNT(*) FILTER (
        WHERE churn_flag = 1
    ) AS churned_customers,

    ROUND(
        SUM(monthly_charges)
        FILTER (
            WHERE churn_flag = 1
        ),
        2
    ) AS monthly_revenue_at_risk,

    ROUND(
        SUM(total_charges)
        FILTER (
            WHERE churn_flag = 1
        ),
        2
    ) AS historical_revenue_from_churned_customers

FROM analytics.customers

GROUP BY contract

ORDER BY monthly_revenue_at_risk DESC;