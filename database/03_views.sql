/* ============================================================
   CUSTOMER REVENUE INTELLIGENCE
   BI / ANALYTICS VIEWS
   ============================================================ */


DROP VIEW IF EXISTS analytics.vw_customer_360 CASCADE;
DROP VIEW IF EXISTS analytics.vw_churn_analysis CASCADE;
DROP VIEW IF EXISTS analytics.vw_revenue_analysis CASCADE;
DROP VIEW IF EXISTS analytics.vw_customer_segments CASCADE;
DROP VIEW IF EXISTS analytics.vw_executive_kpis CASCADE;


/* ============================================================
   VIEW 1
   EXECUTIVE KPIs
   ============================================================ */

CREATE VIEW analytics.vw_executive_kpis AS

SELECT

    COUNT(*) AS total_customers,

    COUNT(*) FILTER (
        WHERE churn_flag = 0
    ) AS active_customers,

    SUM(churn_flag)
        AS churned_customers,

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
    ) AS avg_lifetime_revenue,

    ROUND(
        AVG(tenure),
        2
    ) AS avg_tenure_months,

    ROUND(
        SUM(monthly_revenue_at_risk),
        2
    ) AS monthly_revenue_at_risk

FROM analytics.customers;


/* ============================================================
   VIEW 2
   CHURN ANALYSIS
   ============================================================ */

CREATE VIEW analytics.vw_churn_analysis AS

SELECT

    contract,
    internet_service,
    payment_method,
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
    ) AS avg_monthly_charge,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS monthly_revenue,

    ROUND(
        SUM(monthly_revenue_at_risk),
        2
    ) AS monthly_revenue_at_risk

FROM analytics.customers

GROUP BY
    contract,
    internet_service,
    payment_method,
    tenure_group;


/* ============================================================
   VIEW 3
   REVENUE ANALYSIS
   ============================================================ */

CREATE VIEW analytics.vw_revenue_analysis AS

SELECT

    contract,
    internet_service,
    service_count,

    COUNT(*) AS customers,

    ROUND(
        SUM(monthly_charges),
        2
    ) AS monthly_revenue,

    ROUND(
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        SUM(total_charges),
        2
    ) AS lifetime_revenue,

    ROUND(
        AVG(total_charges),
        2
    ) AS avg_lifetime_revenue,

    ROUND(
        AVG(churn_flag::numeric) * 100,
        2
    ) AS churn_rate_pct

FROM analytics.customers

GROUP BY
    contract,
    internet_service,
    service_count;


/* ============================================================
   VIEW 4
   CUSTOMER SEGMENTS
   ============================================================ */

CREATE VIEW analytics.vw_customer_segments AS

WITH segmented AS (

    SELECT

        *,

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
        AVG(monthly_charges),
        2
    ) AS avg_monthly_charge,

    ROUND(
        AVG(total_charges),
        2
    ) AS avg_lifetime_revenue

FROM segmented

GROUP BY customer_segment;


/* ============================================================
   VIEW 5
   CUSTOMER 360
   Primary dataset for BI + ML operations
   ============================================================ */

CREATE VIEW analytics.vw_customer_360 AS

SELECT

    c.*,

    CASE

        WHEN c.tenure <= 12
            AND c.monthly_charges >= 70
            THEN 'New High Value'

        WHEN c.tenure > 48
            AND c.monthly_charges >= 70
            THEN 'Loyal High Value'

        WHEN c.tenure <= 12
            THEN 'New Customer'

        WHEN c.tenure > 48
            THEN 'Loyal Customer'

        ELSE 'Established Customer'

    END AS customer_segment,

    CASE

        WHEN c.monthly_charges >= 90
            THEN 'High Revenue'

        WHEN c.monthly_charges >= 50
            THEN 'Medium Revenue'

        ELSE 'Low Revenue'

    END AS revenue_band

FROM analytics.customers AS c;