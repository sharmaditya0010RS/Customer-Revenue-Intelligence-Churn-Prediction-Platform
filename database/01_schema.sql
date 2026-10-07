CREATE SCHEMA IF NOT EXISTS analytics;


DROP TABLE IF EXISTS analytics.customers CASCADE;


CREATE TABLE analytics.customers (

    customer_id VARCHAR(20) PRIMARY KEY,

    gender VARCHAR(10) NOT NULL,
    senior_citizen INTEGER NOT NULL,

    partner VARCHAR(3) NOT NULL,
    dependents VARCHAR(3) NOT NULL,

    tenure INTEGER NOT NULL,

    phone_service VARCHAR(20) NOT NULL,
    multiple_lines VARCHAR(30) NOT NULL,

    internet_service VARCHAR(30) NOT NULL,
    online_security VARCHAR(30) NOT NULL,
    online_backup VARCHAR(30) NOT NULL,
    device_protection VARCHAR(30) NOT NULL,
    tech_support VARCHAR(30) NOT NULL,

    streaming_tv VARCHAR(30) NOT NULL,
    streaming_movies VARCHAR(30) NOT NULL,

    contract VARCHAR(30) NOT NULL,
    paperless_billing VARCHAR(3) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,

    monthly_charges NUMERIC(10,2) NOT NULL,
    total_charges NUMERIC(12,2) NOT NULL,

    churn VARCHAR(3) NOT NULL,
    churn_flag INTEGER NOT NULL,

    tenure_group VARCHAR(30) NOT NULL,

    avg_revenue_per_tenure_month NUMERIC(12,2),

    service_count INTEGER NOT NULL,

    monthly_revenue_at_risk NUMERIC(10,2) NOT NULL,

    CONSTRAINT chk_churn_flag
        CHECK (churn_flag IN (0,1)),

    CONSTRAINT chk_tenure
        CHECK (tenure >= 0),

    CONSTRAINT chk_monthly_charges
        CHECK (monthly_charges >= 0),

    CONSTRAINT chk_total_charges
        CHECK (total_charges >= 0),

    CONSTRAINT chk_service_count
        CHECK (service_count >= 0)
);


CREATE INDEX idx_customers_churn
ON analytics.customers(churn_flag);


CREATE INDEX idx_customers_contract
ON analytics.customers(contract);


CREATE INDEX idx_customers_tenure
ON analytics.customers(tenure);


CREATE INDEX idx_customers_internet_service
ON analytics.customers(internet_service);


CREATE INDEX idx_customers_payment_method
ON analytics.customers(payment_method);