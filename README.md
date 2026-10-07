# Customer Revenue Intelligence & Churn Prediction Platform

An end-to-end customer analytics, business intelligence, and machine learning platform for analyzing telecom revenue, understanding churn behavior, predicting customer churn risk, prioritizing retention actions, and monitoring production scoring behavior.

The project combines **Python, PostgreSQL, SQL, Machine Learning, Power BI, Streamlit, Explainable AI, synthetic production data generation, data quality monitoring, drift monitoring, automated testing, and CI/CD** in a single portfolio-grade system.

---

## Executive Summary

Customer churn creates both revenue loss and retention-planning challenges for subscription businesses.

This platform was designed to answer five business questions:

1. How is the customer base and recurring revenue performing?
2. Which customer characteristics are associated with churn?
3. Which customers have the highest predicted churn risk?
4. Where should retention teams prioritize intervention?
5. How can prediction quality, data integrity, and production behavior be monitored?

The solution converts the IBM Telco Customer Churn dataset into a complete analytics and machine learning workflow, from raw data ingestion to executive dashboards and production-style model monitoring.

---

## Platform Highlights

- **7,043** historical customers
- **26.54%** historical churn rate
- **$456,116.60** historical monthly revenue
- **$139,130.85** historical monthly revenue associated with churned customers
- XGBoost production churn model
- **0.8438** held-out test ROC-AUC
- **0.6241** held-out test F1 score
- **71.93%** recall on churners
- Production decision threshold: **0.34**
- **22** production model features
- Customer-level and global model explainability
- Persistent synthetic live-customer generation
- PostgreSQL analytical and production-serving layers
- Power BI executive reporting
- Streamlit decision-support platform
- Population Stability Index (PSI) drift monitoring
- Automated data-quality reconciliation
- **50 automated tests**

---

## Business Problem

A telecom company's leadership is experiencing customer attrition and needs a unified analytics platform that can:

- explain revenue performance,
- identify customer churn patterns,
- quantify revenue exposure,
- predict customers at high risk of leaving,
- prioritize retention activity,
- provide executive and operational dashboards,
- and monitor the production scoring environment.

The project therefore goes beyond building a classification model.

It creates a complete **Customer Revenue Intelligence Platform** connecting analytics, business intelligence, machine learning, explainability, monitoring, and operational decision support.

---

## End-to-End Architecture

```text
                        ┌─────────────────────────────┐
                        │ IBM Telco Customer Dataset  │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │ Python Data Validation      │
                        │ Cleaning + Feature Creation │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │ PostgreSQL                  │
                        │ analytics.customers         │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
             ┌────────────┐     ┌─────────────┐    ┌─────────────┐
             │ SQL Views  │     │ Python EDA  │    │ ML Pipeline │
             │ & KPIs     │     │ & Statistics│    │ XGBoost     │
             └─────┬──────┘     └─────────────┘    └──────┬──────┘
                   │                                      │
                   │                                      ▼
                   │                              ┌──────────────────┐
                   │                              │ Saved Production │
                   │                              │ Model Artifact   │
                   │                              └────────┬─────────┘
                   │                                       │
                   │                                       ▼
                   │                              ┌──────────────────┐
                   │                              │ Batch + Live     │
                   │                              │ Churn Scoring    │
                   │                              └────────┬─────────┘
                   │                                       │
                   └──────────────────┬────────────────────┘
                                      │
                                      ▼
                           ┌───────────────────────┐
                           │ PostgreSQL Analytics  │
                           │ + Prediction Views    │
                           └───────────┬───────────┘
                                       │
                    ┌──────────────────┼───────────────────┐
                    │                  │                   │
                    ▼                  ▼                   ▼
             ┌────────────┐     ┌─────────────┐     ┌──────────────┐
             │ Power BI   │     │ Streamlit   │     │ Monitoring   │
             │ Dashboards │     │ Command     │     │ & Data       │
             │            │     │ Center      │     │ Quality      │
             └────────────┘     └─────────────┘     └──────────────┘
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Programming | Python 3.11 |
| Data Processing | pandas, NumPy |
| Statistical Analysis | SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Database | PostgreSQL |
| Database Access | SQLAlchemy, psycopg2 |
| Machine Learning | scikit-learn, XGBoost |
| Model Persistence | joblib |
| BI | Power BI |
| Application | Streamlit |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Version Control | Git, GitHub |

---

## Repository Structure

```text
customer-revenue-intelligence/
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   └── pull_request_template.md
│
├── app/
│   ├── assets/
│   │   └── styles.css
│   ├── components/
│   ├── views/
│   └── streamlit_app.py
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── database/
│   ├── queries/
│   ├── views/
│   ├── 01_schema.sql
│   ├── 02_analytics.sql
│   ├── 03_views.sql
│   └── 04_live_platform.sql
│
├── models/
│   └── churn_model.joblib
│
├── notebooks/
│   └── eda.ipynb
│
├── powerbi/
│   └── dashboard.pbix
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── config.py
│   ├── customer_generator.py
│   ├── data_cleaning.py
│   ├── database.py
│   ├── eda.py
│   ├── engine_controller.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── features.py
│   ├── generator_worker.py
│   ├── live_database.py
│   ├── load_database.py
│   ├── monitoring.py
│   ├── predict.py
│   ├── run_sql.py
│   ├── scoring.py
│   ├── synthetic_data.py
│   └── train.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Data Engineering

## Dataset

The project uses the **IBM Telco Customer Churn** dataset.

Historical population:

```text
7,043 customers
21 raw columns
```

The dataset contains customer demographics, services, contract information, billing attributes, tenure, charges, and historical churn status.

---

## Data Cleaning Pipeline

The Python cleaning layer performs:

- schema validation,
- duplicate handling,
- column-name normalization,
- whitespace cleanup,
- numeric conversion,
- missing-value treatment,
- churn target encoding,
- business-rule validation,
- feature engineering,
- data-quality reporting.

A known dataset issue occurs in `TotalCharges`, where zero-tenure customers contain blank values.

These records are handled using the business rule:

```text
tenure = 0
→ total_charges = 0
```

The cleaned dataset contains:

```text
7,043 rows
26 columns
```

---

## Engineered Features

Additional analytical and model features include:

- `tenure_group`
- `avg_revenue_per_tenure_month`
- `service_count`
- `monthly_revenue_at_risk`

The historical `monthly_revenue_at_risk` field is an analytical metric derived from customers who actually churned.

It is deliberately excluded from machine-learning features because it contains target information.

---

# PostgreSQL Analytics Layer

The cleaned customer population is loaded into PostgreSQL under the `analytics` schema.

Core tables include:

```text
analytics.customers
analytics.customer_predictions
analytics.live_customers
analytics.live_predictions
analytics.generator_state
analytics.generation_log
```

Analytical views include:

```text
analytics.vw_executive_kpis
analytics.vw_churn_analysis
analytics.vw_revenue_analysis
analytics.vw_customer_segments
analytics.vw_customer_360
analytics.vw_customer_risk_360
analytics.vw_retention_command_center
analytics.vw_all_customers
analytics.vw_all_customer_risk
analytics.vw_platform_kpis
```

The database layer supports historical analytics, production scoring, live synthetic customers, Power BI, and Streamlit from a common serving layer.

---

# Exploratory Data Analysis

EDA combines visual analysis with statistical testing.

## Key Findings

### Contract Type

Historical churn rates:

| Contract | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Contract type showed one of the strongest associations with churn.

Cramér's V:

```text
Contract = 0.4101
```

---

### Internet Service

Historical churn rates:

| Internet Service | Churn Rate |
|---|---:|
| Fiber optic | 41.89% |
| DSL | 18.96% |
| No internet service | 7.40% |

Cramér's V:

```text
Internet Service = 0.3225
```

---

### Tenure

Average tenure:

```text
Churned customers  = 17.98 months
Retained customers = 37.57 months
```

Mann-Whitney U testing produced:

```text
p ≈ 2.42 × 10^-208
```

This indicates a strong statistical difference in tenure distributions between churned and retained historical customers.

---

### Monthly Charges

Average monthly charges:

```text
Churned customers  = $74.44
Retained customers = $61.27
```

Mann-Whitney U:

```text
p ≈ 3.31 × 10^-54
```

---

### Other Strong Associations

Cramér's V results include:

```text
Online Security = 0.3474
Tech Support    = 0.3429
Payment Method  = 0.3034
```

These findings are treated as **associations**, not causal effects.

---

# Machine Learning

## Objective

Predict customer churn probability while preserving a production-safe feature contract and preventing target leakage.

Target:

```text
churn_flag
```

Leakage fields excluded from training:

```text
customer_id
churn
churn_flag
monthly_revenue_at_risk
```

The production feature contract contains:

```text
22 raw features
```

---

## Preprocessing

The machine-learning pipeline uses a `ColumnTransformer`.

Numerical features:

```text
StandardScaler
```

Categorical features:

```text
OneHotEncoder(
    handle_unknown="ignore"
)
```

Preprocessing and the estimator are persisted together in one production pipeline, preventing training-serving transformation mismatch.

---

## Models Evaluated

Three candidate models were compared:

- Logistic Regression
- Random Forest
- XGBoost

### Validation Results

| Model | CV ROC-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.8440 | 0.8480 |
| Random Forest | 0.8441 | 0.8487 |
| XGBoost | 0.8389 | **0.8518** |

XGBoost was selected using the predefined **validation ROC-AUC** criterion.

The result should not be interpreted as XGBoost dominating every evaluation metric: Random Forest achieved a slightly higher cross-validation ROC-AUC.

---

# Production Model Evaluation

After model selection and threshold optimization, the final model was evaluated once on the untouched holdout test set.

Decision threshold:

```text
0.34
```

Held-out performance:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8438** |
| Precision | **0.5512** |
| Recall | **0.7193** |
| F1 Score | **0.6241** |
| Accuracy | **0.7701** |

Confusion matrix:

```text
TN = 816
FP = 219
FN = 105
TP = 269
```

The threshold prioritizes improved churn detection rather than defaulting automatically to `0.50`.

---

# Production Risk Segmentation

Each scored customer receives:

- churn probability,
- binary model decision,
- risk segment,
- retention priority,
- probability-weighted monthly revenue exposure.

Risk bands:

| Probability | Segment |
|---|---|
| `< 0.30` | Low |
| `0.30 – <0.50` | Medium |
| `0.50 – <0.70` | High |
| `>= 0.70` | Critical |

Expected monthly revenue exposure is calculated as:

```text
monthly_charges × churn_probability
```

This produces a commercial prioritization measure rather than treating all high-risk customers as equally valuable.

---

# Explainable AI

The platform includes both global and customer-level model explainability.

## Global Model Drivers

Top production XGBoost drivers include:

| Driver | Importance |
|---|---:|
| Contract: Month-to-month | 0.2984 |
| Internet Service: Fiber optic | 0.0851 |
| Online Security: No | 0.0695 |
| Internet Service: DSL | 0.0623 |
| Tech Support: No | 0.0593 |
| Tenure Group: 0–12 Months | 0.0527 |

Feature importance describes **predictive model influence**, not causality.

---

## Customer-Level Explanation

Prediction Lab uses native XGBoost contribution analysis to explain individual predictions.

For each scenario, the application separates:

```text
Risk-Increasing Drivers
Risk-Reducing Drivers
```

Contributions operate in the XGBoost model's **raw-margin / log-odds space**.

They are not percentage-point changes in churn probability and should not be interpreted as causal effects.

---

# Streamlit Decision Platform

The Streamlit application functions as an operational customer intelligence command center rather than a single prediction form.

Main workspaces:

```text
Overview
├── Command Center
└── Executive Overview

Analytics
├── Churn Analytics
├── Revenue Intelligence
├── ML Risk Intelligence
└── Retention Center

AI & Operations
├── Prediction Lab
├── Live Data Engine
└── Model & Data Monitoring

Business Intelligence
└── Power BI Hub
```

---

## Command Center

Provides platform-level operational visibility across:

- historical customers,
- generated customers,
- revenue,
- churn risk,
- expected revenue exposure,
- live generation state.

---

## Prediction Lab

Users can construct customer scenarios using raw business attributes.

The platform then:

```text
Raw Customer Attributes
        ↓
Feature Engineering
        ↓
Persisted Preprocessing Pipeline
        ↓
XGBoost
        ↓
Churn Probability
        ↓
Risk Segment
        ↓
Retention Priority
        ↓
Expected Revenue Exposure
        ↓
Customer-Level Explanation
```

Prediction Lab scenarios are **what-if analyses only** and are not automatically persisted to PostgreSQL.

---

# Live Customer Generation Engine

The platform includes a persistent synthetic customer generation system for demonstrating production-style scoring.

Users can:

```text
START
STOP
Generate One Customer
Configure generation interval
```

Each generated customer is:

1. created using business rules,
2. assigned a permanent synthetic customer ID,
3. persisted to PostgreSQL,
4. scored using the production ML pipeline,
5. persisted to the live prediction layer,
6. exposed through unified analytical views,
7. displayed in Streamlit,
8. available to Power BI after dataset refresh.

Generated customers persist across application restarts.

---

## Historical vs Live Data Isolation

The project deliberately separates labeled historical data from generated production-style data.

```text
Historical
analytics.customers
analytics.customer_predictions

Live
analytics.live_customers
analytics.live_predictions
```

Synthetic customers are **not appended to the historical training dataset**.

This prevents artificial production examples from contaminating historical ground truth.

---

# Production Monitoring

The platform includes a dedicated **Model & Data Monitoring** workspace.

It monitors four areas.

## Data Quality

Automated checks validate:

- historical ID uniqueness,
- live ID uniqueness,
- null customer IDs,
- historical population preservation,
- historical prediction coverage,
- live prediction coverage,
- unified population reconciliation,
- unified risk coverage,
- historical/live ID isolation.

Current quality gate:

```text
10 / 10 checks passing
```

---

## Prediction Monitoring

The application monitors:

- churn probability distribution,
- risk-segment distribution,
- historical vs generated scoring behavior,
- high/critical risk concentration,
- expected revenue exposure.

---

## Population Drift

Generated customers are compared with the historical baseline using **Population Stability Index (PSI)**.

Interpretation:

| PSI | Status |
|---|---|
| `< 0.10` | Stable |
| `0.10 – 0.25` | Moderate Shift |
| `> 0.25` | Significant Shift |

Drift monitoring is intentionally separated from model-performance monitoring.

A distribution shift does **not** automatically mean the model has degraded.

---

## Why Live Accuracy Is Not Displayed

Generated customers receive predictions immediately but do not have observed future churn outcomes.

Therefore, the platform does **not** fabricate:

- live accuracy,
- live precision,
- live recall,
- live F1,
- live ROC-AUC.

Those metrics require ground-truth outcomes.

Instead, the live system monitors prediction distributions, input stability, data quality, and scoring coverage.

---

# Power BI

The Power BI layer provides management-focused analytics across three pages.

## Executive Overview

Designed for senior management to understand:

- customer population,
- churn,
- recurring revenue,
- risk exposure,
- customer segmentation.

## Churn & Revenue Analysis

Focuses on:

- contract behavior,
- customer tenure,
- internet services,
- churn patterns,
- commercial performance.

## AI-Powered Retention Command Center

Focuses on:

- predicted churn,
- high-risk customers,
- critical-risk customers,
- expected monthly revenue exposure,
- retention prioritization.

Power BI uses PostgreSQL in **Import mode**.

New generated customers become visible after dataset refresh.

---

# Automated Testing

The project includes **50 automated tests** covering:

```text
Configuration
Database integrity
Feature engineering
Target leakage
Prediction outputs
Live platform integrity
Synthetic customer generation
Generator controller
Explainability
Production monitoring
PSI drift calculations
Model artifact contract
Historical/live isolation
Prediction coverage
```

Current local quality gate:

```text
50 passed
```

---

# CI/CD

GitHub Actions provides an automated quality gate for pushes and pull requests.

The CI workflow performs:

```text
Repository Checkout
        ↓
Python 3.11 Setup
        ↓
Dependency Installation
        ↓
Python Compilation
        ↓
PostgreSQL Initialization
        ↓
Historical Data Load
        ↓
SQL Analytics Setup
        ↓
Live Platform Setup
        ↓
Production Prediction Generation
        ↓
Automated Tests
        ↓
Explainability Validation
        ↓
Monitoring Validation
```

This validates the project on a clean environment instead of relying only on the developer's local machine.

---

# Running the Project Locally

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-revenue-intelligence
```

---

## 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure PostgreSQL

Create:

```text
Database: customer_intelligence
Schema: analytics
```

Copy:

```text
.env.example
```

to:

```text
.env
```

Configure:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_intelligence
DB_USER=postgres
DB_PASSWORD=your_password
```

Never commit `.env`.

---

## 5. Clean the source dataset

Place the IBM Telco dataset at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Run:

```bash
python -m src.data_cleaning
```

---

## 6. Load PostgreSQL

```bash
python -m src.load_database
```

---

## 7. Build analytics SQL

```bash
python -m src.run_sql
```

Initialize the live platform using:

```text
database/04_live_platform.sql
```

---

## 8. Run EDA

```bash
python -m src.eda
```

---

## 9. Train the model

```bash
python -m src.train
```

---

## 10. Generate production predictions

```bash
python -m src.predict
```

---

## 11. Validate explainability

```bash
python -m src.explainability
```

---

## 12. Validate monitoring

```bash
python -m src.monitoring
```

---

## 13. Run automated tests

```bash
pytest -v
```

Expected quality gate:

```text
50 passed
```

---

## 14. Launch Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

# Data and ML Lineage

```text
RAW DATA
   │
   ▼
Schema Validation
   │
   ▼
Cleaning
   │
   ▼
Feature Engineering
   │
   ├──────────────► PostgreSQL Analytics
   │
   ▼
Leakage Removal
   │
   ▼
Train / Validation / Test Strategy
   │
   ▼
Candidate Model Comparison
   │
   ▼
Validation-Based Model Selection
   │
   ▼
Threshold Optimization
   │
   ▼
Final Holdout Evaluation
   │
   ▼
Persisted Pipeline
   │
   ▼
Production Scoring
   │
   ├──────────────► Risk Segmentation
   ├──────────────► Revenue Exposure
   ├──────────────► Explainability
   ├──────────────► PostgreSQL
   ├──────────────► Streamlit
   └──────────────► Power BI
```

---

# Business Interpretation

The platform demonstrates how predictive modeling can be connected to business decision-making.

Instead of producing only:

```text
Customer will churn / Customer will not churn
```

the serving layer produces:

```text
Probability
+
Risk Segment
+
Retention Priority
+
Commercial Exposure
+
Prediction Explanation
```

This enables retention teams to consider both **likelihood of churn** and **customer value**.

---

# Important Limitations

This is a portfolio and analytical system built from a public historical dataset.

Important limitations include:

1. Generated customers are synthetic and do not represent real production traffic.
2. Generated customers do not have future ground-truth churn outcomes.
3. Live model-performance metrics therefore cannot yet be calculated.
4. Model feature importance and customer-level contributions are predictive explanations, not causal evidence.
5. Historical churn patterns may not generalize to another telecom business or time period.
6. No retention intervention has been experimentally tested.
7. Expected revenue exposure is a prioritization metric, not a guaranteed financial loss estimate.
8. PSI alerts indicate distribution shift, not automatic model failure.
9. The system would require authentication, authorization, infrastructure hardening, observability, and organizational governance before use with real customer data.

---

# Screenshots

Application screenshots will be added after final deployment validation.

Recommended portfolio screenshots:

```text
1. Streamlit Command Center
2. Executive Overview
3. ML Risk Intelligence + Global Drivers
4. Prediction Lab + Why This Prediction?
5. Live Data Engine
6. Model & Data Monitoring
7. Power BI Executive Overview
8. Power BI Retention Command Center
```

---

# Deployment

Deployment configuration is completed in the final productionization phase.

The deployed architecture will preserve:

- secret-based database credentials,
- PostgreSQL persistence,
- persisted model artifact,
- historical/live data separation,
- production scoring consistency,
- monitoring,
- reproducible dependency installation.

**Live application:** Coming after final deployment validation.

---

# What This Project Demonstrates

This project demonstrates skills across three closely related roles.

### Data Analytics

- data cleaning,
- EDA,
- statistical testing,
- customer segmentation,
- churn analysis,
- revenue analysis,
- business KPI design.

### Business Intelligence

- PostgreSQL analytical modeling,
- reusable SQL views,
- Power BI dashboard design,
- executive reporting,
- retention prioritization,
- commercial KPI communication.

### Machine Learning Engineering

- leakage-safe feature engineering,
- preprocessing pipelines,
- model comparison,
- threshold optimization,
- model persistence,
- production scoring,
- explainability,
- synthetic live inference,
- drift monitoring,
- data-quality validation,
- automated testing,
- CI/CD.

---

# Interview Talking Points

A few design decisions are particularly important:

**Why XGBoost?**  
XGBoost achieved the highest validation ROC-AUC under the predefined selection criterion. Random Forest slightly outperformed it in cross-validation ROC-AUC, so the project does not claim universal model dominance.

**Why threshold 0.34 instead of 0.50?**  
The classification threshold was optimized on validation data rather than assuming the default probability threshold.

**How was leakage prevented?**  
Customer identifiers, the original churn field, encoded churn target, and historical churn-derived revenue-at-risk metric were excluded from model features.

**Why separate historical and generated customers?**  
Historical customers contain ground-truth churn labels. Synthetic generated customers do not. Combining them in the training truth layer would compromise data integrity.

**Why no live accuracy metric?**  
Accuracy requires observed outcomes. Generated customers have predictions but no future churn labels.

**What does PSI measure?**  
PSI identifies population distribution shifts relative to the historical baseline. It is a monitoring signal, not direct evidence of model-performance degradation.

---

# Future Production Enhancements

For a real enterprise deployment, the next evolution could include:

- authenticated role-based access,
- real event-driven customer ingestion,
- scheduled retraining,
- model registry and versioning,
- feature-store integration,
- retention intervention tracking,
- experiment / uplift measurement,
- delayed ground-truth collection,
- performance monitoring after labels mature,
- centralized observability and alerting.

These are intentionally treated as future production enhancements rather than simulated as features that do not exist.

---

## Project Status

```text
Data Engineering             COMPLETE
PostgreSQL Analytics         COMPLETE
Exploratory Analysis         COMPLETE
Machine Learning             COMPLETE
Production Scoring           COMPLETE
Power BI                     COMPLETE
Streamlit Platform           COMPLETE
Live Data Engine             COMPLETE
Explainability               COMPLETE
Production Monitoring        COMPLETE
Automated Tests              50 PASSING
CI/CD Quality Gate           IMPLEMENTED
Deployment                   DONE
```

---

## Author

**Aditya Sharma**


Portfolio Project — Data Analytics | Business Intelligence | Machine Learning Engineering
