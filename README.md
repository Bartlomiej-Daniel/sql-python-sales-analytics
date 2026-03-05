# SQL + Python Sales Analytics & Machine Learning Project

## Project Overview

This project demonstrates an **end‑to‑end data analytics workflow**
using an e‑commerce retail dataset.

Key components: - ETL pipeline and SQLite database - SQL + Python
exploratory analysis - Customer segmentation (RFM) - Machine learning
clustering (K‑Means) - Revenue concentration analysis - Churn risk
simulation - Time‑series forecasting (Prophet, Linear Regression, Naive)

Dataset: **Online Retail -- UCI Machine Learning Repository**\
https://archive.ics.uci.edu/dataset/352/online+retail

------------------------------------------------------------------------

# Project Structure

    sql-python-sales-analytics/
    │
    ├── data/
    │   └── Online Retail.xlsx
    │
    ├── db/
    │   └── sales.db  (SQLite database)
    │
    ├── etl/
    │   └── etl_basic.py
    │
    ├── analysis/
    │   ├── eda.py
    │   └── sql_queries.py
    │
    ├── core/
    │   └── data_loader.py
    │
    ├── modeling/
    |   |
    │   ├── experiments/
    │   │   ├── compare_models.py
    │   │   └── prophet_tuning.py
    │   │  
    |   |
    │   ├── forecasting/
    │   │   ├── evaluation.py
    │   │   ├── main.py
    │   │   └── models.py
    |   |
    │   └── segmentation/
    │       ├── rfm.py
    │       ├── kmeans_segmentation.py
    │       └── segmentation_comparison.py
    │
    ├── reports/
    │   ├── rfm_report.xlsx
    │   ├── clusters_frequency_monetary.png
    │   ├── clusters_recency_monetary.png
    │   ├── clusters_revenue_share.png
    │   ├── revenue_bysegment.png
    │   └── RFM_vs_KMeans_Segment_Comparison.png
    │
    └── README.md


------------------------------------------------------------------------

# ETL Pipeline

The ETL pipeline:

1.  Loads Excel data
2.  Cleans missing values
3.  Creates feature **TotalPrice**
4.  Stores processed data in **SQLite database**

SQLite provides a lightweight analytics database suitable for
prototypes.

------------------------------------------------------------------------

# Exploratory Data Analysis

EDA investigates:

-   Revenue trends
-   Purchase frequency
-   Country revenue distribution
-   Customer purchase patterns

Scripts:

analysis/eda.py\
analysis/sql_queries.py

------------------------------------------------------------------------

# Time Series Forecasting

Revenue forecasting models:

-   Prophet
-   Linear Regression
-   Naive baseline

Pipeline:

1.  Monthly aggregation
2.  Train/test split (80/20)
3.  Model training
4.  Evaluation using **MAE and MAPE**

------------------------------------------------------------------------

# Forecasting Experiments

Additional experiments:

### Prophet Hyperparameter Tuning

`prophet_tuning.py` tests different values of:

changepoint_prior_scale

Example values:

0.01, 0.05, 0.1, 0.3, 0.5

### Daily vs Monthly Forecast Comparison

`compare_models.py` compares forecasting stability between daily and
monthly data.

------------------------------------------------------------------------

# Customer Segmentation (RFM)

Customers segmented by:

-   Recency
-   Frequency
-   Monetary

Segments:

VIP\
Loyal\
Regular\
Occasional\
At Risk

Output report:

reports/rfm_report.xlsx

------------------------------------------------------------------------

# Machine Learning Segmentation

Unsupervised clustering using **K‑Means**.

Process:

1.  Feature scaling
2.  Elbow method
3.  Cluster training
4.  Cluster profiling

Silhouette Score:

≈ **0.59**

------------------------------------------------------------------------

# Key Insight

A very small group of customers generates a disproportionate share of
revenue.

Example:

23 customers generate **\~25% of total revenue**.

------------------------------------------------------------------------

# Revenue Risk Simulation

Revenue loss scenarios:

Losing 5 Ultra VIP customers → \~11--12% revenue loss\
Losing 10 Ultra VIP customers → \~17% revenue loss

This reveals strong **revenue concentration risk**.

------------------------------------------------------------------------

# Ultra VIP Churn Risk Scoring

Risk scoring based on:

-   Recency
-   Frequency
-   Monetary

Example distribution:

High risk: 8\
Medium risk: 14\
Low risk: 4

------------------------------------------------------------------------

# Technologies

Python\
Pandas\
NumPy\
Scikit‑learn\
Prophet\
Matplotlib\
SQLite\
SQL

------------------------------------------------------------------------

# Skills Demonstrated

-   Data Engineering
-   SQL Analytics
-   Exploratory Data Analysis
-   Time Series Forecasting
-   Machine Learning Clustering
-   Customer Segmentation
-   Business Risk Analysis
-   Data Visualization

------------------------------------------------------------------------
