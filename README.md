# NASA Predictive Maintenance — Turbofan Engine RUL Prediction

<p align="center">
  <b>Predicting Remaining Useful Life of aircraft turbofan engines using sensor data, machine learning, and cost analysis</b>
</p>

---

## Overview

This project predicts the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the **NASA C-MAPSS FD004** dataset — a benchmark dataset containing simulated sensor readings from 249 engines operating across 6 different flight conditions.

The project covers the full data science lifecycle: **exploratory data analysis**, **feature engineering**, **machine learning model training**, **model explainability with SHAP**, **cost estimation**, and **interactive visualization** — all built on top of a **Medallion Architecture data pipeline** using Delta Lake.

## Problem Statement

Aircraft engine failures are catastrophic and expensive. Unplanned downtime costs approximately **$15,000/hour** and **$225,000/event** (industry benchmark). Predicting when an engine will fail allows maintenance teams to:

- Schedule proactive maintenance before failure occurs
- Reduce unplanned downtime costs by up to 80%
- Optimize spare parts inventory and crew scheduling

**Goal:** Given real-time sensor readings and operating conditions, predict how many more flight cycles an engine can safely operate (RUL), classify its health state, and estimate the maintenance cost.

## Architecture — Medallion Architecture (Delta Lake)

```
Bronze Layer (Raw)          Silver Layer (Feature Engineered)       Gold Layer (Business Aggregates)
+--------------------+      +-----------------------------+        +------------------------------+
| Raw C-MAPSS files  | ---> | K-Means Clustering (6 regs) | -----> | Predicted RUL                |
| train_FD004.txt    |      | Z-Score Normalization       |        | Health State Classification  |
| test_FD004.txt     |      | Rolling Mean/Std (w=5,10,20)|        | Maintenance Cost Estimate    |
| RUL_FD004.txt      |      | Slope Features (lag=1,5)   |        | Risk Category & Health Index  |
| Delta Tables in UC |      | RUL + Health State Bins     |        | Delta Table in Unity Catalog |
+--------------------+      | Calendar Features           |        +------------------------------+
                            | Delta Tables in UC          |
                            +-----------------------------+
```

**Technologies:** PySpark, Delta Lake, Unity Catalog, Spark ML (KMeans), Window Functions

## Key Results

### Model Performance

| Model | Features | R² | RMSE (cycles) |
|---|---|---|---|
| **Full Model** (RandomForest) | 145 engineered features | 0.874 | 15.04 |
| **Simplified Model** (RandomForest) | 24 raw features | 0.791 | 19.37 |

The simplified model (using only raw sensor values) achieves **91% of the full model's accuracy** — making it practical for real-time prediction without requiring historical rolling data.

### Top 5 Most Important Sensors

| Rank | Sensor | Description | Importance |
|---|---|---|---|
| 1 | sensor_13 | Corrected Fuel Flow | 44.0% |
| 2 | sensor_11 | Static Pressure at HPC Outlet | 16.7% |
| 3 | sensor_15 | Bypass Ratio | 13.3% |
| 4 | sensor_4 | Temperature at LPT Outlet | 6.8% |
| 5 | sensor_12 | Ratio of Fuel Flow to Ps30 | 4.0% |

### Cost Analysis

| Scenario | Downtime | Est. Cost |
|---|---|---|
| Critical (RUL <= 30) | 48-72 hrs | ~$1,000,000+ |
| Degraded (RUL 31-80) | 8-24 hrs | ~$46,000-$48,000 |
| Healthy (RUL > 80) | 1-2 hrs | ~$1,000 |

Calendar multipliers (holiday 1.5x, weekend 1.2x, peak season 1.3x) are applied to labour and downtime costs.

## Tech Stack

| Category | Tools |
|---|---|
| **Data Platform** | Databricks, Delta Lake, Unity Catalog |
| **Data Processing** | PySpark, Spark ML, Spark SQL, Window Functions |
| **Machine Learning** | scikit-learn (RandomForest), SHAP (TreeExplainer) |
| **Data Source** | NASA C-MAPSS FD004 (249 engines, 102,463 records) |
| **Visualization** | Streamlit, Matplotlib, Seaborn, SHAP plots |
| **Deployment** | Python modules (.joblib), Streamlit web app |

## Project Structure

```
Predictive maintenance NASA Data/
|
|-- Predictive Maintenance NASA Analysis.ipynb   # EDA + Model Training + SHAP Analysis
|-- NASA Medallion Pipeline.ipynb                # Bronze -> Silver -> Gold Delta Lake Pipeline
|
|-- rul_model.py                                  # RUL prediction module (full 145-feature model)
|-- cost_model.py                                 # Cost estimation module with scenario analysis
|-- streamlit_app.py                              # Interactive web app (imports both modules)
|-- requirements.txt                              # Python dependencies
|-- README.md                                     # This file
|
|-- artifacts/                                    # Saved models and config (12 files)
|   |-- rul_regressor.joblib                      # Full RUL model (145 features)
|   |-- rul_regressor_raw.joblib                   # Simplified RUL model (24 features)
|   |-- health_state_classifier.joblib            # Health state classifier
|   |-- kmeans_op_regime.joblib                    # K-Means clustering model
|   |-- shap_explainer_raw.joblib                  # SHAP TreeExplainer
|   |-- regime_mu.joblib / regime_sigma.joblib     # Normalization statistics
|   |-- features.json / raw_features.json          # Feature name lists
|   |-- feature_stats.json                         # Min/max/mean/std for app sliders
|   |-- cost_parameters.json                      # Industry benchmark cost parameters
|   |-- model_config.json                         # Sensor lists, bins, labels
|
|-- CMaps/                                        # NASA C-MAPSS raw data files
    |-- train_FD004.txt  (61,249 rows)
    |-- test_FD004.txt   (41,214 rows)
    |-- RUL_FD004.txt    (248 rows)
```

## How to Run

### 1. Run the Analysis Notebook

Open **Predictive Maintenance NASA Analysis.ipynb** in Databricks and run all cells. This notebook:
- Loads and explores the C-MAPSS FD004 dataset
- Performs feature engineering (clustering, normalization, rolling stats)
- Trains RandomForest models for RUL regression and health state classification
- Computes SHAP values for model explainability
- Exports all models and artifacts to `artifacts/`

### 2. Run the Medallion Pipeline

Open **NASA Medallion Pipeline.ipynb** and run all cells. This notebook:
- **Bronze:** Ingests raw C-MAPSS files into Delta tables (`workspace.bronze.*`)
- **Silver:** Feature engineering with PySpark -> Delta tables (`workspace.silver.*`)
- **Gold:** ML predictions + cost estimation -> Delta table (`workspace.gold.engine_health_status`)

### 3. Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app provides:
- **Real-time RUL prediction** via interactive sliders for 24 sensor/operating parameters
- **SHAP Waterfall Plot** showing which features push RUL up or down
- **Sensitivity Analysis Table** quantifying the impact of each feature (+/- 1 std)
- **Maintenance Cost Estimate** with calendar multipliers (holiday/weekend/peak season)
- **Cost Scenario Comparison** for sensitivity analysis on cost parameters

## Feature Engineering Pipeline

```
Raw Sensors (21) + Op Conditions (3)
        |
        v
K-Means Clustering (k=6) --------> op_regime column
        |
        v
Z-Score Normalization per regime -> sensor_X_norm (15 sensors)
        |
        v
Rolling Mean & Std (w=5,10,20) -> 90 new columns
        |
        v
Slope / Rate of Change (lag=1,5) -> 12 new columns
        |
        v
RUL (piecewise, capped at 130) + Health State (3-class)
        |
        v
Calendar Features (is_holiday, is_weekend, is_peak_season)
        |
        v
Total: 145 features
```

## Module Documentation

### rul_model.py

```python
from rul_model import RULPredictor

predictor = RULPredictor(artifacts_dir='./artifacts')

# Predict from raw data (auto feature engineering)
rul, health = predictor.predict_from_raw(raw_df)

# Or predict from pre-engineered features
rul = predictor.predict_rul(features_df)
health = predictor.predict_health_state(features_df)
```

### cost_model.py

```python
from cost_model import CostEstimator

estimator = CostEstimator(artifacts_dir='./artifacts')

# Single estimate
result = estimator.estimate_single(rul=45, health_state='Degraded (Medium Risk)')
# -> {'maintenance_action': 'Proactive Planned Maintenance',
#     'maintenance_cost': 46432, 'downtime_hours': 11.5, ...}

# Scenario comparison
comparison = estimator.compare_scenarios(df, {
    'Baseline': {},
    'High Downtime': {'unplanned_downtime_cost_per_hour': 20000},
})
```

## Future Improvements

- Integrate NASA POWER weather API data as a Bronze layer table for weather stress features
- Train XGBoost/LightGBM models for improved accuracy
- Add time-series forecasting (LSTM/Transformer) for multi-step RUL prediction
- Deploy as Databricks App with authentication
- Add automated data quality checks in the Silver layer
- Implement model monitoring and drift detection

## Dataset

**NASA C-MAPSS FD004** — Commercial Modular Aero-Propulsion System Simulation

- 249 train engines, 248 test engines
- 21 sensors, 3 operating conditions
- 6 operating regimes (altitude, Mach number, throttle)
- 61,249 training cycles, 41,214 test cycles

Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," 2008.

---

Built on **Databricks** with **PySpark**, **Delta Lake**, **scikit-learn**, **SHAP**, and **Streamlit**.
