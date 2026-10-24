# Databricks notebook source
# DBTITLE 1,Pipeline Title
# MAGIC %md
# MAGIC # NASA Predictive Maintenance — Medallion Architecture Pipeline (Delta Lake)
# MAGIC
# MAGIC This notebook implements a **Bronze → Silver → Gold** data pipeline for the NASA C-MAPSS FD004 dataset using **PySpark** and **Delta Lake**.
# MAGIC
# MAGIC | Layer | Purpose | Tables |
# MAGIC |---|---|---|
# MAGIC | **Bronze** | Raw sensor data ingestion (preserve original schema) | `workspace.bronze.train_fd004`, `workspace.bronze.test_fd004`, `workspace.bronze.rul_fd004` |
# MAGIC | **Silver** | Feature engineering: K-Means clustering, Z-score normalization, rolling stats, RUL, calendar features | `workspace.silver.engine_features_train`, `workspace.silver.engine_features_test` |
# MAGIC | **Gold** | Business aggregates: predicted RUL, health state, maintenance cost, risk category | `workspace.gold.engine_health_status` |
# MAGIC
# MAGIC **Key Technologies:** PySpark Window Functions, Spark ML (K-Means), Delta Lake, sklearn model inference via `mapInPandas`

# COMMAND ----------

# DBTITLE 1,Pipeline Config & Constants
# ============================================================
# Configuration — Catalog, Schema, Paths, Constants
# ============================================================

CATALOG = "workspace"
SCHEMA_BRONZE = f"{CATALOG}.bronze"
SCHEMA_SILVER = f"{CATALOG}.silver"
SCHEMA_GOLD   = f"{CATALOG}.gold"

DATA_DIR = "/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/CMaps"
ARTIFACTS_DIR = "/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/artifacts"

# --- Sensor configuration (from notebook analysis) ---
USEFUL_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]
KEY_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']
WINDOWS = [5, 10, 20]
RUL_CAP = 130
N_REGIMES = 6
SETTINGS_COLS = ['op_cond_1', 'op_cond_2', 'op_cond_3']

# --- Raw C-MAPSS column names ---
COLS = ['unit_id', 'cycle', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + [f'sensor_{i}' for i in range(1, 22)]

# --- Create schemas ---
for schema in ['bronze', 'silver', 'gold']:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
print("✓ Schemas ready: bronze, silver, gold")

# COMMAND ----------

# DBTITLE 1,Bronze Layer — Raw Ingestion to Delta
# ============================================================
# BRONZE LAYER — Raw Data Ingestion → Delta Tables
# อ่านไฟล์ C-MAPSS ดิบ (whitespace-delimited) และบันทึกลง Delta Table
# รักษา Schema ดั้งเดิมทั้งหมด เพิ่มเฉพาะ ingestion_timestamp
# ============================================================

from pyspark.sql.functions import split, col, trim, current_timestamp

def read_cmapss(path, cols):
    """อ่านไฟล์ C-MAPSS ที่ใช้ whitespace เป็น separator"""
    raw = spark.read.text(path)
    split_expr = split(trim(col("value")), r"\s+")
    return raw.select(
        split_expr[0].cast("int").alias(cols[0]),
        split_expr[1].cast("int").alias(cols[1]),
        *[split_expr[i].cast("double").alias(cols[i]) for i in range(2, len(cols))]
    ).withColumn("ingestion_timestamp", current_timestamp())

# --- อ่านไฟล์และเขียน Delta Table ---
train_bronze = read_cmapss(f"{DATA_DIR}/train_FD004.txt", COLS)
test_bronze  = read_cmapss(f"{DATA_DIR}/test_FD004.txt", COLS)

# RUL file (single integer column)
rul_raw = spark.read.text(f"{DATA_DIR}/RUL_FD004.txt")
rul_bronze = rul_raw.select(
    split(trim(col("value")), r"\s+")[0].cast("int").alias("RUL")
).withColumn("ingestion_timestamp", current_timestamp())

# Write to Delta tables (overwrite mode for idempotent re-runs)
train_bronze.write.format("delta").mode("overwrite").saveAsTable(f"{SCHEMA_BRONZE}.train_fd004")
test_bronze.write.format("delta").mode("overwrite").saveAsTable(f"{SCHEMA_BRONZE}.test_fd004")
rul_bronze.write.format("delta").mode("overwrite").saveAsTable(f"{SCHEMA_BRONZE}.rul_fd004")

print("✓ Bronze Layer: 3 Delta tables created")
print(f"  {SCHEMA_BRONZE}.train_fd004 : {train_bronze.count():,} rows, {len(train_bronze.columns)} cols")
print(f"  {SCHEMA_BRONZE}.test_fd004  : {test_bronze.count():,} rows, {len(test_bronze.columns)} cols")
print(f"  {SCHEMA_BRONZE}.rul_fd004    : {rul_bronze.count():,} rows")
print("\nSample (train_fd004):")
display(spark.table(f"{SCHEMA_BRONZE}.train_fd004").limit(5))

# COMMAND ----------

# DBTITLE 1,Silver — K-Means & Z-Score Normalization
# ============================================================
# SILVER LAYER — Step 1: K-Means Clustering & Z-Score Normalization
# รัน K-Means (Spark ML, k=6) เพื่อแบ่ง Operating Regimes
# แล้วทำ Z-score normalization แยกตาม regime โดยใช้สถิติจาก Train
# ============================================================

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import mean as spark_mean, stddev, col

# --- Read from Bronze ---
bronze_train = spark.table(f"{SCHEMA_BRONZE}.train_fd004").drop("ingestion_timestamp")
bronze_test  = spark.table(f"{SCHEMA_BRONZE}.test_fd004").drop("ingestion_timestamp")

# --- K-Means Clustering (Spark ML, k=6, seed=42) ---
assembler = VectorAssembler(inputCols=SETTINGS_COLS, outputCol="_settings_vec")
train_vec = assembler.transform(bronze_train)

kmeans = KMeans(k=N_REGIMES, seed=42, featuresCol="_settings_vec",
                predictionCol="op_regime", maxIter=300)
kmeans_model = kmeans.fit(train_vec)

train_silver = kmeans_model.transform(train_vec).drop("_settings_vec")
test_vec = assembler.transform(bronze_test)
test_silver = kmeans_model.transform(test_vec).drop("_settings_vec")

print(f"✓ K-Means: {N_REGIMES} regimes assigned (seed=42)")
display(train_silver.groupBy("op_regime").count().orderBy("op_regime"))

# --- Z-Score Normalization per Regime ---
# คำนวณ mu/sigma ราย regime จาก Train (6 rows — tiny lookup table)
regime_stats = train_silver.groupBy("op_regime").agg(
    *[spark_mean(c).alias(f"{c}_mu") for c in USEFUL_SENSORS],
    *[stddev(c).alias(f"{c}_sigma") for c in USEFUL_SENSORS]
)

# Join stats แล้วคำนวณ Z-score
train_silver = train_silver.join(regime_stats, "op_regime")
test_silver  = test_silver.join(regime_stats, "op_regime")

for s in USEFUL_SENSORS:
    train_silver = train_silver.withColumn(
        f"{s}_norm", (col(s) - col(f"{s}_mu")) / col(f"{s}_sigma")
    )
    test_silver = test_silver.withColumn(
        f"{s}_norm", (col(s) - col(f"{s}_mu")) / col(f"{s}_sigma")
    )
    train_silver = train_silver.drop(f"{s}_mu", f"{s}_sigma")
    test_silver  = test_silver.drop(f"{s}_mu", f"{s}_sigma")

print(f"✓ Z-score normalization: {len(USEFUL_SENSORS)} sensors → _norm columns")

# COMMAND ----------

# DBTITLE 1,Silver — Rolling & Slope Features (Window)
# ============================================================
# SILVER LAYER — Step 2: Rolling Features & Slope (PySpark Window)
# Rolling Mean & Std (window=5,10,20) บน 15 normalized sensors
# Rate of Change (Slope) ของ 6 key sensors ที่ lag=1,5
# ใช้ Window functions แบบ distributed — ไม่ต้อง collect to driver
# ============================================================

from pyspark.sql.window import Window
from pyspark.sql.functions import avg, stddev as spark_stddev, lag, col

norm_cols = [f"{s}_norm" for s in USEFUL_SENSORS]

# --- Rolling Mean & Rolling Std ---
for w_size in WINDOWS:
    w = Window.partitionBy("unit_id").orderBy("cycle").rowsBetween(-(w_size - 1), 0)
    for nc in norm_cols:
        train_silver = train_silver.withColumn(f"{nc}_rmean{w_size}", avg(col(nc)).over(w))
        train_silver = train_silver.withColumn(f"{nc}_rstd{w_size}", spark_stddev(col(nc)).over(w))
        test_silver  = test_silver.withColumn(f"{nc}_rmean{w_size}", avg(col(nc)).over(w))
        test_silver  = test_silver.withColumn(f"{nc}_rstd{w_size}", spark_stddev(col(nc)).over(w))

# Fill NaN std (first row ของแต่ละ window) ด้วย 0
rstd_cols = [f"{nc}_rstd{w}" for w in WINDOWS for nc in norm_cols]
train_silver = train_silver.na.fill(0, rstd_cols)
test_silver  = test_silver.na.fill(0, rstd_cols)

n_rolling = len(WINDOWS) * len(norm_cols) * 2
print(f"✓ Rolling features: {n_rolling} columns (mean + std, windows {WINDOWS})")

# --- Slope (Rate of Change) ---
for ks in KEY_SENSORS:
    nc = f"{ks}_norm"
    for lag_val in [1, 5]:
        w_lag = Window.partitionBy("unit_id").orderBy("cycle")
        train_silver = train_silver.withColumn(
            f"{nc}_slope{lag_val}", col(nc) - lag(col(nc), lag_val).over(w_lag)
        )
        test_silver = test_silver.withColumn(
            f"{nc}_slope{lag_val}", col(nc) - lag(col(nc), lag_val).over(w_lag)
        )

slope_cols = [f"{ks}_norm_slope{lag}" for ks in KEY_SENSORS for lag in [1, 5]]
train_silver = train_silver.na.fill(0, slope_cols)
test_silver  = test_silver.na.fill(0, slope_cols)

print(f"✓ Slope features: {len(slope_cols)} columns (6 key sensors × 2 lags)")

# COMMAND ----------

# DBTITLE 1,Install holidays package
# MAGIC %pip install holidays

# COMMAND ----------

# DBTITLE 1,Silver — RUL, Health State & Calendar
# ============================================================
# SILVER LAYER — Step 3: RUL, Health State & Calendar Features
# - Ground Truth RUL (piecewise linear, capped at 130)
# - Health State bins (Critical / Degraded / Healthy)
# - Synthetic timestamps → calendar features (is_weekend, is_holiday, is_peak_season)
# - Weather placeholder columns (for model compatibility)
# ============================================================

from pyspark.sql.functions import (
    max as spark_max, when, col, lit, to_date, date_format,
    dayofweek, month as month_col, rand, date_add,
    unix_timestamp, from_unixtime
)

# --- RUL (Ground Truth) ---
for label, df in [("train", train_silver), ("test", test_silver)]:
    max_cycle = df.groupBy("unit_id").agg(spark_max("cycle").alias("_max_cycle"))
    df = df.join(max_cycle, "unit_id")
    df = df.withColumn("RUL_raw", col("_max_cycle") - col("cycle"))
    df = df.withColumn("RUL", when(col("RUL_raw") > RUL_CAP, RUL_CAP).otherwise(col("RUL_raw")))
    df = df.drop("_max_cycle", "RUL_raw")
    if label == "train":
        train_silver = df
    else:
        test_silver = df

print(f"✓ RUL calculated (capped at {RUL_CAP} cycles)")

# --- Health State ---
for label, df in [("train", train_silver), ("test", test_silver)]:
    df = df.withColumn("health_state",
        when(col("RUL") <= 30, "Critical (High Risk)")
        .when(col("RUL") <= 80, "Degraded (Medium Risk)")
        .otherwise("Healthy (Low Risk)")
    )
    if label == "train":
        train_silver = df
    else:
        test_silver = df

print("✓ Health state assigned (3-class: Critical / Degraded / Healthy)")

# --- Synthetic Timestamps & Calendar Features ---
# Generate staggered commissioning dates (seed=42 for reproducibility)
all_units = train_silver.select("unit_id").distinct()
commissioning = all_units.withColumn(
    "_comm_date",
    date_add(lit("2024-01-01"), (rand(42) * 548).cast("int"))  # 548 ≈ days in 2024-2025 minus buffer
)

for label, df in [("train", train_silver), ("test", test_silver)]:
    df = df.join(commissioning, "unit_id")
    # 1 cycle = 8 hours → timestamp (via unix timestamp arithmetic)
    df = df.withColumn("timestamp",
        from_unixtime(unix_timestamp(col("_comm_date")) + col("cycle") * 8 * 3600)
    )
    df = df.withColumn("date", to_date(col("timestamp")))
    # Weekday/Weekend (dayofweek: 1=Sun ... 7=Sat)
    df = df.withColumn("is_weekend", dayofweek(col("date")).isin([1, 7]))
    # Peak season
    df = df.withColumn("_md", date_format(col("date"), "MM-dd"))
    df = df.withColumn("is_peak_season",
        when(col("_md").between("06-15", "08-15"), True)
        .when(col("_md").between("11-15", "11-28"), True)
        .when(col("_md") >= "12-15", True)
        .when(col("_md") <= "01-02", True)
        .otherwise(False)
    )
    df = df.drop("_comm_date", "_md")
    if label == "train":
        train_silver = df
    else:
        test_silver = df

# --- Holidays (US/Illinois) ---
import holidays as hl
us_holidays = hl.US(years=[2024, 2025], state='IL')
holiday_rows = [(d, name) for d, name in sorted(us_holidays.items())]
holiday_df = spark.createDataFrame(holiday_rows, ["_hdate", "_hname"])
holiday_df = holiday_df.withColumn("_hdate", to_date(col("_hdate")))

for label, df in [("train", train_silver), ("test", test_silver)]:
    df = df.join(holiday_df, df["date"] == holiday_df["_hdate"], "left")
    df = df.withColumn("is_holiday", col("_hname").isNotNull())
    df = df.drop("_hdate", "_hname")
    if label == "train":
        train_silver = df
    else:
        test_silver = df

# --- Weather placeholder columns (for model compatibility) ---
# TODO: Integrate NASA POWER weather data in a future enhancement
weather_placeholders = [
    'accumulated_high_temp_cycles',
    'rolling_std_temp_15',
    'weather_stress_score',
    'cumulative_weather_stress',
]
for wp in weather_placeholders:
    train_silver = train_silver.withColumn(wp, lit(0).cast("double"))
    test_silver  = test_silver.withColumn(wp, lit(0).cast("double"))

print("✓ Calendar features: is_weekend, is_holiday, is_peak_season")
print(f"✓ Weather placeholders: {weather_placeholders} (TODO: integrate NASA POWER API)")
print(f"\n  Train: {train_silver.count():,} rows, {len(train_silver.columns)} columns")
print(f"  Test:  {test_silver.count():,} rows, {len(test_silver.columns)} columns")

# COMMAND ----------

# DBTITLE 1,Silver — Write to Delta Tables
# ============================================================
# SILVER LAYER — Write to Delta Tables
# บันทึก Silver DataFrame ลง Delta Table พร้อม schema evolution
# ============================================================

train_silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true")\
    .saveAsTable(f"{SCHEMA_SILVER}.engine_features_train")
test_silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true")\
    .saveAsTable(f"{SCHEMA_SILVER}.engine_features_test")

print("✓ Silver Layer: 2 Delta tables written")
print(f"  {SCHEMA_SILVER}.engine_features_train : {train_silver.count():,} rows")
print(f"  {SCHEMA_SILVER}.engine_features_test  : {test_silver.count():,} rows")

# Verify — show key columns
display(
    spark.table(f"{SCHEMA_SILVER}.engine_features_train")
    .select("unit_id", "cycle", "op_regime", "RUL", "health_state",
            "is_weekend", "is_holiday", "is_peak_season")
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,Gold — ML Predictions via mapInPandas
# ============================================================
# GOLD LAYER — Step 1: ML Predictions via mapInPandas
# โหลด sklearn models จาก artifacts และทำนาย RUL + Health State
# ใช้ mapInPandas สำหรับ distributed model inference
# ============================================================

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col, current_timestamp
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType
import json

# --- Read Silver table (train) ---
silver_train = spark.table(f"{SCHEMA_SILVER}.engine_features_train")

# --- Get latest cycle per unit_id ---
w_latest = Window.partitionBy("unit_id").orderBy(col("cycle").desc())
latest_df = (silver_train
    .withColumn("_rn", row_number().over(w_latest))
    .filter(col("_rn") == 1)
    .drop("_rn")
)

print(f"Latest snapshot: {latest_df.count()} engines")

# --- Define output schema for mapInPandas ---
gold_schema = StructType([
    StructField("unit_id", IntegerType(), True),
    StructField("latest_cycle", IntegerType(), True),
    StructField("predicted_rul", DoubleType(), True),
    StructField("predicted_health_state", StringType(), True),
    StructField("actual_rul", DoubleType(), True),
    StructField("actual_health_state", StringType(), True),
    StructField("maintenance_action", StringType(), True),
    StructField("estimated_cost", DoubleType(), True),
    StructField("downtime_hours", DoubleType(), True),
    StructField("pipeline_run_timestamp", TimestampType(), True),
])

# --- Prediction + Cost function (runs on Spark workers) ---
def predict_and_cost(iterator):
    import joblib, json
    import pandas as pd
    import numpy as np

    # Load models inside function (workspace path accessible on all nodes)
    reg = joblib.load(f"{ARTIFACTS_DIR}/rul_regressor.joblib")
    clf = joblib.load(f"{ARTIFACTS_DIR}/health_state_classifier.joblib")
    with open(f"{ARTIFACTS_DIR}/features.json") as f:
        FEATURES = json.load(f)
    with open(f"{ARTIFACTS_DIR}/cost_parameters.json") as f:
        cost_params = json.load(f)

    for pdf in iterator:
        # Add missing feature columns as 0 (for model compatibility)
        for feat in FEATURES:
            if feat not in pdf.columns:
                pdf[feat] = 0

        X = pdf[FEATURES].fillna(0)

        # ML predictions
        pdf['predicted_rul'] = reg.predict(X)
        pdf['predicted_health_state'] = clf.predict(X)

        # Vectorized cost estimation
        rul = pdf['predicted_rul']
        health = pdf['predicted_health_state']
        is_critical = (health == 'Critical (High Risk)') | (rul <= 30)
        is_degraded = (health == 'Degraded (Medium Risk)') | ((rul > 30) & (rul <= 80))

        # Downtime hours
        dt = np.where(is_critical, np.random.uniform(48, 72, len(pdf)),
             np.where(is_degraded, np.random.uniform(8, 24, len(pdf)),
                      np.random.uniform(1, 2, len(pdf))))

        # Maintenance action
        actions = np.where(is_critical, 'Unplanned Failure / Emergency',
                  np.where(is_degraded, 'Proactive Planned Maintenance',
                           'Early Replacement (Preventive)'))

        # Cost
        cost = np.where(is_critical,
                dt * cost_params['unplanned_downtime_cost_per_hour'] + cost_params['unplanned_downtime_cost_per_event'],
                np.where(is_degraded,
                    dt * cost_params['labour_rate_per_hour'] + cost_params['proactive_maintenance_cost_per_event'],
                    dt * cost_params['labour_rate_per_hour'] + cost_params['early_replacement_waste_per_cycle']))

        result = pd.DataFrame({
            'unit_id': pdf['unit_id'].astype(int),
            'latest_cycle': pdf['cycle'].astype(int),
            'predicted_rul': pdf['predicted_rul'],
            'predicted_health_state': pdf['predicted_health_state'],
            'actual_rul': pdf['RUL'],
            'actual_health_state': pdf['health_state'],
            'maintenance_action': actions,
            'estimated_cost': cost,
            'downtime_hours': dt,
            'pipeline_run_timestamp': pd.Timestamp.now(),
        })
        yield result

# --- Apply predictions ---
gold_df = latest_df.mapInPandas(predict_and_cost, schema=gold_schema)

print("✓ ML predictions applied (RUL + Health State + Cost)")
display(gold_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Gold — Enrich & Write Delta Table
# ============================================================
# GOLD LAYER — Step 2: Enrich & Write to Delta Table
# เพิ่ม risk category และ health index
# บันทึก Gold table สำหรับระบบอื่นดึงไปใช้
# ============================================================

from pyspark.sql.functions import when, col, round as spark_round, lit

# --- Enrich with Risk Category & Health Index ---
gold_enriched = gold_df.withColumn(
    "risk_category",
    when(col("predicted_rul") <= 30, "🔴 Critical — Immediate Action Required")
    .when(col("predicted_rul") <= 80, "🟡 Degraded — Schedule Maintenance")
    .otherwise("🟢 Healthy — Monitor")
).withColumn(
    "health_index",
    spark_round((1.0 - (col("predicted_rul") / RUL_CAP)) * 100, 1)
).withColumn(
    "cost_category",
    when(col("estimated_cost") > 500000, "Very High")
    .when(col("estimated_cost") > 200000, "High")
    .when(col("estimated_cost") > 50000, "Medium")
    .otherwise("Low")
)

# --- Write to Delta Table ---
gold_enriched.write.format("delta").mode("overwrite").option("overwriteSchema", "true")\
    .saveAsTable(f"{SCHEMA_GOLD}.engine_health_status")

print(f"✓ Gold Layer: {SCHEMA_GOLD}.engine_health_status written")
print(f"  Engines: {gold_enriched.count()}")

# --- Display final Gold table ---
display(
    spark.table(f"{SCHEMA_GOLD}.engine_health_status")
    .select("unit_id", "latest_cycle", "predicted_rul", "predicted_health_state",
            "risk_category", "health_index", "maintenance_action",
            "estimated_cost", "cost_category")
    .orderBy(col("predicted_rul"))
    .limit(15)
)

# COMMAND ----------

# DBTITLE 1,Pipeline Summary — Medallion Overview
# ============================================================
# Pipeline Summary — Medallion Architecture Overview
# แสดงสรุปตารางทั้งหมด และ business summary stats
# ============================================================

from pyspark.sql.functions import (
    col, mean as spark_mean, count as spark_count,
    round as spark_round, sum as spark_sum
)

print("=" * 70)
print("  NASA Predictive Maintenance — Medallion Architecture Pipeline")
print("=" * 70)

# --- Bronze ---
print("\n🔵 BRONZE LAYER (Raw Data)")
for t in ["train_fd004", "test_fd004", "rul_fd004"]:
    cnt = spark.table(f"{SCHEMA_BRONZE}.{t}").count()
    print(f"  {SCHEMA_BRONZE}.{t:20s} : {cnt:>8,} rows")

# --- Silver ---
print("\n⚪ SILVER LAYER (Feature Engineered)")
for t in ["engine_features_train", "engine_features_test"]:
    df = spark.table(f"{SCHEMA_SILVER}.{t}")
    print(f"  {SCHEMA_SILVER}.{t:25s} : {df.count():>8,} rows, {len(df.columns)} columns")

# --- Gold ---
print("\n🟡 GOLD LAYER (Business Aggregates)")
gold = spark.table(f"{SCHEMA_GOLD}.engine_health_status")
print(f"  {SCHEMA_GOLD}.engine_health_status : {gold.count()} engines")

# Business summary
print("\n" + "-" * 70)
print("  Business Summary")
print("-" * 70)

display(
    gold.groupBy("predicted_health_state").agg(
        spark_round(spark_mean("predicted_rul"), 1).alias("avg_predicted_rul"),
        spark_round(spark_mean("estimated_cost"), 0).alias("avg_cost_usd"),
        spark_round(spark_mean("downtime_hours"), 1).alias("avg_downtime_hrs"),
        spark_count("unit_id").alias("n_engines"),
    ).orderBy(col("avg_predicted_rul"))
)

# Cost by action
display(
    gold.groupBy("maintenance_action").agg(
        spark_count("unit_id").alias("n_engines"),
        spark_round(spark_sum("estimated_cost"), 0).alias("total_cost_usd"),
        spark_round(spark_mean("downtime_hours"), 1).alias("avg_downtime_hrs"),
    ).orderBy(col("total_cost_usd").desc())
)

print("\n✅ Pipeline complete — Bronze → Silver → Gold → Delta Lake tables ready")