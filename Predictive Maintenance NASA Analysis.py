# Databricks notebook source
# MAGIC %md ***Data Load***
# MAGIC

# COMMAND ----------

# DBTITLE 1,List Files in Workspace Directory
#เริ่มทำวันที่ 10/23/2026
import os

# List files in the workspace directory (Databricks equivalent of /kaggle/input)
workspace_dir = "/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data"

print("=== Files in workspace directory ===")
if os.path.exists(workspace_dir):
    for item in os.listdir(workspace_dir):
        full_path = os.path.join(workspace_dir, item)
        size = os.path.getsize(full_path) if os.path.isfile(full_path) else "-"
        print(f"{full_path}  ({size} bytes)")
else:
    print(f"Directory not found: {workspace_dir}")

print("\n=== Tip: Upload the NASA dataset ===")
print("1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
print("2. Upload the CSV files to this folder using 'Upload data' in the Databricks sidebar")
print("3. Re-run this cell to see the uploaded files listed above")



# COMMAND ----------

# DBTITLE 1,Install plotly and seaborn
# MAGIC %pip install plotly seaborn

# COMMAND ----------

# DBTITLE 1,Import Libraries for Data Visualization and Warnings Control
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# DBTITLE 1,Define Columns List Including Unit Cycle and Sensors
cols = ['unit_id', 'cycle',
        'op_cond_1', 'op_cond_2', 'op_cond_3'] + \
       [f'sensor_{i}' for i in range(1, 22)]

# COMMAND ----------

# DBTITLE 1,Load Remaining Useful Life Data from Text File
rul_df = pd.read_csv('/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/CMaps/RUL_FD004.txt',sep = '\s+',header = None,names =['RUL'])
rul_df

# COMMAND ----------

# DBTITLE 1,Load Predictive Maintenance Training Data from Text Fil ...
train_df = pd.read_csv('/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/CMaps/train_FD004.txt',sep = '\s+',header = None,names = cols)
train_df

# COMMAND ----------

# DBTITLE 1,Examine DataFrame Structure and Summary Statistics
train_df.info() 

# COMMAND ----------

# DBTITLE 1,Summarize DataFrame Characteristics and Descriptive Sta ...
train_df.describe()

# COMMAND ----------

# DBTITLE 1,Load Predictive Maintenance Test Dataset from Text File
test_df = pd.read_csv('/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/CMaps/test_FD004.txt',sep = '\s+',header = None,names= cols)
test_df

# COMMAND ----------

# DBTITLE 1,Display DataFrame Structure and Metadata Summary
test_df.info()

# COMMAND ----------

# DBTITLE 1,Display Shapes and Sample Rows of Training DataFrames
print("Train shape:", train_df.shape)   
print("Test shape:", test_df.shape)     
print("RUL shape:", rul_df.shape)       

print("\nFirst few rows of train data:")
train_df.head()

# COMMAND ----------

# DBTITLE 1,Check for NULL Values in Train and Test DataFrames
print('Train_df NULL Value: ',train_df.isnull().sum().sum())
print('Test_df NULL Value: ',test_df.isnull().sum().sum())

# COMMAND ----------

# MAGIC %md ***Data exploreation***
# MAGIC

# COMMAND ----------

# DBTITLE 1,Data Integrity & Cross-File Statistical Check
# ============================================================
# 1. Lifetime Distribution (Train) — ภาพรวมอายุขัยเครื่องยนต์
# ============================================================
train_lifetime = train_df.groupby('unit_id')['cycle'].max()

print("=" * 60)
print("1. Lifetime Distribution — Train (max_cycle ของแต่ละเครื่อง)")
print("=" * 60)
print(f"  จำนวนเครื่องยนต์ทั้งหมด: {len(train_lifetime)} เครื่อง")
print(f"  Min    : {train_lifetime.min()} รอบ  (เครื่องที่พังเร็วที่สุด)")
print(f"  Max    : {train_lifetime.max()} รอบ  (เครื่องที่ทนที่สุด)")
print(f"  Median : {train_lifetime.median():.1f} รอบ")
print(f"  Mean   : {train_lifetime.mean():.2f} รอบ")
print(f"  Std    : {train_lifetime.std():.2f} รอบ  → อายุขัยมีความแปรปรวนกว้างมาก")

# ============================================================
# 2. Cross-File Validation — ตรวจสอบความสัมพันธ์ระหว่าง 3 ไฟล์
# ============================================================
train_units = train_df['unit_id'].nunique()
test_units = test_df['unit_id'].nunique()
rul_rows = len(rul_df)

print("\n" + "=" * 60)
print("2. Cross-File Validation — Train / Test / RUL")
print("=" * 60)
print(f"  Unique unit_id ใน train_FD004 : {train_units}  (คาดหวัง 249)")
print(f"  Unique unit_id ใน test_FD004  : {test_units}  (คาดหวัง 248 หรือ 249)")
print(f"  จำนวนแถวใน RUL_FD004         : {rul_rows}  (ต้องตรงกับจำนวนเครื่องใน Test)")
print(f"  → Test units == RUL rows? {test_units == rul_rows}")

# ============================================================
# 3. Test Total Lifetime = max_cycle_test + RUL Ground Truth
#    และเทียบการกระจายตัวกับ Train
# ============================================================
test_max_cycle = test_df.groupby('unit_id')['cycle'].max().reset_index()
test_max_cycle.columns = ['unit_id', 'max_cycle_test']

# RUL_FD004.txt เรียงตาม unit_id 1, 2, 3, ... → ใช้ .values เพื่อ align
test_max_cycle['RUL_ground_truth'] = rul_df['RUL'].values
test_max_cycle['total_lifetime'] = test_max_cycle['max_cycle_test'] + test_max_cycle['RUL_ground_truth']

test_total = test_max_cycle['total_lifetime']

print("\n" + "=" * 60)
print("3. Test Total Lifetime (max_cycle_test + RUL Ground Truth)")
print("=" * 60)
print(f"  Min    : {test_total.min()} รอบ")
print(f"  Max    : {test_total.max()} รอบ")
print(f"  Median : {test_total.median():.1f} รอบ")
print(f"  Mean   : {test_total.mean():.2f} รอบ")
print(f"  Std    : {test_total.std():.2f} รอบ")

# ============================================================
# สรุปเทียบการกระจายตัว: Train vs Test Total Lifetime
# ============================================================
print("\n" + "=" * 60)
print("4. Comparison: Train Lifetime vs Test Total Lifetime")
print("=" * 60)

comparison = pd.DataFrame({
    'Metric'    : ['Min', 'Max', 'Median', 'Mean', 'Std'],
    'Train'     : [train_lifetime.min(), train_lifetime.max(),
                   train_lifetime.median(), train_lifetime.mean(),
                   train_lifetime.std()],
    'Test_Total': [test_total.min(), test_total.max(),
                   test_total.median(), test_total.mean(),
                   test_total.std()]
})
comparison['Train']      = comparison['Train'].round(2)
comparison['Test_Total'] = comparison['Test_Total'].round(2)
display(comparison)

print("\nสรุป: หาก Train และ Test_Total อยู่ในช่วงใกล้เคียงกัน")
print("→ แสดงว่าการแบ่ง Train/Test มาจากประชากรเดียวกัน (same distribution)")

# COMMAND ----------

# DBTITLE 1,3D & 2D Scatter - 6 Operating Conditions
import plotly.express as px

# ============================================================
# 0. Identify unique operating conditions (expect 6)
#    Round values to remove sensor noise before grouping
# ============================================================
train_plot = train_df.copy()
train_plot['cond_1_r'] = train_plot['op_cond_1'].round(0)
train_plot['cond_2_r'] = train_plot['op_cond_2'].round(2)
train_plot['cond_3_r'] = train_plot['op_cond_3'].round(0)

settings = train_plot[['cond_1_r', 'cond_2_r', 'cond_3_r']].drop_duplicates().sort_values('cond_1_r').reset_index(drop=True)
settings['condition'] = settings.index + 1

print('=' * 60)
print('Unique Operating Conditions (Settings 1-3, rounded)')
print('=' * 60)
print(f'Total conditions: {len(settings)}')
print(settings.to_string(index=False))

# Assign condition label using merge on rounded keys
cond_map = settings.set_index(['cond_1_r', 'cond_2_r', 'cond_3_r'])['condition'].to_dict()
train_plot['condition'] = list(map(
    lambda r: cond_map.get((r['cond_1_r'], r['cond_2_r'], r['cond_3_r']), -1),
    train_plot.to_dict('records')
))

# Sample for plotting performance (3,000 points from 61,249 rows)
sample_df = train_plot.sample(n=3000, random_state=42)

# ============================================================
# 1. 3D Scatter Plot - Settings 1, 2, 3
# ============================================================
fig_3d = px.scatter_3d(
    sample_df,
    x='op_cond_1',
    y='op_cond_2',
    z='op_cond_3',
    color='condition',
    title='3D Scatter: 6 Operating Conditions (Altitude x Mach x Throttle)',
    labels={
        'op_cond_1': 'Setting 1 - Altitude',
        'op_cond_2': 'Setting 2 - Mach Number',
        'op_cond_3': 'Setting 3 - Throttle-Resolver Angle'
    },
    opacity=0.6,
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={'condition': sorted(sample_df['condition'].unique())}
)
fig_3d.update_traces(marker_size=3)
fig_3d.update_layout(width=900, height=700, margin=dict(l=0, r=0, t=50, b=0))
fig_3d.show()

# ============================================================
# 2. 2D Scatter Plots - pairwise settings
# ============================================================
pairs = [
    ('op_cond_1', 'op_cond_2', 'Altitude', 'Mach Number'),
    ('op_cond_1', 'op_cond_3', 'Altitude', 'Throttle-Resolver Angle'),
    ('op_cond_2', 'op_cond_3', 'Mach Number', 'Throttle-Resolver Angle')
]

for x_col, y_col, x_label, y_label in pairs:
    fig_2d = px.scatter(
        sample_df,
        x=x_col,
        y=y_col,
        color='condition',
        title=f'2D Scatter: {x_label} vs {y_label}',
        labels={x_col: x_label, y_col: y_label, 'condition': 'Condition'},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set1,
        category_orders={'condition': sorted(sample_df['condition'].unique())}
    )
    fig_2d.update_traces(marker_size=4)
    fig_2d.update_layout(width=700, height=450)
    fig_2d.show()

print('\n' + '=' * 60)
print('Observation Summary:')
print('=' * 60)
print('If points form 6 distinct spherical clusters separated clearly,')
print('it confirms the aircraft was programmed to cycle between 6 fixed conditions')
print('(e.g. sea level, 10K ft, 20K ft, 25K ft, 35K ft, cruise)')
print('rather than flying in a continuous free manner.')

# COMMAND ----------

# DBTITLE 1,Dead Sensors Detection - Std & Variance Check
# ============================================================
# Dead Sensors (Flatlines) - คัดกรองเซนเซอร์ที่ค่าไม่ขยับ
# คำนวณ Std ทั้งแบบรวม และแบบแยกตาม operating condition
# (FD004 มี 6 conditions → เซนเซอร์บางตัวขยับตาม condition
#  แต่เป็น flatline ภายใน condition เดียวกัน → ไม่บอกการเสื่อม)
# ============================================================

sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# สร้าง condition key จากค่า settings ที่ rounded
train_eval = train_df.copy()
train_eval['cond_key'] = (
    train_eval['op_cond_1'].round(0).astype(int).astype(str) + '_' +
    train_eval['op_cond_2'].round(2).astype(str) + '_' +
    train_eval['op_cond_3'].round(0).astype(int).astype(str)
)

# คำนวณ Std แบบรวม และ Std ภายในแต่ละ condition
rows = []
for c in sensor_cols:
    overall_std = train_eval[c].std()
    overall_var = train_eval[c].var()
    n_unique_overall = train_eval[c].nunique()
    # Std ภายในแต่ละ condition
    within_std = train_eval.groupby('cond_key')[c].std()
    mean_within_std = within_std.mean()
    max_within_std = within_std.max()
    rows.append({
        'sensor': c,
        'overall_std': overall_std,
        'overall_var': overall_var,
        'n_unique': n_unique_overall,
        'mean_within_std': mean_within_std,
        'max_within_std': max_within_std,
    })

sensor_stats = pd.DataFrame(rows)

# ตัดสินใจ:
#  1) overall_std == 0 → ค่าคงที่ตลอดกาล → DROP
#  2) max_within_std < 0.01 → flatline ภายในทุก condition → DROP
#     (เซนเซอร์ขยับตาม condition แต่ไม่ขยับภายใน condition → ไม่บอกการเสื่อม)
sensor_stats['is_dead'] = (
    (sensor_stats['overall_std'] == 0) |
    (sensor_stats['max_within_std'] < 0.01)
)
sensor_stats['drop?'] = sensor_stats['is_dead'].map({True: 'DROP', False: 'keep'})

print('=' * 80)
print('Sensor Statistics - Overall Std + Within-Condition Std')
print('=' * 80)
display(sensor_stats.round(6))

# ============================================================
# สรุปเซนเซอร์ที่ตายแล้ว
# ============================================================
dead_sensors = sensor_stats[sensor_stats['is_dead']]['sensor'].tolist()
keep_sensors = sensor_stats[~sensor_stats['is_dead']]['sensor'].tolist()

print('=' * 80)
print('Summary')
print('=' * 80)
print(f'  Total sensors       : {len(sensor_cols)}')
print(f'  Dead sensors (DROP) : {len(dead_sensors)}  -> {dead_sensors}')
print(f'  Useful sensors      : {len(keep_sensors)}  -> {keep_sensors}')

# ============================================================
# Bar chart: max_within_std ของทุกเซนเซอร์ (เรียงจากน้อยไปมา)
# ============================================================
sensor_sorted = sensor_stats.sort_values('max_within_std')
fig = px.bar(
    sensor_sorted,
    x='sensor',
    y='max_within_std',
    color='drop?',
    color_discrete_map={'DROP': 'red', 'keep': 'steelblue'},
    title='Max Within-Condition Std (sorted) - Red = Dead / Flatline within each condition',
    labels={'sensor': 'Sensor', 'max_within_std': 'Max Std Within Condition', 'drop?': 'Status'},
)
fig.update_layout(width=900, height=450, xaxis_tickangle=-45)
fig.show()

# ============================================================
# เก็บรายชื่อ dead sensors ไว้ใช้ในเซลล์ถัดไป
# ============================================================
DEAD_SENSORS = dead_sensors
USEFUL_SENSORS = keep_sensors
print(f'\nDEAD_SENSORS = {DEAD_SENSORS}')
print(f'USEFUL_SENSORS = {USEFUL_SENSORS}')

# COMMAND ----------

# DBTITLE 1,Operating Regime Clustering - K-Means k=6
from sklearn.cluster import KMeans

# ============================================================
# Operating Regime Clustering - K-Means (k=6) บน Settings 1-3
# ============================================================

settings_cols = ['op_cond_1', 'op_cond_2', 'op_cond_3']

# --- Fit K-Means บน Train ---
X_train = train_df[settings_cols].values
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
train_df['op_regime'] = kmeans.fit_predict(X_train)

# --- Predict บน Test (ใช้ model เดียวกัน) ---
X_test = test_df[settings_cols].values
test_df['op_regime'] = kmeans.predict(X_test)

# ============================================================
# สรุป centroids ของแต่ละ regime
# ============================================================
centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=['Altitude', 'Mach_Number', 'Throttle_Angle']
)
centroids.index.name = 'regime'

print('=' * 70)
print('K-Means Cluster Centroids (k=6)')
print('=' * 70)
display(centroids.round(4))

# ============================================================
# สรุปจำนวนแถวในแต่ละ regime
# ============================================================
train_counts = train_df['op_regime'].value_counts().sort_index()
test_counts  = test_df['op_regime'].value_counts().sort_index()

regime_summary = pd.DataFrame({
    'regime': range(6),
    'centroid_alt': centroids['Altitude'].round(2).values,
    'centroid_mach': centroids['Mach_Number'].round(4).values,
    'centroid_thr': centroids['Throttle_Angle'].round(1).values,
    'train_rows': train_counts.values,
    'test_rows': test_counts.reindex(range(6), fill_value=0).values,
})

print('=' * 70)
print('Regime Distribution - Train vs Test')
print('=' * 70)
display(regime_summary)

# ============================================================
# 3D Scatter: สีตาม K-Means regime (sample 3,000 จุด)
# ============================================================
sample_df = train_df.sample(n=3000, random_state=42)

fig = px.scatter_3d(
    sample_df,
    x='op_cond_1',
    y='op_cond_2',
    z='op_cond_3',
    color='op_regime',
    title='K-Means Clustering (k=6) - Operating Regimes',
    labels={
        'op_cond_1': 'Setting 1 - Altitude',
        'op_cond_2': 'Setting 2 - Mach Number',
        'op_cond_3': 'Setting 3 - Throttle-Resolver Angle',
        'op_regime': 'Regime (0-5)'
    },
    opacity=0.6,
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={'op_regime': list(range(6))}
)
fig.update_traces(marker_size=3)
fig.update_layout(width=900, height=700, margin=dict(l=0, r=0, t=50, b=0))
fig.show()

print('\nNew column "op_regime" (0-5) added to both train_df and test_df')
print(f'Train op_regime distribution: {dict(train_counts)}')
print(f'Test  op_regime distribution: {dict(test_counts)}')

# COMMAND ----------

# DBTITLE 1,Regime-based Normalization - Z-score per op_regime
# ============================================================
# Regime-based Normalization (Z-score แยกตาม op_regime)
# ใช้สถิติจาก Train คำนวณ mu/sigma ราย regime
# แล้วแปลงค่าเซนเซอร์ 15 ตัวที่เหลือให้เป็น Z-score
# ============================================================

# Useful sensors (จาก Cell 17 Dead Sensors analysis)
USEFUL_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

# --- Step 1: คำนวณ mu และ sigma ราย regime จาก Train ---
regime_mu = train_df.groupby('op_regime')[USEFUL_SENSORS].mean()
regime_sigma = train_df.groupby('op_regime')[USEFUL_SENSORS].std()

print('=' * 70)
print('Mu (mean) per Regime — 15 useful sensors')
print('=' * 70)
display(regime_mu.round(4))

print('\n' + '=' * 70)
print('Sigma (std) per Regime — 15 useful sensors')
print('=' * 70)
display(regime_sigma.round(4))

# --- Step 2: แปลงค่าเป็น Z-score: (value - mu_regime) / sigma_regime ---
norm_cols = [s + '_norm' for s in USEFUL_SENSORS]

for regime in range(6):
    mask_tr = train_df['op_regime'] == regime
    mask_te = test_df['op_regime'] == regime
    for s in USEFUL_SENSORS:
        mu = regime_mu.loc[regime, s]
        sigma = regime_sigma.loc[regime, s]
        train_df.loc[mask_tr, s + '_norm'] = (train_df.loc[mask_tr, s] - mu) / sigma
        test_df.loc[mask_te, s + '_norm'] = (test_df.loc[mask_te, s] - mu) / sigma

print('\nNew columns added:')
print(f'  {len(norm_cols)} normalized sensor columns (_norm) in train_df & test_df')
print(f'  Example: {norm_cols[:3]} ...')

# --- Step 3: Before vs After Visualization ---
# เลือกเครื่องยนต์ที่มีอายุขัยยาวพอให้เห็นแนวโน้ม
sample_engines = [1, 10, 50, 100, 150, 200]

for demo_sensor in ['sensor_2', 'sensor_9']:
    norm_col = demo_sensor + '_norm'

    # BEFORE: raw values (ฟันปลาจาก 6 regimes)
    fig_before = px.line(
        train_df[train_df['unit_id'].isin(sample_engines)],
        x='cycle', y=demo_sensor, color='unit_id',
        title=f'BEFORE Normalization: {demo_sensor} (raw) — sawtooth from 6 regimes',
        labels={'cycle': 'Cycle', demo_sensor: f'{demo_sensor} (raw)', 'unit_id': 'Engine ID'},
    )
    fig_before.update_layout(width=900, height=350)
    fig_before.show()

    # AFTER: Z-score normalized (รวมบนระนาบเดียวกัน)
    fig_after = px.line(
        train_df[train_df['unit_id'].isin(sample_engines)],
        x='cycle', y=norm_col, color='unit_id',
        title=f'AFTER Regime-based Normalization: {demo_sensor} (Z-score) — degradation trend visible',
        labels={'cycle': 'Cycle', norm_col: f'{demo_sensor} (Z-score)', 'unit_id': 'Engine ID'},
    )
    fig_after.update_layout(width=900, height=350)
    fig_after.show()

print('\n' + '=' * 70)
print('Summary')
print('=' * 70)
print('Before: Sensor values jump between 6 baselines (sawtooth pattern)')
print('After:  All regimes aligned to mean=0, std=1 — degradation trend clearly visible')
print(f'Normalized columns: {norm_cols}')

# COMMAND ----------

# DBTITLE 1,Time-Series Feature Extraction - Rolling Stats & Slope
# ============================================================
# Time-Series Feature Extraction
# - Rolling Mean & Std (window = 5, 10, 20) บน 15 normalized sensors
# - Rate of Change (Slope) ของเซนเซอร์สำคัญ
# ============================================================

USEFUL_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_6', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]
norm_cols = [s + '_norm' for s in USEFUL_SENSORS]
WINDOWS = [5, 10, 20]

# เซนเซอร์สำคัญที่เกี่ยวกับ HPC (ใช้พล็อต slope ให้เห็นชัด)
KEY_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']

# ============================================================
# Step 1: Rolling Mean & Rolling Std (หน้าต่างย้อนหลัง)
# ============================================================
for w in WINDOWS:
    for nc in norm_cols:
        # Rolling Mean — ลด noise / ดูแนวโน้ม
        train_df[nc + f'_rmean{w}'] = train_df.groupby('unit_id')[nc].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        test_df[nc + f'_rmean{w}'] = test_df.groupby('unit_id')[nc].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        # Rolling Std — วัดความผันผวนสะสม
        train_df[nc + f'_rstd{w}'] = train_df.groupby('unit_id')[nc].transform(
            lambda x: x.rolling(window=w, min_periods=1).std()
        ).fillna(0)
        test_df[nc + f'_rstd{w}'] = test_df.groupby('unit_id')[nc].transform(
            lambda x: x.rolling(window=w, min_periods=1).std()
        ).fillna(0)

n_rolling = len(WINDOWS) * len(norm_cols) * 2
print('=' * 70)
print(f'Step 1: Rolling Features Created')
print('=' * 70)
print(f'  Windows         : {WINDOWS}')
print(f'  Sensors         : {len(norm_cols)} normalized sensors')
print(f'  Stats per window: rolling_mean, rolling_std')
print(f'  Total new cols  : {n_rolling} per dataframe (train & test)')

# ============================================================
# Step 2: Rate of Change (Slope) ของเซนเซอร์สำคัญ
# ============================================================
for ks in KEY_SENSORS:
    nc = ks + '_norm'
    # Slope = ค่าปัจจุบัน - ค่า N รอบก่อนหน้า
    for lag in [1, 5]:
        train_df[nc + f'_slope{lag}'] = train_df.groupby('unit_id')[nc].diff(lag).fillna(0)
        test_df[nc + f'_slope{lag}'] = test_df.groupby('unit_id')[nc].diff(lag).fillna(0)

n_slope = len(KEY_SENSORS) * 2 * 2  # 6 sensors x 2 lags x 2 dataframes
print('\n' + '=' * 70)
print(f'Step 2: Rate of Change (Slope) Features')
print('=' * 70)
print(f'  Key sensors : {KEY_SENSORS}')
print(f'  Lags        : [1, 5]')
print(f'  Total new cols: {len(KEY_SENSORS) * 2} per dataframe')

# ============================================================
# Step 3: สรุปจำนวนฟีเจอร์ทั้งหมด
# ============================================================
n_base = len(norm_cols)
print('\n' + '=' * 70)
print('Feature Summary')
print('=' * 70)
print(f'  Base normalized sensors       : {n_base}')
print(f'  Rolling mean/std features    : {n_rolling}')
print(f'  Slope features               : {len(KEY_SENSORS) * 2}')
print(f'  Total engineered features    : {n_base + n_rolling + len(KEY_SENSORS) * 2}')
print(f'  train_df shape               : {train_df.shape}')
print(f'  test_df shape                : {test_df.shape}')

# ============================================================
# Step 4: Visualization — Rolling Mean vs Raw (sensor_3_norm)
# ============================================================
sample_engines = [1, 50, 100, 200]
plot_df = train_df[train_df['unit_id'].isin(sample_engines)].copy()

fig = px.line(
    plot_df, x='cycle', y='sensor_3_norm', color='unit_id',
    title='sensor_3_norm (raw Z-score) — noisy',
    labels={'cycle': 'Cycle', 'sensor_3_norm': 'sensor_3 (Z-score)', 'unit_id': 'Engine'},
)
fig.update_layout(width=900, height=300)
fig.show()

for w in [5, 10, 20]:
    fig_r = px.line(
        plot_df, x='cycle', y=f'sensor_3_norm_rmean{w}', color='unit_id',
        title=f'sensor_3_norm — Rolling Mean (window={w}) — smoothed trend',
        labels={'cycle': 'Cycle', f'sensor_3_norm_rmean{w}': f'sensor_3 (rmean{w})', 'unit_id': 'Engine'},
    )
    fig_r.update_layout(width=900, height=300)
    fig_r.show()

# ============================================================
# Step 5: Visualization — Slope (sensor_3_norm_slope5)
# ============================================================
fig_s = px.line(
    plot_df, x='cycle', y='sensor_3_norm_slope5', color='unit_id',
    title='sensor_3_norm — Rate of Change (slope lag=5) — acceleration of degradation',
    labels={'cycle': 'Cycle', 'sensor_3_norm_slope5': 'slope (lag=5)', 'unit_id': 'Engine'},
)
fig_s.update_layout(width=900, height=300)
fig_s.show()

print('\nDone! All rolling & slope features added to train_df and test_df.')

# COMMAND ----------

# DBTITLE 1,Explanation: Why 117 Features in Cell 20
# MAGIC %md
# MAGIC # 📐 อธิบาย Cell 20: ทำไมฟีเจอร์ถึงเพิ่มขึ้นเป็น 117 ตัว?
# MAGIC
# MAGIC ## สูตรการคำนวณจำนวนฟีเจอร์
# MAGIC
# MAGIC | ประเภท | สูตร | จำนวน | ตัวอย่างคอลัมน์ |
# MAGIC |---|---|---|---|
# MAGIC | **Base Z-score** | 15 sensors × 1 | **15** | `sensor_2_norm`, `sensor_3_norm`, ... |
# MAGIC | **Rolling Mean** | 15 sensors × 3 windows | **45** | `sensor_2_norm_rmean5`, `sensor_2_norm_rmean10`, `sensor_2_norm_rmean20` |
# MAGIC | **Rolling Std** | 15 sensors × 3 windows | **45** | `sensor_2_norm_rstd5`, `sensor_2_norm_rstd10`, `sensor_2_norm_rstd20` |
# MAGIC | **Slope** | 6 key sensors × 2 lags | **12** | `sensor_3_norm_slope1`, `sensor_3_norm_slope5` |
# MAGIC | **รวม** | | **117** | |
# MAGIC
# MAGIC ## ทำไมต้องเพิ่มฟีเจอร์เยอะขนาดนี้?
# MAGIC
# MAGIC ### 1. ปัญหาของข้อมูลดิบ: Noise สูงมาก
# MAGIC เซนเซอร์ในเครื่องยนต์จริงมีสัญญาณรบกวน (sensor noise) สูง ค่าในแต่ละรอบบินกระโดดขึ้นลง ทำให้โมเดลเห็นแนวโน้มการเสื่อมไม่ได้ หากใช้แค่ค่าดิบ
# MAGIC
# MAGIC ### 2. Rolling Mean (45 ฟีเจอร์) — ทำไมต้อง 3 ขนาดหน้าต่าง?
# MAGIC - **window=5**: ลด noise เล็กน้อย ยังเห็นการเปลี่ยนแปลงละเอียด → เหมาะกับการตรวจจับการเสื่อมตั้งแต่ระยะเริ่มต้น
# MAGIC - **window=10**: สมดุลระหว่าง noise reduction และ responsiveness → ใช้สำหรับ trend ระดับกลาง
# MAGIC - **window=20**: เรียบมาก เห็นแนวโน้มระยะยาวชัด → ตรวจจับการเสื่อมสภาพสะสม
# MAGIC
# MAGIC โมเดลเลือกเองว่าจะใช้ขนาดหน้าต่างไหนเหมาะกับสภาพเครื่องยนต์ในขณะนั้น
# MAGIC
# MAGIC ### 3. Rolling Std (45 ฟีเจอร์) — ทำไมต้องวัดความผันผวน?
# MAGIC - เครื่องยนต์ที่ใกล้พังมักมีความผันผวนของเซนเซอร์สูงขึ้น (เครื่องยนต์สั่น เสียงดัง อุณหภูมิไม่เสถียร)
# MAGIC - Rolling Std จับสัญญาณนี้ได้ดี โดยที่ Rolling Mean มองไม่เห็น
# MAGIC - เปรียบเหมือนการวัด "ความปั่นป่วน" ไม่ใช่แค่ "ค่าเฉลี่ย"
# MAGIC
# MAGIC ### 4. Slope / Rate of Change (12 ฟีเจอร์) — ทำไมใช้แค่ 6 sensors?
# MAGIC - คำนวณเฉพาะเซนเซอร์สำคัญที่เกี่ยวกับ HPC (High-Pressure Compressor): `sensor_2, 3, 4, 7, 11, 15`
# MAGIC - **lag=1**: อัตราการเปลี่ยนแปลงระหว่างรอบบินติดกัน (สั้นมาก)
# MAGIC - **lag=5**: อัตราการเปลี่ยนแปลงในระยะ 5 รอบ (เห็นแนวโน้มเร่ง/ชะลอของการเสื่อม)
# MAGIC - ไม่คำนวณทุกตัวเพราะ Slope ของเซนเซอร์ที่ไม่สำคัญจะเป็น noise มากกว่าสัญญาณจริง
# MAGIC
# MAGIC ## สรุป
# MAGIC
# MAGIC ```
# MAGIC ฟีเจอร์ทั้งหมด 117 ตัว มาจาก:
# MAGIC
# MAGIC   15 (base Z-score)     — ค่าปัจจุบันหลัง normalize
# MAGIC + 45 (rolling mean)     — แนวโน้ม 3 มาตรา (short/medium/long)
# MAGIC + 45 (rolling std)      — ความผันผวนสะสม 3 มาตรา
# MAGIC + 12 (slope)            — อัตราเปลี่ยนแปลงของเซนเซอร์ HPC สำคัญ
# MAGIC = 117 ฟีเจอร์ทั้งหมด
# MAGIC ```
# MAGIC
# MAGIC จำนวนนี้ไม่ใช่เยอะเกินไปสำหรับ Machine Learning เพราะ:
# MAGIC - มีข้อมูล 61,249 แถว (ratio ~525:1 แถวต่อฟีเจอร์) → เพียงพอที่โมเดลจะเรียนรู้ได้โดยไม่ overfit
# MAGIC - โมเดลสมัยใหม่ (XGBoost, Random Forest) มี feature selection ในตัว → ฟีเจอร์ที่ไม่มีประโยชน์จะถูกลดน้ำหนักอัตโนมัติ
# MAGIC - สามารถใช้ Feature Importance หรือ PCA ลดมิติภายหลังได้

# COMMAND ----------

# DBTITLE 1,Target Labeling - RUL with Piecewise Clipping
# ============================================================
# Target Labeling - Remaining Useful Life (RUL)
# 1. RUL = max_cycle - cycle สำหรับแต่ละเครื่องยนต์ใน Train
# 2. Piecewise Linear Clipping: ตัดเพดานที่ RUL_MAX (130 cycles)
#    เพื่อป้องกันโมเดลจำค่าในช่วงที่เครื่องยังสมบูรณ์
# ============================================================

RUL_MAX = 130  # เพดาน RUL — ช่วงที่เครื่องยนต์ยังสมบูรณ์ ไม่มีการเสื่อมสภาพ (consistent with Cell 34 & Medallion Pipeline)

# --- Step 1: คำนวณ RUL จริง (Raw RUL) ---
train_df['RUL_raw'] = train_df.groupby('unit_id')['cycle'].transform(
    lambda x: x.max() - x
)

# --- Step 2: Piecewise Clipping (RUL clipped at RUL_MAX) ---
train_df['RUL'] = train_df['RUL_raw'].clip(upper=RUL_MAX)

# ============================================================
# สรุปสถิติ RUL ก่อนและหลัง clipping
# ============================================================
print('=' * 70)
print(f'Target Labeling - RUL (max clip = {RUL_MAX})')
print('=' * 70)

rul_stats = pd.DataFrame({
    'Metric'   : ['Min', 'Max', 'Mean', 'Median', 'Std'],
    'RUL_raw'  : [train_df['RUL_raw'].min(), train_df['RUL_raw'].max(),
                  train_df['RUL_raw'].mean(), train_df['RUL_raw'].median(),
                  train_df['RUL_raw'].std()],
    'RUL_clipped': [train_df['RUL'].min(), train_df['RUL'].max(),
                    train_df['RUL'].mean(), train_df['RUL'].median(),
                    train_df['RUL'].std()],
})
rul_stats['RUL_raw']    = rul_stats['RUL_raw'].round(2)
rul_stats['RUL_clipped'] = rul_stats['RUL_clipped'].round(2)
display(rul_stats)

# จำนวนแถวที่ถูก clip
n_clipped = (train_df['RUL_raw'] > RUL_MAX).sum()
pct_clipped = n_clipped / len(train_df) * 100
print(f'\nRows clipped at {RUL_MAX}: {n_clipped} ({pct_clipped:.1f}%)')
print(f'Rows kept as-is        : {len(train_df) - n_clipped} ({100 - pct_clipped:.1f}%)')

# ============================================================
# Visualization: RUL distribution & Piecewise function
# ============================================================
fig1 = px.histogram(
    train_df, x='RUL_raw', nbins=80,
    title='RUL Distribution: Raw (before clipping)',
    labels={'RUL_raw': 'RUL (raw)', 'count': 'Count'},
    color_discrete_sequence=['steelblue'],
)
fig1.add_vline(x=RUL_MAX, line_dash='dash', line_color='red',
               annotation_text=f'Clip at {RUL_MAX}')
fig1.update_layout(width=800, height=350)
fig1.show()

fig2 = px.histogram(
    train_df, x='RUL', nbins=80,
    title=f'RUL Distribution: After Piecewise Clipping (max={RUL_MAX})',
    labels={'RUL': 'RUL (clipped)', 'count': 'Count'},
    color_discrete_sequence=['indianred'],
)
fig2.update_layout(width=800, height=350)
fig2.show()

# ============================================================
# Piecewise Linear Function: แสดง concept ของ clipping
# ============================================================
import plotly.graph_objects as go

x_demo = list(range(0, 544))
y_raw = x_demo[::-1]  # RUL_raw = max_cycle - cycle
y_clipped = [min(r, RUL_MAX) for r in y_raw]

fig_pw = go.Figure()
fig_pw.add_trace(go.Scatter(x=x_demo, y=y_raw, mode='lines',
    name='RUL raw', line=dict(color='steelblue', width=2)))
fig_pw.add_trace(go.Scatter(x=x_demo, y=y_clipped, mode='lines',
    name=f'RUL clipped (max={RUL_MAX})', line=dict(color='red', width=2)))
fig_pw.add_hrect(y0=RUL_MAX, y1=max(y_raw), fillcolor='red', opacity=0.05,
    line_width=0, annotation_text='Flat zone (no degradation signal)')
fig_pw.update_layout(
    title='Piecewise Linear RUL Function',
    xaxis_title='Cycle', yaxis_title='RUL', width=800, height=400,
)
fig_pw.show()

# ============================================================
# ตัวอย่าง RUL ของเครื่องยนต์ 3 เครื่อง
# ============================================================
sample_engines = [1, 50, 100]
sample_rul = train_df[train_df['unit_id'].isin(sample_engines)][
    ['unit_id', 'cycle', 'RUL_raw', 'RUL']
].copy()
print('\nSample RUL for engines:', sample_engines)
display(sample_rul.head(10))

print('\nNew columns added to train_df:')
print(f'  RUL_raw  = max_cycle - cycle (unclipped)')
print(f'  RUL      = min(RUL_raw, {RUL_MAX})  (piecewise clipped target)')
print(f'\ntrain_df shape: {train_df.shape}')

# COMMAND ----------

# MAGIC %md เหตุผลที่ clip ที่ 125: ช่วง RUL > 125 เครื่องยนต์ยังสมบูรณ์ สัญญาณเซนเซอร์ไม่แสดงการเสื่อม → โมเดลไม่ควรจำค่าในช่วงนี้ ควรเรียนรู้แค่ช่วงที่มี degradation signal จริงๆ (RUL 0-125)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Synthetic Timestamp Generation - Staggered Commissioning & Cycle-to-Datetime
import numpy as np

# ============================================================
# Synthetic Timestamp Generation
# จำลองวันเริ่มต้นใช้งานของแต่ละเครื่องยนต์ (Staggered Entry)
# และแปลง cycle → timestamp สำหรับเชื่อม Weather/Calendar ในเซลล์ถัดไป
# ============================================================

# --- กรอบเวลาหลักของฝูงบิน (Fleet Operational Horizon) ---
FLEET_START = pd.Timestamp('2024-01-01')
FLEET_END   = pd.Timestamp('2025-12-31')

# --- สมมติฐาน: 1 cycle = 8 ชั่วโมง (เครื่องบินพาณิชย์ 2-4 ไฟลต์/วัน) ---
DELTA_HOURS = 8

# --- รวม unit_id ทั้งหมดจาก train และ test ---
all_units = sorted(set(train_df['unit_id'].unique()) | set(test_df['unit_id'].unique()))
n_units = len(all_units)

# --- สุ่มวันเริ่มต้น (Commissioning Date) แบบ Staggered Entry ---
# กระจายวันเริ่มตั้งแต่ FLEET_START ไปจนถึง 6 เดือนก่อน FLEET_END
# เพื่อให้ทุกเครื่องมีเวลาทำงานเพียงพอ
np.random.seed(42)
max_start_days = (FLEET_END - FLEET_START).days - 180  # เว้น buffer 6 เดือน
start_offsets = np.random.randint(0, max_start_days, size=n_units)

commissioning_dates = {
    unit: FLEET_START + pd.Timedelta(days=int(offset))
    for unit, offset in zip(all_units, start_offsets)
}

# --- แปลง cycle → timestamp สำหรับ train_df ---
train_df['start_date'] = train_df['unit_id'].map(commissioning_dates)
train_df['timestamp'] = train_df['start_date'] + pd.to_timedelta(
    train_df['cycle'] * DELTA_HOURS, unit='h'
)

# --- แปลง cycle → timestamp สำหรับ test_df (ใช้ commissioning date เดียวกัน) ---
test_df['start_date'] = test_df['unit_id'].map(commissioning_dates)
test_df['timestamp'] = test_df['start_date'] + pd.to_timedelta(
    test_df['cycle'] * DELTA_HOURS, unit='h'
)

# --- สกัดฟีเจอร์พื้นฐานจาก timestamp (ไว้ใช้ต่อกับ Calendar/Weather) ---
for df in [train_df, test_df]:
    df['date']        = df['timestamp'].dt.date
    df['month']       = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0=Mon ... 6=Sun
    df['hour']        = df['timestamp'].dt.hour

# ============================================================
# สรุปสถิติ
# ============================================================
comm_dates_list = list(commissioning_dates.values())

print('=' * 70)
print('Synthetic Timestamp Generation — Summary')
print('=' * 70)
print(f'  Fleet Horizon        : {FLEET_START.date()} → {FLEET_END.date()}')
print(f'  Total units          : {n_units}')
print(f'  Delta t per cycle    : {DELTA_HOURS} hours')
print(f'  Commissioning range  : {min(comm_dates_list).date()} → {max(comm_dates_list).date()}')
print(f'  Train timestamp range : {train_df["timestamp"].min()} → {train_df["timestamp"].max()}')
print(f'  Test  timestamp range : {test_df["timestamp"].min()} → {test_df["timestamp"].max()}')
print(f'\n  New columns: start_date, timestamp, date, month, day_of_week, hour')
print(f'  train_df shape: {train_df.shape}')
print(f'  test_df shape : {test_df.shape}')

# ============================================================
# ตารางวันเริ่มต้นของเครื่องยนต์ตัวอย่าง
# ============================================================
sample_comm = pd.DataFrame([
    {'unit_id': u, 'commissioning_date': d.date(), 'max_cycle_train': train_df[train_df['unit_id']==u]['cycle'].max()}
    for u, d in list(commissioning_dates.items())[:10]
])
print('\nSample Commissioning Dates (first 10 engines):')
display(sample_comm)

# ============================================================
# Visualization 1: การกระจายวันเริ่มต้น (Staggered Entry)
# ============================================================
fig1 = px.histogram(
    x=[d.date() for d in comm_dates_list],
    nbins=30,
    title='Commissioning Date Distribution (Staggered Entry) — 249 engines',
    labels={'x': 'Commissioning Date', 'count': 'Number of Engines'},
    color_discrete_sequence=['steelblue'],
)
fig1.update_layout(width=800, height=350)
fig1.show()

# ============================================================
# Visualization 2: Timeline ของเครื่องยนต์ 8 ลำ (Gantt-style)
# ============================================================
sample_engines = [1, 30, 60, 90, 120, 150, 180, 210]
timeline_rows = []
for unit in sample_engines:
    unit_data = train_df[train_df['unit_id'] == unit]
    timeline_rows.append({
        'unit_id': f'Engine {unit}',
        'start': unit_data['timestamp'].min(),
        'end':   unit_data['timestamp'].max(),
        'n_cycles': len(unit_data),
    })
timeline_df = pd.DataFrame(timeline_rows)

fig2 = px.timeline(
    timeline_df, x_start='start', x_end='end', y='unit_id',
    color='n_cycles',
    title='Engine Operational Timeline (sample 8 engines)',
    labels={'unit_id': 'Engine', 'n_cycles': 'Cycles'},
    color_continuous_scale='Viridis',
)
fig2.update_layout(width=900, height=400)
fig2.show()

# ============================================================
# Visualization 3: ตัวอย่าง timestamp ของ Engine 1 (10 แถวแรก)
# ============================================================
print('\nSample timestamps for Engine 1:')
display(train_df[train_df['unit_id'] == 1][
    ['unit_id', 'cycle', 'start_date', 'timestamp', 'date', 'month', 'day_of_week', 'hour']
].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ยิง API ดึงข้อมูลจริงจาก NASA POWER

# COMMAND ----------

# DBTITLE 1,NASA POWER Weather API Fetch & Merge with C-MAPSS
import pandas as pd
import requests

# ============================================================
# NASA POWER API — ดึงข้อมูลสภาพอากาศจริงและ Merge เข้า C-MAPSS
# ตัวแปร: T2M (อุณหภูมิ), RH2M (ความชื้น), PS (ความดัน), WS10M (ลม)
# ============================================================

# --- พิกัดฐานบินสมมติ: Chicago O'Hare (ศูนย์กลางการบิน) ---
LAT, LON = 41.97, -87.90

# --- ช่วงเวลาตาม Synthetic Timeline (Fleet Horizon 2024-2025) ---
WEATHER_START = "20240101"
WEATHER_END   = "20251231"

# --- ตัวแปรสภาพอากาศที่ดึง (ตรงกับโจทย์เครื่องยนต์ไอพ่น) ---
# T2M  = Temperature at 2 Meters (อุณหภูมิอากาศภายนอก)
# RH2M = Relative Humidity at 2 Meters (ความชื้นสัมพัทธ์)
# PS   = Surface Pressure (ความดันอากาศระดับพื้นดิน)
# WS10M= Wind Speed at 10 Meters (ความเร็วลม)
PARAMETERS = "T2M,RH2M,PS,WS10M"

print("=" * 70)
print("NASA POWER API — Weather Data Fetch")
print("=" * 70)
print(f"  Coordinates : ({LAT}, {LON}) — Chicago O'Hare")
print(f"  Date range  : {WEATHER_START} → {WEATHER_END}")
print(f"  Parameters  : {PARAMETERS}")

# --- ยิง Request ไปยัง NASA POWER API ---
url = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters={PARAMETERS}&community=RE&longitude={LON}&latitude={LAT}&"
    f"start={WEATHER_START}&end={WEATHER_END}&format=JSON"
)

try:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    print("  Status      : API request successful")
except Exception as e:
    print(f"  ERROR       : {e}")
    raise

# --- แปลงผลลัพธ์เป็น DataFrame รายวัน (Daily Weather Lookup) ---
weather_records = data["properties"]["parameter"]

weather_df = pd.DataFrame({
    "date": pd.to_datetime(list(weather_records["T2M"].keys()), format="%Y%m%d"),
    "nasa_ambient_temp":     list(weather_records["T2M"].values()),
    "nasa_humidity":         list(weather_records["RH2M"].values()),
    "nasa_surface_pressure": list(weather_records["PS"].values()),
    "nasa_wind_speed":       list(weather_records["WS10M"].values()),
})

# กรองค่า missing (-999 = missing ใน NASA POWER)
weather_df = weather_df[weather_df["nasa_ambient_temp"] != -999].copy()
weather_df = weather_df[weather_df["nasa_wind_speed"] != -999].copy()

print(f"\nWeather data fetched: {len(weather_df)} days")
print(f"  Date range  : {weather_df['date'].min().date()} → {weather_df['date'].max().date()}")
print(f"  Columns     : {list(weather_df.columns)}")
print("\nWeather Statistics:")
display(weather_df.describe().round(2))

# ============================================================
# Merge ข้อมูล Weather เข้ากับ C-MAPSS (Left Join ด้วย date)
# ============================================================
# สร้าง date key สำหรับ merge
train_df['_date_key'] = pd.to_datetime(train_df['date'])
test_df['_date_key']  = pd.to_datetime(test_df['date'])

# Left Join: ทุกแถวใน train/test จะได้สภาพอากาศของวันนั้น
train_df = train_df.merge(
    weather_df, left_on='_date_key', right_on='date', how='left', suffixes=('', '_w')
)
test_df = test_df.merge(
    weather_df, left_on='_date_key', right_on='date', how='left', suffixes=('', '_w')
)

# ทำความสะอาดคอลัมน์ซ้ำ
for df in [train_df, test_df]:
    cols_to_drop = [c for c in ['_date_key', 'date_w'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

# ตรวจสอบ missing
n_missing_tr = train_df['nasa_ambient_temp'].isna().sum()
n_missing_te = test_df['nasa_ambient_temp'].isna().sum()

print('=' * 70)
print('Merge Results — Weather × C-MAPSS')
print('=' * 70)
print(f"  train_df: {len(train_df):,} rows, {n_missing_tr} missing weather ({n_missing_tr/len(train_df)*100:.1f}%)")
print(f"  test_df : {len(test_df):,} rows, {n_missing_te} missing weather ({n_missing_te/len(test_df)*100:.1f}%)")
print(f"  train_df shape: {train_df.shape}")
print(f"  test_df shape : {test_df.shape}")

# ============================================================
# Visualization: Weather trends over Fleet Horizon
# ============================================================
fig = px.line(
    weather_df, x='date',
    y=['nasa_ambient_temp', 'nasa_humidity', 'nasa_wind_speed'],
    title="NASA POWER Weather Data — Chicago O'Hare (2024-2025)",
    labels={'date': 'Date', 'value': 'Value', 'variable': 'Weather Variable'},
)
fig.update_layout(width=900, height=400)
fig.show()

fig2 = px.line(
    weather_df, x='date', y='nasa_surface_pressure',
    title='NASA POWER — Surface Pressure (2024-2025)',
    labels={'date': 'Date', 'nasa_surface_pressure': 'Surface Pressure (kPa)'},
    color_discrete_sequence=['green'],
)
fig2.update_layout(width=900, height=300)
fig2.show()

# ============================================================
# Sample: Engine 1 พร้อมสภาพอากาศ
# ============================================================
print('\nSample: Engine 1 with weather data')
display(train_df[train_df['unit_id'] == 1][
    ['unit_id', 'cycle', 'date', 'timestamp',
     'nasa_ambient_temp', 'nasa_humidity', 'nasa_surface_pressure', 'nasa_wind_speed']
].head(10))

# COMMAND ----------

# DBTITLE 1,Weather x Sensor Correlation Analysis
# ============================================================
# Weather × Sensor Correlation Analysis
# ตรวจสอบว่าสภาพอากาศมีผลต่อสัญญาณเซนเซอร์จริงหรือไม่
# ก่อนนำไปคำนวณ Stress Index
# ============================================================

WEATHER_COLS = ['nasa_ambient_temp', 'nasa_humidity', 'nasa_surface_pressure', 'nasa_wind_speed']
KEY_NORM_SENSORS = ['sensor_2_norm', 'sensor_3_norm', 'sensor_4_norm',
                    'sensor_7_norm', 'sensor_11_norm', 'sensor_15_norm',
                    'sensor_9_norm', 'sensor_14_norm', 'sensor_17_norm']

# ============================================================
# 1. Correlation: Weather vs Normalized Sensors
# ============================================================
corr_cols = WEATHER_COLS + KEY_NORM_SENSORS
corr_matrix = train_df[corr_cols].corr()

# สกัลเฉพาะส่วน Weather × Sensor
weather_sensor_corr = corr_matrix.loc[WEATHER_COLS, KEY_NORM_SENSORS]

print('=' * 70)
print('1. Weather × Normalized Sensor Correlation')
print('=' * 70)
display(weather_sensor_corr.round(3))

# ============================================================
# 2. Correlation: Weather vs RUL
# ============================================================
weather_rul_corr = train_df[WEATHER_COLS + ['RUL']].corr()['RUL'].drop('RUL')

print('\n' + '=' * 70)
print('2. Weather × RUL Correlation')
print('=' * 70)
for col, val in weather_rul_corr.items():
    direction = 'positive' if val > 0 else 'negative'
    strength = 'strong' if abs(val) > 0.3 else 'moderate' if abs(val) > 0.15 else 'weak' if abs(val) > 0.05 else 'negligible'
    print(f'  {col:30s}: {val:+.4f}  ({strength} {direction})')

# ============================================================
# 3. Heatmap: Weather × Sensor Correlation
# ============================================================
fig = px.imshow(
    weather_sensor_corr,
    title='Correlation Heatmap: Weather Variables x Normalized Sensors',
    labels=dict(color='Pearson r'),
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    text_auto='.3f',
    aspect='auto',
)
fig.update_layout(width=800, height=350)
fig.show()

# ============================================================
# 4. Scatter: Weather vs Key Sensor (colored by RUL)
# ============================================================
sample_df = train_df.sample(n=5000, random_state=42)

for weather_col in ['nasa_ambient_temp', 'nasa_wind_speed']:
    for sensor_col in ['sensor_3_norm', 'sensor_9_norm']:
        fig_s = px.scatter(
            sample_df, x=weather_col, y=sensor_col,
            color='RUL',
            title=f'{weather_col} vs {sensor_col} (colored by RUL)',
            labels={weather_col: weather_col, sensor_col: sensor_col, 'RUL': 'RUL'},
            opacity=0.5,
            color_continuous_scale='Viridis',
        )
        fig_s.update_layout(width=600, height=400)
        fig_s.show()

# ============================================================
# 5. Summary & Interpretation
# ============================================================
print('\n' + '=' * 70)
print('Summary & Interpretation')
print('=' * 70)

# หา weather variable ที่ correlate แรงที่สุดกับ sensor ใดๆ
max_corr = weather_sensor_corr.abs().max()
max_weather = weather_sensor_corr.abs().max(axis=1).idxmax()
max_sensor = weather_sensor_corr.abs().max(axis=0).idxmax()
max_val = weather_sensor_corr.loc[max_weather, max_sensor]

print(f'  Strongest weather-sensor correlation:')
print(f'    {max_weather} x {max_sensor}: r = {max_val:+.4f}')
print(f'\n  Weather x RUL correlations:')
for col, val in weather_rul_corr.items():
    print(f'    {col:30s}: r = {val:+.4f}')

# ตัดสินใจ
n_strong = (weather_sensor_corr.abs() > 0.3).sum().sum()
n_moderate = ((weather_sensor_corr.abs() > 0.15) & (weather_sensor_corr.abs() <= 0.3)).sum().sum()
n_weak = ((weather_sensor_corr.abs() > 0.05) & (weather_sensor_corr.abs() <= 0.15)).sum().sum()
print(f'\n  Correlation strength counts (Weather x Sensor):')
print(f'    Strong  (|r| > 0.30): {n_strong}')
print(f'    Moderate(0.15-0.30): {n_moderate}')
print(f'    Weak    (0.05-0.15): {n_weak}')

if n_strong + n_moderate > 0:
    print(f'\n  -> Weather data shows meaningful correlation with sensor signals')
    print(f'     Proceeding to build Stress Index is justified.')
else:
    print(f'\n  -> Weather correlation is weak; Stress Index may have limited value')
    print(f'     Consider using weather as auxiliary features rather than primary.')

# COMMAND ----------

# DBTITLE 1,Diagnostic: Synthetic Date Bias & Seasonal Confounding Check
# ============================================================
# Diagnostic: วันที่สมมติบิดเบี้ยวหรือไม่?
# ตรวจสอบว่า weather ที่แต่ละเครื่องได้รับ สอดคล้องกับฤดูกาลจริงไหม
# และ correlation หากแยกตามฤดู จะเปลี่ยนไปไหม
# ============================================================

# --- กำหนดฤดูกาลจากเดือน ---
season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall',
}
train_df['season'] = train_df['month'].map(season_map)

WEATHER_COLS = ['nasa_ambient_temp', 'nasa_humidity', 'nasa_surface_pressure', 'nasa_wind_speed']
KEY_SENSORS = ['sensor_2_norm', 'sensor_3_norm', 'sensor_9_norm', 'sensor_14_norm']

# ============================================================
# 1. ตรวจสอบ: weather ของแต่ละฤดูเป็นอย่างไรจริงๆ?
# ============================================================
print('=' * 70)
print("1. Weather by Season (Chicago O'Hare 2024-2025)")
print('=' * 70)
season_weather = train_df.groupby('season')[WEATHER_COLS].mean().round(2)
season_weather = season_weather.reindex(['Winter', 'Spring', 'Summer', 'Fall'])
display(season_weather)

# ============================================================
# 2. แต่ละเครื่องยนต์ได้ weather แบบไหน? (อิงจากเดือนเริ่มต้น)
# ============================================================
engine_season = train_df.groupby('unit_id').agg(
    start_month=('month', 'first'),
    start_season=('season', 'first'),
    n_cycles=('cycle', 'max'),
).reset_index()

print('\n' + '=' * 70)
print('2. Engine Commissioning Season Distribution')
print('=' * 70)
season_counts = engine_season['start_season'].value_counts().reindex(
    ['Winter', 'Spring', 'Summer', 'Fall'], fill_value=0
)
for season, count in season_counts.items():
    print(f'  {season:8s}: {count} engines')

# ============================================================
# 3. Correlation แยกตามฤดู (Within-Season Correlation)
# ============================================================
print('\n' + '=' * 70)
print('3. Weather x Sensor Correlation — Pooled vs Within-Season')
print('=' * 70)

# Pooled (ทั้งหมดรวมกัน)
pooled_corr = train_df[WEATHER_COLS + KEY_SENSORS].corr().loc[WEATHER_COLS, KEY_SENSORS]
print('\nPooled (all seasons combined):')
display(pooled_corr.round(3))

# แยกตามฤดู
for season in ['Winter', 'Summer']:
    season_data = train_df[train_df['season'] == season]
    if len(season_data) > 100:
        season_corr = season_data[WEATHER_COLS + KEY_SENSORS].corr().loc[WEATHER_COLS, KEY_SENSORS]
        print(f'\nWithin {season} ({len(season_data):,} rows):')
        display(season_corr.round(3))

# ============================================================
# 4. เปรียบเทียบ: Pooled vs Within-Season ในตารางเดียว
# ============================================================
comparison_rows = []
for w in WEATHER_COLS:
    for s in KEY_SENSORS:
        pooled_r = pooled_corr.loc[w, s]
        winter_r = train_df[train_df['season']=='Winter'][[w, s]].corr().iloc[0, 1]
        summer_r = train_df[train_df['season']=='Summer'][[w, s]].corr().iloc[0, 1]
        comparison_rows.append({
            'weather': w, 'sensor': s,
            'pooled_r': round(pooled_r, 3),
            'winter_r': round(winter_r, 3),
            'summer_r': round(summer_r, 3),
        })

comparison_df = pd.DataFrame(comparison_rows)
print('\n' + '=' * 70)
print('4. Pooled vs Within-Season Correlation Comparison')
print('=' * 70)
display(comparison_df)

# ============================================================
# 5. สรุป: ปัญหาอยู่ที่วันที่ หรือที่ simulation?
# ============================================================
print('\n' + '=' * 70)
print('5. Diagnosis')
print('=' * 70)

# ดูว่า within-season correlation แตกต่างจาก pooled มากไหม
diff_winter = (comparison_df['winter_r'] - comparison_df['pooled_r']).abs().mean()
diff_summer = (comparison_df['summer_r'] - comparison_df['pooled_r']).abs().mean()

print(f'  Avg |delta| Pooled vs Winter: {diff_winter:.4f}')
print(f'  Avg |delta| Pooled vs Summer: {diff_summer:.4f}')
print()
print('  คำอธิบาย:')
print('  - หาก within-season correlation สูงขึ้นอย่างชัดเจน')
print('    -> ปัญหาคือวันที่สมมติบิดเบี้ยว (seasonal confounding)')
print('  - หาก within-season correlation ยังต่ำเท่าเดิม')
print('    -> ปัญหาคือ C-MAPSS เป็น simulation ที่ไม่ใช้ weather เป็น input')
print('    -> correlation ต่ำไม่ใช่เพราะใส่จิ๊กซอว์ผิดช่อง')
print('    -> แต่เพราะภาพ jigsaw ตัวนั้น (weather) ไม่ได้อยู่ในภาพ (simulation) อยู่แล้ว')

# COMMAND ----------

# MAGIC %md
# MAGIC ข้อมูลสภาพอากาศสำหรับ C-MAPSS : มีผลน้อย
# MAGIC เพราะเป็น simulation ที่ไม่ใช้ weather 
# MAGIC เป็น input รวมถึงการที่
# MAGIC เราตรวจสอบแล้วพบว่ามี 
# MAGIC seasonal confounding เกิดขึ้นด้วย เปลี่ยนเป็นการ pivot จาก "weather ทำนาย RUL ไม่ได้" ไปเป็น "weather มีผลต่อการวางแผนซ่อมบำรุง" ซึ่งเป็นโจทย์ทางธุรกิจที่มีความหมายจริง

# COMMAND ----------

# DBTITLE 1,Cost Parameter Object: Maintenance Cost Benchmarks
# ============================================================
# Cost Parameter Object: Maintenance Cost Benchmarks
# แหล่งอ้างอิงมาตรฐานอุตสาหกรรม (IATA, FAA, Boeing/Airbus)
# ============================================================

cost_parameters = {
    'unplanned_downtime_cost_per_hour': 15000,   # USD/hour (range: 10k–20k or 150k–300k per event)
    'proactive_maintenance_cost_per_event': 45000,  # USD/event (range: 30k–60k)
    'early_replacement_waste_per_cycle': 225,    # USD/cycle (range: 150–300)
    'labour_rate_per_hour': 125,                 # USD/hour-person (range: 100–150)
    'unplanned_downtime_cost_per_event': 225000, # USD/event (range: 150k–300k)
}

cost_param_df = pd.DataFrame([
    {'name': 'Unplanned Downtime Cost (per hour)', 'key': 'unplanned_downtime_cost_per_hour', 'value': cost_parameters['unplanned_downtime_cost_per_hour'], 'unit': 'USD/hour'},
    {'name': 'Unplanned Downtime Cost (per event)', 'key': 'unplanned_downtime_cost_per_event', 'value': cost_parameters['unplanned_downtime_cost_per_event'], 'unit': 'USD/event'},
    {'name': 'Proactive Maintenance Cost (per event)', 'key': 'proactive_maintenance_cost_per_event', 'value': cost_parameters['proactive_maintenance_cost_per_event'], 'unit': 'USD/event'},
    {'name': 'Early Replacement Waste (per cycle)', 'key': 'early_replacement_waste_per_cycle', 'value': cost_parameters['early_replacement_waste_per_cycle'], 'unit': 'USD/cycle'},
    {'name': 'Labour Rate (per hour-person)', 'key': 'labour_rate_per_hour', 'value': cost_parameters['labour_rate_per_hour'], 'unit': 'USD/hour-person'},
])

print("\nMaintenance Cost Parameters (Benchmark)")
display(cost_param_df)

# สามารถปรับค่าให้อยู่ในรูปแบบ scenario ได้ เช่น sensitivity analysis
# ค่าใน object นี้ถูกใช้โดย Cost Function นำไปคูณกับ Estimated Duration


# COMMAND ----------

# DBTITLE 1,Convert date column to datetime (before calendar feature)
# =============================================================
# Ensure date columns are datetime types before calendar feature
# =============================================================
for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
print('Date columns converted to datetime (train_df, test_df)')
print('Sample:', train_df[['date']].head())


# COMMAND ----------

# DBTITLE 1,Install holidays package (for calendar features)
# MAGIC %uv pip install holidays
# MAGIC

# COMMAND ----------

# DBTITLE 1,Calendar Feature Engineering — Holiday, Weekday, Peak Travel
# =============================================================
# Calendar Feature Engineering — Holiday, Weekday, Peak Travel
# เพิ่มฟีเจอร์วันหยุดนักขัตฤกษ์, วันธรรมดา, และ High Season เพื่อปรับค่าแรงและ downtime cost
# =============================================================

import pandas as pd
from datetime import datetime
import holidays

# --- กำหนดบริเวณและช่วงปี ---
COUNTRY = 'US'
STATE = 'IL'  # Illinois — Chicago O'Hare
YEARS = [2024, 2025]

# --- ฟีเจอร์ Weekday/Weekend ---
for df in [train_df, test_df]:
    df['is_weekend'] = df['date'].dt.weekday >= 5  # เสาร์/อาทิตย์: 0=False, 1=True
    df['weekday_name'] = df['date'].dt.day_name()

# --- ฟีเจอร์ Holiday ---
us_holidays = holidays.US(years=YEARS, state=STATE)
for df in [train_df, test_df]:
    df['is_holiday'] = df['date'].dt.date.isin(us_holidays)
    df['holiday_name'] = df['date'].dt.date.map(lambda x: us_holidays.get(x, None))

# --- ฟีเจอร์ High Season (Peak Travel) ---
# ตัวอย่าง: ช่วง Mid-June ถึง Mid-August + Thanksgiving และ Christmas
peak_ranges = [
    ('06-15', '08-15'),  # Summer peak
    ('11-15', '11-28'),  # Thanksgiving week
    ('12-15', '01-02'),  # Christmas + New Year peak
]

def is_peak_season(date):
    month_day = date.strftime('%m-%d')
    for start, end in peak_ranges:
        # Christmas/NY: ข้ามปี
        if start < end:
            if start <= month_day <= end:
                return True
        else:
            # สำหรับช่วงที่ข้ามปี เช่น 12-15 ถึง 01-02
            if (month_day >= start) or (month_day <= end):
                return True
    return False

for df in [train_df, test_df]:
    df['is_peak_season'] = df['date'].apply(is_peak_season)

print('\nCalendar Features Added:')
print('  is_weekend    : วันหยุดเสาร์อาทิตย์')
print('  is_holiday    : วันหยุดนักขัตฤกษ์ (US/Illinois)')
print('  holiday_name  : ชื่อวันหยุด')
print('  is_peak_season: High season ช่วง travel demand สูง')
print('\nตัวอย่างข้อมูล:')
display(train_df[['date', 'is_weekend', 'is_holiday', 'holiday_name', 'is_peak_season']].head(20))

# เหตุผล: ฟีเจอร์เหล่านี้ถูกใช้เพื่อปรับ multiplier ค่าแรงและ downtime cost ตามสถานการณ์ธุรกิจ
# เช่น วันหยุดนักขัตฤกษ์ = ค่าแรง 1.5x, peak season = downtime cost สูงขึ้น
# ผลงาน Research/Industry: US Dept. of Labor, Airline Scheduling, IATA/FAA studies


# COMMAND ----------

# DBTITLE 1,Step 1: สร้าง RUL Features + Health State Bins
# =============================================================
# Step 1: สร้างฟีเจอร์ RUL และ Health State
# =============================================================

# (A) Regression Target: Piecewise Linear RUL (capped to 130 cycles)
def capped_rul(df, cap=130):
    # หา max cycle ต่อ unit_id ก่อน (จุดที่เครื่องเจ๊ง)
    max_cycle = df.groupby('unit_id')['cycle'].transform('max')
    rul = (max_cycle - df['cycle']).clip(lower=0, upper=cap)
    return rul

train_df['RUL'] = capped_rul(train_df, cap=130)
test_df['RUL'] = capped_rul(test_df, cap=130)

# (B) Classification Target: Health State
# Healthy: RUL > 80, Degraded: 30 < RUL <= 80, Critical: RUL <= 30

bins = [-1, 30, 80, 130]
labels = ['Critical (High Risk)', 'Degraded (Medium Risk)', 'Healthy (Low Risk)']
train_df['health_state'] = pd.cut(train_df['RUL'], bins=bins, labels=labels)
test_df['health_state'] = pd.cut(test_df['RUL'], bins=bins, labels=labels)

print('Preview train_df with RUL + health_state:')
display(train_df[['unit_id', 'cycle', 'RUL', 'health_state']].head(20))

print('\nRUL (capped) stats:')
print(train_df['RUL'].describe().to_string())
print('\nHealth state counts:')
print(train_df['health_state'].value_counts())

# COMMAND ----------

# DBTITLE 1,RUL & Health State Distribution Visualization
# =============================================================
# Visualization: RUL Distribution & Health State Proportions
# =============================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(train_df['RUL'], bins=30, kde=True, color='dodgerblue')
plt.title('RUL Distribution (Capped at 130 cycles)')
plt.xlabel('Remaining Useful Life (cycles)')

plt.subplot(1,2,2)
health_counts = train_df['health_state'].value_counts().loc[['Healthy (Low Risk)', 'Degraded (Medium Risk)', 'Critical (High Risk)']]
sns.barplot(y=health_counts.index, x=health_counts.values, palette='viridis')
plt.title('Health State Proportions')
plt.xlabel('Number of Samples')
plt.ylabel('')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,ML Classification: Health State (RUL Window) Prediction
# =============================================================
# ML Classification: Predict Health State from Features (RUL Window)
# 3-class: Healthy (>80), Degraded (31–80), Critical (<=30) cycles
# =============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = [c for c in train_df.columns if c.startswith('sensor_') or c in [
    'accumulated_high_temp_cycles', 'rolling_std_temp_15', 'weather_stress_score',
    'cumulative_weather_stress', 'is_holiday', 'is_weekend', 'is_peak_season']]

X = train_df[FEATURES].fillna(0)
y = train_df['health_state']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2026)

clf = RandomForestClassifier(n_estimators=140, max_depth=9, class_weight='balanced', random_state=2026)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)

print('RUL Health State (3-class) Classification')
print('Accuracy:', accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_valid, y_pred, labels=['Healthy (Low Risk)','Degraded (Medium Risk)','Critical (High Risk)']))

importances = pd.Series(clf.feature_importances_, index=FEATURES)
print('\nTop Important Features:')
display(importances.sort_values(ascending=False).head(8))

# Visualization: Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_valid, y_pred, labels=['Healthy (Low Risk)','Degraded (Medium Risk)','Critical (High Risk)'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy','Degraded','Critical'], yticklabels=['Healthy','Degraded','Critical'])
plt.title('Confusion Matrix: Predicted vs Actual Health State')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,ML Regression: RUL (Piecewise Linear, Random Forest)
# =============================================================
# ML Regression: Predict RUL (Remaining Useful Life, capped at 130 cycles)
# Piecewise Linear Regressor, suitable for C-MAPSS predictive maintenance
# =============================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [c for c in train_df.columns if c.startswith('sensor_') or c in [
    'accumulated_high_temp_cycles', 'rolling_std_temp_15', 'weather_stress_score',
    'cumulative_weather_stress', 'is_holiday', 'is_weekend', 'is_peak_season']]

X = train_df[FEATURES].fillna(0)
y = train_df['RUL']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2026)

reg = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=2026)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print('RUL Regression (Random Forest, capped)')
print(f'MAE: {mae:.2f} cycles')
print(f'RMSE: {rmse:.2f} cycles')
print(f'R2: {r2:.3f}')

importances = pd.Series(reg.feature_importances_, index=FEATURES)
print('\nTop Important Features:')
display(importances.sort_values(ascending=False).head(8))

# Plot: true vs predicted RUL
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.scatter(y_valid, y_pred, s=10, alpha=0.28, color='dodgerblue')
plt.xlabel('True RUL (cycles)')
plt.ylabel('Predicted RUL (cycles)')
plt.title('Predicted vs True RUL (Validation set)')
plt.plot([0, 130], [0, 130], '--', color='grey', lw=1)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Step 2: Cost Mapping from RUL / Health State
# =============================================================
# Step 2: Cost Mapping from RUL / Health State
# =============================================================

# --- Function: Map RUL/health state to maintenance action + cost ---
def estimate_maintenance_cost(row, cost_parameters):
    rul = row['RUL']
    health = row['health_state']
    # Map health_state (classification) & RUL (regression) to action/cost
    if health == 'Critical (High Risk)' or rul <= 30:
        # Unplanned Failure scenario
        downtime_hours = np.random.uniform(48, 72)  # model downtime distribution
        base_cost = cost_parameters['unplanned_downtime_cost_per_event']
        hourly_cost = cost_parameters['unplanned_downtime_cost_per_hour']
        total_cost = downtime_hours * hourly_cost + base_cost
        action = 'Unplanned Failure / Emergency'
    elif health == 'Degraded (Medium Risk)' or 30 < rul <= 80:
        # Proactive Maintenance scenario
        downtime_hours = np.random.uniform(8, 24)   # model scheduled downtime distribution
        base_cost = cost_parameters['proactive_maintenance_cost_per_event']
        hourly_cost = cost_parameters['labour_rate_per_hour']
        total_cost = downtime_hours * hourly_cost + base_cost
        action = 'Proactive Planned Maintenance'
    else:
        # Early Replacement scenario
        downtime_hours = np.random.uniform(1, 2)    # minimal downtime
        base_cost = 0
        cycle_waste = cost_parameters['early_replacement_waste_per_cycle']
        total_cost = downtime_hours * cost_parameters['labour_rate_per_hour'] + cycle_waste
        action = 'Early Replacement (Preventive)'
    return pd.Series({
        'maintenance_action': action,
        'downtime_hours': downtime_hours,
        'maintenance_cost': total_cost
    })

# --- Apply mapping to validation set ---
cost_map_df = X_valid.copy()
cost_map_df['RUL'] = y_valid.values
cost_map_df['health_state'] = y_valid.index.map(lambda i: train_df.loc[i, 'health_state'])

cost_outputs = cost_map_df.apply(estimate_maintenance_cost, axis=1, cost_parameters=cost_parameters)
cost_map_df = pd.concat([cost_map_df, cost_outputs], axis=1)

# --- Summary Stats ---
print("\n=== Maintenance Action Cost Distribution ===")
display(cost_map_df[['maintenance_action','maintenance_cost','downtime_hours']].groupby('maintenance_action').describe())

# --- Example Records ---
print("\nExample mapping from RUL/health state to action & cost:")
display(cost_map_df[['RUL','health_state','maintenance_action','maintenance_cost','downtime_hours']].head(10))

# --- Visualization: Maintenance Cost by Action ---
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
sns.boxplot(x='maintenance_action', y='maintenance_cost', data=cost_map_df, palette='Set2')
plt.title('Distribution of Maintenance Cost by Action Type')
plt.ylabel('Cost (USD)')
plt.xlabel('Maintenance Action')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Install SHAP for Explainability
# =============================================================
# Install SHAP for Explainability
# =============================================================
%pip install shap


# COMMAND ----------

# DBTITLE 1,Explainable RUL Model: SHAP & PDP Analysis
# =============================================================
# Explainable RUL Model: SHAP & PDP Analysis
# =============================================================
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- SHAP Analysis ---
explainer = shap.TreeExplainer(reg)  # reg = model from Cell 37
shap_values = explainer.shap_values(X_valid)

# --- Summary Plot ---
plt.figure(figsize=(13,5))
shap.summary_plot(shap_values, X_valid, feature_names=FEATURES, plot_type='bar')
plt.title('SHAP Feature Importance for RUL Prediction')
plt.show()

# --- Dependence Plot: ดูผลการเปลี่ยนค่าแต่ละฟีเจอร์ต่อ RUL ---
for feature_name in ['sensor_11_norm_rmean5', 'sensor_17_norm_rmean20', 'cumulative_weather_stress', 'op_cond_1']:
    if feature_name in X_valid.columns:
        shap.dependence_plot(feature_name, shap_values, X_valid, feature_names=FEATURES)
        plt.title(f'SHAP: Effect of {feature_name} on RUL Prediction')
        plt.tight_layout()
        plt.show()

# --- Partial Dependence Plot (PDP) NEW SYNTAX ---
from sklearn.inspection import PartialDependenceDisplay
features_to_plot = ['sensor_11_norm_rmean5', 'sensor_17_norm_rmean20', 'cumulative_weather_stress', 'op_cond_1']
for feat in features_to_plot:
    if feat in X_valid.columns:
        fig, ax = plt.subplots(figsize=(7,4))
        PartialDependenceDisplay.from_estimator(reg, X_valid, [feat], ax=ax)
        plt.title(f'PDP: {feat} vs RUL')
        plt.tight_layout()
        plt.show()

print("\nหมายเหตุ: SHAP/PDP แสดงให้เห็นว่าการเพิ่ม/ลดค่าแต่ละฟีเจอร์จะทำให้ RUL เพิ่มหรือลดเท่าไร")
print("  ใช้สำหรับ sensitivity analysis, policy simulation, root cause analysis หรือวางแผน optimal setting/usage ได้เลย.")


# COMMAND ----------

# DBTITLE 1,Export Models & Artifacts for Deployment
# ============================================================
# Export Models & Artifacts for Deployment
# บันทึก ML models, feature lists, cost params, normalization stats
# เป็น .joblib ไฟล์สำหรับนำไป deploy (Streamlit / API / etc.)
# ============================================================

import joblib
import os
import json

# --- กำหนดโฟลเดอร์สำหรับเก็บ artifacts ---
ARTIFACTS_DIR = '/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- บันทึก ML Models ---
joblib.dump(reg, os.path.join(ARTIFACTS_DIR, 'rul_regressor.joblib'))
joblib.dump(clf, os.path.join(ARTIFACTS_DIR, 'health_state_classifier.joblib'))
joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, 'kmeans_op_regime.joblib'))
print('✓ Models saved: rul_regressor, health_state_classifier, kmeans_op_regime')

# --- บันทึก Feature List ---
with open(os.path.join(ARTIFACTS_DIR, 'features.json'), 'w') as f:
    json.dump(FEATURES, f, indent=2)
print(f'✓ Features list saved ({len(FEATURES)} features)')

# --- บันทึก Cost Parameters ---
with open(os.path.join(ARTIFACTS_DIR, 'cost_parameters.json'), 'w') as f:
    json.dump(cost_parameters, f, indent=2)
print('✓ Cost parameters saved')

# --- บันทึก Normalization Stats (regime mu/sigma) ---
joblib.dump(regime_mu, os.path.join(ARTIFACTS_DIR, 'regime_mu.joblib'))
joblib.dump(regime_sigma, os.path.join(ARTIFACTS_DIR, 'regime_sigma.joblib'))
print('✓ Regime normalization stats saved (regime_mu, regime_sigma)')

# --- บันทึก Sensor Lists & Config ---
config = {
    'USEFUL_SENSORS': USEFUL_SENSORS,
    'KEY_SENSORS': KEY_SENSORS,
    'WINDOWS': [5, 10, 20],
    'RUL_CAP': 130,
    'HEALTH_BINS': [-1, 30, 80, 130],
    'HEALTH_LABELS': ['Critical (High Risk)', 'Degraded (Medium Risk)', 'Healthy (Low Risk)'],
    'N_REGIMES': 6,
    'SETTINGS_COLS': ['op_cond_1', 'op_cond_2', 'op_cond_3'],
}
with open(os.path.join(ARTIFACTS_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print('✓ Model config saved (sensors, windows, bins, labels)')

# --- สรุปไฟล์ทั้งหมด ---
print('\n' + '=' * 60)
print('Artifacts Export Summary')
print('=' * 60)
for fname in sorted(os.listdir(ARTIFACTS_DIR)):
    fpath = os.path.join(ARTIFACTS_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f'  {fname:40s} {size_kb:>10.1f} KB')
print(f'\n  Directory: {ARTIFACTS_DIR}')

# COMMAND ----------

# DBTITLE 1,Train Simplified RUL Model for Streamlit (24 Raw Features)
# ============================================================
# Train Simplified RUL Model for Streamlit App (24 Raw Features)
# โมเดลนี้ใช้เฉพาะ op_cond + raw sensors (ไม่มี rolling/normalized)
# เพื่อให้ผู้ใช้ใส่ค่าได้โดยตรงใน Streamlit app
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib, json, os, shap
import numpy as np

# --- กำหนด Raw Features (24 ตัว) ---
RAW_FEATURES = ['op_cond_1', 'op_cond_2', 'op_cond_3'] + [f'sensor_{i}' for i in range(1, 22)]
print(f"Raw features: {len(RAW_FEATURES)} (3 op_conds + 21 sensors)")

# --- เตรียมข้อมูล ---
X_raw = train_df[RAW_FEATURES].fillna(0)
y_rul = train_df['RUL']

X_train, X_valid, y_train, y_valid = train_test_split(
    X_raw, y_rul, test_size=0.2, random_state=2026
)

# --- เทรนโมเดล ---
reg_raw = RandomForestRegressor(
    max_depth=12, n_estimators=200, random_state=2026, n_jobs=-1
)
reg_raw.fit(X_train, y_train)

# --- ประเมินผล ---
pred = reg_raw.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, pred))
mae = mean_absolute_error(y_valid, pred)
r2 = r2_score(y_valid, pred)

print(f"\n{'='*60}")
print(f"Simplified RUL Model (24 Raw Features)")
print(f"{'='*60}")
print(f"  RMSE : {rmse:.2f} cycles")
print(f"  MAE  : {mae:.2f} cycles")
print(f"  R²   : {r2:.4f}")

# เปรียบเทียบกับโมเดลเต็ม (145 features)
pred_full = reg.predict(train_df.loc[X_valid.index, FEATURES].fillna(0))
rmse_full = np.sqrt(mean_squared_error(y_valid, pred_full))
r2_full = r2_score(y_valid, pred_full)
print(f"\n  vs Full model (145 features): RMSE={rmse_full:.2f}, R²={r2_full:.4f}")
print(f"  → โมเดลง่ายใช้ได้จริงถ้า R² > 0.5")

# --- Feature Importance ---
importance_df = pd.DataFrame({
    'feature': RAW_FEATURES,
    'importance': reg_raw.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*60}")
print("Top 10 Most Important Features:")
print(f"{'='*60}")
print(importance_df.head(10).to_string(index=False))

# --- สถิติข้อมูลสำหรับ Streamlit sliders ---
feature_stats = {}
for col in RAW_FEATURES:
    feature_stats[col] = {
        'min':   float(X_raw[col].min()),
        'max':   float(X_raw[col].max()),
        'mean':  float(X_raw[col].mean()),
        'std':   float(X_raw[col].std()),
        'p25':   float(X_raw[col].quantile(0.25)),
        'p75':   float(X_raw[col].quantile(0.75)),
    }

# --- SHAP Explainer ---
explainer_raw = shap.TreeExplainer(reg_raw)
shap_sample = X_valid.sample(n=min(500, len(X_valid)), random_state=42)
shap_values_raw = explainer_raw.shap_values(shap_sample)

print(f"\n✓ SHAP values computed ({len(shap_sample)} samples)")

# --- บันทึก Artifacts ---
ARTIFACTS_DIR = '/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/artifacts'
joblib.dump(reg_raw, os.path.join(ARTIFACTS_DIR, 'rul_regressor_raw.joblib'))
with open(os.path.join(ARTIFACTS_DIR, 'raw_features.json'), 'w') as f:
    json.dump(RAW_FEATURES, f, indent=2)
with open(os.path.join(ARTIFACTS_DIR, 'feature_stats.json'), 'w') as f:
    json.dump(feature_stats, f, indent=2)
joblib.dump(explainer_raw, os.path.join(ARTIFACTS_DIR, 'shap_explainer_raw.joblib'))

print(f"\n✓ Artifacts saved:")
print(f"  rul_regressor_raw.joblib  (simplified model)")
print(f"  raw_features.json        (24 feature names)")
print(f"  feature_stats.json       (min/max/mean/std for sliders)")
print(f"  shap_explainer_raw.joblib (SHAP explainer)")

# COMMAND ----------

# DBTITLE 1,Compress rul_regressor_raw.joblib to reduce size
# ============================================================
# Compress joblib model file to reduce size
# ============================================================
import joblib
import os

ARTIFACTS_DIR = '/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/artifacts'
model_path = os.path.join(ARTIFACTS_DIR, 'rul_regressor_raw.joblib')
compressed_path = os.path.join(ARTIFACTS_DIR, 'rul_regressor_raw_compress.joblib')

# Reload model if needed
reg_raw = joblib.load(model_path)
# Compress and save with compression level=3 (adjust if needed, max 9)
joblib.dump(reg_raw, compressed_path, compress=3)

# Check size after compress
size_mb = os.path.getsize(compressed_path) / (1024*1024)
print(f'Compressed model file size: {size_mb:.2f} MB')
print(f'✓ Saved file at: {compressed_path}\n')

# Optional: delete old file if size < 52MB
if size_mb < 52:
    os.remove(model_path)
    os.rename(compressed_path, model_path)
    print('✓ Replaced old model file with compressed version.')
else:
    print('⚠️ Compressed file still exceeds 52MB. Try higher compression or retrain with fewer trees/features.')

# COMMAND ----------

# DBTITLE 1,Install SHAP for Joblib Decompression
# MAGIC %pip install shap
# MAGIC

# COMMAND ----------

# DBTITLE 1,Compress SHAP Explainer Joblib to Reduce File Size
import os
from joblib import load, dump

artifact_path = '/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/artifacts/shap_explainer_raw.joblib'
compressed_path = '/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data/artifacts/shap_explainer_raw_compressed.joblib'

explainer = load(artifact_path)
dump(explainer, compressed_path, compress=3)

orig_size = os.path.getsize(artifact_path)
comp_size = os.path.getsize(compressed_path)
print(f'Original size: {orig_size/1024/1024:.2f} MB')
print(f'Compressed size: {comp_size/1024/1024:.2f} MB')
print(f'Compressed file saved to: {compressed_path}')