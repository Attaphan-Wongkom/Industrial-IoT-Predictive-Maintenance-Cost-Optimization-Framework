

"""
NASA Predictive Maintenance — RUL Prediction & Sensitivity Analysis
Streamlit App for Interactive What-If Analysis

Usage:
    streamlit run streamlit_app.py

Features:
    1. ปรับค่า Operating Conditions + Sensors ด้วย sliders
    2. ทำนาย RUL (Remaining Useful Life) แบบ real-time
    3. SHAP Waterfall Plot: บอกว่าแต่ละ feature ดัน RUL ขึ้นหรือลด
    4. Sensitivity Analysis: ถ้าเพิ่ม/ลด feature ละ 1 std → RUL เปลี่ยนเท่าไหร่
    5. Feature Importance: เรียงลำดับความสำคัญของแต่ละ sensor
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# --- Page config MUST be the first Streamlit command ---
st.set_page_config(
    page_title="NASA RUL Predictor",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Import cost_model module from same directory
try:
    _BD = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BD = "/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data"
if _BD not in sys.path:
    sys.path.insert(0, _BD)
from cost_model import CostEstimator

# ========================================================
# Configuration
# ========================================================
try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = "/Workspace/Users/prabcrab23@gmail.com/Predictive maintenance NASA Data"
ARTIFACTS_DIR = os.path.join(_BASE_DIR, "artifacts")
RUL_CAP = 130

# Sensor descriptions (NASA C-MAPSS FD004)
SENSOR_INFO = {
    'op_cond_1':  ('Altitude (ft)', 'Operating Condition 1'),
    'op_cond_2':  ('Mach Number', 'Operating Condition 2'),
    'op_cond_3':  ('Throttle Resolver Angle (°)', 'Operating Condition 3'),
    'sensor_1':   ('Total Temperature at Fan Inlet (°R)', 'T24'),
    'sensor_2':   ('Total Temperature at LPC Outlet (°R)', 'T30'),
    'sensor_3':   ('Total Temperature at HPC Outlet (°R)', 'T50'),
    'sensor_4':   ('Total Temperature at LPT Outlet (°R)', 'T48'),
    'sensor_5':   ('Pressure at Fan Inlet (psia)', 'P15'),
    'sensor_6':   ('Total Pressure at LPC Outlet (psia)', 'P20'),
    'sensor_7':   ('Total Pressure at HPC Outlet (psia)', 'P30'),
    'sensor_8':   ('Physical Fan Speed (rpm)', 'Nf'),
    'sensor_9':   ('Physical Core Speed (rpm)', 'Nc'),
    'sensor_10':  ('Engine Pressure Ratio (P50/P2)', 'epr'),
    'sensor_11':  ('Static Pressure at HPC Outlet (psia)', 'Ps30'),
    'sensor_12':  ('Ratio of Fuel Flow to Ps30', 'phi'),
    'sensor_13':  ('Corrected Fuel Flow (pps)', 'farB'),
    'sensor_14':  ('Corrected Fan Speed (rpm)', 'NFRf'),
    'sensor_15':  ('Bypass Ratio', 'B'),
    'sensor_16':  ('Burner Fuel-Air Ratio', 'htBleed'),
    'sensor_17':  ('Bleed Enthalpy', 'W31'),
    'sensor_18':  ('Demand Fan Speed (rpm)', 'W32'),
    'sensor_19':  ('Demand Corrected Fan Speed (rpm)', 'W48'),
    'sensor_20':  ('HPT Coolant Bleed (lbm/s)', 'SmFan'),
    'sensor_21':  ('LPT Coolant Bleed (lbm/s)', 'SmLPC'),
}

# ========================================================
# Load Artifacts (cached)
# ========================================================
@st.cache_resource
def load_artifacts():
    """โหลด model, feature list, feature stats, SHAP explainer"""
    model = joblib.load(os.path.join(ARTIFACTS_DIR, 'rul_regressor_raw.joblib'))
    with open(os.path.join(ARTIFACTS_DIR, 'raw_features.json')) as f:
        features = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_stats.json')) as f:
        stats = json.load(f)
    explainer = joblib.load(os.path.join(ARTIFACTS_DIR, 'shap_explainer_raw_compressed.joblib'))
    return model, features, stats, explainer

model, FEATURES, feature_stats, shap_explainer = load_artifacts()

# --- Load Cost Estimator (from cost_model.py) ---
@st.cache_resource
def load_cost_estimator():
    return CostEstimator(artifacts_dir=ARTIFACTS_DIR)

cost_estimator = load_cost_estimator()

# ========================================================
# Page Setup
# ========================================================
st.title("🛩️ NASA Predictive Maintenance — RUL Prediction & Sensitivity")
st.markdown("""
ปรับค่า **Operating Conditions** และ **Sensor Readings** เพื่อทำนาย **Remaining Useful Life (RUL)**
และดูว่าแต่ละตัวแปรมีผลต่อ RUL อย่างไร
""")

# ========================================================
# Sidebar — Feature Input
# ========================================================
st.sidebar.header("⚙️ Engine Configuration")

# Initialize session_state for current_values if not present
if 'current_values' not in st.session_state:
    st.session_state['current_values'] = {feat: feature_stats[feat]['mean'] for feat in FEATURES}


st.sidebar.subheader("📝 RUL Health Criteria")
st.sidebar.markdown("""
- **Healthy:** RUL > 80 cycles
- **Degraded:** 30 < RUL ≤ 80 cycles
- **Critical:** RUL ≤ 30 cycles

ปรับค่าสำหรับ Operating Conditions และ Sensors ด้วย sliders หรือ preset เพื่อดูว่าจะส่งผลต่อ RUL และ health state อย่างไร
""")
st.sidebar.divider()

# --- Calendar Context (affects cost multipliers) ---
st.sidebar.subheader("📅 Maintenance Context")
ctx_col1, ctx_col2, ctx_col3 = st.sidebar.columns(3)
is_holiday = ctx_col1.toggle("Holiday", help="Holiday = 1.5x labour cost")
is_weekend = ctx_col2.toggle("Weekend", help="Weekend = 1.2x labour cost")
is_peak_season = ctx_col3.toggle("Peak Season", help="Peak = 1.3x downtime cost")

st.sidebar.divider()

# Sliders for Operating Conditions
st.sidebar.subheader("🎛️ Operating Conditions")
for feat in ['op_cond_1', 'op_cond_2', 'op_cond_3']:
    label, code = SENSOR_INFO.get(feat, (feat, feat))
    s = feature_stats[feat]
    step = (s['max'] - s['min']) / 100
    val = st.sidebar.slider(
        f"{label} ({code})",
        min_value=s['min'],
        max_value=s['max'],
        value=st.session_state['current_values'][feat],
        step=step,
        format="%.2f",
        key=f"slider_{feat}",
    )
    st.session_state['current_values'][feat] = val

# Sliders for Key Sensors (top importance)
KEY_SENSORS_DISPLAY = ['sensor_13', 'sensor_11', 'sensor_15', 'sensor_4', 'sensor_12', 'sensor_9']
st.sidebar.subheader("📊 Key Sensors (Top Importance)")
# Mapping: sensor_13, sensor_11, sensor_15, sensor_4, sensor_12, sensor_9 (เรียงตาม slider)
SLIDER_INDEX = 1
for feat in KEY_SENSORS_DISPLAY:
    label, code = SENSOR_INFO.get(feat, (feat, feat))
    s = feature_stats[feat]
    step = max((s['max'] - s['min']) / 100, 0.01)
    val = st.sidebar.slider(
        f"{code}: {label} (Slider {SLIDER_INDEX}: {feat})",
        min_value=s['min'],
        max_value=s['max'],
        value=st.session_state['current_values'][feat],
        step=step,
        format="%.2f",
        key=f"slider_{feat}",
    )
    st.session_state['current_values'][feat] = val
    SLIDER_INDEX += 1

# Remaining sensors in expander
OTHER_SENSORS = [f for f in FEATURES if f.startswith('sensor_') and f not in KEY_SENSORS_DISPLAY]
with st.sidebar.expander("🔧 Other Sensors", expanded=False):
    for feat in OTHER_SENSORS:
        label, code = SENSOR_INFO.get(feat, (feat, feat))
        s = feature_stats[feat]
        step = max((s['max'] - s['min']) / 100, 0.01)
        val = st.slider(
            f"{code}: {label}",
            min_value=s['min'],
            max_value=s['max'],
            value=st.session_state['current_values'][feat],
            step=step,
            format="%.2f",
            key=f"slider_{feat}",
        )
        st.session_state['current_values'][feat] = val

# ========================================================
# Build Input DataFrame
# ========================================================
input_values = {feat: st.session_state['current_values'][feat] for feat in FEATURES}
input_df = pd.DataFrame([input_values])

# ========================================================
# Prediction
# ========================================================
predicted_rul = float(model.predict(input_df)[0])
predicted_rul_clipped = min(predicted_rul, RUL_CAP)

# Health state
if predicted_rul_clipped <= 30:
    health_state = "🔴 Critical (High Risk)"
    health_color = "red"
elif predicted_rul_clipped <= 80:
    health_state = "🟡 Degraded (Medium Risk)"
    health_color = "orange"
else:
    health_state = "🟢 Healthy (Low Risk)"
    health_color = "green"

# ========================================================
# Main Panel — Results
# ========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted RUL", f"{predicted_rul_clipped:.1f} cycles")
with col2:
    st.metric("Health State", health_state)
with col3:
    st.metric("Health Index", f"{(1 - predicted_rul_clipped / RUL_CAP) * 100:.1f}%")

st.divider()

# ========================================================
# Maintenance Cost Estimate (from cost_model.py)
# ========================================================
st.subheader("💰 Maintenance Cost Estimate")

cost_result = cost_estimator.estimate_single(
    rul=predicted_rul_clipped,
    health_state=health_state,
    is_holiday=is_holiday,
    is_weekend=is_weekend,
    is_peak_season=is_peak_season,
)

cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
with cost_col1:
    st.metric("Est. Cost", f"${cost_result['maintenance_cost']:,.0f}")
with cost_col2:
    st.metric("Downtime", f"{cost_result['downtime_hours']:.1f} hrs")
with cost_col3:
    st.metric("Action", cost_result['maintenance_action'].split(' / ')[0])
with cost_col4:
    mults = []
    if is_holiday: mults.append("1.5x")
    if is_weekend: mults.append("1.2x")
    if is_peak_season: mults.append("1.3x")
    st.metric("Multipliers", ", ".join(mults) if mults else "None")

with st.expander("📋 Cost Breakdown & Parameters"):
    breakdown = cost_result['cost_breakdown']
    for item, value in breakdown.items():
        if isinstance(value, float):
            if 'cost' in item.lower() or 'waste' in item.lower():
                st.write(f"  - **{item}**: ${value:,.2f}")
            else:
                st.write(f"  - **{item}**: {value:.2f}")
        else:
            st.write(f"  - **{item}**: {value}")
    st.markdown("---")
    st.markdown("**Cost Parameters (Industry Benchmark):**")
    st.dataframe(cost_estimator.get_cost_parameter_table(), use_container_width=True, hide_index=True)

st.divider()

# ========================================================
# SHAP Waterfall Plot
# ========================================================
st.subheader("🔍 SHAP Waterfall — แต่ละ Feature มีผลต่อ RUL อย่างไร")
st.markdown("""
- 🔵 น้ำเงิน (ค่า SHAP ติดลบ): เพิ่มค่านี้แล้ว RUL **ลดลง**
- 🔴 แดง (ค่า SHAP เป็นบวก): เพิ่มค่านี้แล้ว RUL **เพิ่มขึ้น**
หมายเหตุ: ผลของการ "เพิ่ม" หรือ "ลด" แต่ละ feature ต่อ RUL ขึ้นกับทิศทาง SHAP ของแต่ละ sensor ที่ทำงานร่วมกัน
""")

shap_values = shap_explainer.shap_values(input_df)

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=shap_explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=FEATURES,
    ),
    max_display=15,
    show=False,
)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# ========================================================
# Sensitivity Analysis
# ========================================================
st.subheader("📈 Sensitivity Analysis — ถ้าเปลี่ยน Feature จะกระทบ RUL เท่าไหร่?")
st.markdown("""
ตารางแสดงว่าถ้าเพิ่มหรือลดแต่ละ feature ไป **1 standard deviation** จะทำให้ RUL เปลี่ยนเท่าไหร่
""")

sensitivity_data = []
base_rul = predicted_rul

for feat in FEATURES:
    s = feature_stats[feat]
    std = s['std'] if s['std'] > 0 else 0.001

    # Increase by 1 std
    input_plus = input_df.copy()
    input_plus[feat] = input_plus[feat] + std
    rul_plus = float(model.predict(input_plus)[0])

    # Decrease by 1 std
    input_minus = input_df.copy()
    input_minus[feat] = input_minus[feat] - std
    rul_minus = float(model.predict(input_minus)[0])

    label, code = SENSOR_INFO.get(feat, (feat, feat))
    sensitivity_data.append({
        'Feature': f"{code}",
        'Description': label,
        'Current': st.session_state['current_values'][feat],
        '+1 Std → ΔRUL': rul_plus - base_rul,
        '-1 Std → ΔRUL': rul_minus - base_rul,
        'Range': f"{s['min']:.1f} – {s['max']:.1f}",
    })

sensitivity_df = pd.DataFrame(sensitivity_data)
sensitivity_df = sensitivity_df.sort_values('+1 Std → ΔRUL', ascending=True)

# Display as styled table
def highlight_rul(val):
    if val > 0:
        return 'background-color: rgba(0, 128, 0, 0.15)'
    elif val < 0:
        return 'background-color: rgba(255, 0, 0, 0.15)'
    return ''

styled = sensitivity_df.style.map(
    highlight_rul, subset=['+1 Std → ΔRUL', '-1 Std → ΔRUL']
).format({
    '+1 Std → ΔRUL': '{:+.1f}',
    '-1 Std → ΔRUL': '{:+.1f}',
    'Current': '{:.2f}',
})

st.dataframe(styled, use_container_width=True, height=500)

st.markdown("""
**วิธีอ่าน:**
- 🟢 เขียว = การเปลี่ยนแปลงนั้นทำให้ RUL **เพิ่มขึ้น** (ดีขึ้น)
- 🔴 แดง = การเปลี่ยนแปลงนั้นทำให้ RUL **ลดลง** (แย่ลง)
- ตัวเลขบวก/ลบ = จำนวน cycles ที่ RUL เปลี่ยน
""")

# ========================================================
# Feature Importance (Global)
# ========================================================
st.subheader("🏆 Global Feature Importance")

importance = model.feature_importances_
imp_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': importance,
}).sort_values('Importance', ascending=True).tail(15)

# Add labels
imp_df['Label'] = imp_df['Feature'].apply(
    lambda x: f"{SENSOR_INFO.get(x, (x, x))[1]}: {SENSOR_INFO.get(x, (x, x))[0][:30]}"
)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(imp_df['Label'], imp_df['Importance'], color='steelblue')
ax2.set_xlabel('Feature Importance')
ax2.set_title('Top 15 Most Important Features for RUL Prediction')
plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()

# ========================================================
# Summary
# ========================================================
st.divider()
st.subheader("📝 สรุปการวิเคราะห์")

# Find most impactful features from SHAP
shap_impact = pd.DataFrame({
    'Feature': FEATURES,
    'SHAP Value': shap_values[0],
}).sort_values('SHAP Value', key=abs, ascending=False)

top_positive = shap_impact[shap_impact['SHAP Value'] > 0].head(5)
top_negative = shap_impact[shap_impact['SHAP Value'] < 0].head(5)

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("#### 🟢 Features ที่เพิ่ม RUL")
    for _, row in top_positive.iterrows():
        label, code = SENSOR_INFO.get(row['Feature'], (row['Feature'], row['Feature']))
        st.write(f"  • **{code}** ({label}): +{row['SHAP Value']:.1f} cycles")

with col_neg:
    st.markdown("#### 🔴 Features ที่ลด RUL")
    for _, row in top_negative.iterrows():
        label, code = SENSOR_INFO.get(row['Feature'], (row['Feature'], row['Feature']))
        st.write(f"  • **{code}** ({label}): {row['SHAP Value']:.1f} cycles")

st.info(f"""
**ผลการทำนาย:** RUL = {predicted_rul_clipped:.1f} cycles -> {health_state}
**ต้นทุนโดยประมาณ:** ${cost_result['maintenance_cost']:,.0f} ({cost_result['maintenance_action']})

**คำแนะนำ:**
- ลดค่า features ที่ทำให้ RUL ลดลง (แดง) เพื่อยืดอายุการใช้งาน
- รักษาค่า features ที่ทำให้ RUL เพิ่มขึ้น (เขียว) ในระดับที่เหมาะสม
- Feature ที่สำคัญที่สุดคือ **sensor_13** (Corrected Fuel Flow) คิดเป็น {importance[FEATURES.index('sensor_13')]*100:.1f}% ของความสำคัญทั้งหมด
""")

# ========================================================
# Cost Scenario Comparison (from cost_model.py)
# ========================================================
st.divider()
st.subheader("📊 Cost Scenario Analysis")
st.markdown("ปรับ cost parameters เพื่อดูว่าถ้าสภาวะตลาดเปลี่ยน ต้นทุนจะเปลี่ยนเท่าไหร่")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    st.markdown("**Adjust Parameters:**")
    new_downtime_rate = st.slider("Unplanned Downtime ($/hr)", 10000, 25000, 15000, step=1000)
    new_labour_rate = st.slider("Labour Rate ($/hr)", 80, 200, 125, step=5)

with scenario_col2:
    cost_row = pd.DataFrame([{
        'RUL': predicted_rul_clipped,
        'health_state': health_state,
    }])
    scenarios = {
        'Baseline': {},
        'High Downtime': {'unplanned_downtime_cost_per_hour': new_downtime_rate},
        'High Labour': {'labour_rate_per_hour': new_labour_rate},
        'Both High': {
            'unplanned_downtime_cost_per_hour': new_downtime_rate,
            'labour_rate_per_hour': new_labour_rate,
        },
    }
    try:
        comparison = cost_estimator.compare_scenarios(cost_row, scenarios)
        st.dataframe(comparison.style.format({
            'total_cost': '${:,.0f}',
            'avg_cost': '${:,.0f}',
        }), use_container_width=True)
    except Exception:
        results = []
        original = cost_estimator.get_cost_parameters()
        for name, overrides in scenarios.items():
            cost_estimator.update_cost_parameters(overrides)
            r = cost_estimator.estimate_single(rul=predicted_rul_clipped, health_state=health_state)
            results.append({
                'Scenario': name,
                'Cost': f"${r['maintenance_cost']:,.0f}",
                'Action': r['maintenance_action'].split(' / ')[0],
            })
        cost_estimator.update_cost_parameters(original)
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
