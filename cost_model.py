"""
cost_model.py — Maintenance Cost Estimation Module

โมดูลสำหรับคำนวณต้นทุนการซ่อมบำรุงจากค่า RUL และ Health State
อิงตามมาตรฐานอุตสาหกรรม (IATA, FAA, Boeing/Airbus)

Cost Parameters (benchmark):
    - Unplanned Downtime:  $15,000/hour, $225,000/event
    - Proactive Maintenance: $45,000/event
    - Early Replacement Waste: $225/cycle
    - Labour Rate: $125/hour-person

Usage:
    from cost_model import CostEstimator

    estimator = CostEstimator(artifacts_dir='./artifacts')
    result = estimator.estimate(df_with_rul_and_health)
    # หรือ
    result = estimator.estimate_single(rul=45, health_state='Degraded (Medium Risk)')

    # ปรับ cost parameters แบบ scenario / sensitivity analysis
    estimator.update_cost_parameters({'unplanned_downtime_cost_per_hour': 18000})
    result = estimator.estimate(df)
"""

import os
import json
import numpy as np
import pandas as pd


# ========================================================
# Default Cost Parameters (Industry Benchmark)
# ========================================================

DEFAULT_COST_PARAMETERS = {
    'unplanned_downtime_cost_per_hour': 15000,    # USD/hour
    'proactive_maintenance_cost_per_event': 45000, # USD/event
    'early_replacement_waste_per_cycle': 225,      # USD/cycle
    'labour_rate_per_hour': 125,                   # USD/hour-person
    'unplanned_downtime_cost_per_event': 225000,   # USD/event
}

# Cost multiplier ตามสถานการณ์ (calendar features)
COST_MULTIPLIERS = {
    'is_holiday': 1.5,       # วันหยุดนักขัตฤกษ์ = ค่าแรง 1.5x
    'is_peak_season': 1.3,   # Peak travel season = downtime cost สูงขึ้น 30%
    'is_weekend': 1.2,       # วันหยุดสุดสัปดาห์ = ค่าแรง 1.2x
}


class CostEstimator:
    """
    คำนวณต้นทุนการซ่อมบำรุงจาก RUL และ Health State
    รองรับการปรับ cost parameters สำหรับ sensitivity analysis
    """

    def __init__(self, artifacts_dir='./artifacts'):
        """
        โหลด cost parameters จาก artifacts_dir หรือใช้ค่า default

        Args:
            artifacts_dir: path ไปยังโฟลเดอร์ artifacts
        """
        self.artifacts_dir = artifacts_dir

        # โหลด cost parameters จาก JSON (ถ้ามี) ไม่งั้นใช้ default
        cost_path = os.path.join(artifacts_dir, 'cost_parameters.json')
        if os.path.exists(cost_path):
            with open(cost_path, 'r') as f:
                self.cost_parameters = json.load(f)
        else:
            self.cost_parameters = DEFAULT_COST_PARAMETERS.copy()

        self.multipliers = COST_MULTIPLIERS.copy()

    def update_cost_parameters(self, updates):
        """
        อัปเดต cost parameters สำหรับ sensitivity analysis / scenario planning

        Args:
            updates: dict ของ parameters ที่ต้องการแก้ เช่น
                {'unplanned_downtime_cost_per_hour': 18000}
        """
        self.cost_parameters.update(updates)

    def get_cost_parameters(self):
        """คืน cost parameters ปัจจุบัน"""
        return self.cost_parameters.copy()

    def get_cost_parameter(self, key):
        """คืนค่า cost parameter เฉพาะตัว"""
        return self.cost_parameters.get(key)

    # ========================================================
    # Core Cost Estimation Logic (จาก Notebook Cell 38)
    # ========================================================

    def estimate_maintenance_cost(self, row):
        """
        คำนวณ maintenance action และ cost จาก RUL และ health_state
        ฟังก์ชันนี้เหมือนกับใน Notebook Cell 38 แต่รองรับ cost multipliers

        Args:
            row: dict หรือ Series ที่มี 'RUL' และ 'health_state'
                  และอาจมี 'is_holiday', 'is_weekend', 'is_peak_season'

        Returns:
            dict ของ {maintenance_action, downtime_hours, maintenance_cost, cost_breakdown}
        """
        rul = row.get('RUL', 130)
        health = row.get('health_state', 'Healthy (Low Risk)')

        # กำหนด cost multipliers จาก calendar features
        labour_multiplier = 1.0
        downtime_multiplier = 1.0

        if row.get('is_holiday', False):
            labour_multiplier *= self.multipliers['is_holiday']
        if row.get('is_weekend', False):
            labour_multiplier *= self.multipliers['is_weekend']
        if row.get('is_peak_season', False):
            downtime_multiplier *= self.multipliers['is_peak_season']

        if health == 'Critical (High Risk)' or rul <= 30:
            # Unplanned Failure scenario
            downtime_hours = np.random.uniform(48, 72)
            base_cost = self.cost_parameters['unplanned_downtime_cost_per_event']
            hourly_cost = self.cost_parameters['unplanned_downtime_cost_per_hour'] * downtime_multiplier
            labour_cost = downtime_hours * self.cost_parameters['labour_rate_per_hour'] * labour_multiplier
            total_cost = downtime_hours * hourly_cost + base_cost + labour_cost
            action = 'Unplanned Failure / Emergency'
            breakdown = {
                'base_event_cost': base_cost,
                'hourly_downtime_cost': downtime_hours * hourly_cost,
                'labour_cost': labour_cost,
                'downtime_multiplier': downtime_multiplier,
                'labour_multiplier': labour_multiplier,
            }
        elif health == 'Degraded (Medium Risk)' or 30 < rul <= 80:
            # Proactive Maintenance scenario
            downtime_hours = np.random.uniform(8, 24)
            base_cost = self.cost_parameters['proactive_maintenance_cost_per_event']
            labour_cost = downtime_hours * self.cost_parameters['labour_rate_per_hour'] * labour_multiplier
            total_cost = base_cost + labour_cost
            action = 'Proactive Planned Maintenance'
            breakdown = {
                'base_event_cost': base_cost,
                'labour_cost': labour_cost,
                'downtime_multiplier': downtime_multiplier,
                'labour_multiplier': labour_multiplier,
            }
        else:
            # Early Replacement / Preventive scenario
            downtime_hours = np.random.uniform(1, 2)
            base_cost = 0
            cycle_waste = self.cost_parameters['early_replacement_waste_per_cycle']
            labour_cost = downtime_hours * self.cost_parameters['labour_rate_per_hour'] * labour_multiplier
            total_cost = labour_cost + cycle_waste
            action = 'Early Replacement (Preventive)'
            breakdown = {
                'cycle_waste_cost': cycle_waste,
                'labour_cost': labour_cost,
                'labour_multiplier': labour_multiplier,
            }

        return {
            'maintenance_action': action,
            'downtime_hours': round(downtime_hours, 1),
            'maintenance_cost': round(total_cost, 2),
            'cost_breakdown': {k: round(v, 2) for k, v in breakdown.items()},
        }

    def estimate_single(self, rul, health_state=None, is_holiday=False,
                        is_weekend=False, is_peak_season=False):
        """
        คำนวณ cost สำหรับ 1 record — convenience method

        Args:
            rul: ค่า RUL (cycles)
            health_state: label หรือ None (จะ infer จาก RUL)
            is_holiday, is_weekend, is_peak_season: calendar flags

        Returns:
            dict ของ {maintenance_action, downtime_hours, maintenance_cost, cost_breakdown}
        """
        if health_state is None:
            if rul <= 30:
                health_state = 'Critical (High Risk)'
            elif rul <= 80:
                health_state = 'Degraded (Medium Risk)'
            else:
                health_state = 'Healthy (Low Risk)'

        row = {
            'RUL': rul,
            'health_state': health_state,
            'is_holiday': is_holiday,
            'is_weekend': is_weekend,
            'is_peak_season': is_peak_season,
        }
        return self.estimate_maintenance_cost(row)

    def estimate(self, df):
        """
        คำนวณ maintenance cost สำหรับทุกแถวใน DataFrame

        Args:
            df: DataFrame ที่มี 'RUL' และ 'health_state' (required)
                และ 'is_holiday', 'is_weekend', 'is_peak_season' (optional)

        Returns:
            DataFrame ที่เพิ่มคอลัมน์: maintenance_action, downtime_hours, maintenance_cost
        """
        results = df.apply(self.estimate_maintenance_cost, axis=1)
        result_df = pd.concat([df, pd.DataFrame(list(results))], axis=1)
        return result_df

    def estimate_batch(self, df):
        """
        คำนวณ maintenance cost เป็น batch — เหมือน estimate() แต่คืน summary stats ด้วย

        Returns:
            tuple (result_df, summary_df)
        """
        result_df = self.estimate(df)

        summary = result_df.groupby('maintenance_action').agg(
            count=('maintenance_cost', 'count'),
            avg_cost=('maintenance_cost', 'mean'),
            min_cost=('maintenance_cost', 'min'),
            max_cost=('maintenance_cost', 'max'),
            avg_downtime=('downtime_hours', 'mean'),
        ).round(2)

        return result_df, summary

    def compare_scenarios(self, df, scenarios):
        """
        เปรียบเทียบต้นทุนในหลายสถานการณ์ (sensitivity analysis)

        Args:
            df: DataFrame ที่มี RUL และ health_state
            scenarios: dict ของ {scenario_name: {cost_param: value, ...}}

        Returns:
            DataFrame เปรียบเทียบ total cost ของแต่ละ scenario

        Example:
            scenarios = {
                'baseline': {},
                'high_downtime': {'unplanned_downtime_cost_per_hour': 20000},
                'low_labour': {'labour_rate_per_hour': 100},
            }
            comparison = estimator.compare_scenarios(df, scenarios)
        """
        original_params = self.cost_parameters.copy()
        results = {}

        for name, overrides in scenarios.items():
            self.cost_parameters = original_params.copy()
            self.update_cost_parameters(overrides)
            result_df = self.estimate(df)
            results[name] = {
                'total_cost': result_df['maintenance_cost'].sum(),
                'avg_cost': result_df['maintenance_cost'].mean(),
                'n_critical': (result_df['maintenance_action'] == 'Unplanned Failure / Emergency').sum(),
                'n_proactive': (result_df['maintenance_action'] == 'Proactive Planned Maintenance').sum(),
                'n_preventive': (result_df['maintenance_action'] == 'Early Replacement (Preventive)').sum(),
            }

        self.cost_parameters = original_params
        return pd.DataFrame(results).T

    def get_cost_parameter_table(self):
        """คืน DataFrame ของ cost parameters สำหรับ display"""
        return pd.DataFrame([
            {'name': 'Unplanned Downtime Cost (per hour)', 'key': 'unplanned_downtime_cost_per_hour',
             'value': self.cost_parameters['unplanned_downtime_cost_per_hour'], 'unit': 'USD/hour'},
            {'name': 'Unplanned Downtime Cost (per event)', 'key': 'unplanned_downtime_cost_per_event',
             'value': self.cost_parameters['unplanned_downtime_cost_per_event'], 'unit': 'USD/event'},
            {'name': 'Proactive Maintenance Cost (per event)', 'key': 'proactive_maintenance_cost_per_event',
             'value': self.cost_parameters['proactive_maintenance_cost_per_event'], 'unit': 'USD/event'},
            {'name': 'Early Replacement Waste (per cycle)', 'key': 'early_replacement_waste_per_cycle',
             'value': self.cost_parameters['early_replacement_waste_per_cycle'], 'unit': 'USD/cycle'},
            {'name': 'Labour Rate (per hour-person)', 'key': 'labour_rate_per_hour',
             'value': self.cost_parameters['labour_rate_per_hour'], 'unit': 'USD/hour-person'},
        ])


# ========================================================
# Convenience Functions (function-based API)
# ========================================================

def estimate_maintenance_cost(row, cost_parameters=None):
    """
    คำนวณ maintenance cost จาก RUL และ health_state
    ฟังก์ชันนี้เข้ากันได้กับโค้ดใน Notebook Cell 38

    Args:
        row: Series/dict ที่มี 'RUL' และ 'health_state'
    cost_parameters: dict ของ cost params (ถ้า None จะใช้ default)

    Returns:
        pd.Series ของ {maintenance_action, downtime_hours, maintenance_cost}
    """
    if cost_parameters is None:
        cost_parameters = DEFAULT_COST_PARAMETERS.copy()

    rul = row['RUL']
    health = row['health_state']

    if health == 'Critical (High Risk)' or rul <= 30:
        downtime_hours = np.random.uniform(48, 72)
        base_cost = cost_parameters['unplanned_downtime_cost_per_event']
        hourly_cost = cost_parameters['unplanned_downtime_cost_per_hour']
        total_cost = downtime_hours * hourly_cost + base_cost
        action = 'Unplanned Failure / Emergency'
    elif health == 'Degraded (Medium Risk)' or 30 < rul <= 80:
        downtime_hours = np.random.uniform(8, 24)
        base_cost = cost_parameters['proactive_maintenance_cost_per_event']
        hourly_cost = cost_parameters['labour_rate_per_hour']
        total_cost = downtime_hours * hourly_cost + base_cost
        action = 'Proactive Planned Maintenance'
    else:
        downtime_hours = np.random.uniform(1, 2)
        base_cost = 0
        cycle_waste = cost_parameters['early_replacement_waste_per_cycle']
        total_cost = downtime_hours * cost_parameters['labour_rate_per_hour'] + cycle_waste
        action = 'Early Replacement (Preventive)'

    return pd.Series({
        'maintenance_action': action,
        'downtime_hours': downtime_hours,
        'maintenance_cost': total_cost
    })


# ========================================================
# Main (สำหรับทดสอบ)
# ========================================================

if __name__ == '__main__':
    import sys

    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else './artifacts'
    estimator = CostEstimator(artifacts_dir)

    print('CostEstimator loaded')
    print('\nCost Parameters:')
    print(estimator.get_cost_parameter_table().to_string(index=False))

    # ทดสอบ single estimate
    print('\n--- Single Estimate Tests ---')
    for rul, label in [(15, 'Critical'), (50, 'Degraded'), (100, 'Healthy')]:
        result = estimator.estimate_single(rul=rul, health_state=f'{label}')
        print(f'  RUL={rul:3d} ({label:10s}): {result["maintenance_action"]:35s} '
              f'${result["maintenance_cost"]:>10,.0f}  ({result["downtime_hours"]}h)')

    # ทดสอบ calendar multipliers
    print('\n--- Calendar Multiplier Test (Critical, Peak Season, Holiday) ---')
    result = estimator.estimate_single(
        rul=15, health_state='Critical (High Risk)',
        is_holiday=True, is_peak_season=True
    )
    print(f'  Cost: ${result["maintenance_cost"]:,.0f}  (includes multipliers)')
    print(f'  Breakdown: {result["cost_breakdown"]}')
