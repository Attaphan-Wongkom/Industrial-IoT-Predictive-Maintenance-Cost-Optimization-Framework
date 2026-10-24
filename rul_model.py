"""
rul_model.py — RUL (Remaining Useful Life) & Health State Prediction Module

โมดูลสำหรับทำนาย RUL และ Health State ของเครื่องยนต์จากข้อมูลเซนเซอร์
ใช้ควบคู่กับ artifacts ที่ export จาก notebook (Cell 41)

Usage:
    from rul_model import RULPredictor

    predictor = RULPredictor(artifacts_dir='./artifacts')
    rul = predictor.predict_rul(X_features)
    health = predictor.predict_health_state(X_features)

    # หรือใช้ full pipeline ที่รวม feature engineering ตั้งแต่ raw data
    rul, health = predictor.predict_from_raw(raw_df)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


class RULPredictor:
    """โหลด models และ artifacts ที่ export จาก notebook เพื่อทำนาย RUL & Health State"""

    def __init__(self, artifacts_dir='./artifacts'):
        """
        โหลด ML models, feature lists, normalization stats จาก artifacts_dir

        Args:
            artifacts_dir: path ไปยังโฟลเดอร์ที่เก็บ .joblib และ .json ไฟล์
        """
        self.artifacts_dir = artifacts_dir

        # --- โหลด Models ---
        self.reg = joblib.load(os.path.join(artifacts_dir, 'rul_regressor.joblib'))
        self.clf = joblib.load(os.path.join(artifacts_dir, 'health_state_classifier.joblib'))
        self.kmeans = joblib.load(os.path.join(artifacts_dir, 'kmeans_op_regime.joblib'))

        # --- โหลด Feature List ---
        with open(os.path.join(artifacts_dir, 'features.json'), 'r') as f:
            self.features = json.load(f)

        # --- โหลด Model Config ---
        with open(os.path.join(artifacts_dir, 'model_config.json'), 'r') as f:
            self.config = json.load(f)

        # --- โหลด Normalization Stats ---
        self.regime_mu = joblib.load(os.path.join(artifacts_dir, 'regime_mu.joblib'))
        self.regime_sigma = joblib.load(os.path.join(artifacts_dir, 'regime_sigma.joblib'))

        # --- Extract config values ---
        self.useful_sensors = self.config['USEFUL_SENSORS']
        self.key_sensors = self.config['KEY_SENSORS']
        self.windows = self.config['WINDOWS']
        self.rul_cap = self.config['RUL_CAP']
        self.health_bins = self.config['HEALTH_BINS']
        self.health_labels = self.config['HEALTH_LABELS']
        self.n_regimes = self.config['N_REGIMES']
        self.settings_cols = self.config['SETTINGS_COLS']

    # ========================================================
    # Feature Engineering Pipeline
    # ========================================================

    def assign_op_regime(self, df):
        """กำหนด op_regime ด้วย KMeans model ที่เทรนจาก Train data"""
        X = df[self.settings_cols].values
        df['op_regime'] = self.kmeans.predict(X)
        return df

    def normalize_sensors(self, df):
        """
        Z-score normalization แยกตาม op_regime โดยใช้ mu/sigma จาก Train data
        สร้างคอลัมน์ sensor_X_norm
        """
        for regime in range(self.n_regimes):
            mask = df['op_regime'] == regime
            for s in self.useful_sensors:
                mu = self.regime_mu.loc[regime, s]
                sigma = self.regime_sigma.loc[regime, s]
                df.loc[mask, s + '_norm'] = (df.loc[mask, s] - mu) / sigma
        return df

    def add_rolling_features(self, df):
        """
        Rolling Mean & Rolling Std (window = 5, 10, 20) บน normalized sensors
        คำนวณแยกตาม unit_id เพื่อไม่ให้ข้ามเครื่องยนต์
        """
        norm_cols = [s + '_norm' for s in self.useful_sensors]
        for w in self.windows:
            for nc in norm_cols:
                df[nc + f'_rmean{w}'] = df.groupby('unit_id')[nc].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
                df[nc + f'_rstd{w}'] = df.groupby('unit_id')[nc].transform(
                    lambda x: x.rolling(window=w, min_periods=1).std()
                ).fillna(0)
        return df

    def add_slope_features(self, df):
        """Rate of Change (Slope) ของเซนเซอร์สำคัญ ที่ lag = 1 และ 5"""
        for ks in self.key_sensors:
            nc = ks + '_norm'
            for lag in [1, 5]:
                df[nc + f'_slope{lag}'] = df.groupby('unit_id')[nc].diff(lag).fillna(0)
        return df

    def add_calendar_features(self, df, date_col='date'):
        """
        เพิ่ม calendar features: is_weekend, is_holiday, is_peak_season
        ต้องมีคอลัมน์ date (datetime) ใน df
        """
        if date_col not in df.columns:
            return df

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Weekday/Weekend
        df['is_weekend'] = df[date_col].dt.weekday >= 5

        # Holiday (US/Illinois) — ใช้ holidays package
        try:
            import holidays as holidays_lib
            us_holidays = holidays_lib.US(years=[2024, 2025], state='IL')
            df['is_holiday'] = df[date_col].dt.date.isin(us_holidays)
        except ImportError:
            df['is_holiday'] = False

        # Peak Season
        peak_ranges = [
            ('06-15', '08-15'),   # Summer peak
            ('11-15', '11-28'),   # Thanksgiving week
            ('12-15', '01-02'),   # Christmas + New Year peak
        ]

        def _is_peak(d):
            if pd.isna(d):
                return False
            md = d.strftime('%m-%d')
            for start, end in peak_ranges:
                if start < end:
                    if start <= md <= end:
                        return True
                else:
                    if md >= start or md <= end:
                        return True
            return False

        df['is_peak_season'] = df[date_col].apply(_is_peak)
        return df

    def engineer_features(self, df):
        """
        Full Feature Engineering Pipeline สำหรับ raw C-MAPSS data
        ทำตามลำดับ: regime → normalize → rolling → slope → calendar

        Args:
            df: DataFrame ที่มีคอลัมน์ unit_id, cycle, op_cond_1/2/3, sensor_1..21
               และ date (optional, สำหรับ calendar features)

        Returns:
            DataFrame ที่มี features ครบทั้งหมด (145 features)
        """
        df = df.copy()
        df = self.assign_op_regime(df)
        df = self.normalize_sensors(df)
        df = self.add_rolling_features(df)
        df = self.add_slope_features(df)
        if 'date' in df.columns:
            df = self.add_calendar_features(df)
        return df

    # ========================================================
    # Prediction Functions
    # ========================================================

    def _prepare_features(self, df):
        """เลือกเฉพาะ features ที่โมเดลใช้ และ fillna(0)"""
        X = df[self.features].fillna(0)
        return X

    def predict_rul(self, df):
        """
        ทำนาย RUL (Remaining Useful Life) ในหน่วย cycles

        Args:
            df: DataFrame ที่มี features ครบ (raw หรือ engineered)
                 ถ้าเป็น raw จะ auto-run feature engineering

        Returns:
            np.array ของค่า RUL (0 - 130 cycles)
        """
        # ตรวจสอบว่า features ครบไหม ถ้าไม่ครบให้ engineer
        if not all(f in df.columns for f in self.features):
            df = self.engineer_features(df)
        X = self._prepare_features(df)
        return self.reg.predict(X)

    def predict_health_state(self, df):
        """
        ทำนาย Health State (3-class classification)

        Classes:
            - 'Critical (High Risk)': RUL <= 30 cycles
            - 'Degraded (Medium Risk)': 30 < RUL <= 80 cycles
            - 'Healthy (Low Risk)': RUL > 80 cycles

        Args:
            df: DataFrame ที่มี features ครบ (raw หรือ engineered)

        Returns:
            np.array ของ health state labels
        """
        if not all(f in df.columns for f in self.features):
            df = self.engineer_features(df)
        X = self._prepare_features(df)
        return self.clf.predict(X)

    def predict_health_state_proba(self, df):
        """
        ทำนาย probability ของแต่ละ health state class

        Returns:
            np.array ของ probabilities shape (n_samples, 3)
        """
        if not all(f in df.columns for f in self.features):
            df = self.engineer_features(df)
        X = self._prepare_features(df)
        return self.clf.predict_proba(X)

    def predict_from_raw(self, raw_df):
        """
        Full pipeline: raw data → feature engineering → predict RUL & health state

        Args:
            raw_df: DataFrame ที่มี unit_id, cycle, op_cond_1/2/3, sensor_1..21
                    และ date (optional)

        Returns:
            tuple (rul_array, health_state_array)
        """
        df = self.engineer_features(raw_df)
        rul = self.predict_rul(df)
        health = self.predict_health_state(df)
        return rul, health

    def rul_to_health_state(self, rul_values):
        """แปลงค่า RUL เป็น health state label โดยใช้ bins ที่กำหนด"""
        return pd.cut(
            pd.Series(rul_values),
            bins=self.health_bins,
            labels=self.health_labels,
        ).values


# ========================================================
# Convenience Functions (สำหรับใช้แบบ function-based ไม่ต้องสร้าง class)
# ========================================================

_default_predictor = None

def _get_predictor(artifacts_dir='./artifacts'):
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = RULPredictor(artifacts_dir)
    return _default_predictor

def predict_rul(X, artifacts_dir='./artifacts'):
    """ทำนาย RUL — convenience function"""
    predictor = _get_predictor(artifacts_dir)
    return predictor.predict_rul(X)

def predict_health_state(X, artifacts_dir='./artifacts'):
    """ทำนาย Health State — convenience function"""
    predictor = _get_predictor(artifacts_dir)
    return predictor.predict_health_state(X)


# ========================================================
# Main (สำหรับทดสอบ)
# ========================================================

if __name__ == '__main__':
    import sys

    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else './artifacts'
    predictor = RULPredictor(artifacts_dir)

    print(f'RULPredictor loaded from {artifacts_dir}')
    print(f'  Features: {len(predictor.features)}')
    print(f'  RUL cap: {predictor.rul_cap} cycles')
    print(f'  Health labels: {predictor.health_labels}')
    print(f'  Useful sensors: {len(predictor.useful_sensors)}')
    print(f'  Key sensors: {predictor.key_sensors}')
    print(f'  Windows: {predictor.windows}')
    print(f'  Regimes: {predictor.n_regimes}')
    print('\nReady for prediction!')
