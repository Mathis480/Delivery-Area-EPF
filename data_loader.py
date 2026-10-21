"""
Data loader for rolling quarter-hour (QH) forecasting.

Loads the master dataset, lags ex-post features by 2 hours, selects the feature sets S1-S4 from
features.csv and provides standardized calibration/test arrays per (zone, product, test day).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BENCHMARK_COL_PATTERN, CORR_THRESHOLDS, DATA_START, DUPLICATE_CORR, FEATURE_SET_BALANCE, FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR, FEATURES_CSV, LOOKBACK_DAYS, MASTER_PARQUET, N_QH, SHIFT_QH,
    TARGET_COL_PATTERN, ZONES,
)

SET_COLUMNS = {FEATURE_SET_MACRO: "S1", FEATURE_SET_NEIGHBOR: "S2", FEATURE_SET_FUNDAMENTAL: "S3", FEATURE_SET_BALANCE: "S4"}


@dataclass
class WindowData:
    """Standardized calibration and test arrays for one (zone, product, test day)."""
    X_train: np.ndarray       # filtered features (LASSO, cSVR)
    y_train: np.ndarray       # standardized price difference
    X_test: np.ndarray
    X_train_full: np.ndarray  # all features of the set (FA-NN)
    X_test_full: np.ndarray
    bm_train: np.ndarray      # naive price of the calibration days (cSVR correction kernel)
    col_mask: np.ndarray
    y_mean: float
    y_std: float


class QHDataLoader:
    """Holds the master dataset split into 96 tables (one per delivery quarter-hour, pure UTC)."""

    def __init__(self, parquet_path: str = MASTER_PARQUET, features_csv: str = FEATURES_CSV):
        catalog = pd.read_csv(features_csv)
        df = pd.read_parquet(parquet_path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)

        missing = sorted(set(catalog["name"]) - set(df.columns))
        if missing:
            raise ValueError(f"features.csv lists columns missing in the master dataset: {missing[:10]}")

        # Ex-post features are lagged by 2 hours
        shift_cols = catalog.loc[catalog["shift_needed_2h"] == "x", "name"].tolist()
        df[shift_cols] = df[shift_cols].shift(SHIFT_QH)

        # Feature sets per zone; catalog order defines column order
        inputs = catalog[catalog["tag"] == "input"]
        self._features = {
            (zone, fs): inputs.loc[inputs["model"].isin(["all", zone]) & inputs[col].notna(), "name"].tolist()
            for zone in ZONES for fs, col in SET_COLUMNS.items()
        }

        df["_qh_idx"] = (df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15).astype(np.int8)
        df["_delivery_date"] = df["Date"].dt.date
        df = df[df["Date"] >= pd.Timestamp(DATA_START, tz="UTC")].reset_index(drop=True)

        self._qh_tables = {q: t.reset_index(drop=True) for q, t in df.groupby("_qh_idx", sort=True)}
        self._qh_dates = {q: t["_delivery_date"].values for q, t in self._qh_tables.items()}
        assert len(self._qh_tables) == N_QH

        cols = ["Date", "_delivery_date", "_qh_idx"]
        cols += [p.format(zone=z) for z in ZONES for p in (TARGET_COL_PATTERN, BENCHMARK_COL_PATTERN)]
        self._targets = df[cols]

    def feature_names(self, zone: str, feature_set: str) -> list[str]:
        return self._features[(zone, feature_set)]

    def n_features(self, zone: str, feature_set: str) -> int:
        return len(self.feature_names(zone, feature_set))

    def test_targets(self, zone: str, start: str, end: str) -> pd.DataFrame:
        """Target, naive price, delivery day and product index of all test quarter-hours in [start, end]."""
        t = self._targets
        days = pd.to_datetime(t["_delivery_date"])
        sel = t[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
        return pd.DataFrame({
            "date": sel["_delivery_date"].astype(str).values,
            "qh": sel["_qh_idx"].values,
            "y": sel[TARGET_COL_PATTERN.format(zone=zone)].values,
            "naive": sel[BENCHMARK_COL_PATTERN.format(zone=zone)].values,
        })

    def get_window(
        self, zone: str, qh_idx: int, test_date: str, feature_set: str,
        corr_filter: bool = False, lookback_days: int = LOOKBACK_DAYS,
    ) -> WindowData | None:
        """Calibration window of `lookback_days` preceding days plus the test day for one product."""
        dates = self._qh_dates[qh_idx]
        target_date = pd.Timestamp(test_date).date()
        test_idx = np.searchsorted(dates, target_date)
        if test_idx >= len(dates) or dates[test_idx] != target_date:
            return None

        w_df = self._qh_tables[qh_idx].iloc[max(0, test_idx - lookback_days): test_idx + 1]
        n_train = len(w_df) - 1
        X_all = w_df[self.feature_names(zone, feature_set)].values.astype(np.float64)
        bm = w_df[BENCHMARK_COL_PATTERN.format(zone=zone)].values.astype(np.float64)
        y_diff = w_df[TARGET_COL_PATTERN.format(zone=zone)].values.astype(np.float64) - bm

        # Standardization and variable filters are estimated on the calibration window only
        y_mean, y_std = float(np.mean(y_diff[:n_train])), float(np.std(y_diff[:n_train]))
        X_train_raw = X_all[:n_train]
        # Constant variables and exact duplicates are removed; the correlation filter removes highly correlated ones
        col_mask = np.var(X_train_raw, axis=0) > 1e-8
        threshold = CORR_THRESHOLDS[feature_set] if corr_filter and feature_set in CORR_THRESHOLDS else DUPLICATE_CORR
        corr = np.corrcoef(X_train_raw[:, col_mask], rowvar=False)
        pairs = np.argwhere(np.triu(np.abs(corr) >= threshold, 1))
        col_mask[np.where(col_mask)[0][np.unique(pairs[:, 1])]] = False

        X_full = StandardScaler().fit(X_train_raw).transform(X_all)
        return WindowData(
            X_train=X_full[:n_train, col_mask],
            y_train=(y_diff[:n_train] - y_mean) / y_std,
            X_test=X_full[-1:, col_mask],
            X_train_full=X_full[:n_train],
            X_test_full=X_full[-1:],
            bm_train=bm[:-1],
            col_mask=col_mask,
            y_mean=y_mean,
            y_std=y_std,
        )
