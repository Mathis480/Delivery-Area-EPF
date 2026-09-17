"""
Data loader for rolling quarter-hour (QH) EPF forecasting.

Loads master_dataset.parquet into memory, applies the 2h (8 QH) lag to ex-post features,
groups observations by delivery quarter-hour (qh_idx 0..95, Pure UTC),
removes zero-variance columns per-QH window, and provides scaled train/val/test slices.
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from config import (
    BENCHMARK_COL_PATTERN,
    FEATURES_CSV,
    LOOKBACK_DAYS,
    MASTER_PARQUET,
    N_QH,
    NEIGHBOR_SAFE_VWAP_WINDOWS,
    SCALERS_DIR,
    SCENARIO,
    SHIFT_QH,
    TARGET_COL_PATTERN,
    VAL_DAYS,
    ZONES,
)

# Feature Catalogue Parser
def load_feature_catalogue(
    features_csv: str = FEATURES_CSV, scenario: str = SCENARIO
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Parse features.csv to retrieve:
      1. Zone-isolated active feature lists: mapping zone ('DE1'..'DE4') -> active feature names
         combining national ('all') features and zone-specific ('DEx') features.
      2. All ex-post/realized columns across all zones requiring 2h backward shift (Rule 2.2 / 2.3).
    """
    feat = pd.read_csv(features_csv)
    mask_keep = feat["keep"].astype(str).str.strip().str.lower() == "yes"
    mask_sc = feat[scenario].astype(str).str.strip().str.lower() == "x"
    active_df = feat[mask_keep & mask_sc]

    zone_features: dict[str, list[str]] = {}
    for zone in ["DE1", "DE2", "DE3", "DE4"]:
        mask_z = active_df["model"].str.strip().isin(["all", zone])
        zone_features[zone] = active_df.loc[mask_z, "name"].str.strip().tolist()

    # All columns requiring 2h backward shift across the dataset
    shift_mask = feat["shift_needed_2h"].astype(str).str.strip().str.lower() == "x"
    shift_cols = feat.loc[shift_mask, "name"].str.strip().tolist()

    return zone_features, shift_cols

# Data Container
@dataclass
class WindowData:
    """Scaled training, validation, and test arrays for a single (zone, QH, test_day)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_trainval: np.ndarray
    y_trainval: np.ndarray
    X_test: np.ndarray
    y_test: float
    benchmark_eur: float
    scaler_X: StandardScaler
    dates_train: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    dates_val: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)

    def save_scaler(self, zone: str, qh_idx: int, date_str: str, folder: str = SCALERS_DIR) -> str:
        """Persist fitted StandardScaler to Models/scalers/."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"scaler_{zone}_QH{qh_idx:02d}_{date_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.scaler_X, f)
        return path


# Main Loader
class QHDataLoader:
    """
    In-memory data provider for rolling QH.

    Pipeline:
      1. Loads master_dataset.parquet and parses features.csv for active zone features.
      2. Shifts ex-post features (actuals, flows, reserves) by 8 QH (2h) backwards.
      3. Maps UTC timestamps to German delivery days and quarter-hour indices (qh_idx 0..95).
      4. Splits data into 96 separate tables (one per QH position) for rolling slicing.
      5. Provides get_window() to extract scaled train/val/test arrays and differenced targets.
    """
    def __init__(
        self,
        parquet_path: str = MASTER_PARQUET,
        features_csv: str = FEATURES_CSV,
        scenario: str = SCENARIO,
        start_delivery_date: str = "2021-10-01",
        verbose: bool = False,
    ):
        self.parquet_path = parquet_path
        self.verbose = verbose
        self.start_delivery_date = start_delivery_date

        # 1. Load active feature lists & all shiftable ex-post columns
        self._zone_feature_map, raw_shift_cols = load_feature_catalogue(features_csv, scenario)

        # 2. In-Memory ingestion from Parquet
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)

        # Retain only features present in parquet
        available_cols = set(df.columns)
        self._zone_features: dict[str, list[str]] = {
            z: [c for c in cols if c in available_cols]
            for z, cols in self._zone_feature_map.items()
        }
        self._shift_cols = [c for c in raw_shift_cols if c in available_cols]

        # Shift ex-post features by 8 QH (2h)
        if self._shift_cols:
            df[self._shift_cols] = df[self._shift_cols].shift(SHIFT_QH)

        # Build leakage-safe neighbor VWAP price trajectories on continuous 15-minute grid
        # Leakage-safe rule: window AtoB ending A min before neighbor delivery requires A >= 90 + 15*k
        nb_series = {}
        for zone in ZONES:
            for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
                src_col = f"{zone}_VWAP_{win}"
                nb_col = f"{zone}_nb_k{k:+d}_VWAP_{win}"
                if src_col in df.columns:
                    # Row i gets value of neighbor k (row i + k) -> shift(-k)
                    nb_series[nb_col] = df[src_col].shift(-k).ffill()
                    if zone in self._zone_features and nb_col not in self._zone_features[zone]:
                        self._zone_features[zone].append(nb_col)

        if nb_series:
            df = pd.concat([df, pd.DataFrame(nb_series, index=df.index)], axis=1)

        # Map to delivery day and QH index (0..95) — PURE UTC, no timezone conversion
        # Rule 2.1: All data uses strictly Pure UTC. No tz_convert to Europe/Berlin.
        df["_qh_idx"] = (df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15).astype(np.int8)
        df["_delivery_date"] = df["Date"].dt.date

        if start_delivery_date:
            min_date = pd.to_datetime(start_delivery_date).date()
            df = df[df["_delivery_date"] >= min_date].reset_index(drop=True)

        # Pre-group by QH position for fast slicing (no DST duplicates in UTC)
        self._qh_tables: dict[int, pd.DataFrame] = {
            q: df[df["_qh_idx"] == q].sort_values("Date").reset_index(drop=True)
            for q in range(N_QH)
        }
        self._qh_dates: dict[int, np.ndarray] = {
            q: self._qh_tables[q]["_delivery_date"].values for q in range(N_QH)
        }

        if verbose:
            print(f"[QHDataLoader] Loaded {len(df):,} rows into RAM ({table.nbytes / 1e6:.1f} MB), "
                  f"{len(self._shift_cols)} ex-post features shifted by 2h (8 QH).")
            for z, z_cols in self._zone_features.items():
                n_inputs = len([c for c in z_cols if c not in {'Date', TARGET_COL_PATTERN.format(zone=z)}])
                print(f"  └─ {z}: {n_inputs} active inputs ({len(z_cols)} total)")

    def get_window(
        self,
        zone: str,
        qh_idx: int,
        test_date: str | pd.Timestamp,
        lookback_days: int = LOOKBACK_DAYS,
        val_days: int = VAL_DAYS,
        save_scaler: bool = False,
        filter_zero_var: bool = True,
    ) -> Optional[WindowData]:
        """
        Returns scaled train, validation, and test arrays for a given (zone, qh_idx, test_date).
        Target is differenced against the naive benchmark: y_diff = VWAP_0to30 - VWAP_90to105.
        StandardScaler is fitted strictly on X_train.

        Args:
            filter_zero_var: If True, removes columns with zero variance in the training
                set (e.g. Hour/Quarter dummies constant within a QH, nighttime solar).
                Set to False for MAML which requires fixed input dimensions across QH.
        """
        target_col = TARGET_COL_PATTERN.format(zone=zone)
        benchmark_col = BENCHMARK_COL_PATTERN.format(zone=zone)
        feature_cols = self.feature_names(zone)

        qh_df = self._qh_tables[qh_idx]
        dates = self._qh_dates[qh_idx]

        target_date = pd.to_datetime(test_date).date()
        test_idx = np.searchsorted(dates, target_date)

        if test_idx >= len(dates) or dates[test_idx] != target_date:
            return None

        history_df = qh_df.iloc[test_idx - lookback_days : test_idx]
        if len(history_df) < lookback_days:
            warnings.warn(
                f"Lookback underflow: {zone} QH{qh_idx} {test_date} — "
                f"requested {lookback_days} days, got {len(history_df)}",
                UserWarning,
                stacklevel=2,
            )
        test_row = qh_df.iloc[test_idx]

        # Extract matrices (clean arrays without NaNs)
        X_raw = history_df[feature_cols].values.astype(np.float64)
        y_raw = history_df[target_col].values.astype(np.float64)
        bm_raw = history_df[benchmark_col].values.astype(np.float64)

        # Differenced target: y = VWAP_0to30 - VWAP_90to105
        y_diff = y_raw - bm_raw

        # Train/Validation temporal split
        n_train = len(history_df) - val_days
        X_train_raw, y_train = X_raw[:n_train], y_diff[:n_train]
        X_val_raw, y_val = X_raw[n_train:], y_diff[n_train:]

        # Optionally remove zero-variance columns (e.g., Hour/Quarter dummies
        # constant within a single QH, nighttime solar = 0). The mask is
        # computed on X_train only to avoid look-ahead bias.
        # Disabled for MAML which needs a fixed dimension across all 96 QH.
        if filter_zero_var:
            col_var = np.var(X_train_raw, axis=0)
            nonzero_mask = col_var > 0
            X_train_raw = X_train_raw[:, nonzero_mask]
            X_val_raw = X_val_raw[:, nonzero_mask]
        else:
            nonzero_mask = np.ones(X_train_raw.shape[1], dtype=bool)

        # Standardize: fit ONLY on training set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        X_trainval = np.vstack([X_train_scaled, X_val_scaled])
        y_trainval = np.concatenate([y_train, y_val])

        # Test observation
        X_test_raw = test_row[feature_cols].values.astype(np.float64).reshape(1, -1)
        X_test_raw = X_test_raw[:, nonzero_mask]
        X_test_scaled = scaler.transform(X_test_raw)

        y_test_raw = float(test_row[target_col])
        benchmark_test = float(test_row[benchmark_col])
        y_test_diff = y_test_raw - benchmark_test

        dates_history = pd.DatetimeIndex(history_df["Date"].values)

        window = WindowData(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            X_test=X_test_scaled,
            y_test=y_test_diff,
            benchmark_eur=benchmark_test,
            scaler_X=scaler,
            dates_train=dates_history[:n_train],
            dates_val=dates_history[n_train:],
        )

        if save_scaler:
            window.save_scaler(zone, qh_idx, str(target_date))

        return window

    def feature_names(self, zone: str | None = None) -> list[str]:
        """Return active feature names for the specified zone excluding target and Date."""
        excl = {"Date"}
        if zone:
            excl.add(TARGET_COL_PATTERN.format(zone=zone))
            cols = self._zone_features.get(zone, [])
            return [c for c in cols if c not in excl]
        for z in ZONES:
            excl.add(TARGET_COL_PATTERN.format(zone=z))
        all_cols = sorted(set(c for cols in self._zone_features.values() for c in cols))
        return [c for c in all_cols if c not in excl]

    def n_features(self, zone: str | None = None) -> int:
        return len(self.feature_names(zone))

    def close(self):
        """No-op for in-memory backend, maintains context manager compatibility."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
