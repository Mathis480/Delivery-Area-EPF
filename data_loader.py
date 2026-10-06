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
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional
from config import (
    BENCHMARK_COL_PATTERN, FEATURES_CSV, LOOKBACK_DAYS, MASTER_PARQUET,
    N_QH, NEIGHBOR_SAFE_VWAP_WINDOWS, FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR, FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE, SCALERS_DIR,
    SHIFT_QH, TARGET_COL_PATTERN, VAL_DAYS, ZONES,
)

def load_feature_catalogue(
    features_csv: str = FEATURES_CSV, feature_set: str = "S1", scenario: Optional[str] = None
) -> tuple[dict[str, list[str]], list[str]]:

    """Parse features.csv to retrieve:
      - zone-specific active feature lists ('DE1'..'DE4') -> active feature names
        combining national ('all') features and zone-specific ('DEx') features.
        Only features with keep='yes' and the respective feature set (S1..S4) marked are selected.
        Features with keep='only past products' (windows 30to45, 45to60, 60to75, 75to90)
        are ex-post neighbor inputs constructed dynamically via NEIGHBOR_SAFE_VWAP_WINDOWS
        with safe .shift(-k) and are never loaded directly to prevent look-ahead bias.
      - All ex-post/realized columns across all zones requiring 2h backward shift.""" 

    feat = pd.read_csv(features_csv)
    fs = (scenario or feature_set).upper()
    active_df = feat[feat[fs].notna()]

    zone_features = {
        zone: active_df.loc[active_df["model"].isin(["all", zone]), "name"].tolist()
        for zone in ["DE1", "DE2", "DE3", "DE4"]
    }
    shift_cols = feat.loc[feat["shift_needed_2h"] == "x", "name"].tolist()
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
    y_mean: float = 0.0              # Mean of y_diff on training set (for inverse transform)
    y_std: float = 1.0               # Std of y_diff on training set (for inverse transform)
    dates_train: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    dates_val: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    col_mask: Optional[np.ndarray] = None
    X_train_full: Optional[np.ndarray] = None
    X_val_full: Optional[np.ndarray] = None
    X_test_full: Optional[np.ndarray] = None
    bm_trainval: Optional[np.ndarray] = None

    def save_scaler(self, zone: str, qh_idx: int, date_str: str, folder: str = SCALERS_DIR) -> str:
        """Persist fitted StandardScaler to Models/scalers/."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"scaler_{zone}_QH{qh_idx:02d}_{date_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.scaler_X, f)
        return path

# Main Loader
class QHDataLoader:
    """Data provider for rolling QH:
      1. Loads master_dataset.parquet and parses features.csv for active zone features.
      2. Shifts ex-post features (actuals, flows, reserves) by 8 QH (2h) backwards.
      3. Maps UTC timestamps to quarter-hour indices (qh_idx 0..95).
      4. Splits data into 96 separate tables (one per QH position) for rolling slicing.
      5. Provides get_window() to extract scaled train/val/test arrays and differenced targets. """

    def __init__(
        self,
        parquet_path: str = MASTER_PARQUET,
        features_csv: str = FEATURES_CSV,
        start_delivery_date: str = "2021-10-01",
        verbose: bool = False,
    ):
        self.parquet_path = parquet_path
        self.verbose = verbose
        self.start_delivery_date = start_delivery_date

        feat_df = pd.read_csv(features_csv)
        df = pq.read_table(parquet_path).to_pandas()
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)
        available_cols = set(df.columns)

        # 1. Shift ex-post features by 8 QH (2h)
        self._shift_cols = [c for c in feat_df.loc[feat_df["shift_needed_2h"] == "x", "name"] if c in available_cols]
        if self._shift_cols:
            df[self._shift_cols] = df[self._shift_cols].shift(SHIFT_QH)

        # 2. Build leakage-safe neighbor trajectories (national + regional)
        df, self._neighbor_features = self._build_neighbor_features(df)

        # 3. Load feature sets S1, S3, S4 directly from features.csv
        # Pre-filter valid features once: drop targets, unobserved columns, and Date
        valid_df = feat_df[
            feat_df["name"].isin(available_cols)
            & ~feat_df["name"].str.endswith(("VWAP_0to30", "VWAP_0to15", "VWAP_15to30"))
            & ~feat_df["name"].isin(["Date", "VWAP_0to30"])
        ]

        self._s1_features: dict[str, list[str]] = {}
        self._s3_features: dict[str, list[str]] = {}
        self._balance_features: dict[str, list[str]] = {}

        for zone in ZONES:
            z_df = valid_df[valid_df["model"].isin(["all", zone])]
            bench = BENCHMARK_COL_PATTERN.format(zone=zone)

            self._s1_features[zone] = z_df.loc[z_df["S1"].notna(), "name"].tolist()
            self._balance_features[zone] = [c for c in z_df.loc[z_df["S4"].notna(), "name"] if c != bench]
            s3_raw = [c for c in z_df.loc[z_df["S3"].notna(), "name"] if c != bench]
            self._s3_features[zone] = [bench] + s3_raw if bench in available_cols else s3_raw

        self._zone_features = {
            z: sorted(set(self._s1_features[z] + self._neighbor_features[z] + self._s3_features[z] + self._balance_features[z]))
            for z in ZONES
        }

        # 4. Map to delivery day and QH index (0..95) — PURE UTC
        df["_qh_idx"] = (df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15).astype(np.int8)
        df["_delivery_date"] = df["Date"].dt.date

        if start_delivery_date:
            df = df[df["_delivery_date"] >= pd.to_datetime(start_delivery_date).date()].reset_index(drop=True)

        self._qh_tables: dict[int, pd.DataFrame] = {
            q: df[df["_qh_idx"] == q].sort_values("Date").reset_index(drop=True)
            for q in range(N_QH)
        }
        self._qh_dates: dict[int, np.ndarray] = {
            q: self._qh_tables[q]["_delivery_date"].values for q in range(N_QH)
        }

    @staticmethod
    def _build_neighbor_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Construct leakage-safe neighbor trajectory features (S2) and spreads."""
        nb_series = {}
        # National neighbor VWAP
        for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
            col = f"VWAP_{win}"
            if col in df.columns:
                s = df[col].shift(-k)
                nb_series[f"DE_nb_k{k:+d}_VWAP_{win}"] = s.ffill() if k > 0 else s

        # Regional neighbor VWAP, spreads, and liquidity
        neighbor_features: dict[str, list[str]] = {z: [] for z in ZONES}
        for zone in ZONES:
            bench = df[f"{zone}_VWAP_90to105"]
            for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
                pfx = f"{zone}_nb_k{k:+d}"
                nat_col = f"DE_nb_k{k:+d}_VWAP_{win}"
                nat_s = nb_series.get(nat_col)

                # Price trajectories and spreads
                reg_src = f"{zone}_VWAP_{win}"
                if reg_src in df.columns:
                    val = df[reg_src].shift(-k)
                    if k > 0:
                        val = val.ffill()
                    if nat_s is not None:
                        val = val.fillna(nat_s)

                    price_col = f"{pfx}_VWAP_{win}"
                    spread_col = f"{pfx}_spread"
                    nb_series[price_col] = val
                    nb_series[spread_col] = val - bench
                    neighbor_features[zone].extend([price_col, spread_col])

                    if nat_s is not None:
                        cross_col = f"{zone}_vs_DE_nb_k{k:+d}"
                        nb_series[cross_col] = val - nat_s
                        neighbor_features[zone].append(cross_col)

                # Liquidity (volume & trades)
                for metric, suffix in (("total_volume", "vol"), ("total_trades", "trades")):
                    src = f"{zone}_VWAP_{metric}_{win}"
                    if src in df.columns:
                        col = f"{pfx}_{suffix}"
                        nb_series[col] = df[src].shift(-k).fillna(0.0)
                        neighbor_features[zone].append(col)

        if nb_series:
            df = pd.concat([df, pd.DataFrame(nb_series, index=df.index)], axis=1)
        return df, neighbor_features

    def get_window(
        self,
        zone: str,
        qh_idx: int,
        test_date: str | pd.Timestamp,
        feature_set: str = FEATURE_SET_MACRO,
        lookback_days: int = LOOKBACK_DAYS,
        val_days: int = VAL_DAYS,
        save_scaler: bool = False,
        filter_zero_var: bool = True,
        filter_corr: bool = False,
        corr_threshold: float = 0.80,
    ) -> Optional[WindowData]:
        qh_df = self._qh_tables[qh_idx]
        dates = self._qh_dates[qh_idx]
        target_date = pd.to_datetime(test_date).date()
        test_idx = np.searchsorted(dates, target_date)
        if test_idx >= len(dates) or dates[test_idx] != target_date:
            return None

        # 1. Contiguous slice: Lookback + test row in one go
        w_df = qh_df.iloc[test_idx - lookback_days : test_idx + 1]
        n_train = len(w_df) - 1 - val_days
        feat_cols = self.feature_names(zone, feature_set=feature_set)
        target_col = TARGET_COL_PATTERN.format(zone=zone)
        bench_col = BENCHMARK_COL_PATTERN.format(zone=zone)

        # 2. Arrays & target difference (VWAP_0to30 - VWAP_90to105)
        X_all = w_df[feat_cols].values.astype(np.float64)
        y_raw = w_df[target_col].values.astype(np.float64)
        bm_raw = w_df[bench_col].values.astype(np.float64)
        y_diff = y_raw - bm_raw

        # 3. Target standardization (strictly on training set)
        y_mean = float(np.mean(y_diff[:n_train]))
        y_std = float(np.std(y_diff[:n_train]))
        if y_std <= 1e-8:
            y_std = 1.0
        y_std_all = (y_diff - y_mean) / y_std

        # 4. Column filtering (strictly on training set)
        X_train_raw = X_all[:n_train]
        col_mask = (np.var(X_train_raw, axis=0) > 1e-8) if filter_zero_var else np.ones(X_all.shape[1], bool)
        if filter_corr and col_mask.sum() > 1:
            corr = np.nan_to_num(np.corrcoef(X_train_raw[:, col_mask], rowvar=False))
            p = np.argwhere(np.triu(np.abs(corr) >= corr_threshold, 1))
            if len(p):
                col_mask[np.where(col_mask)[0][np.unique(p[:, 1])]] = False

        # 5. Single-pass fit and transform for all rows
        scaler = StandardScaler().fit(X_train_raw)
        X_full = scaler.transform(X_all)
        X_masked = X_full[:, col_mask]

        hist_dates = pd.DatetimeIndex(w_df["Date"].values[:-1])
        window = WindowData(
            X_train=X_masked[:n_train],
            y_train=y_std_all[:n_train],
            X_val=X_masked[n_train:-1],
            y_val=y_std_all[n_train:-1],
            X_trainval=X_masked[:-1],
            y_trainval=y_std_all[:-1],
            X_test=X_masked[-1:],
            y_test=float(y_std_all[-1]),
            benchmark_eur=float(bm_raw[-1]),
            scaler_X=scaler,
            y_mean=y_mean,
            y_std=y_std,
            dates_train=hist_dates[:n_train],
            dates_val=hist_dates[n_train:],
            col_mask=col_mask,
            X_train_full=X_full[:n_train],
            X_val_full=X_full[n_train:-1],
            X_test_full=X_full[-1:],
            bm_trainval=bm_raw[:-1],
        )
        if save_scaler:
            window.save_scaler(zone, qh_idx, str(target_date))
        return window

    def feature_names(self, zone: str | None = None, feature_set: str = FEATURE_SET_MACRO) -> list[str]:
        """Return active feature names for the specified zone and feature set."""
        if zone:
            bench = BENCHMARK_COL_PATTERN.format(zone=zone)
            nb = self._neighbor_features.get(zone, [])
            bal = self._balance_features.get(zone, [])
            mapping = {
                FEATURE_SET_MACRO: self._s1_features.get(zone, []),
                "S1": self._s1_features.get(zone, []),
                FEATURE_SET_NEIGHBOR: [bench] + nb if bench not in nb else nb,
                "S2": [bench] + nb if bench not in nb else nb,
                FEATURE_SET_FUNDAMENTAL: self._s3_features.get(zone, []),
                "S3": self._s3_features.get(zone, []),
                FEATURE_SET_BALANCE: [bench] + bal if bench not in bal else bal,
                "S4": [bench] + bal if bench not in bal else bal,
            }
            return mapping.get(feature_set, self._s1_features.get(zone, []))

        excl = {"Date"}
        for z in ZONES:
            excl.add(TARGET_COL_PATTERN.format(zone=z))
        all_cols = sorted(set(c for cols in self._zone_features.values() for c in cols))
        return [c for c in all_cols if c not in excl]

    def n_features(self, zone: str | None = None, feature_set: str = FEATURE_SET_MACRO) -> int:
        return len(self.feature_names(zone, feature_set=feature_set))

    def close(self):
        """No-op for in-memory backend, maintains context manager compatibility."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()