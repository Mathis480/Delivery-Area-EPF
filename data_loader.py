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
    FEATURE_SET_ALL,
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
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

    """Parse features.csv to retrieve:
      - zone-specific active feature lists ('DE1'..'DE4') -> active feature names
        combining national ('all') features and zone-specific ('DEx') features.
        Only features with keep='yes' and scenario/S1='x' are selected as direct features.
        Features with keep='only past products' (windows 30to45, 45to60, 60to75, 75to90)
        are ex-post neighbor inputs constructed dynamically via NEIGHBOR_SAFE_VWAP_WINDOWS
        with safe .shift(-k) and are never loaded directly to prevent look-ahead bias.
      - All ex-post/realized columns across all zones requiring 2h backward shift.""" 

    feat = pd.read_csv(features_csv)
    sc_col = scenario if scenario in feat.columns else ("S1" if "S1" in feat.columns else "scenario1")
    mask_sc = feat[sc_col].astype(str).str.strip().str.lower().isin(["x", sc_col.lower()])
    active_df = feat[mask_sc]

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
    """In-memory data provider for rolling QH.
    Pipeline:
      1. Loads master_dataset.parquet and parses features.csv for active zone features.
      2. Shifts ex-post features (actuals, flows, reserves) by 8 QH (2h) backwards.
      3. Maps UTC timestamps to quarter-hour indices (qh_idx 0..95).
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

        # Retain only features present in parquet (strictly purging target columns and Date at runtime)
        available_cols = set(df.columns)
        self._zone_features: dict[str, list[str]] = {
            z: [c for c in cols if c in available_cols and c not in {"Date", "VWAP_0to30", TARGET_COL_PATTERN.format(zone=z)} and not c.endswith("VWAP_0to30")]
            for z, cols in self._zone_feature_map.items()
        }
        self._shift_cols = [c for c in raw_shift_cols if c in available_cols]

        # Shift ex-post features by 8 QH (2h)
        if self._shift_cols:
            df[self._shift_cols] = df[self._shift_cols].shift(SHIFT_QH)

        # Build leakage-safe neighbor VWAP price trajectories on continuous 15-minute grid
        # Leakage-safe rule: window AtoB ending A min before neighbor delivery requires A >= 90 + 15*k
        nb_series = {}

        # 1. National neighbor series
        # For future contracts (k > 0) at the terminal boundary (31.12. 22:30/22:45 UTC), 2025 products lie outside the dataset.
        # We forward-fill the last observed neighbor values to avoid spurious self-proxying with the target contract's own windows.
        for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
            nat_src = f"VWAP_{win}"
            nat_nb = f"DE_nb_k{k:+d}_VWAP_{win}"
            if nat_src in df.columns:
                nat_val = df[nat_src].shift(-k)
                if k > 0:
                    nat_val = nat_val.ffill()
                nb_series[nat_nb] = nat_val

        # 2. Regional neighbor series, spreads, volumes, trades
        self._neighbor_features: dict[str, list[str]] = {z: [] for z in ZONES}
        for zone in ZONES:
            bench_col = f"{zone}_VWAP_90to105"
            for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
                reg_src = f"{zone}_VWAP_{win}"
                reg_nb = f"{zone}_nb_k{k:+d}_VWAP_{win}"
                nat_nb = f"DE_nb_k{k:+d}_VWAP_{win}"

                if reg_src in df.columns:
                    nb_val = df[reg_src].shift(-k)
                    # Strictly boundary forward-fill for future contracts (k > 0)
                    if k > 0:
                        nb_val = nb_val.ffill()
                    # Regional fallback to national aggregate in the same neighbor window (Rule 4)
                    if nat_nb in nb_series:
                        nb_val = nb_val.fillna(nb_series[nat_nb])

                    nb_series[reg_nb] = nb_val
                    self._neighbor_features[zone].append(reg_nb)

                    # Spread relative to target naive benchmark: (P_k - P_target_naive)
                    spread_col = f"{zone}_nb_k{k:+d}_spread"
                    nb_series[spread_col] = nb_val - df[bench_col]
                    self._neighbor_features[zone].append(spread_col)

                    # Cross-zone spread vs national: (P_k_zone - P_k_national)
                    if nat_nb in nb_series:
                        cross_col = f"{zone}_vs_DE_nb_k{k:+d}"
                        nb_series[cross_col] = nb_val - nb_series[nat_nb]
                        self._neighbor_features[zone].append(cross_col)

                # Volumes and trades for neighbor windows: strictly fillna(0.0) per AGENTS.md
                vol_src = f"{zone}_VWAP_total_volume_{win}"
                if vol_src in df.columns:
                    vol_col = f"{zone}_nb_k{k:+d}_vol"
                    nb_series[vol_col] = df[vol_src].shift(-k).fillna(0.0)
                    self._neighbor_features[zone].append(vol_col)

                trades_src = f"{zone}_VWAP_total_trades_{win}"
                if trades_src in df.columns:
                    trades_col = f"{zone}_nb_k{k:+d}_trades"
                    nb_series[trades_col] = df[trades_src].shift(-k).fillna(0.0)
                    self._neighbor_features[zone].append(trades_col)

            # Ensure all neighbor features are registered in _zone_features for FEATURE_SET_ALL
            if zone in self._zone_features:
                for c in self._neighbor_features[zone]:
                    if c not in self._zone_features[zone]:
                        self._zone_features[zone].append(c)

        if nb_series:
            df = pd.concat([df, pd.DataFrame(nb_series, index=df.index)], axis=1)

        # 3. Zone Balances & Sets S1..S4 (loaded directly from features.csv if present)
        feat_df = pd.read_csv(features_csv)
        self._balance_features: dict[str, list[str]] = {z: [] for z in ZONES}
        self._s1_features: dict[str, list[str]] = {z: [] for z in ZONES}
        self._s3_features: dict[str, list[str]] = {z: [] for z in ZONES}

        for zone in ZONES:
            mask_z = feat_df["model"].str.strip().isin(["all", zone])
            bench = BENCHMARK_COL_PATTERN.format(zone=zone)
            target_col = TARGET_COL_PATTERN.format(zone=zone)

            # Set 4 (Balances)
            if "S4" in feat_df.columns:
                s4_cols = feat_df.loc[mask_z & feat_df["S4"].astype(str).str.strip().str.lower().isin(["x", "s4"]), "name"].str.strip().tolist()
                self._balance_features[zone] = [c for c in s4_cols if c in available_cols and c != bench]
            else:
                safe_bal_windows = (
                    "345to360", "330to345", "315to330", "300to315", "285to300", "270to285",
                    "255to270", "240to255", "225to240", "210to225", "195to210", "180to195",
                    "165to180", "150to165", "135to150", "120to135", "105to120", "90to105",
                )
                for col in df.columns:
                    if col.startswith(zone) and ("_id_balance" in col or "_id_trade_to_self" in col):
                        if any(col.endswith(f"_{w}") for w in safe_bal_windows):
                            self._balance_features[zone].append(col)

            # Set 1 (Macro) — strictly purge national and regional VWAP_0to30 targets
            if "S1" in feat_df.columns:
                s1_cols = feat_df.loc[mask_z & feat_df["S1"].astype(str).str.strip().str.lower().isin(["x", "s1"]), "name"].str.strip().tolist()
                self._s1_features[zone] = [
                    c for c in s1_cols 
                    if c in available_cols 
                    and c not in {target_col, "VWAP_0to30", "Date"} 
                    and not c.endswith("VWAP_0to30") 
                    and not c.endswith("VWAP_0to15") 
                    and not c.endswith("VWAP_15to30")
                ]

            # Set 3 (Fundamental)
            if "S3" in feat_df.columns:
                s3_raw = feat_df.loc[mask_z & feat_df["S3"].astype(str).str.strip().str.lower().isin(["x", "s3"]), "name"].str.strip().tolist()
                fund_cols = [
                    c for c in s3_raw 
                    if c in available_cols 
                    and c not in {target_col, "VWAP_0to30", "Date"} 
                    and c != bench 
                    and not c.endswith("VWAP_0to30")
                ]
                if bench in available_cols:
                    fund_cols = [bench] + fund_cols
                self._s3_features[zone] = fund_cols

            # Register balance, S1, and S3 features in _zone_features for FEATURE_SET_ALL
            if zone in self._zone_features:
                for col in self._balance_features[zone] + self._s1_features[zone] + self._s3_features[zone]:
                    if col not in self._zone_features[zone]:
                        self._zone_features[zone].append(col)


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
        feature_set: str = FEATURE_SET_ALL,
        lookback_days: int = LOOKBACK_DAYS,
        val_days: int = VAL_DAYS,
        save_scaler: bool = False,
        filter_zero_var: bool = True,
        filter_corr: bool = False,
        corr_threshold: float = 0.80,
    ) -> Optional[WindowData]:
        """
        Returns scaled train, validation, and test arrays for a given (zone, qh_idx, test_date).
        Target is differenced against the naive benchmark: y_diff = VWAP_0to30 - VWAP_90to105.
        StandardScaler is fitted strictly on X_train.

        Args:
            feature_set: Feature subset to use ('all', 'set1_macro', 'set2_neighbor', 'set3_fundamental').
            filter_zero_var: If True, removes columns with zero variance in the training
                set (e.g. constant dummies within a QH, nighttime solar).
                Set to False for MAML which requires fixed input dimensions across QH.
            filter_corr: If True, applies Puc et al. correlation filter (|r| >= corr_threshold)
                strictly on X_train to remove redundant collinear regressors.
            corr_threshold: Correlation cutoff threshold (default 0.80 matching Puć et al.).
        """
        target_col = TARGET_COL_PATTERN.format(zone=zone)
        benchmark_col = BENCHMARK_COL_PATTERN.format(zone=zone)
        feature_cols = self.feature_names(zone, feature_set=feature_set)

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
        y_train_raw, y_val_raw = y_diff[:n_train], y_diff[n_train:]
        X_train_raw = X_raw[:n_train]
        X_val_raw = X_raw[n_train:]

        # Standardize target: fit ONLY on training set
        y_mean = float(np.mean(y_train_raw))
        y_std = float(np.std(y_train_raw))
        if y_std <= 1e-8:
            y_std = 1.0
        y_train = (y_train_raw - y_mean) / y_std
        y_val = (y_val_raw - y_mean) / y_std

        # Column selection mask (fit strictly on X_train to prevent look-ahead leakage)
        col_mask = np.ones(X_raw.shape[1], dtype=bool)

        if filter_zero_var:
            col_var = np.var(X_train_raw, axis=0)
            col_mask = col_mask & (col_var > 1e-8)

        if filter_corr and np.sum(col_mask) > 1:
            X_curr = X_train_raw[:, col_mask]
            corr = np.corrcoef(X_curr, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            p = np.argwhere(np.triu(np.abs(corr) >= corr_threshold, 1))
            if len(p) > 0:
                cols_to_del = np.unique(p[:, 1])
                curr_indices = np.where(col_mask)[0]
                col_mask[curr_indices[cols_to_del]] = False

        # Standardize features: fit strictly on full unmasked training set
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_raw)
        X_val_full = scaler.transform(X_val_raw)
        X_test_raw_full = test_row[feature_cols].values.astype(np.float64).reshape(1, -1)
        X_test_full = scaler.transform(X_test_raw_full)

        # Apply col_mask to get filtered features for linear/SVR models
        X_train_scaled = X_train_full[:, col_mask]
        X_val_scaled = X_val_full[:, col_mask]
        X_trainval = np.vstack([X_train_scaled, X_val_scaled])
        y_trainval = np.concatenate([y_train, y_val])
        X_test_scaled = X_test_full[:, col_mask]

        y_test_raw = float(test_row[target_col])
        benchmark_test = float(test_row[benchmark_col])
        y_test_diff = y_test_raw - benchmark_test
        y_test_scaled = (y_test_diff - y_mean) / y_std

        dates_history = pd.DatetimeIndex(history_df["Date"].values)

        window = WindowData(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            X_test=X_test_scaled,
            y_test=y_test_scaled,
            benchmark_eur=benchmark_test,
            scaler_X=scaler,
            y_mean=y_mean,
            y_std=y_std,
            dates_train=dates_history[:n_train],
            dates_val=dates_history[n_train:],
            col_mask=col_mask,
            X_train_full=X_train_full,
            X_val_full=X_val_full,
            X_test_full=X_test_full,
            bm_trainval=bm_raw,
        )

        if save_scaler:
            window.save_scaler(zone, qh_idx, str(target_date))

        return window

    def feature_names(self, zone: str | None = None, feature_set: str = FEATURE_SET_ALL) -> list[str]:
        """
        Return active feature names for the specified zone and feature set.
        Feature sets:
          - 'all': All features (macro + neighbor).
          - 'set1_macro': Own VWAP history, auction prices, AR lags, fundamentals, balancing (no neighbors).
          - 'set2_neighbor': Neighbor prices, spreads, cross-zone spreads, volumes, trades + naive anchor.
          - 'set3_fundamental': Pure fundamentals (load, solar, wind, balancing, calendar) + naive anchor.
        """
        excl = {"Date"}
        if zone:
            excl.add(TARGET_COL_PATTERN.format(zone=zone))
            all_cols = [c for c in self._zone_features.get(zone, []) if c not in excl]

            if feature_set == FEATURE_SET_ALL:
                return all_cols
            elif feature_set == FEATURE_SET_MACRO:
                if self._s1_features.get(zone):
                    return self._s1_features[zone]
                return [
                    c for c in all_cols
                    if "_nb_" not in c and "_vs_DE_nb_" not in c and not c.startswith("DE_nb_")
                    and "_id_balance" not in c and "_id_trade_to_self" not in c
                ]
            elif feature_set == FEATURE_SET_NEIGHBOR:
                bench = BENCHMARK_COL_PATTERN.format(zone=zone)
                nb_cols = [c for c in self._neighbor_features.get(zone, [])]
                if bench in all_cols and bench not in nb_cols:
                    return [bench] + nb_cols
                return nb_cols
            elif feature_set == FEATURE_SET_FUNDAMENTAL:
                if self._s3_features.get(zone):
                    return self._s3_features[zone]
                bench = BENCHMARK_COL_PATTERN.format(zone=zone)
                fund_keywords = ("load", "solar", "wind", "MRL", "SRL", "Weekday", "prediction_error")
                fund_cols = [c for c in all_cols if any(kw in c for kw in fund_keywords) and "_nb_" not in c and "_vs_DE_nb_" not in c and "_id_balance" not in c and "_id_trade_to_self" not in c]
                if bench in all_cols and bench not in fund_cols:
                    return [bench] + fund_cols
                return fund_cols
            elif feature_set == FEATURE_SET_BALANCE:
                bench = BENCHMARK_COL_PATTERN.format(zone=zone)
                bal_cols = [c for c in self._balance_features.get(zone, [])]
                if bench in all_cols and bench not in bal_cols:
                    return [bench] + bal_cols
                return bal_cols
            else:
                raise ValueError(f"Unknown feature_set: '{feature_set}'. Choose from: 'all', 'set1_macro', 'set2_neighbor', 'set3_fundamental', 'set4_balance'.")

        for z in ZONES:
            excl.add(TARGET_COL_PATTERN.format(zone=z))
        all_cols = sorted(set(c for cols in self._zone_features.values() for c in cols))
        return [c for c in all_cols if c not in excl]

    def n_features(self, zone: str | None = None, feature_set: str = FEATURE_SET_ALL) -> int:
        return len(self.feature_names(zone, feature_set=feature_set))

    def close(self):
        """No-op for in-memory backend, maintains context manager compatibility."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
