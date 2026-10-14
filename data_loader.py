"""
Data loader for rolling quarter-hour (QH) EPF forecasting.

Loads master_dataset.parquet into memory, applies the 2h (8 QH) lag to ex-post features,
groups observations by delivery quarter-hour (qh_idx 0..95, Pure UTC),
removes zero-variance columns per-QH window, and provides scaled train/val/test slices.
"""
from __future__ import annotations

import os
import pickle
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
    SHIFT_QH, TARGET_COL_PATTERN, ZONES, S4_USE_STRUCTURED_RATIOS,
)

def load_feature_catalogue(
    features_csv: str = FEATURES_CSV, feature_set: str = "S1"
) -> tuple[dict[str, list[str]], list[str]]:
    """Parse features.csv to return:
      1. Zone-specific active feature lists for the specified feature set (S1..S4).
      2. Ex-post columns requiring a 2h backward shift (shift_needed_2h == 'x')."""
    feat = pd.read_csv(features_csv)
    active_df = feat[feat[feature_set].notna()]

    zone_features = {
        zone: active_df.loc[active_df["model"].isin(["all", zone]), "name"].tolist()
        for zone in ZONES
    }
    shift_cols = feat.loc[feat["shift_needed_2h"] == "x", "name"].tolist()
    return zone_features, shift_cols

# Data Container
@dataclass
class WindowData:
    """Scaled training and test arrays for a single (zone, QH, test_day)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: float
    benchmark_eur: float
    scaler_X: StandardScaler
    y_mean: float = 0.0              # Mean of y_diff on training set (for inverse transform)
    y_std: float = 1.0               # Std of y_diff on training set (for inverse transform)
    dates_train: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    col_mask: Optional[np.ndarray] = None
    X_train_full: Optional[np.ndarray] = None
    X_test_full: Optional[np.ndarray] = None
    bm_train: Optional[np.ndarray] = None

    def save_scaler(self, zone: str, qh_idx: int, date_str: str, folder: str = SCALERS_DIR) -> str:
        """Persist fitted StandardScaler to Models/scalers/."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"scaler_{zone}_QH{qh_idx:02d}_{date_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.scaler_X, f)
        return path

# Main Loader
class QHDataLoader:
    """High-performance data provider for quarter-hourly EPF:
      1. Shifts ex-post features by 8 QH (2h) to eliminate lookahead bias.
      2. Builds leakage-safe neighbor trajectories and spreads (S2).
      3. Loads partition feature sets S1, S3, S4 directly from features.csv.
      4. Maps UTC timestamps to delivery date and quarter-hour index (0..95).
      5. Splits data into 96 separate tables (one per QH position) for rolling slicing.
      Provides get_window() to extract scaled train/test arrays and differenced targets."""

    def __init__(
        self,
        parquet_path: str = MASTER_PARQUET,
        features_csv: str = FEATURES_CSV,
        start_delivery_date: str = "2021-10-01",
    ):
        self.parquet_path = parquet_path
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

        # 2. Build neighbor trajectories (national + regional)
        df, self._s2_features = self._build_neighbor_features(df)

        # 3. Build structured balance features (S4) if enabled
        if S4_USE_STRUCTURED_RATIOS:
            df, self._s4_features = self._build_balance_features(df)
        else:
            self._s4_features = {}

        # 4. Load feature sets S1, S3 (and S4 legacy fallback) directly from features.csv
        # Pre-filter valid features: strictly require tag == "input"
        valid_df = feat_df[
            feat_df["name"].isin(available_cols)
            & (feat_df["tag"] == "input")
        ]

        self._s1_features: dict[str, list[str]] = {}
        self._s3_features: dict[str, list[str]] = {}

        for zone in ZONES:
            z_df = valid_df[valid_df["model"].isin(["all", zone])]
            self._s1_features[zone] = z_df.loc[z_df["S1"].notna(), "name"].tolist()
            self._s3_features[zone] = z_df.loc[z_df["S3"].notna(), "name"].tolist()
            if not S4_USE_STRUCTURED_RATIOS:
                self._s4_features[zone] = z_df.loc[z_df["S4"].notna(), "name"].tolist()

        self._zone_features = {
            z: sorted(set(self._s1_features[z] + self._s2_features[z] + self._s3_features[z] + self._s4_features[z]))
            for z in ZONES
        }

        # 5. Map to delivery day and QH index (0..95) — PURE UTC
        df["_qh_idx"] = (df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15).astype(np.int8)
        df["_delivery_date"] = df["Date"].dt.date

        if start_delivery_date:
            df = df[df["Date"] >= pd.to_datetime(start_delivery_date, utc=True)].reset_index(drop=True)

        # 6. Split into 96 separate tables (one per QH position)
        grouped = dict(tuple(df.groupby("_qh_idx", sort=True)))
        self._qh_tables: dict[int, pd.DataFrame] = {}
        self._qh_dates: dict[int, np.ndarray] = {}
        for q in range(N_QH):
            sub = grouped[q].reset_index(drop=True)
            self._qh_tables[q] = sub
            self._qh_dates[q] = sub["_delivery_date"].values

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
        neighbor_features: dict[str, list[str]] = {z: [f"{z}_VWAP_90to105"] for z in ZONES}
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

                    # Intertemporal spread: neighbor price vs own naive benchmark (k=0)
                    price_col = f"{pfx}_VWAP_{win}"
                    spread_col = f"{pfx}_spread"
                    nb_series[price_col] = val
                    nb_series[spread_col] = val - bench
                    neighbor_features[zone].extend([price_col, spread_col])

                    # Cross-zone basis spread: regional price vs national benchmark
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

    @staticmethod
    def _build_balance_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Construct structured flow-to-load ratio and momentum features (S4)."""
        bal_series = {}
        s4_features: dict[str, list[str]] = {z: [] for z in ZONES}

        early_wins = ['345to360', '330to345', '315to330', '300to315', '285to300', '270to285', '255to270', '240to255']
        mid_wins   = ['225to240', '210to225', '195to210', '180to195', '165to180', '150to165', '135to150']
        late_wins  = ['120to135', '105to120', '90to105']
        all_wins   = early_wins + mid_wins + late_wins

        phases = [
            ('240to360', early_wins),
            ('135to240', mid_wins),
            ('90to135',  late_wins),
            ('90to360',  all_wins),
        ]

        for zone in ZONES:
            load = np.maximum(df[f"{zone}_load_forecast"].values, 100.0)
            partner_zones = [pz for pz in ZONES if pz != zone]

            # 1. Total balance ratios across phases (4 features)
            for p_name, wins in phases:
                cols = [f"{zone}_id_balance_{w}" for w in wins if f"{zone}_id_balance_{w}" in df.columns]
                col_name = f"{zone}_id_balance_ratio_{p_name}"
                bal_series[col_name] = df[cols].sum(axis=1).values / load
                s4_features[zone].append(col_name)

            # 2. Bilateral balance ratios to partner zones in mid and late phases (6 features)
            for pz in partner_zones:
                for p_name, wins in [('135to240', mid_wins), ('90to135', late_wins)]:
                    cols = [f"{zone}_id_balance_to_{pz}_{w}" for w in wins if f"{zone}_id_balance_to_{pz}_{w}" in df.columns]
                    col_name = f"{zone}_id_balance_to_{pz}_ratio_{p_name}"
                    val = df[cols].sum(axis=1).values / load if cols else np.zeros(len(df), dtype=np.float32)
                    bal_series[col_name] = val
                    s4_features[zone].append(col_name)

            # 3. Trade to self ratios across phases (4 features)
            for p_name, wins in phases:
                cols = [f"{zone}_id_trade_to_self_{w}" for w in wins if f"{zone}_id_trade_to_self_{w}" in df.columns]
                col_name = f"{zone}_id_trade_to_self_ratio_{p_name}"
                bal_series[col_name] = df[cols].sum(axis=1).values / load
                s4_features[zone].append(col_name)

            # 4. Momentum & Closing dynamics (2 features)
            mom_col = f"{zone}_id_balance_diff_ratio_90to120"
            b_90 = df[f"{zone}_id_balance_90to105"].values if f"{zone}_id_balance_90to105" in df.columns else np.zeros(len(df))
            b_105 = df[f"{zone}_id_balance_105to120"].values if f"{zone}_id_balance_105to120" in df.columns else np.zeros(len(df))
            bal_series[mom_col] = (b_90 - b_105) / load
            s4_features[zone].append(mom_col)

            clos_col = f"{zone}_id_trade_to_self_closing_ratio_90to105"
            t_90 = df[f"{zone}_id_trade_to_self_90to105"].values if f"{zone}_id_trade_to_self_90to105" in df.columns else np.zeros(len(df))
            bal_series[clos_col] = t_90 / load
            s4_features[zone].append(clos_col)

        if bal_series:
            df = pd.concat([df, pd.DataFrame(bal_series, index=df.index)], axis=1)

        return df, s4_features

    def get_window(
        self,
        zone: str,
        qh_idx: int,
        test_date: str | pd.Timestamp,
        feature_set: str = FEATURE_SET_MACRO,
        lookback_days: int = LOOKBACK_DAYS,
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
        start_idx = max(0, test_idx - lookback_days)
        w_df = qh_df.iloc[start_idx : test_idx + 1]
        n_train = len(w_df) - 1
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
            X_test=X_masked[-1:],
            y_test=float(y_std_all[-1]),
            benchmark_eur=float(bm_raw[-1]),
            scaler_X=scaler,
            y_mean=y_mean,
            y_std=y_std,
            dates_train=hist_dates,
            col_mask=col_mask,
            X_train_full=X_full[:n_train],
            X_test_full=X_full[-1:],
            bm_train=bm_raw[:-1],
        )
        if save_scaler:
            window.save_scaler(zone, qh_idx, str(target_date))
        return window

    def feature_names(self, zone: str | None = None, feature_set: str = FEATURE_SET_MACRO) -> list[str]:
        """Return active feature names for the specified zone and feature set."""
        if zone:
            mapping = {
                FEATURE_SET_MACRO: self._s1_features.get(zone, []),
                "S1": self._s1_features.get(zone, []),
                FEATURE_SET_NEIGHBOR: self._s2_features.get(zone, []),
                "S2": self._s2_features.get(zone, []),
                FEATURE_SET_FUNDAMENTAL: self._s3_features.get(zone, []),
                "S3": self._s3_features.get(zone, []),
                FEATURE_SET_BALANCE: self._s4_features.get(zone, []),
                "S4": self._s4_features.get(zone, []),
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