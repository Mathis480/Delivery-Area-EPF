"""
Data loader for rolling quarter-hour (QH) forecasting.

Loads the master dataset, lags ex-post features until their publication (features.csv), selects the feature sets S1-S4 from
features.csv and provides standardized calibration/test arrays per (zone, product, test day).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BENCHMARK_COL_PATTERN, CORR_FILTER_EXEMPT_PREFIXES, CORR_THRESHOLDS, DATA_START, DUPLICATE_CORR, FEATURE_SET_BALANCE, FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR, FEATURES_CSV, LOOKBACK_DAYS, MASTER_PARQUET, N_QH,
    DST_PRODUCTS, TARGET_COL_PATTERN, TIMEZONE, ZONES,
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
    """Holds the master dataset split into 96 tables (one per local delivery quarter-hour)."""

    def __init__(self, parquet_path: str = MASTER_PARQUET, features_csv: str = FEATURES_CSV):
        catalog = pd.read_csv(features_csv)
        df = pd.read_parquet(parquet_path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)

        missing = sorted(set(catalog["name"]) - set(df.columns))
        if missing:
            raise ValueError(f"features.csv lists columns missing in the master dataset: {missing[:10]}")

        # Ex-post features are lagged until they are published at the forecast time (lag per column in features.csv)
        for lag, cols in catalog.dropna(subset=["ex_post_lag_qh"]).groupby("ex_post_lag_qh")["name"]:
            df[cols.tolist()] = df[cols.tolist()].shift(int(lag))

        # Feature sets per zone; catalog order defines column order
        inputs = catalog[catalog["tag"] == "input"]
        self._features = {
            (zone, fs): inputs.loc[inputs["model"].isin(["all", zone]) & inputs[col].notna(), "name"].tolist()
            for zone in ZONES for fs, col in SET_COLUMNS.items()
        }
        self._exempt = {k: np.array([n.startswith(CORR_FILTER_EXEMPT_PREFIXES) for n in v]) for k, v in self._features.items()}

        local = df["Date"].dt.tz_convert(TIMEZONE)
        df["_qh_idx"] = (local.dt.hour * 4 + local.dt.minute // 15).astype(np.int8)
        df["_delivery_date"] = local.dt.date
        df = df[local >= pd.Timestamp(DATA_START, tz=TIMEZONE)].reset_index(drop=True)

        # The second delivery of 02:00-02:45 on the autumn DST day keeps its product but gets a slot behind the 96 products
        second = df.groupby(["_delivery_date", "_qh_idx"]).cumcount() > 0
        df["_slot"] = df["_qh_idx"].astype(int) + second * (N_QH - DST_PRODUCTS[0])

        self._qh_tables = {q: t.reset_index(drop=True) for q, t in df.groupby("_qh_idx", sort=True)}
        self._qh_dates = {q: t["_delivery_date"].values for q, t in self._qh_tables.items()}
        assert len(self._qh_tables) == N_QH

        # Numeric copy of all used columns per product: window rows are gathered in numpy (a pandas row take copies all columns)
        used = list(dict.fromkeys(
            [n for names in self._features.values() for n in names]
            + [p.format(zone=z) for z in ZONES for p in (TARGET_COL_PATTERN, BENCHMARK_COL_PATTERN)]
        ))
        pos = {n: i for i, n in enumerate(used)}
        self._qh_arrays = {q: t[used].to_numpy(np.float64) for q, t in self._qh_tables.items()}
        self._col_idx = {k: np.array([pos[n] for n in v]) for k, v in self._features.items()}
        self._bm_idx = {z: pos[BENCHMARK_COL_PATTERN.format(zone=z)] for z in ZONES}
        self._target_idx = {z: pos[TARGET_COL_PATTERN.format(zone=z)] for z in ZONES}

        cols = ["Date", "_delivery_date", "_slot"]
        cols += [p.format(zone=z) for z in ZONES for p in (TARGET_COL_PATTERN, BENCHMARK_COL_PATTERN)]
        self._targets = df[cols]

    def feature_names(self, zone: str, feature_set: str) -> list[str]:
        return self._features[(zone, feature_set)]

    def n_features(self, zone: str, feature_set: str) -> int:
        return len(self.feature_names(zone, feature_set))

    def test_targets(self, zone: str, start: str, end: str) -> pd.DataFrame:
        """Target, naive price, local delivery day and slot of all test quarter-hours in [start, end]."""
        t = self._targets
        days = pd.to_datetime(t["_delivery_date"])
        sel = t[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
        return pd.DataFrame({
            "date": sel["_delivery_date"].astype(str).values,
            "qh": sel["_slot"].values,
            "y": sel[TARGET_COL_PATTERN.format(zone=zone)].values,
            "naive": sel[BENCHMARK_COL_PATTERN.format(zone=zone)].values,
        })

    def get_window(
        self, zone: str, qh_idx: int, test_date: str, feature_set: str,
        corr_filter: bool = False, lookback_days: int = LOOKBACK_DAYS,
    ) -> WindowData | None:
        """Calibration window of `lookback_days` preceding observations plus the test row for one slot (product)."""
        product, occurrence = (qh_idx, 0) if qh_idx < N_QH else (qh_idx - N_QH + DST_PRODUCTS[0], 1)
        dates = self._qh_dates[product]
        target_date = pd.Timestamp(test_date).date()
        first = np.searchsorted(dates, target_date)
        if first + occurrence >= len(dates) or dates[first + occurrence] != target_date:
            return None

        # Calibration ends before the test day, also for the second delivery of the autumn DST day
        rows = np.r_[max(0, first - lookback_days): first, first + occurrence]
        arr = self._qh_arrays[product]
        n_train = len(rows) - 1
        # Column-major as pandas .values: the layout sets the summation order of the scaler and thus the last bits
        X_all = np.asfortranarray(arr[rows[:, None], self._col_idx[(zone, feature_set)]])
        bm = arr[rows, self._bm_idx[zone]]
        y_diff = arr[rows, self._target_idx[zone]] - bm

        # Standardization and variable filters are estimated on the calibration window only
        y_mean, y_std = float(np.mean(y_diff[:n_train])), float(np.std(y_diff[:n_train]))
        X_train_raw = X_all[:n_train]
        # Constant variables and exact duplicates are removed; the correlation filter removes highly correlated ones
        # Exact constancy (scale-free): the load-normalized S4 ratios are of order 1e-4, so a variance threshold would drop real signals
        col_mask = np.ptp(X_train_raw, axis=0) > 0
        threshold = CORR_THRESHOLDS[feature_set] if corr_filter and feature_set in CORR_THRESHOLDS else DUPLICATE_CORR
        corr = np.abs(np.triu(np.corrcoef(X_train_raw[:, col_mask], rowvar=False), 1))
        # As in Puć & Janczura (2024), exogenous variables are exempt from the correlation filter (not from exact duplicates)
        filterable = ~self._exempt[(zone, feature_set)][col_mask]
        hit = (corr >= DUPLICATE_CORR) | ((corr >= threshold) & filterable[:, None] & filterable[None, :])
        pairs = np.argwhere(hit)
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
