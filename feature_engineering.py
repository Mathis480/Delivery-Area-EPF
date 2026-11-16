"""
Deterministic, row-wise feature engineering for the master dataset.

Derived features that do not depend on a calibration window are materialized here as columns of
master_dataset.parquet, so that every model input exists in the master dataset and is registered in
features.csv. Ex-post lags (`ex_post_lag_qh`), standardization and variable filters stay in data_loader.py.

* S2 (neighboring contracts): VWAP of the latest admissible window of contract s+k, spread to the own
  naive price, spread to the national neighbor VWAP, traded volume and trade count. The window of
  contract s+k ends 90 + 15k minutes before its own delivery, i.e. 90 minutes before delivery of s.
* S4 (inter-zonal trade balances): net balance (German areas plus external XBID counterparties),
  bilateral balances to the other German areas and intra-area traded volume, aggregated over trading
  phases and normalized by the zonal day-ahead load forecast.

Usage: python feature_engineering.py   (appends/overwrites the derived columns in MASTER_PARQUET)
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

from config import MASTER_PARQUET, NEIGHBOR_SAFE_VWAP_WINDOWS, ZONES

# Trading phases for S4 (windows in minutes before delivery of the own product)
EARLY_WINS = ["345to360", "330to345", "315to330", "300to315", "285to300", "270to285", "255to270", "240to255"]
MID_WINS = ["225to240", "210to225", "195to210", "180to195", "165to180", "150to165", "135to150"]
LATE_WINS = ["120to135", "105to120", "90to105"]
PHASES = [
    ("240to360", EARLY_WINS),
    ("135to240", MID_WINS),
    ("90to135", LATE_WINS),
    ("90to360", EARLY_WINS + MID_WINS + LATE_WINS),
]
LOAD_FLOOR_MW = 100.0


def build_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    """S2: neighbor prices, spreads and liquidity from admissible windows. Requires a continuous 15-min grid."""
    out = {}
    # National neighbor VWAP (helper: fallback and reference of the zonal-national spread)
    for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
        s = df[f"VWAP_{win}"].shift(-k)
        out[f"DE_nb_k{k:+d}_VWAP_{win}"] = s.ffill() if k > 0 else s  # k > 0: contracts beyond the end of the dataset
    for zone in ZONES:
        bench = df[f"{zone}_VWAP_90to105"]
        for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
            pfx = f"{zone}_nb_k{k:+d}"
            national = out[f"DE_nb_k{k:+d}_VWAP_{win}"]
            val = df[f"{zone}_VWAP_{win}"].shift(-k)
            val = (val.ffill() if k > 0 else val).fillna(national)
            out[f"{pfx}_VWAP_{win}"] = val
            out[f"{pfx}_spread"] = val - bench
            out[f"{zone}_vs_DE_nb_k{k:+d}"] = val - national
            out[f"{pfx}_vol"] = df[f"{zone}_VWAP_total_volume_{win}"].shift(-k).fillna(0.0)
            out[f"{pfx}_trades"] = df[f"{zone}_VWAP_total_trades_{win}"].shift(-k).fillna(0.0)
    return pd.DataFrame(out, index=df.index)


def build_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """S4: phase-aggregated trade balances normalized by the zonal day-ahead load forecast."""
    out = {}
    for zone in ZONES:
        load = np.maximum(df[f"{zone}_load_forecast"].values, LOAD_FLOOR_MW)

        def ratio(metric: str, wins: list[str]) -> np.ndarray:
            return df[[f"{zone}_{metric}_{w}" for w in wins]].sum(axis=1).values / load

        # 1. Net balance ratios across phases (4)
        for p_name, wins in PHASES:
            out[f"{zone}_id_balance_ratio_{p_name}"] = ratio("id_balance", wins)
        # 2. Bilateral balance ratios to the other zones in the middle and late phase (6)
        for pz in (z for z in ZONES if z != zone):
            for p_name, wins in (("135to240", MID_WINS), ("90to135", LATE_WINS)):
                out[f"{zone}_id_balance_to_{pz}_ratio_{p_name}"] = ratio(f"id_balance_to_{pz}", wins)
        # 3. Intra-area trade ratios across phases (4)
        for p_name, wins in PHASES:
            out[f"{zone}_id_trade_intra_ratio_{p_name}"] = ratio("id_trade_intra", wins)
        # 4. Change of the net balance and closing intra-area trade ratio (2)
        out[f"{zone}_id_balance_diff_ratio_90to120"] = (
            df[f"{zone}_id_balance_90to105"].values - df[f"{zone}_id_balance_105to120"].values) / load
        out[f"{zone}_id_trade_intra_closing_ratio_90to105"] = ratio("id_trade_intra", ["90to105"])
    return pd.DataFrame(out, index=df.index)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (sorted by Date) with S2 and S4 columns appended; existing derived columns are replaced."""
    df = df.sort_values("Date").reset_index(drop=True)
    step = pd.to_datetime(df["Date"], utc=True).diff().dropna().unique()
    if len(step) != 1 or step[0] != np.timedelta64(15, "m"):
        raise ValueError("Neighbor features require a continuous 15-minute grid.")

    derived = pd.concat([build_neighbor_features(df), build_balance_features(df)], axis=1)
    df = df.drop(columns=[c for c in derived.columns if c in df.columns])
    return pd.concat([df, derived.astype(np.float64)], axis=1)


if __name__ == "__main__":
    master = pd.read_parquet(MASTER_PARQUET)
    out = add_derived_features(master)
    backup = os.path.join(os.path.dirname(os.path.dirname(MASTER_PARQUET)), "old", "data_archive", "master_dataset_before_feature_engineering.parquet")
    if not os.path.exists(backup):
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        shutil.copy2(MASTER_PARQUET, backup)
    out.to_parquet(MASTER_PARQUET, index=False)
    print(f"Master dataset: {master.shape} -> {out.shape} (backup: {backup})")
