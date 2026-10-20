"""
Deterministic, row-wise feature engineering for the master dataset.

Derived features that do not depend on a calibration window are materialized here as columns of
master_dataset.parquet, so that every model input exists physically in the master dataset and is
registered in features.csv. Window-dependent preprocessing (2h ex-post shift, standardization,
variable filters) stays in data_loader.py.

* S2 (neighboring contracts): VWAP of the latest admissible window of contract s+k, spread to the own
  naive price, spread to the national neighbor VWAP, traded volume and trade count. The window of
  contract s+k ends 90 + 15k minutes before its own delivery, i.e. 90 minutes before delivery of s.
* S4 (inter-zonal trade balances): net, bilateral and self-trade balances aggregated over trading
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


def build_neighbor_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """S2: leakage-safe neighbor trajectories, spreads and liquidity. Requires a continuous 15-min grid."""
    nb_series = {}
    # National neighbor VWAP (helper for the zonal-national spread)
    for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
        col = f"VWAP_{win}"
        if col in df.columns:
            s = df[col].shift(-k)
            nb_series[f"DE_nb_k{k:+d}_VWAP_{win}"] = s.ffill() if k > 0 else s

    neighbor_features: dict[str, list[str]] = {z: [] for z in ZONES}
    for zone in ZONES:
        bench = df[f"{zone}_VWAP_90to105"]
        for k, win in NEIGHBOR_SAFE_VWAP_WINDOWS.items():
            pfx = f"{zone}_nb_k{k:+d}"
            nat_s = nb_series.get(f"DE_nb_k{k:+d}_VWAP_{win}")

            reg_src = f"{zone}_VWAP_{win}"
            if reg_src in df.columns:
                val = df[reg_src].shift(-k)
                if k > 0:
                    val = val.ffill()
                if nat_s is not None:
                    val = val.fillna(nat_s)

                price_col, spread_col = f"{pfx}_VWAP_{win}", f"{pfx}_spread"
                nb_series[price_col] = val
                nb_series[spread_col] = val - bench
                neighbor_features[zone].extend([price_col, spread_col])

                if nat_s is not None:
                    cross_col = f"{zone}_vs_DE_nb_k{k:+d}"
                    nb_series[cross_col] = val - nat_s
                    neighbor_features[zone].append(cross_col)

            for metric, suffix in (("total_volume", "vol"), ("total_trades", "trades")):
                src = f"{zone}_VWAP_{metric}_{win}"
                if src in df.columns:
                    col = f"{pfx}_{suffix}"
                    nb_series[col] = df[src].shift(-k).fillna(0.0)
                    neighbor_features[zone].append(col)

    return pd.DataFrame(nb_series, index=df.index), neighbor_features


def build_balance_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """S4: phase-aggregated trade balances normalized by the zonal day-ahead load forecast."""
    bal_series = {}
    s4_features: dict[str, list[str]] = {z: [] for z in ZONES}

    def _phase_sum(cols: list[str]) -> np.ndarray:
        cols = [c for c in cols if c in df.columns]
        return df[cols].sum(axis=1).values if cols else np.zeros(len(df))

    for zone in ZONES:
        load = np.maximum(df[f"{zone}_load_forecast"].values, LOAD_FLOOR_MW)
        partner_zones = [pz for pz in ZONES if pz != zone]

        # 1. Net balance ratios across phases (4)
        for p_name, wins in PHASES:
            col = f"{zone}_id_balance_ratio_{p_name}"
            bal_series[col] = _phase_sum([f"{zone}_id_balance_{w}" for w in wins]) / load
            s4_features[zone].append(col)

        # 2. Bilateral balance ratios to partner zones in middle and late phase (6)
        for pz in partner_zones:
            for p_name, wins in (("135to240", MID_WINS), ("90to135", LATE_WINS)):
                col = f"{zone}_id_balance_to_{pz}_ratio_{p_name}"
                bal_series[col] = _phase_sum([f"{zone}_id_balance_to_{pz}_{w}" for w in wins]) / load
                s4_features[zone].append(col)

        # 3. Self-trade ratios across phases (4)
        for p_name, wins in PHASES:
            col = f"{zone}_id_trade_to_self_ratio_{p_name}"
            bal_series[col] = _phase_sum([f"{zone}_id_trade_to_self_{w}" for w in wins]) / load
            s4_features[zone].append(col)

        # 4. Balance momentum and closing self-trade ratio (2)
        b_90, b_105 = _phase_sum([f"{zone}_id_balance_90to105"]), _phase_sum([f"{zone}_id_balance_105to120"])
        col = f"{zone}_id_balance_diff_ratio_90to120"
        bal_series[col] = (b_90 - b_105) / load
        s4_features[zone].append(col)

        col = f"{zone}_id_trade_to_self_closing_ratio_90to105"
        bal_series[col] = _phase_sum([f"{zone}_id_trade_to_self_90to105"]) / load
        s4_features[zone].append(col)

    return pd.DataFrame(bal_series, index=df.index), s4_features


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (sorted by Date) with S2 and S4 columns appended; existing derived columns are replaced."""
    df = df.sort_values("Date").reset_index(drop=True)
    step = pd.to_datetime(df["Date"], utc=True).diff().dropna().unique()
    if len(step) != 1 or step[0] != np.timedelta64(15, "m"):
        raise ValueError("Neighbor features require a continuous 15-minute grid.")

    nb_df, _ = build_neighbor_features(df)
    bal_df, _ = build_balance_features(df)
    derived = pd.concat([nb_df, bal_df], axis=1)
    df = df.drop(columns=[c for c in derived.columns if c in df.columns])
    return pd.concat([df, derived.astype(np.float64)], axis=1)


if __name__ == "__main__":
    master = pd.read_parquet(MASTER_PARQUET)
    out = add_derived_features(master)
    backup = os.path.join(os.path.dirname(os.path.dirname(MASTER_PARQUET)), "old", "data_archive", "master_dataset_before_feature_engineering.parquet")
    if not os.path.exists(backup):
        shutil.copy2(MASTER_PARQUET, backup)
    out.to_parquet(MASTER_PARQUET, index=False)
    print(f"Master dataset: {master.shape} -> {out.shape} (backup: {backup})")
