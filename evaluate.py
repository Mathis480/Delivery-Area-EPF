"""
Results aggregation and evaluation for the EPF Delivery Area backtest.

Metrics:
  MAE   — Mean Absolute Error (EUR/MWh)
  rMAE  — Relative MAE = MAE_model / MAE_benchmark (naive VWAP_90to105)
  DM    — Diebold-Mariano test statistic vs. naive benchmark
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# DM test
# ---------------------------------------------------------------------------

def dm_test(
    e_model: np.ndarray,
    e_bm: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test: H0 = models have equal predictive accuracy.

    Parameters
    ----------
    e_model : forecast errors of the model (y_true - y_hat_model)
    e_bm    : forecast errors of the benchmark
    h       : forecast horizon (1 for one-step-ahead)

    Returns
    -------
    (dm_statistic, p_value)  p_value < 0.05 → model significantly beats benchmark
    """
    d = np.abs(e_model) - np.abs(e_bm)
    n = len(d)
    if n < 2:
        return 0.0, 1.0
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(var_d / n)
    p_val = 2 * t_dist.cdf(-abs(dm_stat), df=n - 1)
    return float(dm_stat), float(p_val)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_results(results_path: str) -> pd.DataFrame:
    """Load and validate a results CSV."""
    df = pd.read_csv(results_path, parse_dates=["test_date"])
    required = {"zone", "qh_idx", "test_date", "model", "pred_eur", "y_true_eur", "benchmark_eur"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results: {missing}")
    return df


def aggregate_results(
    df: pd.DataFrame,
    by: list[str] = ["zone", "model"],
) -> pd.DataFrame:
    """
    Compute MAE, rMAE, and DM test statistics aggregated over the given grouping.
    """
    rows = []

    for keys, grp in df.groupby(by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(by, keys))

        # Errors
        e_model = grp["y_true_eur"].values - grp["pred_eur"].values
        e_bm    = grp["y_true_eur"].values - grp["benchmark_eur"].values

        mae_model = float(np.mean(np.abs(e_model)))
        mae_bm    = float(np.mean(np.abs(e_bm)))
        rmae      = mae_model / mae_bm if mae_bm > 0 else np.nan

        dm_stat, p_val = dm_test(e_model, e_bm)

        row = {
            **key_dict,
            "n_obs": len(grp),
            "mae_eur": mae_model,
            "mae_benchmark": mae_bm,
            "rmae": rmae,
            "dm_stat": dm_stat,
            "p_val": p_val,
            "beats_bm": rmae < 1.0,
            "sig_5pct": p_val < 0.05,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by + ["rmae"] if "rmae" not in by else by)


def print_summary(df_agg: pd.DataFrame):
    """Pretty-print aggregated results table."""
    print(f"\n{'Zone':<6} {'Model':<8} {'MAE':>8} {'rMAE':>8} {'DM':>8} {'p':>7}  Beats?  Sig?")
    print("-" * 70)
    for _, row in df_agg.iterrows():
        zone  = row.get("zone", "")
        model = row.get("model", "")
        mae   = row.get("mae_eur", np.nan)
        rmae  = row.get("rmae", np.nan)
        dm    = row.get("dm_stat", np.nan)
        p     = row.get("p_val", np.nan)
        beats = "✓" if row.get("beats_bm") else " "
        sig   = "✓" if row.get("sig_5pct") else " "
        print(f"{zone:<6} {model:<8} {mae:>8.3f} {rmae:>8.4f} {dm:>8.3f} {p:>7.4f}  {beats}       {sig}")


def rMAE_by_qh(df: pd.DataFrame, zone: str, model: str) -> pd.Series:
    """Compute rMAE for each of the 96 QH positions for a given zone+model."""
    sub = df[(df["zone"] == zone) & (df["model"] == model)]
    result = {}

    for qh in range(96):
        g = sub[sub["qh_idx"] == qh]
        if len(g) == 0:
            result[qh] = np.nan
            continue
        e_m  = g["y_true_eur"].values - g["pred_eur"].values
        e_bm = g["y_true_eur"].values - g["benchmark_eur"].values
        mae_m  = np.mean(np.abs(e_m))
        mae_bm = np.mean(np.abs(e_bm))
        result[qh] = mae_m / mae_bm if mae_bm > 0 else np.nan

    return pd.Series(result, name=f"{zone}_{model}_rMAE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", help="Path to results CSV from backtest.py")
    parser.add_argument("--by", nargs="+", default=["zone", "model"])
    parser.add_argument("--qh_detail", action="store_true",
                        help="Also print per-QH rMAE summary")
    args = parser.parse_args()

    df = load_results(args.results_csv)
    df_agg = aggregate_results(df, by=args.by)
    print_summary(df_agg)

    if args.qh_detail:
        for zone in df["zone"].unique():
            for model in df["model"].unique():
                s = rMAE_by_qh(df, zone, model)
                print(f"\nrMAE by QH — {zone} {model}:")
                print(s.describe())
