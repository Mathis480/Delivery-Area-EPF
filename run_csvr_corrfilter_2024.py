"""
cSVR Re-Run with Puć & Janczura Correlation Filter (2024 Annual Backtest).

Loads the existing full4sets .npz results, recomputes cSVR predictions for
all 4 feature sets with filter_corr=True (|r| >= 0.80), and saves updated
results with "_corrfilter" suffix and also overwrites the canonical files.

This avoids retraining LASSO/MAML which remain unchanged.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

from config import (
    RESULTS_DIR,
    LOOKBACK_DAYS,
    VAL_DAYS,
    N_QH,
    ZONES,
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
    CSVR_USE_GAUSSIAN_CORRECTION,
    ENSEMBLE_CALIB_DAYS,
    ENSEMBLE_POWER,
)
from data_loader import QHDataLoader
from Models.csvr import train_csvr, predict_csvr
from Models.ensemble import compute_rolling_intelligent_ensemble

# Puć & Janczura (2024) differential correlation thresholds per feature set:
#   S1 (Macro):       |r| >= 0.80  — broad price information, many correlated lags
#   S2 (Neighbor):    |r| >= 0.95  — limited variables, higher threshold to preserve information
#   S3 (Fundamental): NO FILTER    — hand-picked exogenous variables (load, solar, wind, calendar)
#   S4 (Balance):     NO FILTER    — curated bilateral flow balances (our extension, not in Puć)
SET_CONFIG = [
    {"fs": FEATURE_SET_MACRO,       "label": "s1", "name": "S1 Macro",       "filter_corr": True,  "corr_threshold": 0.80},
    {"fs": FEATURE_SET_NEIGHBOR,    "label": "s2", "name": "S2 Neighbor",    "filter_corr": True,  "corr_threshold": 0.95},
    {"fs": FEATURE_SET_FUNDAMENTAL, "label": "s3", "name": "S3 Fundamental", "filter_corr": False, "corr_threshold": None},
    {"fs": FEATURE_SET_BALANCE,     "label": "s4", "name": "S4 Balance",     "filter_corr": False, "corr_threshold": None},
]
SET_LABELS = [c["label"] for c in SET_CONFIG]


def _backtransform(y_pred_scaled: float, y_mean: float, y_std: float, benchmark_eur: float) -> float:
    return float(y_pred_scaled * y_std + y_mean + benchmark_eur)


def dm_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray):
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    stat = mean_d / np.sqrt(var_d / len(d))
    p_val = 2 * (1 - stats.norm.cdf(np.abs(stat)))
    return stat, p_val


def run_zone_csvr_corrfilter(zone: str, loader: QHDataLoader):
    """Re-run cSVR with correlation filter for a single zone."""
    print(f"\n{'='*60}")
    print(f"  cSVR RE-RUN with Puć & Janczura Corr. Filter — {zone}")
    print(f"  S1: |r|≥0.80  |  S2: |r|≥0.95  |  S3,S4: exempt")
    print(f"{'='*60}")

    # Load existing results
    npz_path = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_multiset_2024.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No existing results for {zone} at {npz_path}")

    base = np.load(npz_path, allow_pickle=True)
    dates_arr = base["dates"]
    qh_arr = base["qh_idx"]
    y_true_arr = base["y_true"]
    bm_arr = base["benchmark"]
    n_total = len(dates_arr)
    mae_naive = float(np.mean(np.abs(y_true_arr - bm_arr)))

    # Store old cSVR MAEs for comparison
    old_mae = {}
    for sl in SET_LABELS:
        key = f"pred_csvr_{sl}"
        if key in base:
            old_mae[sl] = float(np.mean(np.abs(y_true_arr - base[key])))

    from joblib import Parallel, delayed

    # ---- Re-run cSVR with correlation filter (parallelized across 8 threads) ----
    preds = {sl: [0.0] * n_total for sl in SET_LABELS}
    cols_removed = {sl: [0] * n_total for sl in SET_LABELS}

    def process_qh(i):
        d_str = str(dates_arr[i])
        q = int(qh_arr[i])
        bench_eur = float(bm_arr[i])
        qh_preds = {}
        qh_cols = {}

        for cfg in SET_CONFIG:
            fs = cfg["fs"]
            sl = cfg["label"]
            w = loader.get_window(
                zone, q, d_str,
                feature_set=fs,
                lookback_days=LOOKBACK_DAYS,
                val_days=VAL_DAYS,
                filter_zero_var=True,
                filter_corr=cfg["filter_corr"],
                corr_threshold=cfg["corr_threshold"] if cfg["corr_threshold"] is not None else 0.80,
            )
            if w is not None and w.X_trainval is not None:
                n_orig = w.col_mask.shape[0] if w.col_mask is not None else 0
                n_kept = w.X_trainval.shape[1]
                qh_cols[sl] = n_orig - n_kept
                try:
                    model = train_csvr(
                        w.X_trainval, w.y_trainval,
                        bm_trainval=w.bm_trainval if CSVR_USE_GAUSSIAN_CORRECTION else None,
                    )
                    y_hat = predict_csvr(
                        model, w.X_test,
                        bm_test=np.array([bench_eur]) if CSVR_USE_GAUSSIAN_CORRECTION else None,
                    )
                    y_val = float(np.asarray(y_hat).ravel()[0])
                    qh_preds[sl] = _backtransform(y_val, w.y_mean, w.y_std, bench_eur)
                except Exception as ex:
                    print(f"[{zone}] Exception at index {i} ({sl}): {ex}")
                    qh_preds[sl] = bench_eur
            else:
                qh_cols[sl] = 0
                qh_preds[sl] = bench_eur

        return i, qh_preds, qh_cols

    t0 = time.time()
    results = Parallel(n_jobs=8, backend="threading")(
        delayed(process_qh)(i) for i in tqdm(range(n_total), desc=f"{zone} cSVR+CorrFilter (8 threads)")
    )
    for i, qh_p, qh_c in results:
        for sl in SET_LABELS:
            preds[sl][i] = qh_p[sl]
            cols_removed[sl][i] = qh_c[sl]

    elapsed = time.time() - t0
    print(f"[{zone}] cSVR+CorrFilter finished in {elapsed:.1f}s ({elapsed/n_total*1000:.1f}ms/QH)")

    # ---- Build updated data dict ----
    data = {k: base[k] for k in base.files}

    for sl in SET_LABELS:
        arr = np.array(preds[sl], dtype=np.float32)
        data[f"pred_csvr_{sl}"] = arr

    data["pred_csvr"] = data["pred_csvr_s1"]

    # Rebuild 2D matrices for cSVR
    dates_daily = data["dates_daily"]
    n_days = len(dates_daily)
    date_to_idx = {d: idx for idx, d in enumerate(dates_daily)}

    for sl in SET_LABELS:
        key = f"pred_csvr_{sl}"
        mat_2d = np.full((n_days, N_QH), np.nan, dtype=np.float64)
        arr_1d = data[key]
        for idx in range(n_total):
            d_i = date_to_idx[dates_arr[idx]]
            q_i = int(qh_arr[idx])
            mat_2d[d_i, q_i] = float(arr_1d[idx])
        data[f"{key}_2d"] = mat_2d

    data["pred_csvr_2d"] = data["pred_csvr_s1_2d"]

    # ---- Rebuild ensembles ----
    y_true_2d = data["y_true_2d"]
    bm_2d = data["benchmark_2d"]

    csvr_4sets = {
        f"csvr_{sl}": data[f"pred_csvr_{sl}_2d"] for sl in SET_LABELS
    }
    p_csvr_ens_pure_2d, _, m_csvr_pure = compute_rolling_intelligent_ensemble(
        csvr_4sets, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    p_csvr_ens_wn_2d, _, m_csvr_wn = compute_rolling_intelligent_ensemble(
        {**csvr_4sets, "naive": bm_2d}, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )

    # MAML ensembles (unchanged, but recomputed from stored predictions for consistency)
    maml_4sets = {
        f"maml_{sl}": data[f"pred_maml_{sl}_2d"] for sl in SET_LABELS
    }
    p_maml_ens_pure_2d, _, m_maml_pure = compute_rolling_intelligent_ensemble(
        maml_4sets, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )

    def _to_1d(mat_2d):
        res = np.zeros(n_total, dtype=np.float32)
        for idx in range(n_total):
            d_i = date_to_idx[dates_arr[idx]]
            q_i = int(qh_arr[idx])
            res[idx] = mat_2d[d_i, q_i]
        return res

    data["pred_csvr_ens_pure_2d"] = p_csvr_ens_pure_2d
    data["pred_csvr_ens_pure"] = _to_1d(p_csvr_ens_pure_2d)
    data["pred_csvr_ens_wn_2d"] = p_csvr_ens_wn_2d
    data["pred_csvr_ens_wn"] = _to_1d(p_csvr_ens_wn_2d)
    data["pred_maml_ens_pure_2d"] = p_maml_ens_pure_2d
    data["pred_maml_ens_pure"] = _to_1d(p_maml_ens_pure_2d)

    # ---- Save ----
    # Save with suffix for A/B comparison
    out_cf = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_corrfilter_2024.npz")
    np.savez_compressed(out_cf, **data)
    print(f"[{zone}] Saved correlation-filtered results to {out_cf}")

    # Also update canonical files for downstream evaluation
    out_full = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    np.savez_compressed(out_full, **data)
    out_multi = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_multiset_2024.npz")
    np.savez_compressed(out_multi, **data)
    print(f"[{zone}] Updated canonical results files.")

    # ---- Report ----
    print(f"\n{'─'*78}")
    print(f"  {zone} — cSVR A/B Comparison: Without vs With Puć & Janczura Corr. Filter")
    print(f"{'─'*78}")
    print(f"  Naive Benchmark:  {mae_naive:.4f} EUR/MWh (rMAE = 1.0000)")
    print(f"{'─'*78}")
    print(f"  {'Set':<22} {'Thresh':>7} {'OLD MAE':>10} {'NEW MAE':>10} {'Δ MAE':>10} {'OLD rMAE':>10} {'NEW rMAE':>10}")
    print(f"  {'─'*69}")

    for cfg in SET_CONFIG:
        sl = cfg["label"]
        label = cfg["name"]
        thresh_str = f"|r|≥{cfg['corr_threshold']}" if cfg["filter_corr"] else "—none—"
        new_mae = float(np.mean(np.abs(y_true_arr - data[f"pred_csvr_{sl}"])))
        old = old_mae.get(sl, np.nan)
        delta = new_mae - old
        print(f"  {label:<22} {thresh_str:>7} {old:>10.4f} {new_mae:>10.4f} {delta:>+10.4f} {old/mae_naive:>10.4f} {new_mae/mae_naive:>10.4f}")

    print(f"  {'─'*62}")
    print(f"  {'Ens cSVR Pure':<22} {'':>10} {m_csvr_pure['mae_ensemble']:>10.4f} {'':>10} {'':>10} {m_csvr_pure['rmae_ensemble']:>10.4f}")
    print(f"  {'Ens cSVR +Naive':<22} {'':>10} {m_csvr_wn['mae_ensemble']:>10.4f} {'':>10} {'':>10} {m_csvr_wn['rmae_ensemble']:>10.4f}")
    print(f"  {'Ens MAML Pure':<22} {'':>10} {m_maml_pure['mae_ensemble']:>10.4f} {'':>10} {'':>10} {m_maml_pure['rmae_ensemble']:>10.4f}")

    # Feature removal stats
    print(f"\n  Feature Removal Stats (avg cols removed per QH by correlation filter):")
    for cfg in SET_CONFIG:
        sl = cfg["label"]
        label = cfg["name"]
        removed = cols_removed[sl]
        if cfg["filter_corr"] and removed:
            print(f"    {label} (|r|≥{cfg['corr_threshold']}): mean={np.mean(removed):.1f}, median={np.median(removed):.0f}, max={np.max(removed)}")
        elif not cfg["filter_corr"]:
            print(f"    {label}: EXEMPT (hand-picked exogenous variables)")

    # DM: new cSVR vs old cSVR (using naive as proxy if old not available)
    dm_stat, dm_p = dm_test(y_true_arr, data["pred_csvr_ens_pure"], data["pred_maml_ens_pure"])
    print(f"\n  DM Test (MAML Ens vs new cSVR Ens): stat={dm_stat:.3f}, p={dm_p:.4e}")

    return {
        "zone": zone,
        "mae_naive": mae_naive,
        "mae_csvr_ens_pure": m_csvr_pure["mae_ensemble"],
        "rmae_csvr_ens_pure": m_csvr_pure["rmae_ensemble"],
        "mae_maml_ens_pure": m_maml_pure["mae_ensemble"],
        "rmae_maml_ens_pure": m_maml_pure["rmae_ensemble"],
    }


def main():
    parser = argparse.ArgumentParser(description="Re-run cSVR with Puć & Janczura correlation filter")
    parser.add_argument("--zone", type=str, default="all", choices=["all", "DE1", "DE2", "DE3", "DE4"])
    args = parser.parse_args()

    target_zones = ZONES if args.zone == "all" else [args.zone]

    loader = QHDataLoader()
    summaries = []

    for z in target_zones:
        s = run_zone_csvr_corrfilter(z, loader)
        summaries.append(s)

    df = pd.DataFrame(summaries)
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — cSVR with Correlation Filter")
    print(f"{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
