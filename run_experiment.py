"""
Out-of-sample 2024 benchmark across German delivery areas (DE1..DE4):
  - Naive Benchmark & Standard LASSO
  - Pure cSVR Ensemble (Puć & Janczura, 2024)
  - Pure MAML-NN Ensemble (Finn et al., 2017)
"""
from __future__ import annotations
import os
import sys

# Silence TensorFlow C++ & oneDNN informational logs before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time

_t0 = 0.0
def tic(): global _t0; _t0 = time.perf_counter()
def toc(): return time.perf_counter() - _t0

import multiprocessing
import numpy as np
from tqdm import tqdm

from config import (
    RESULTS_DIR, LOOKBACK_DAYS, N_QH, ZONES,
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE,
    CSVR_CORR_THRESH_S1, CSVR_CORR_THRESH_S2,
    ENSEMBLE_CALIB_DAYS, ENSEMBLE_POWER, ENSEMBLE_QH_ADAPTIVE, RANDOM_SEED,
    MAML_BYPASS_MODES, N_WORKERS,
)
from data_loader import QHDataLoader
from Models.linear import train_lasso, predict_lasso
from Models.csvr import train_csvr, predict_csvr
from Models.maml_nn import MAMLManager
from Models.ensemble import compute_rolling_weighted_average, dm_test

FEATURE_SETS = [
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE,
]
SET_TAGS = {
    FEATURE_SET_MACRO: "s1",
    FEATURE_SET_NEIGHBOR: "s2",
    FEATURE_SET_FUNDAMENTAL: "s3",
    FEATURE_SET_BALANCE: "s4",
}


def _backtransform(y_pred_scaled: float, y_mean: float, y_std: float, benchmark_eur: float) -> float:
    return float(y_pred_scaled * y_std + y_mean + benchmark_eur)


def _ensemble_and_save_results(zone: str, preds: dict[str, list[float]], data: dict, in_npz: str) -> None:
    """Computes rolling inverse-MAE ensembles (LASSO, cSVR, MAML) and saves compressed NPZ."""
    dates_arr = data["dates"]
    dates_daily = data["dates_daily"]
    n_days = len(dates_daily)
    date_to_idx = {d: idx for idx, d in enumerate(dates_daily)}
    qh_arr = data["qh_idx"]
    y_true_2d = data["y_true_2d"]
    bm_2d = data["benchmark_2d"]
    n_total = len(dates_arr)

    def _to_2d(arr_1d: np.ndarray) -> np.ndarray:
        mat = np.full((n_days, N_QH), np.nan, dtype=np.float64)
        for idx in range(n_total):
            mat[date_to_idx[dates_arr[idx]], int(qh_arr[idx])] = float(arr_1d[idx])
        return mat

    def _to_1d(mat_2d: np.ndarray) -> np.ndarray:
        res = np.zeros(n_total, dtype=np.float32)
        for idx in range(n_total):
            res[idx] = mat_2d[date_to_idx[dates_arr[idx]], int(qh_arr[idx])]
        return res

    # 1. Standard LASSO Ensembling
    lasso_sets_2d = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        arr = np.array(preds[f"lasso_std_{st}"], dtype=np.float32)
        data[f"pred_lasso_{st}"] = arr
        mat = _to_2d(arr)
        data[f"pred_lasso_{st}_2d"] = mat
        lasso_sets_2d[f"lasso_{st}"] = mat
    p_l_ens_2d, _, _ = compute_rolling_weighted_average(
        lasso_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER, qh_adaptive=ENSEMBLE_QH_ADAPTIVE
    )
    data["pred_lasso_ens_pure_2d"] = p_l_ens_2d
    data["pred_lasso_ens_pure"] = _to_1d(p_l_ens_2d)

    # 1b. LASSO + Naive Benchmark (Scenario 1b: Marcjasz et al., 2020)
    lasso_wn_sets_2d = dict(lasso_sets_2d)
    lasso_wn_sets_2d["naive"] = bm_2d
    p_l_wn_2d, _, _ = compute_rolling_weighted_average(
        lasso_wn_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER, qh_adaptive=ENSEMBLE_QH_ADAPTIVE
    )
    data["pred_lasso_ens_wn_2d"] = p_l_wn_2d
    data["pred_lasso_ens_wn"] = _to_1d(p_l_wn_2d)

    # 2a. Pure cSVR Ensembling (with Corr Filter - Puć & Janczura 2024 Standard)
    csvr_sets_2d = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        arr = np.array(preds[f"csvr_{st}"], dtype=np.float32)
        data[f"pred_csvr_{st}"] = arr
        mat = _to_2d(arr)
        data[f"pred_csvr_{st}_2d"] = mat
        csvr_sets_2d[f"csvr_{st}"] = mat
    p_c_ens_2d, _, _ = compute_rolling_weighted_average(
        csvr_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER, qh_adaptive=ENSEMBLE_QH_ADAPTIVE
    )
    data["pred_csvr_ens_pure_2d"] = p_c_ens_2d
    data["pred_csvr_ens_pure"] = _to_1d(p_c_ens_2d)

    # 2b. Pure cSVR Ensembling (No Corr Filter - Ablation Study)
    csvr_nocorr_sets_2d = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        arr = np.array(preds[f"csvr_nocorr_{st}"], dtype=np.float32)
        data[f"pred_csvr_nocorr_{st}"] = arr
        mat = _to_2d(arr)
        data[f"pred_csvr_nocorr_{st}_2d"] = mat
        csvr_nocorr_sets_2d[f"csvr_nocorr_{st}"] = mat
    p_c_nocorr_ens_2d, _, _ = compute_rolling_weighted_average(
        csvr_nocorr_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER, qh_adaptive=ENSEMBLE_QH_ADAPTIVE
    )
    data["pred_csvr_nocorr_ens_pure_2d"] = p_c_nocorr_ens_2d
    data["pred_csvr_nocorr_ens_pure"] = _to_1d(p_c_nocorr_ens_2d)

    # 3. MAML Ensembling across feature sets (Puć & Janczura, 2024 inverse-MAE weights)
    for mode in MAML_BYPASS_MODES:
        maml_sets_2d = {}
        for s in FEATURE_SETS:
            st = SET_TAGS[s]
            arr = np.array(preds[f"maml_{mode}_{st}"], dtype=np.float32)
            data[f"pred_maml_{mode}_{st}"] = arr
            mat = _to_2d(arr)
            data[f"pred_maml_{mode}_{st}_2d"] = mat
            maml_sets_2d[f"maml_{st}"] = mat

        p_m_ens_2d, _, _ = compute_rolling_weighted_average(
            maml_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER, qh_adaptive=ENSEMBLE_QH_ADAPTIVE
        )
        data[f"pred_maml_{mode}_ens_pure_2d"] = p_m_ens_2d
        data[f"pred_maml_{mode}_ens_pure"] = _to_1d(p_m_ens_2d)

        # Backward compatibility alias for Scenario 3
        if mode == "optlasso":
            data["pred_maml_ens_pure_2d"] = p_m_ens_2d
            data["pred_maml_ens_pure"] = _to_1d(p_m_ens_2d)

    # Save to canonical NPZ
    np.savez_compressed(in_npz, **data)


def _run_zone(zone: str) -> None:
    """Worker process computing all benchmark models for a single delivery area across 2024."""
    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    if not os.path.exists(in_npz):
        raise FileNotFoundError(f"Missing base NPZ file: {in_npz}")

    base = np.load(in_npz)
    data = {k: base[k] for k in base.files}
    dates_arr, qh_arr, bm_arr = data["dates"], data["qh_idx"], data["benchmark"]
    n_total = len(dates_arr)

    preds: dict[str, list[float]] = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        preds[f"lasso_std_{st}"] = []
        preds[f"csvr_{st}"] = []
        preds[f"csvr_nocorr_{st}"] = []
        for mode in MAML_BYPASS_MODES:
            preds[f"maml_{mode}_{st}"] = []

    loader = QHDataLoader()
    t_models = {"lasso": 0.0, "csvr": 0.0, "maml": 0.0}

    # Pretrain shared backbones per zone and bypass mode (deterministic RANDOM_SEED)
    maml_managers = {}
    for mode in MAML_BYPASS_MODES:
        maml_managers[mode] = {}
        for s in FEATURE_SETS:
            mgr = MAMLManager(zone=zone, n_features=loader.n_features(zone, s), feature_set=s, seed=RANDOM_SEED, bypass_mode=mode)
            mgr.pretrain_backbone(loader, test_start="2024-01-01")
            maml_managers[mode][s] = mgr

    # Main interval inference loop
    for i in tqdm(range(n_total), desc=f"Zone {zone:4s}", unit="QH"):
        d_str = str(dates_arr[i])
        q = int(qh_arr[i])
        bm_eur = float(bm_arr[i])

        for s in FEATURE_SETS:
            st = SET_TAGS[s]

            # 1. Standard LASSO
            use_corr = s in [FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR]
            corr_thresh = CSVR_CORR_THRESH_S1 if s == FEATURE_SET_MACRO else CSVR_CORR_THRESH_S2
            w = loader.get_window(
                zone, q, d_str, feature_set=s,
                lookback_days=LOOKBACK_DAYS,
                filter_zero_var=True, filter_corr=use_corr,
                corr_threshold=corr_thresh,
            )
            tic()
            m_l, _ = train_lasso(w.X_train, w.y_train)
            p_l = float(predict_lasso(m_l, w.X_test)[0])
            t_models["lasso"] += toc()
            preds[f"lasso_std_{st}"].append(_backtransform(p_l, w.y_mean, w.y_std, bm_eur))

            # 2a. Pure cSVR (with Corr Filter - Puć & Janczura 2024 Standard)
            tic()
            m_c = train_csvr(w.X_train, w.y_train, bm_train=w.bm_train)
            p_c = float(predict_csvr(m_c, w.X_test, bm_test=bm_eur)[0])
            t_models["csvr"] += toc()
            p_c_real = _backtransform(p_c, w.y_mean, w.y_std, bm_eur)
            preds[f"csvr_{st}"].append(p_c_real)

            # 2b. Pure cSVR (No Corr Filter - Ablation Study)
            if use_corr:
                w_c_no = loader.get_window(
                    zone, q, d_str, feature_set=s,
                    lookback_days=LOOKBACK_DAYS,
                    filter_zero_var=True, filter_corr=False,
                )
                m_c_no = train_csvr(w_c_no.X_train, w_c_no.y_train, bm_train=w_c_no.bm_train)
                p_c_no = float(predict_csvr(m_c_no, w_c_no.X_test, bm_test=bm_eur)[0])
                preds[f"csvr_nocorr_{st}"].append(_backtransform(p_c_no, w_c_no.y_mean, w_c_no.y_std, bm_eur))
            else:
                preds[f"csvr_nocorr_{st}"].append(p_c_real)

            # 3. MAML inference loop across configured bypass modes
            for mode in MAML_BYPASS_MODES:
                tic()
                if mode == "optlasso":
                    p_l_sup = m_l.predict(w.X_train)
                    p_scaled = maml_managers[mode][s].adapt_and_predict(
                        X_support=w.X_train_full,
                        y_support=w.y_train,
                        X_test=w.X_test_full,
                        linear_support=p_l_sup,
                        linear_test=p_l,
                    )
                else:
                    p_scaled = maml_managers[mode][s].adapt_and_predict(
                        X_support=w.X_train_full,
                        y_support=w.y_train,
                        X_test=w.X_test_full,
                    )
                t_models["maml"] += toc()
                preds[f"maml_{mode}_{st}"].append(_backtransform(p_scaled, w.y_mean, w.y_std, bm_eur))

    # Rolling Inverse-MAE Ensembling & Save
    _ensemble_and_save_results(zone, preds, data, in_npz)
    print(f"[{zone} Laufzeit] LASSO: {t_models['lasso']/60:.1f}m | cSVR: {t_models['csvr']/60:.1f}m | MAML: {t_models['maml']/60:.1f}m")


# Results Evaluation & Master Table
def get_zone_summary(zone: str) -> dict:
    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    d = np.load(in_npz)
    y = d["y_true"]
    bm = d["benchmark"]
    naive = float(np.mean(np.abs(y - bm)))

    summary = {"zone": zone, "naive": naive}
    keys_to_track = [
        ("lasso_std", "pred_lasso_ens_pure" if "pred_lasso_ens_pure" in d else "pred_lasso"),
        ("lasso_wn", "pred_lasso_ens_wn"),
        ("csvr_pure", "pred_csvr_ens_pure"),
        ("csvr_nocorr", "pred_csvr_nocorr_ens_pure"),
        ("maml_byp_m015", "pred_maml_optlasso_ens_pure" if "pred_maml_optlasso_ens_pure" in d else "pred_maml_ens_pure"),
        ("maml_pure", "pred_maml_nobypass_ens_pure"),
    ]

    for label, k in keys_to_track:
        summary[label] = float(np.mean(np.abs(y - d[k]))) if k in d else np.nan

    if "pred_maml_nobypass_ens_pure" in d and "pred_csvr_ens_pure" in d:
        s_val, p_val = dm_test(y, d["pred_maml_nobypass_ens_pure"], d["pred_csvr_ens_pure"])
        summary["dm_stat_maml_vs_csvr"] = s_val
        summary["dm_pval_maml_vs_csvr"] = p_val

    return summary


def print_master_table(summaries: list[dict]):
    print(f"\n{'='*104}")
    print(f"               MASTER BENCHMARK: ALL MODELS & SCENARIOS (4 GERMAN DELIVERY AREAS 2024)")
    print(f"{'='*104}")
    print(f"  {'Model / Scenario':50s} | {'DE1':>9s} | {'DE2':>9s} | {'DE3':>9s} | {'DE4':>9s} | {'Average':>9s}")
    print(f"  {'-'*102}")

    models = [
        ("Naive Benchmark (VWAP_90to105)", "naive"),
        ("Scenario 1: Standard LASSO", "lasso_std"),
        ("Scenario 1b: LASSO Ensemble (+ Naive)", "lasso_wn"),
        ("Scenario 2: Pure cSVR Ensemble (with Corr Filter)", "csvr_pure"),
        ("Scenario 2b: Pure cSVR Ensemble (No Corr Filter)", "csvr_nocorr"),
        ("Scenario 3: MAML-NN (Linear Bypass, m=0.15)", "maml_byp_m015"),
        ("Scenario 4: Pure MAML-NN Ensemble", "maml_pure"),
    ]

    for label, key in models:
        vals = []
        row = f"  {label:50s}"
        for s in summaries:
            v = s.get(key, np.nan)
            vals.append(v)
            row += f" | {v:9.4f}" if not np.isnan(v) else f" | {'—':>9s}"
        mean_v = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
        row += f" | {mean_v:9.4f}" if not np.isnan(mean_v) else f" | {'—':>9s}"
        print(row)

    print(f"  {'-'*102}")
    
    # HAC Diebold-Mariano test row (Pure MAML Ensemble vs Pure cSVR Ensemble)
    dm_stats = [s.get("dm_stat_maml_vs_csvr", np.nan) for s in summaries]
    dm_pvals = [s.get("dm_pval_maml_vs_csvr", np.nan) for s in summaries]
    dm_row_s = f"  {'HAC DM Stat (Pure MAML vs Pure cSVR)':50s}"
    dm_row_p = f"  {'HAC DM p-value':50s}"
    for s_val, p_val in zip(dm_stats, dm_pvals):
        dm_row_s += f" | {s_val:9.4f}" if not np.isnan(s_val) else f" | {'—':>9s}"
        dm_row_p += f" | {p_val:9.4f}" if not np.isnan(p_val) else f" | {'—':>9s}"
    mean_dm = float(np.nanmean(dm_stats)) if len(dm_stats) > 0 else np.nan
    dm_row_s += f" | {mean_dm:9.4f}" if not np.isnan(mean_dm) else f" | {'—':>9s}"
    dm_row_p += f" | {'—':>9s}"
    print(dm_row_s)
    print(dm_row_p)
    print(f"  {'-'*102}\n")


# Main Orchestrator
def main():
    if "--summary_only" in sys.argv or "--summary" in sys.argv:
        summaries = [get_zone_summary(z) for z in ZONES]
        print_master_table(summaries)
        return

    multiprocessing.set_start_method("spawn", force=True)
    num_workers = min(N_WORKERS, len(ZONES))
    print(f"[RUN] Starting 2024 benchmark ({', '.join(ZONES)}) with {num_workers} parallel workers...")

    tic()
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(_run_zone, ZONES)

    print(f"[DONE] Benchmark finished in {toc()/60:.1f} minutes.\n")
    summaries = [get_zone_summary(z) for z in ZONES]
    print_master_table(summaries)


if __name__ == "__main__":
    main()
