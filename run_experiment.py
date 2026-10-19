"""
Out-of-sample 2024 benchmark across German delivery areas (DE1..DE4):
  - Naive Benchmark & Standard LASSO
  - Pure cSVR Ensemble (Puć & Janczura, 2024)
  - Pure MAML-NN Ensemble (Finn et al., 2017)
"""
#%%
from __future__ import annotations
import os
import sys
import time
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import multiprocessing
import numpy as np
from tqdm import tqdm

from config import (
    RESULTS_DIR, LOOKBACK_DAYS, N_QH, ZONES,
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE,
    CSVR_CORR_THRESH_S1, CSVR_CORR_THRESH_S2,
    ENSEMBLE_CALIB_DAYS, ENSEMBLE_POWER, ENSEMBLE_QH_ADAPTIVE, RANDOM_SEED,
    MAML_BYPASS_MODES, MAML_TRAINING_MODES, N_WORKERS,
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
    dates_arr, qh_arr = data["dates"], data["qh_idx"]
    n_days, n_total = len(data["dates_daily"]), len(dates_arr)
    d2i = {d: idx for idx, d in enumerate(data["dates_daily"])}
    y_true_2d, bm_2d = data["y_true_2d"], data["benchmark_2d"]

    def _to_2d(arr: np.ndarray) -> np.ndarray:
        m = np.full((n_days, N_QH), np.nan, dtype=np.float64)
        for i in range(n_total):
            m[d2i[dates_arr[i]], int(qh_arr[i])] = float(arr[i])
        return m

    def _to_1d(mat: np.ndarray) -> np.ndarray:
        return np.array([mat[d2i[dates_arr[i]], int(qh_arr[i])] for i in range(n_total)], dtype=np.float32)

    def _ens(sets: dict[str, np.ndarray], prefix: str, add_naive: bool = False):
        s = dict(sets)
        if add_naive:
            s["naive"] = bm_2d
        ens_2d, _, _ = compute_rolling_weighted_average(
            s, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER, qh_adaptive=ENSEMBLE_QH_ADAPTIVE
        )
        data[f"{prefix}_2d"] = ens_2d
        data[prefix] = _to_1d(ens_2d)
        return ens_2d

    def _collect(pred_prefix: str, store_prefix: str | None = None):
        sp = store_prefix or pred_prefix
        sets = {}
        for s in FEATURE_SETS:
            st = SET_TAGS[s]
            arr = np.array(preds[f"{pred_prefix}_{st}"], dtype=np.float32)
            data[f"pred_{sp}_{st}"] = arr
            mat = _to_2d(arr)
            data[f"pred_{sp}_{st}_2d"] = mat
            sets[f"{sp}_{st}"] = mat
        return sets

    # 1. LASSO Ensembles
    lasso_sets = _collect("lasso_std", store_prefix="lasso")
    _ens(lasso_sets, "pred_lasso_ens_pure")
    _ens(lasso_sets, "pred_lasso_ens_wn", add_naive=True)

    # 2. cSVR Ensembles
    if any(len(preds.get(f"csvr_{SET_TAGS[s]}", [])) == n_total for s in FEATURE_SETS):
        _ens(_collect("csvr"), "pred_csvr_ens_pure")
    if any(len(preds.get(f"csvr_nocorr_{SET_TAGS[s]}", [])) == n_total for s in FEATURE_SETS):
        _ens(_collect("csvr_nocorr"), "pred_csvr_nocorr_ens_pure")

    # 3. MAML Ensembles across configured training and bypass modes
    for tm in MAML_TRAINING_MODES:
        for mode in MAML_BYPASS_MODES:
            sets = _collect(f"maml_{tm}_{mode}")
            p_2d = _ens(sets, f"pred_maml_{tm}_{mode}_ens_pure")
            if tm == "warmstart" or len(MAML_TRAINING_MODES) == 1:
                for s in FEATURE_SETS:
                    st = SET_TAGS[s]
                    data[f"pred_maml_{mode}_{st}"] = data[f"pred_maml_{tm}_{mode}_{st}"]
                    data[f"pred_maml_{mode}_{st}_2d"] = data[f"pred_maml_{tm}_{mode}_{st}_2d"]
                data[f"pred_maml_{mode}_ens_pure_2d"] = p_2d
                data[f"pred_maml_{mode}_ens_pure"] = _to_1d(p_2d)
                if mode == "optlasso":
                    data["pred_maml_ens_pure_2d"] = p_2d
                    data["pred_maml_ens_pure"] = _to_1d(p_2d)

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
        for tm in MAML_TRAINING_MODES:
            for mode in MAML_BYPASS_MODES:
                preds[f"maml_{tm}_{mode}_{st}"] = []

    skip_csvr = "--skip_csvr" in sys.argv or "--maml_only" in sys.argv
    loader = QHDataLoader()
    t_models = {"lasso": 0.0, "csvr": 0.0, **{f"maml_{tm}": 0.0 for tm in MAML_TRAINING_MODES}}

    # Pretrain shared backbones per zone, training mode, and bypass mode
    maml_managers = {}
    for tm in MAML_TRAINING_MODES:
        maml_managers[tm] = {}
        for mode in MAML_BYPASS_MODES:
            maml_managers[tm][mode] = {}
            for s in FEATURE_SETS:
                mgr = MAMLManager(
                    zone=zone, n_features=loader.n_features(zone, s),
                    feature_set=s, seed=RANDOM_SEED, bypass_mode=mode,
                    training_mode=tm,
                )
                mgr.pretrain_backbone(loader, test_start="2024-01-01")
                maml_managers[tm][mode][s] = mgr

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
                zone, q, d_str, feature_set=s, lookback_days=LOOKBACK_DAYS,
                filter_zero_var=True, filter_corr=use_corr, corr_threshold=corr_thresh,
            )
            t0 = time.perf_counter()
            m_l, _ = train_lasso(w.X_train, w.y_train)
            p_l = float(predict_lasso(m_l, w.X_test)[0])
            t_models["lasso"] += time.perf_counter() - t0
            preds[f"lasso_std_{st}"].append(_backtransform(p_l, w.y_mean, w.y_std, bm_eur))

            # 2. Pure cSVR models
            if not skip_csvr:
                t0 = time.perf_counter()
                m_c = train_csvr(w.X_train, w.y_train, bm_train=w.bm_train)
                p_c = float(predict_csvr(m_c, w.X_test, bm_test=bm_eur)[0])
                t_models["csvr"] += time.perf_counter() - t0
                p_c_real = _backtransform(p_c, w.y_mean, w.y_std, bm_eur)
                preds[f"csvr_{st}"].append(p_c_real)

                if use_corr:
                    w_c_no = loader.get_window(zone, q, d_str, feature_set=s, lookback_days=LOOKBACK_DAYS, filter_zero_var=True, filter_corr=False)
                    m_c_no = train_csvr(w_c_no.X_train, w_c_no.y_train, bm_train=w_c_no.bm_train)
                    p_c_no = float(predict_csvr(m_c_no, w_c_no.X_test, bm_test=bm_eur)[0])
                    preds[f"csvr_nocorr_{st}"].append(_backtransform(p_c_no, w_c_no.y_mean, w_c_no.y_std, bm_eur))
                else:
                    preds[f"csvr_nocorr_{st}"].append(p_c_real)

            # 3. MAML inference loop across configured training and bypass modes
            p_l_sup = m_l.predict(w.X_train) if "optlasso" in MAML_BYPASS_MODES else None
            for tm in MAML_TRAINING_MODES:
                t0 = time.perf_counter()
                for mode in MAML_BYPASS_MODES:
                    lin_sup = p_l_sup if mode == "optlasso" else None
                    lin_test = p_l if mode == "optlasso" else None
                    p_scaled = maml_managers[tm][mode][s].adapt_and_predict(
                        X_support=w.X_train_full, y_support=w.y_train, X_test=w.X_test_full,
                        linear_support=lin_sup, linear_test=lin_test,
                    )
                    preds[f"maml_{tm}_{mode}_{st}"].append(_backtransform(p_scaled, w.y_mean, w.y_std, bm_eur))
                t_models[f"maml_{tm}"] += time.perf_counter() - t0

    # Rolling Inverse-MAE Ensembling & Save
    _ensemble_and_save_results(zone, preds, data, in_npz)
    t_parts = [f"{k.upper()}: {v/60:.1f}m ({v/n_total*1000:.1f} ms/QH)" for k, v in t_models.items()]
    return f"[{zone} Laufzeit] {' | '.join(t_parts)}"


def get_zone_summary(zone: str) -> dict:
    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    d = np.load(in_npz)
    y = d["y_true"]
    bm = d["benchmark"]
    summary = {"zone": zone, "naive": float(np.mean(np.abs(y - bm)))}

    keys_to_track = [
        ("lasso_std", "pred_lasso_ens_pure" if "pred_lasso_ens_pure" in d else "pred_lasso"),
        ("lasso_wn", "pred_lasso_ens_wn"),
        ("csvr_pure", "pred_csvr_ens_pure"),
        ("csvr_nocorr", "pred_csvr_nocorr_ens_pure"),
        ("maml_ws_optlasso", "pred_maml_warmstart_optlasso_ens_pure" if "pred_maml_warmstart_optlasso_ens_pure" in d else "pred_maml_optlasso_ens_pure"),
        ("maml_maml_optlasso", "pred_maml_maml_optlasso_ens_pure"),
        ("maml_ws_pure", "pred_maml_warmstart_nobypass_ens_pure" if "pred_maml_warmstart_nobypass_ens_pure" in d else "pred_maml_nobypass_ens_pure"),
        ("maml_maml_pure", "pred_maml_maml_nobypass_ens_pure"),
    ]

    for label, k in keys_to_track:
        summary[label] = float(np.mean(np.abs(y - d[k]))) if k in d else np.nan

    ref_maml = "pred_maml_warmstart_nobypass_ens_pure" if "pred_maml_warmstart_nobypass_ens_pure" in d else "pred_maml_nobypass_ens_pure"
    if ref_maml in d and "pred_csvr_ens_pure" in d:
        s_val, p_val = dm_test(y, d[ref_maml], d["pred_csvr_ens_pure"])
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
        ("Scenario 3a: Pure NN Warmstart (+ OptLASSO)", "maml_ws_optlasso"),
        ("Scenario 3b: Pure FOMAML-NN (+ OptLASSO)", "maml_maml_optlasso"),
        ("Scenario 4a: Pure NN Warmstart (no bypass)", "maml_ws_pure"),
        ("Scenario 4b: Pure FOMAML-NN (no bypass)", "maml_maml_pure"),
    ]

    for label, key in models:
        vals = []
        row = f"  {label:50s}"
        for s in summaries:
            v = s.get(key, np.nan)
            vals.append(v)
            row += f" | {v:9.4f}" if not np.isnan(v) else f" | {'—':>9s}"
        mean_v = float(np.nanmean(vals)) if any(not np.isnan(v) for v in vals) else np.nan
        row += f" | {mean_v:9.4f}" if not np.isnan(mean_v) else f" | {'—':>9s}"
        print(row)

    print(f"  {'-'*102}")

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

    t_start = time.perf_counter()
    with multiprocessing.Pool(processes=num_workers) as pool:
        for msg in pool.map(_run_zone, ZONES): print(msg)

    print(f"[DONE] Benchmark finished in {(time.perf_counter() - t_start)/60:.1f} minutes.\n")
    summaries = [get_zone_summary(z) for z in ZONES]
    print_master_table(summaries)


if __name__ == "__main__":
    main()
