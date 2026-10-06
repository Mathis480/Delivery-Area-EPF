"""
Out-of-sample 2024 benchmark across German delivery areas (DE1..DE4):
  - Naive Benchmark & Standard LASSO
  - Pure cSVR Ensemble (Puć & Janczura, 2024)
  - MAML-NN (Pure & Linear Bypass)
"""
from __future__ import annotations
import os
import sys
import time
import traceback
import multiprocessing
import numpy as np
import scipy.stats
from tqdm import tqdm
from config import (
    RESULTS_DIR, LOOKBACK_DAYS, VAL_DAYS, N_QH, ZONES,
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE,
    CSVR_USE_CORR_FILTER, CSVR_CORR_THRESH_S1, CSVR_CORR_THRESH_S2,
    ENSEMBLE_CALIB_DAYS, ENSEMBLE_POWER, MAML_SEEDS,)
from data_loader import QHDataLoader
from Models.linear import train_lasso, predict_lasso
from Models.csvr import train_csvr, predict_csvr
from Models.maml_nn import MAMLManager
from Models.ensemble import compute_rolling_weighted_average

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TARGET_ZONES = ZONES              # ["DE1", "DE2", "DE3", "DE4"]
FORCE_RETRAIN = False             # False: resume from checkpoints | True: retrain from scratch
CHECKPOINT_INTERVAL = 100         # Save intermediate checkpoints every N intervals

FEATURE_SETS = [
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE
]
SET_TAGS = {
    FEATURE_SET_MACRO: "s1",
    FEATURE_SET_NEIGHBOR: "s2",
    FEATURE_SET_FUNDAMENTAL: "s3",
    FEATURE_SET_BALANCE: "s4",
}


def dm_test(actual: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano test for predictive accuracy (two-sided)."""
    mask = ~(np.isnan(actual) | np.isnan(p1) | np.isnan(p2))
    d = np.abs(actual[mask] - p1[mask]) - np.abs(actual[mask] - p2[mask])
    s = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))
    p = float(2 * (1 - scipy.stats.norm.cdf(abs(s))))
    return float(s), p

def _backtransform(y_pred_scaled: float, y_mean: float, y_std: float, benchmark_eur: float) -> float:
    return float(y_pred_scaled * y_std + y_mean + benchmark_eur)

# Backbone Pretraining Utilities
def pretrain_all_backbones(zones: list[str],seeds: list[int] = MAML_SEEDS,force_retrain: bool = False,verbose: bool = True,):
    """Pretrains shared backbones for Pure MAML across requested zones and seeds."""
    print(f"\n{'='*75}")
    print(f"  PRETRAINING PURE MAML BACKBONES (FORCE_RETRAIN={force_retrain})")
    print(f"  Zones: {zones} | Seeds: {seeds} | Features: S1..S4")
    print(f"{'='*75}\n", flush=True)

    loader = QHDataLoader()
    for z in zones:
        for s in FEATURE_SETS:
            for seed in seeds:
                mgr = MAMLManager(zone=z, n_features=loader.n_features(z, s), feature_set=s, seed=seed)
                mgr.pretrain_backbone(loader, test_start="2024-01-01", verbose=verbose, force_retrain=force_retrain)

# Zone Process Runner (Parallel Execution per Zone)
def _load_zone_checkpoint(zone: str,n_total: int,force_retrain: bool,) -> tuple[dict[str, list[float]], int, str]:
    """Initializes model prediction containers and resumes from checkpoint if available."""
    ckpt_file = os.path.join(RESULTS_DIR, "annual_run_2024", f"checkpoint_{zone}_master.npz")
    if force_retrain and os.path.exists(ckpt_file):
        try:
            os.remove(ckpt_file)
        except OSError:
            pass

    preds: dict[str, list[float]] = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        preds[f"lasso_std_{st}"] = []
        preds[f"csvr_{st}"] = []
        for sd in MAML_SEEDS:
            preds[f"maml_nobypass_{st}_seed{sd}"] = []

    start_idx = 0
    if os.path.exists(ckpt_file) and not force_retrain:
        try:
            ckpt = np.load(ckpt_file)
            first_key = list(preds.keys())[0] if preds else None
            if first_key and first_key in ckpt:
                done_len = len(ckpt[first_key])
                if done_len == n_total:
                    for k in preds:
                        if k in ckpt:
                            preds[k] = list(ckpt[k])
                    start_idx = n_total
                elif 0 < done_len < n_total:
                    for k in preds:
                        if k in ckpt:
                            preds[k] = list(ckpt[k])
                    start_idx = done_len
        except Exception:
            start_idx = 0

    return preds, start_idx, ckpt_file

def _ensemble_and_save_results(zone: str,preds: dict[str, list[float]],data: dict,in_npz: str,) -> None:
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
        lasso_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    data["pred_lasso_ens_pure_2d"] = p_l_ens_2d
    data["pred_lasso_ens_pure"] = _to_1d(p_l_ens_2d)

    # 2. Pure cSVR Ensembling
    csvr_sets_2d = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        arr = np.array(preds[f"csvr_{st}"], dtype=np.float32)
        data[f"pred_csvr_{st}"] = arr
        mat = _to_2d(arr)
        data[f"pred_csvr_{st}_2d"] = mat
        csvr_sets_2d[f"csvr_{st}"] = mat
    p_c_ens_2d, _, _ = compute_rolling_weighted_average(
        csvr_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    data["pred_csvr_ens_pure_2d"] = p_c_ens_2d
    data["pred_csvr_ens_pure"] = _to_1d(p_c_ens_2d)

    # 3. Pure MAML Single & Deep Ensemble
    single_sets_2d = {}
    deep_sets_2d = {}
    for s in FEATURE_SETS:
        st = SET_TAGS[s]
        seed_arrays = []
        for sd in MAML_SEEDS:
            arr_sd = np.array(preds[f"maml_nobypass_{st}_seed{sd}"], dtype=np.float32)
            data[f"pred_maml_nobypass_{st}_seed{sd}"] = arr_sd
            seed_arrays.append(arr_sd)
            if sd == 42:
                data[f"pred_maml_nobypass_{st}"] = arr_sd
                mat_s42 = _to_2d(arr_sd)
                data[f"pred_maml_nobypass_{st}_2d"] = mat_s42
                single_sets_2d[f"single_{st}"] = mat_s42

        avg_1d = np.mean(seed_arrays, axis=0).astype(np.float32)
        data[f"pred_maml_nobypass_deep_{st}"] = avg_1d
        mat_deep = _to_2d(avg_1d)
        data[f"pred_maml_nobypass_deep_{st}_2d"] = mat_deep
        deep_sets_2d[f"deep_{st}"] = mat_deep

    # Single-Seed 42 Ensemble
    p_single_2d, _, _ = compute_rolling_weighted_average(
        single_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    data["pred_maml_nobypass_ens_pure_2d"] = p_single_2d
    data["pred_maml_nobypass_ens_pure"] = _to_1d(p_single_2d)

    # Deep Ensemble across sets
    p_deep_2d, _, _ = compute_rolling_weighted_average(
        deep_sets_2d, y_true_2d, bm_2d, calib_window=ENSEMBLE_CALIB_DAYS, power=ENSEMBLE_POWER
    )
    data["pred_maml_nobypass_deep_ens_pure_2d"] = p_deep_2d
    data["pred_maml_nobypass_deep_ens_pure"] = _to_1d(p_deep_2d)

    # Save to NPZ
    np.savez_compressed(in_npz, **data)
    ms_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_multiset_2024.npz")
    if os.path.exists(ms_npz):
        np.savez_compressed(ms_npz, **data)

def _run_zone_process_impl(zone: str,force_retrain: bool,checkpoint_interval: int,queue: multiprocessing.Queue,):
    """Worker process computing all benchmark models for a single delivery area across 2024."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    if not os.path.exists(in_npz):
        raise FileNotFoundError(f"Missing base NPZ file: {in_npz}")

    base = np.load(in_npz)
    data = {k: base[k] for k in base.files}
    dates_arr, qh_arr, bm_arr = data["dates"], data["qh_idx"], data["benchmark"]
    n_total = len(dates_arr)

    # 1. Initialize prediction containers and load checkpoint
    preds, start_idx, ckpt_file = _load_zone_checkpoint(zone, n_total, force_retrain)
    if queue is not None:
        queue.put((zone, start_idx, False))

    loader = QHDataLoader()

    # 2. Initialize MAML managers
    maml_managers = {}
    for sd in MAML_SEEDS:
        maml_managers[sd] = {}
        for s in FEATURE_SETS:
            mgr = MAMLManager(zone=zone, n_features=loader.n_features(zone, s), feature_set=s, seed=sd)
            mgr.pretrain_backbone(loader, test_start="2024-01-01", verbose=False)
            maml_managers[sd][s] = mgr

    # 3. Main interval inference loop
    batch_progress = 0
    for i in range(start_idx, n_total):
        d_str = str(dates_arr[i])
        q = int(qh_arr[i])
        bm_eur = float(bm_arr[i])

        for s in FEATURE_SETS:
            st = SET_TAGS[s]

            # Standard window for LASSO and MAML
            w = loader.get_window(
                zone, q, d_str, feature_set=s,
                lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS,
                filter_zero_var=False,
            )

            # Standard LASSO
            if w is not None and w.X_trainval is not None:
                try:
                    m_l, _ = train_lasso(w.X_trainval, w.y_trainval)
                    p_l = float(predict_lasso(m_l, w.X_test)[0])
                    preds[f"lasso_std_{st}"].append(_backtransform(p_l, w.y_mean, w.y_std, bm_eur))
                except Exception:
                    preds[f"lasso_std_{st}"].append(bm_eur)
            else:
                preds[f"lasso_std_{st}"].append(bm_eur)

            # Pure cSVR (configurable correlation filter per Puć & Janczura 2024)
            use_corr = CSVR_USE_CORR_FILTER and s in [FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR]
            corr_thresh = CSVR_CORR_THRESH_S1 if s == FEATURE_SET_MACRO else CSVR_CORR_THRESH_S2
            w_c = loader.get_window(
                zone, q, d_str, feature_set=s,
                lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS,
                filter_zero_var=True, filter_corr=use_corr,
                corr_threshold=corr_thresh,
            )
            if w_c is not None and w_c.X_trainval is not None:
                try:
                    m_c = train_csvr(w_c.X_trainval, w_c.y_trainval, bm_trainval=w_c.bm_trainval)
                    p_c = float(predict_csvr(m_c, w_c.X_test, bm_test=bm_eur)[0])
                    preds[f"csvr_{st}"].append(_backtransform(p_c, w_c.y_mean, w_c.y_std, bm_eur))
                except Exception:
                    preds[f"csvr_{st}"].append(bm_eur)
            else:
                preds[f"csvr_{st}"].append(bm_eur)

            # Pure MAML (Seeds 42, 123, 999)
            for sd in MAML_SEEDS:
                if w is not None and w.X_trainval is not None:
                    try:
                        p_scaled = maml_managers[sd][s].adapt_and_predict(
                            X_support=w.X_trainval,
                            y_support=w.y_trainval,
                            X_test=w.X_test_full,
                        )
                        preds[f"maml_nobypass_{st}_seed{sd}"].append(_backtransform(p_scaled, w.y_mean, w.y_std, bm_eur))
                    except Exception:
                        preds[f"maml_nobypass_{st}_seed{sd}"].append(bm_eur)
                else:
                    preds[f"maml_nobypass_{st}_seed{sd}"].append(bm_eur)

        # Update live progress queue (batched every 10 intervals)
        batch_progress += 1
        if batch_progress >= 10 or i == n_total - 1:
            if queue is not None:
                queue.put((zone, batch_progress, False))
            batch_progress = 0

        # Periodic checkpoint save
        done_count = i - start_idx + 1
        if done_count % checkpoint_interval == 0 or i == n_total - 1:
            save_dict = {k: np.array(v, dtype=np.float32) for k, v in preds.items()}
            np.savez_compressed(ckpt_file, **save_dict)

    # 4. Rolling Inverse-MAE Ensembling & NPZ Save
    _ensemble_and_save_results(zone, preds, data, in_npz)

    # Signal completion
    if queue is not None:
        queue.put((zone, None, True))

def _run_zone_process(zone: str,force_retrain: bool,checkpoint_interval: int,queue: multiprocessing.Queue,):
    try:
        _run_zone_process_impl(zone, force_retrain, checkpoint_interval, queue)
    except Exception as e:
        print(f"\n[FATAL ERROR in Zone {zone}]: {e}", file=sys.stderr)
        traceback.print_exc()
        if queue is not None:
            queue.put((zone, None, True))
        raise

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
        ("csvr_pure", "pred_csvr_ens_pure"),
        ("maml_byp_m015", "pred_maml_ens_pure"),
        ("maml_nobypass_single", "pred_maml_nobypass_ens_pure"),
        ("maml_nobypass_deep", "pred_maml_nobypass_deep_ens_pure"),
    ]

    for label, k in keys_to_track:
        if k in d:
            mae = float(np.mean(np.abs(y - d[k])))
            rmse = float(np.sqrt(np.mean((y - d[k])**2)))
            summary[label] = mae
            summary[f"{label}_rmse"] = rmse
            summary[f"{label}_rmae"] = mae / naive
        else:
            summary[label] = np.nan

    if "pred_maml_nobypass_deep_ens_pure" in d and "pred_csvr_ens_pure" in d:
        s_val, p_val = dm_test(y, d["pred_maml_nobypass_deep_ens_pure"], d["pred_csvr_ens_pure"])
        summary["dm_stat_deep_vs_csvr"] = s_val
        summary["dm_pval_deep_vs_csvr"] = p_val

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
        ("Scenario 2: Pure cSVR Ensemble", "csvr_pure"),
        ("Scenario 3: MAML-NN (Linear Bypass, m=0.15)", "maml_byp_m015"),
        ("Scenario 4: Pure MAML-NN (Single Seed 42)", "maml_nobypass_single"),
        ("Scenario 5: Pure MAML-NN (Deep Ensemble, 3 Seeds)", "maml_nobypass_deep"),
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

    print(f"  {'-'*102}\n")

# Main Orchestrator
def main():
    if "--summary_only" in sys.argv or "--summary" in sys.argv:
        summaries = [get_zone_summary(z) for z in TARGET_ZONES]
        print_master_table(summaries)
        return

    multiprocessing.set_start_method("spawn", force=True)

    force_retrain = FORCE_RETRAIN or ("--force" in sys.argv) or ("--retrain" in sys.argv)

    # 1. Warmup and secure shared backbones for Pure MAML
    pretrain_all_backbones(TARGET_ZONES, MAML_SEEDS, force_retrain=force_retrain)

    # 2. Determine total test intervals
    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{TARGET_ZONES[0]}_full4sets_2024.npz")
    n_total = len(np.load(in_npz)["dates"])

    print(f"\n{'='*80}")
    print(f"  STARTING PARALLEL MASTER BENCHMARK RUN (2024)")
    print(f"  Zones: {TARGET_ZONES} | Force: {force_retrain}")
    print(f"  Total Intervals per Zone: {n_total:,} | Total Forecasts: {n_total * len(TARGET_ZONES):,}")
    print(f"{'='*80}\n", flush=True)

    # 3. Initialize stacked progress bars for all delivery areas
    queue = multiprocessing.Queue()
    bars = {
        z: tqdm(
            total=n_total,
            desc=f"Zone {z:4s}",
            position=idx,
            leave=True,
            unit="QH",
            dynamic_ncols=True,
        )
        for idx, z in enumerate(TARGET_ZONES)
    }

    # 4. Launch parallel worker processes
    procs = []
    for z in TARGET_ZONES:
        p = multiprocessing.Process(
            target=_run_zone_process,
            args=(z, force_retrain, CHECKPOINT_INTERVAL, queue)
        )
        p.start()
        procs.append(p)

    # 5. Synchronize live progress in the main thread
    finished_zones = set()
    while len(finished_zones) < len(TARGET_ZONES):
        try:
            z, step, is_done = queue.get(timeout=1.0)
            if is_done:
                finished_zones.add(z)
                bars[z].n = n_total
                bars[z].refresh()
            elif step is not None:
                bars[z].update(step)
        except Exception:
            for p in procs:
                if not p.is_alive() and p.exitcode != 0:
                    raise RuntimeError(f"Zone worker process for {z} died with exit code {p.exitcode}")

    for p in procs:
        p.join()

    for b in bars.values():
        b.close()

    print(f"\nAll zone computations finished successfully!")

    # 6. Display master benchmark table across all 4 zones
    summaries = [get_zone_summary(z) for z in TARGET_ZONES]
    print_master_table(summaries)

if __name__ == "__main__":
    main()
