"""
Rolling out-of-sample backtest for all control areas (DE1-DE4):
Naive, LASSO, LASSO+N, cSVR replication (with correlation filter, main benchmark), cSVR without filter (post hoc),
FA-NN and its variants (FOMAML without pre-training, FOMAML after pre-training, LASSO bypass (post hoc), static backbone
without adaptation).

Usage:
    python run_experiment.py                 # backtest all zones in parallel, then print the summary
    python run_experiment.py --zones DE2     # selected zones
    python run_experiment.py --summary       # summary of existing results only
    python run_experiment.py --models fomaml,fomaml_pre       # recompute selected models, merge into existing results
    python run_experiment.py --models fann --seed 1           # seed robustness, written to RUN_DIR/seed_1
Outputs per zone in RUN_DIR: results_<zone>.npz (forecasts, weights) and runtime_<zone>.csv.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
import warnings
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU run, as reported in the paper
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # deterministic CPU kernels

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from config import (
    CORR_THRESHOLDS, FANN_TRAINING_MODES, FEATURE_SETS, N_SLOTS, N_WORKERS, RANDOM_SEED, RUN_DIR,
    SET_TAGS, TEST_END, TEST_START, ZONES,
)
from data_loader import QHDataLoader
from Models.csvr import predict_csvr, train_csvr
from Models.ensemble import inverse_mae_average
from Models.fann import FastAdaptNN
from Models.linear import train_lasso

# LARS reports degenerate (collinear) regressors on almost every fit; the path remains valid.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


FANN_MODELS = {tm: (tm, "none") for tm in FANN_TRAINING_MODES}
FANN_MODELS["fann_bypass"] = ("fann", "lasso")  # LASSO bypass, FA-NN only (post hoc)
FANN_MODELS["fann_static"] = ("fann", "none")  # backbone without local adaptation (G = 0)
MODELS = ["lasso", "csvr", "csvr_corr", *FANN_MODELS]


def run_zone(job: tuple[str, list[str], int, str]) -> str:
    zone, models, seed, run_dir = job
    fann_models = {k: v for k, v in FANN_MODELS.items() if k in models}
    need_lasso = "lasso" in models or any(bp == "lasso" for _, bp in fann_models.values())
    loader = QHDataLoader()
    test = loader.test_targets(zone, TEST_START, TEST_END)
    days = sorted(test["date"].unique())
    d2i = {d: i for i, d in enumerate(days)}
    y_2d = np.full((len(days), N_SLOTS), np.nan)
    naive_2d = np.full((len(days), N_SLOTS), np.nan)
    rows, cols = test["date"].map(d2i).to_numpy(), test["qh"].to_numpy()
    y_2d[rows, cols] = test["y"].to_numpy()
    naive_2d[rows, cols] = test["naive"].to_numpy()

    preds = {f"{m}_{SET_TAGS[fs]}": np.full_like(y_2d, np.nan) for m in models for fs in FEATURE_SETS}
    runtime = defaultdict(float)

    # FA-NN stage 1: one backbone per training mode, bypass mode and feature set
    nets = {}
    for key, (tm, bp) in fann_models.items():
        for fs in FEATURE_SETS:
            t0 = time.perf_counter()
            nets[key, fs] = FastAdaptNN(zone, loader.n_features(zone, fs), fs, training_mode=tm, bypass_mode=bp,
                                        seed=seed, adapt=key != "fann_static")
            nets[key, fs].pretrain(loader)
            runtime[f"{key}_stage1"] += time.perf_counter() - t0

    for row in tqdm(test.itertuples(index=False), total=len(test), desc=f"Zone {zone}", unit="QH"):
        di, q = d2i[row.date], row.qh

        def back(p: float, w) -> float:
            return float(p) * w.y_std + w.y_mean + row.naive

        for fs in FEATURE_SETS:
            tag = SET_TAGS[fs]
            t0 = time.perf_counter()
            w = loader.get_window(zone, q, row.date, fs, corr_filter=True)
            runtime["data"] += time.perf_counter() - t0

            if need_lasso:
                t0 = time.perf_counter()
                lasso = train_lasso(w.X_train, w.y_train)
                p_lasso = lasso.predict(w.X_test)[0]
                runtime["lasso"] += time.perf_counter() - t0
                if "lasso" in models:
                    preds[f"lasso_{tag}"][di, q] = back(p_lasso, w)

            # cSVR with correlation filter (csvr_corr) and without (csvr); the filter only affects S1 and S2,
            # so in S3/S4 both models share one fit
            filtered = fs in CORR_THRESHOLDS
            if "csvr_corr" in models or ("csvr" in models and not filtered):
                t0 = time.perf_counter()
                p = predict_csvr(train_csvr(w.X_train, w.y_train, w.bm_train), w.X_test, row.naive)[0]
                dt = time.perf_counter() - t0
                p = back(p, w)
                if "csvr_corr" in models:
                    runtime["csvr_corr"] += dt
                    preds[f"csvr_corr_{tag}"][di, q] = p
                if "csvr" in models and not filtered:
                    runtime["csvr"] += dt
                    preds[f"csvr_{tag}"][di, q] = p
            if "csvr" in models and filtered:
                w_nf = loader.get_window(zone, q, row.date, fs, corr_filter=False)
                t0 = time.perf_counter()
                p = predict_csvr(train_csvr(w_nf.X_train, w_nf.y_train, w_nf.bm_train), w_nf.X_test, row.naive)[0]
                runtime["csvr"] += time.perf_counter() - t0
                preds[f"csvr_{tag}"][di, q] = back(p, w_nf)

            lasso_support = lasso.predict(w.X_train) if need_lasso else None
            for key, (_, bp) in fann_models.items():
                t0 = time.perf_counter()
                p = nets[key, fs].adapt_and_predict(
                    w.X_train_full, w.y_train, w.X_test_full,
                    lasso_support=lasso_support if bp == "lasso" else None,
                    lasso_test=p_lasso if bp == "lasso" else None,
                )
                runtime[f"{key}_stage2"] += time.perf_counter() - t0
                preds[f"{key}_{tag}"][di, q] = back(p, w)

    # Forecast averaging over the four feature sets (per architecture) and LASSO+N (Marcjasz et al., 2020)
    path = os.path.join(run_dir, f"results_{zone}.npz")
    out = dict(np.load(path)) if os.path.exists(path) and set(models) != set(MODELS) else {}
    out.update({"dates": np.array(days), "y_true": y_2d, "naive": naive_2d})
    for m in models:
        members = {SET_TAGS[fs]: preds[f"{m}_{SET_TAGS[fs]}"] for fs in FEATURE_SETS}
        out[f"pred_{m}"], out[f"weights_{m}"] = inverse_mae_average(members, y_2d)
        out.update({f"pred_{m}_{tag}": p for tag, p in members.items()})
    if "lasso" in models:
        out["pred_lasso_n"] = 0.5 * out["pred_lasso"] + 0.5 * naive_2d

    os.makedirs(run_dir, exist_ok=True)
    np.savez_compressed(path, **out)
    rt = pd.DataFrame({"component": list(runtime), "seconds": list(runtime.values())})
    rt["n_forecasts"] = len(test)
    rt_path = os.path.join(run_dir, f"runtime_{zone}.csv")
    if os.path.exists(rt_path) and set(models) != set(MODELS):
        old = pd.read_csv(rt_path)
        rt = pd.concat([old[~old["component"].isin(rt["component"])], rt], ignore_index=True)
    rt.to_csv(rt_path, index=False)
    return f"[{zone}] " + " | ".join(f"{k}: {v / 60:.1f} min" for k, v in runtime.items())


def summarize(run_dir: str = RUN_DIR) -> pd.DataFrame:
    """MAE per model and zone (tests and paper tables: Evaluation/make_tables.py)."""
    table = {}
    for zone in ZONES:
        path = os.path.join(run_dir, f"results_{zone}.npz")
        if not os.path.exists(path):  # zones not (yet) computed in this run directory
            continue
        d = np.load(path)
        table[zone] = {m: np.nanmean(np.abs(d["naive" if m == "naive" else f"pred_{m}"] - d["y_true"]))
                       for m in ["naive", *MODELS, "lasso_n"] if m == "naive" or f"pred_{m}" in d}
    table = pd.DataFrame(table)
    table.to_csv(os.path.join(run_dir, "summary.csv"))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zones", default=",".join(ZONES), help="comma-separated zones")
    parser.add_argument("--summary", action="store_true", help="only summarize existing results")
    parser.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {MODELS}")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="FA-NN seed (non-default: RUN_DIR/seed_<seed>)")
    args = parser.parse_args()
    run_dir = RUN_DIR if args.seed == RANDOM_SEED else os.path.join(RUN_DIR, f"seed_{args.seed}")

    if not args.summary:
        zones = args.zones.split(",")
        multiprocessing.set_start_method("spawn", force=True)
        t0 = time.perf_counter()
        with multiprocessing.Pool(processes=min(N_WORKERS, len(zones))) as pool:
            for msg in pool.map(run_zone, [(z, args.models.split(","), args.seed, run_dir) for z in zones]):
                print(msg)
        print(f"Backtest finished in {(time.perf_counter() - t0) / 60:.1f} min.")

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(summarize(run_dir))


if __name__ == "__main__":
    main()
