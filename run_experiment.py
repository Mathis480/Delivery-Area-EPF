"""
Rolling out-of-sample backtest for all control areas (DE1-DE4):
Naive, LASSO, LASSO+N, cSVR (main, without correlation filter), cSVR replication (with filter),
FA-NN and its ablations (FOMAML, FOMAML from scratch, LASSO bypass).

Usage:
    python run_experiment.py                 # backtest all zones in parallel, then print the summary
    python run_experiment.py --zones DE2     # selected zones
    python run_experiment.py --summary       # summary of existing results only
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
    CORR_THRESHOLDS, FANN_BYPASS_MODES, FANN_TRAINING_MODES, FEATURE_SETS, N_QH, N_WORKERS, RUN_DIR, SET_TAGS,
    TEST_END, TEST_START, ZONES,
)
from data_loader import QHDataLoader
from Models.csvr import predict_csvr, train_csvr
from Models.ensemble import daily_loss, dm_test, gw_test, inverse_mae_average
from Models.fann import FastAdaptNN
from Models.linear import train_lasso

# LARS reports degenerate (collinear) regressors on almost every fit; the path remains valid.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fann_key(training_mode: str, bypass_mode: str) -> str:
    return training_mode if bypass_mode == "none" else f"{training_mode}_bypass"


FANN_MODELS = {fann_key(tm, bp): (tm, bp) for tm in FANN_TRAINING_MODES for bp in FANN_BYPASS_MODES}
MODELS = ["lasso", "csvr", "csvr_corr", *FANN_MODELS]


def run_zone(zone: str) -> str:
    loader = QHDataLoader()
    test = loader.test_targets(zone, TEST_START, TEST_END)
    days = sorted(test["date"].unique())
    d2i = {d: i for i, d in enumerate(days)}
    y_2d = np.full((len(days), N_QH), np.nan)
    naive_2d = np.full((len(days), N_QH), np.nan)
    rows, cols = test["date"].map(d2i).to_numpy(), test["qh"].to_numpy()
    y_2d[rows, cols] = test["y"].to_numpy()
    naive_2d[rows, cols] = test["naive"].to_numpy()

    preds = {f"{m}_{SET_TAGS[fs]}": np.full_like(y_2d, np.nan) for m in MODELS for fs in FEATURE_SETS}
    runtime = defaultdict(float)

    # FA-NN stage 1: one backbone per training mode, bypass mode and feature set
    nets = {}
    for key, (tm, bp) in FANN_MODELS.items():
        for fs in FEATURE_SETS:
            t0 = time.perf_counter()
            nets[key, fs] = FastAdaptNN(zone, loader.n_features(zone, fs), fs, training_mode=tm, bypass_mode=bp)
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

            t0 = time.perf_counter()
            lasso = train_lasso(w.X_train, w.y_train)
            p_lasso = lasso.predict(w.X_test)[0]
            runtime["lasso"] += time.perf_counter() - t0
            preds[f"lasso_{tag}"][di, q] = back(p_lasso, w)

            t0 = time.perf_counter()
            p = predict_csvr(train_csvr(w.X_train, w.y_train, w.bm_train), w.X_test, row.naive)[0]
            dt = time.perf_counter() - t0
            runtime["csvr_corr"] += dt
            preds[f"csvr_corr_{tag}"][di, q] = back(p, w)
            if fs in CORR_THRESHOLDS:  # the filter only affects S1 and S2
                w_nf = loader.get_window(zone, q, row.date, fs, corr_filter=False)
                t0 = time.perf_counter()
                p = predict_csvr(train_csvr(w_nf.X_train, w_nf.y_train, w_nf.bm_train), w_nf.X_test, row.naive)[0]
                dt = time.perf_counter() - t0
                preds[f"csvr_{tag}"][di, q] = back(p, w_nf)
            else:
                preds[f"csvr_{tag}"][di, q] = preds[f"csvr_corr_{tag}"][di, q]
            runtime["csvr"] += dt

            lasso_support = lasso.predict(w.X_train) if "lasso" in FANN_BYPASS_MODES else None
            for key, (_, bp) in FANN_MODELS.items():
                t0 = time.perf_counter()
                p = nets[key, fs].adapt_and_predict(
                    w.X_train_full, w.y_train, w.X_test_full,
                    lasso_support=lasso_support if bp == "lasso" else None,
                    lasso_test=p_lasso if bp == "lasso" else None,
                )
                runtime[f"{key}_stage2"] += time.perf_counter() - t0
                preds[f"{key}_{tag}"][di, q] = back(p, w)

    # Forecast averaging over the four feature sets (per architecture) and LASSO+N (Marcjasz et al., 2020)
    out = {"dates": np.array(days), "y_true": y_2d, "naive": naive_2d}
    for m in MODELS:
        members = {SET_TAGS[fs]: preds[f"{m}_{SET_TAGS[fs]}"] for fs in FEATURE_SETS}
        out[f"pred_{m}"], out[f"weights_{m}"] = inverse_mae_average(members, y_2d)
        out.update({f"pred_{m}_{tag}": p for tag, p in members.items()})
    out["pred_lasso_n"] = 0.5 * out["pred_lasso"] + 0.5 * naive_2d

    os.makedirs(RUN_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(RUN_DIR, f"results_{zone}.npz"), **out)
    rt = pd.DataFrame({"component": list(runtime), "seconds": list(runtime.values())})
    rt["n_forecasts"] = len(test)
    rt.to_csv(os.path.join(RUN_DIR, f"runtime_{zone}.csv"), index=False)
    return f"[{zone}] " + " | ".join(f"{k}: {v / 60:.1f} min" for k, v in runtime.items())


def summarize() -> pd.DataFrame:
    """MAE per model and zone, plus DM and GW tests of FA-NN against the main cSVR (H1: FA-NN more accurate)."""
    rows = []
    for zone in ZONES:
        d = np.load(os.path.join(RUN_DIR, f"results_{zone}.npz"))
        y = d["y_true"]
        row = {"zone": zone, "naive": np.nanmean(np.abs(d["naive"] - y))}
        for m in [*MODELS, "lasso_n"]:
            row[m] = np.nanmean(np.abs(d[f"pred_{m}"] - y))
        l_csvr, l_fann = daily_loss(d["pred_csvr"], y), daily_loss(d["pred_fann"], y)
        row["dm_p_fann_vs_csvr"] = dm_test(l_csvr, l_fann)[1]
        row["gw_p_fann_vs_csvr"] = gw_test(l_csvr, l_fann)[1]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("zone")
    table.to_csv(os.path.join(RUN_DIR, "summary.csv"))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zones", default=",".join(ZONES), help="comma-separated zones")
    parser.add_argument("--summary", action="store_true", help="only summarize existing results")
    args = parser.parse_args()

    if not args.summary:
        zones = args.zones.split(",")
        multiprocessing.set_start_method("spawn", force=True)
        t0 = time.perf_counter()
        with multiprocessing.Pool(processes=min(N_WORKERS, len(zones))) as pool:
            for msg in pool.map(run_zone, zones):
                print(msg)
        print(f"Backtest finished in {(time.perf_counter() - t0) / 60:.1f} min.")

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(summarize().T)


if __name__ == "__main__":
    main()
