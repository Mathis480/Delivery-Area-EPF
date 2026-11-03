"""
Single-thread runtime benchmark (paper table T2): one process, one thread, identical sample of forecasts for all models.

Usage:
    python Evaluation/runtime_benchmark.py                 # DE2, default sample
    python Evaluation/runtime_benchmark.py --zone DE1 --days 24
Sample: --days test days spread evenly over the test year x all 96 products x all four feature sets.
Stage 1 (backbone / FOMAML pre-training) is timed once per feature set; per-forecast costs are summed over the
four feature sets and extrapolated to the full test year (35,136 quarter-hours per zone).
Results: RESULTS_DIR/runtime_benchmark/runtime_<zone>.csv
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections import defaultdict

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import CORR_THRESHOLDS, FEATURE_SETS, N_QH, RESULTS_DIR, TEST_END, TEST_START
from data_loader import QHDataLoader
from Models.csvr import predict_csvr, train_csvr
from Models.fann import FastAdaptNN
from Models.linear import train_lasso

OUT_DIR = os.path.join(RESULTS_DIR, "runtime_benchmark")
NN_MODES = ["fann", "fomaml_scratch"]


def cpu_name() -> str:
    with open("/proc/cpuinfo") as f:
        return next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), platform.processor())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zone", default="DE2")
    parser.add_argument("--days", type=int, default=12, help="number of test days in the sample")
    args = parser.parse_args()

    loader = QHDataLoader()
    test = loader.test_targets(args.zone, TEST_START, TEST_END)
    all_days = sorted(test["date"].unique())
    days = [all_days[i] for i in np.linspace(0, len(all_days) - 1, args.days).round().astype(int)]
    sample = test[test["date"].isin(days)]
    stage1, per_fc = defaultdict(float), defaultdict(float)

    nets = {}
    for mode in NN_MODES:
        for fs in FEATURE_SETS:
            t0 = time.perf_counter()
            nets[mode, fs] = FastAdaptNN(args.zone, loader.n_features(args.zone, fs), fs, training_mode=mode)
            nets[mode, fs].pretrain(loader)
            stage1[mode] += time.perf_counter() - t0

    def forecast(row, timed: bool) -> None:
        for fs in FEATURE_SETS:
            times = {}
            t0 = time.perf_counter()
            w = loader.get_window(args.zone, row.qh, row.date, fs, corr_filter=True)
            times["data"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            train_lasso(w.X_train, w.y_train).predict(w.X_test)
            times["lasso"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            predict_csvr(train_csvr(w.X_train, w.y_train, w.bm_train), w.X_test, row.naive)
            times["csvr_corr"] = time.perf_counter() - t0
            if fs in CORR_THRESHOLDS:  # the filter only affects S1 and S2
                w_nf = loader.get_window(args.zone, row.qh, row.date, fs, corr_filter=False)
                t0 = time.perf_counter()
                predict_csvr(train_csvr(w_nf.X_train, w_nf.y_train, w_nf.bm_train), w_nf.X_test, row.naive)
                times["csvr"] = time.perf_counter() - t0
            else:
                times["csvr"] = times["csvr_corr"]

            for mode in NN_MODES:
                t0 = time.perf_counter()
                nets[mode, fs].adapt_and_predict(w.X_train_full, w.y_train, w.X_test_full)
                times[f"{mode}_stage2"] = time.perf_counter() - t0
            if timed:
                for k, v in times.items():
                    per_fc[k] += v

    forecast(next(sample.itertuples(index=False)), timed=False)  # warm-up (tf.function tracing)
    for row in sample.itertuples(index=False):
        forecast(row, timed=True)

    n, n_year = len(sample), len(test)
    rows = [{"component": f"{m}_stage1", "ms_per_forecast": np.nan, "min_per_zone_year": s / 60} for m, s in stage1.items()]
    rows += [{"component": k, "ms_per_forecast": 1e3 * v / n, "min_per_zone_year": v / n * n_year / 60}
             for k, v in per_fc.items()]
    out = pd.DataFrame(rows)
    out["zone"], out["n_sample"], out["n_year"] = args.zone, n, n_year
    out["cpu"], out["python"] = cpu_name(), platform.python_version()
    out["tensorflow"], out["sklearn"], out["numpy"] = tf.__version__, sklearn.__version__, np.__version__
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, f"runtime_{args.zone}.csv"), index=False)
    print(f"{args.zone}: {n} forecasts x {len(FEATURE_SETS)} feature sets, 1 thread, {cpu_name()}")
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(out[["component", "ms_per_forecast", "min_per_zone_year"]].to_string(index=False))


if __name__ == "__main__":
    main()
