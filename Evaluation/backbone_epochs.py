"""
Diagnostic: FA-NN backbone pre-training length (Stage 1 epochs) on pre-test blocks of 2023.

One backbone per (zone, period, feature set) is trained for MAX_EPOCHS epochs in a single fit; snapshots are taken at
CHECKPOINTS (epoch 6 = FANN_EPOCHS, identical to the main model). Every snapshot is evaluated with local adaptation
(FA-NN) and without (static backbone, G = 0).

Usage:
    python Evaluation/backbone_epochs.py --zones DE2 --periods 2023-01-01:2023-03-31,2023-04-01:2023-06-30,...
    python Evaluation/backbone_epochs.py --summary
Results: RESULTS_DIR/backbone_epochs/<zone>_<start>_<end>_<set>.npz
"""
from __future__ import annotations

import argparse
import glob
import itertools
import multiprocessing
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import FEATURE_SETS, N_SLOTS, RESULTS_DIR, SET_TAGS

OUT_DIR = os.path.join(RESULTS_DIR, "backbone_epochs")
MAX_EPOCHS = 200
CHECKPOINTS = (6, 12, 25, 50, 100, 200)
PERIODS_2023 = ("2023-01-01:2023-03-31,2023-04-01:2023-06-30,"
                "2023-07-01:2023-09-30,2023-10-01:2023-12-31")


def run(job) -> str:
    zone, start, end, fs = job
    import tensorflow as tf
    from tensorflow import keras
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from config import FANN_BATCH, N_QH
    from data_loader import QHDataLoader
    from Models.fann import FastAdaptNN

    loader = QHDataLoader()
    tag = SET_TAGS[fs]
    net = FastAdaptNN(zone, loader.n_features(zone, fs), fs)

    # Stage 1 as in FastAdaptNN.pretrain (training_mode="fann", bypass "none"), extended to MAX_EPOCHS
    windows = [loader.get_window(zone, q, start, fs, corr_filter=True) for q in range(N_QH)]
    X = np.vstack([w.X_train_full.astype(np.float32) for w in windows])
    y = np.concatenate([w.y_train.astype(np.float32) for w in windows])
    snaps, t0 = {}, time.perf_counter()

    class Snap(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch + 1 in CHECKPOINTS:
                snaps[epoch + 1] = self.model.get_weights()

    keras.utils.set_random_seed(net.seed)
    hist = net._model.fit(X, y, epochs=MAX_EPOCHS, batch_size=FANN_BATCH, shuffle=True, verbose=0, callbacks=[Snap()])
    t_train = time.perf_counter() - t0

    # Stage 2 for every snapshot, with and without local adaptation
    test = loader.test_targets(zone, start, end)
    days = sorted(test["date"].unique())
    d2i = {d: i for i, d in enumerate(days)}
    yt, naive = np.full((len(days), N_SLOTS), np.nan), np.full((len(days), N_SLOTS), np.nan)
    rows, cols = test["date"].map(d2i).to_numpy(), test["qh"].to_numpy()
    yt[rows, cols], naive[rows, cols] = test["y"].to_numpy(), test["naive"].to_numpy()
    preds = {f"{m}_e{e}": np.full_like(yt, np.nan) for e in CHECKPOINTS for m in ("adapt", "static")}
    t0 = time.perf_counter()
    for r in test.itertuples(index=False):
        w = loader.get_window(zone, r.qh, r.date, fs, corr_filter=True)
        for e in CHECKPOINTS:
            net._backbone = snaps[e]
            for m in ("adapt", "static"):
                net.adapt = m == "adapt"
                p = net.adapt_and_predict(w.X_train_full, w.y_train, w.X_test_full)
                preds[f"{m}_e{e}"][d2i[r.date], r.qh] = p * w.y_std + w.y_mean + r.naive
    t_eval = time.perf_counter() - t0

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"{zone}_{start}_{end}_{tag}.npz"), dates=np.array(days), y_true=yt,
                        naive=naive, train_loss=np.array(hist.history["loss"]), train_seconds=t_train,
                        **{f"pred_{k}": v for k, v in preds.items()})
    mae = lambda a: np.nanmean(np.abs(a - yt))
    return (f"{zone} {start} {tag} train={t_train / 60:.1f} min eval={t_eval / 60:.1f} min | "
            + " ".join(f"e{e}: {mae(preds[f'adapt_e{e}']):.3f}/{mae(preds[f'static_e{e}']):.3f}" for e in CHECKPOINTS))


def summary() -> None:
    import pandas as pd
    from Models.ensemble import inverse_mae_average
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    blocks = sorted({tuple(os.path.basename(f).split("_")[:3]) for f in files})
    rows = []
    for zone, start, end in blocks:
        d = {SET_TAGS[fs]: np.load(os.path.join(OUT_DIR, f"{zone}_{start}_{end}_{SET_TAGS[fs]}.npz")) for fs in FEATURE_SETS}
        y = d["s1"]["y_true"]
        for e, m in itertools.product(CHECKPOINTS, ("adapt", "static")):
            ens, _ = inverse_mae_average({k: v[f"pred_{m}_e{e}"] for k, v in d.items()}, y)
            row = {"zone": zone, "start": start, "epochs": e, "mode": m, "ens": np.nanmean(np.abs(ens - y))}
            row.update({k: np.nanmean(np.abs(v[f"pred_{m}_e{e}"] - y)) for k, v in d.items()})
            row.update({f"loss_{k}": float(v["train_loss"][e - 1]) for k, v in d.items()})
            rows.append(row)
        rows.append({"zone": zone, "start": start, "epochs": 0, "mode": "naive", "ens": np.nanmean(np.abs(d["s1"]["naive"] - y))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 250):
        print(df.to_string(index=False))
        print(df[df["mode"] != "naive"].groupby(["mode", "epochs"])[["ens", "s1", "s2", "s3", "s4"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", default="DE2")
    parser.add_argument("--periods", default=PERIODS_2023, help="comma-separated start:end blocks")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if not args.summary:
        periods = [tuple(p.split(":")) for p in args.periods.split(",")]
        jobs = [(z, s, e, fs) for z, (s, e), fs in itertools.product(args.zones.split(","), periods, FEATURE_SETS)
                if not os.path.exists(os.path.join(OUT_DIR, f"{z}_{s}_{e}_{SET_TAGS[fs]}.npz"))]
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(min(len(jobs), 8)) as pool:  # ~2.4 GB RAM per worker
            for msg in pool.imap_unordered(run, jobs):
                print(msg, flush=True)
    summary()
