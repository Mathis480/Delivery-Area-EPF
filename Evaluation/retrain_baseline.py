"""
Diagnostic: conventional per-forecast neural network vs. FA-NN (pre-test year 2023, sampled days).

For every (product, test day, feature set), a fresh MLP with the FA-NN architecture is trained from scratch on that
product's calibration window: default Keras initialization (including the output layer), Adam (lr 1e-3), batch 32,
MAE loss, early stopping on the last VAL_FRAC of the calibration days (patience PATIENCE, best weights restored,
at most MAX_EPOCHS). FA-NN reference on the same days: Evaluation/backbone_epochs.py, snapshot e6.

Usage:
    python Evaluation/retrain_baseline.py --zones DE2 --every 13
    python Evaluation/retrain_baseline.py --summary
    python Evaluation/retrain_baseline.py --timing   # unloaded runtime: one day, each set in a fresh single process
    RETRAIN_ZERO_OUTPUT=1 python Evaluation/retrain_baseline.py   # zero-initialized output layer
    RETRAIN_ZERO_OUTPUT=1 RETRAIN_LR=5e-4 python Evaluation/retrain_baseline.py   # plus FA-NN learning rate
Results: RESULTS_DIR/retrain_baseline/<zone>_<date>_<set>.npz
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

LR = float(os.environ.get("RETRAIN_LR", 1e-3))
ZERO_OUTPUT = os.environ.get("RETRAIN_ZERO_OUTPUT") == "1"  # variant: zero-initialized output layer (as in the FA-NN)
OUT_DIR = os.path.join(RESULTS_DIR, "retrain_baseline" + ("_zero_output" if ZERO_OUTPUT else "")
                       + (f"_lr{LR:g}" if LR != 1e-3 else ""))
REF_DIR = os.path.join(RESULTS_DIR, "backbone_epochs")
YEAR = ("2023-01-01", "2023-12-31")
BATCH, MAX_EPOCHS, PATIENCE, VAL_FRAC = 32, 200, 10, 0.2
N_PROC = int(os.environ.get("RETRAIN_N_PROC", 8))  # ~2.4 GB RAM per worker (master dataset)
TIMING_DAY = "2023-06-14"  # a sampled day of the stride-13 sample


def _set_out_dir(path: str) -> None:
    global OUT_DIR
    OUT_DIR = path


def build(n_features: int, zero_output: bool = False):
    from tensorflow import keras
    from tensorflow.keras import layers
    from config import FANN_ACTIVATION, FANN_HIDDEN_SIZES
    x = inp = keras.Input(shape=(n_features,))
    for h in FANN_HIDDEN_SIZES:
        x = layers.Dense(h, activation=FANN_ACTIVATION)(x)
    out = layers.Dense(1, kernel_initializer="zeros") if zero_output else layers.Dense(1)
    model = keras.Model(inp, out(x))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss="mae")
    return model


def run(job) -> str:
    zone, day, fs, seed = job
    import tensorflow as tf
    from tensorflow import keras
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from data_loader import QHDataLoader

    loader = QHDataLoader()
    test = loader.test_targets(zone, day, day)
    pred, y, naive = (np.full(N_SLOTS, np.nan) for _ in range(3))
    epochs, seconds = np.zeros(N_SLOTS), np.zeros(N_SLOTS)
    for r in test.itertuples(index=False):
        w = loader.get_window(zone, r.qh, r.date, fs, corr_filter=True)
        t0 = time.perf_counter()
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed + r.qh)
        model = build(w.X_train_full.shape[1], ZERO_OUTPUT)
        hist = model.fit(w.X_train_full, w.y_train, epochs=MAX_EPOCHS, batch_size=BATCH, validation_split=VAL_FRAC,
                         shuffle=True, verbose=0, callbacks=[keras.callbacks.EarlyStopping(
                             monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
        p = float(model(w.X_test_full, training=False).numpy()[0, 0])
        seconds[r.qh], epochs[r.qh] = time.perf_counter() - t0, len(hist.history["loss"])
        pred[r.qh], y[r.qh], naive[r.qh] = p * w.y_std + w.y_mean + r.naive, r.y, r.naive

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"{zone}_{day}_{SET_TAGS[fs]}.npz"), pred=pred, y_true=y, naive=naive,
                        epochs=epochs, seconds=seconds)
    return (f"{zone} {day} {SET_TAGS[fs]} MAE={np.nanmean(np.abs(pred - y)):.3f} naive={np.nanmean(np.abs(naive - y)):.3f}"
            f" | {np.nanmean(seconds[seconds > 0]):.2f} s/fit, {np.mean(epochs[epochs > 0]):.0f} epochs")


def reference(zone: str, day: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """FA-NN (adapt) and static backbone forecasts of the pre-test block containing `day`."""
    for f in glob.glob(os.path.join(REF_DIR, f"{zone}_*_{tag}.npz")):
        d = np.load(f)
        idx = np.flatnonzero(d["dates"] == day)
        if idx.size:
            return d["pred_adapt_e6"][idx[0]], d["pred_static_e6"][idx[0]]
    raise FileNotFoundError(f"No FA-NN reference for {zone} {day} {tag}")


def summary() -> None:
    import pandas as pd
    from Models.ensemble import dm_test
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    keys = sorted({tuple(os.path.basename(f)[:-4].split("_")[:2]) for f in files})
    tags = [SET_TAGS[fs] for fs in FEATURE_SETS]
    rows, sec, ep = [], [], []
    for zone, day in keys:
        paths = {t: os.path.join(OUT_DIR, f"{zone}_{day}_{t}.npz") for t in tags}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        for t, p in paths.items():
            d = np.load(p)
            fann, static = reference(zone, day, t)
            y = d["y_true"]
            ok = ~np.isnan(y)
            sec.append(d["seconds"][ok]), ep.append(d["epochs"][ok])
            rows.append({"zone": zone, "day": day, "set": t, "retrain": np.mean(np.abs(d["pred"] - y)[ok]),
                         "fann": np.mean(np.abs(fann - y)[ok]), "static_once": np.mean(np.abs(static - y)[ok]),
                         "naive": np.mean(np.abs(d["naive"] - y)[ok])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    cols = ["naive", "retrain", "static_once", "fann"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        per_set = df.groupby("set")[cols].mean()
        per_set.loc["mean S1-S4"] = df[cols].mean()
        print(f"{df['day'].nunique()} days x 96 products:")
        print(per_set)
    daily = df.groupby("day")[cols].mean()
    print(f"DM (daily MAE averaged over sets), p for H1 FA-NN better than retrain: "
          f"{dm_test(daily['retrain'].to_numpy(), daily['fann'].to_numpy(), h=1)[1]:.4f}; "
          f"retrain better than FA-NN: {dm_test(daily['fann'].to_numpy(), daily['retrain'].to_numpy(), h=1)[1]:.4f}")
    sec, ep = np.concatenate(sec), np.concatenate(ep)
    print(f"Per fit: {sec.mean():.2f} s (median {np.median(sec):.2f}), epochs mean {ep.mean():.0f} (median {np.median(ep):.0f})")
    print(f"Extrapolated test year per control area (35,136 QH x 4 sets, 1 thread): {sec.mean() * 35136 * 4 / 3600:.1f} h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", default="DE2")
    parser.add_argument("--every", type=int, default=13, help="every n-th day of 2023")
    parser.add_argument("--thin", type=int, default=1, help="use every n-th of the sampled days (nested subsample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--timing", action="store_true", help="unloaded runtime on TIMING_DAY (one worker, fresh process per set)")
    args = parser.parse_args()
    if args.timing:
        OUT_DIR = os.path.join(OUT_DIR, "timing")
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(1, maxtasksperchild=1, initializer=_set_out_dir, initargs=(OUT_DIR,)) as pool:
            for msg in pool.imap(run, [("DE2", TIMING_DAY, fs, args.seed) for fs in FEATURE_SETS]):
                print(msg, flush=True)
        sec = np.concatenate([np.load(f)["seconds"] for f in glob.glob(os.path.join(OUT_DIR, "*.npz"))])
        sec = sec[sec > 0]
        print(f"Unloaded: {sec.mean():.2f} s per fit, extrapolated test year per control area: {sec.mean() * 35136 * 4 / 3600:.1f} h")
        sys.exit()
    if not args.summary:
        import pandas as pd
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(*YEAR, freq="D")][args.every // 2::args.every][::args.thin]
        jobs = [(z, d, fs, args.seed) for z, d, fs in itertools.product(args.zones.split(","), days, FEATURE_SETS)
                if not os.path.exists(os.path.join(OUT_DIR, f"{z}_{d}_{SET_TAGS[fs]}.npz"))]
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(min(len(jobs), N_PROC), maxtasksperchild=4) as pool:
            for msg in pool.imap_unordered(run, jobs):
                print(msg, flush=True)
    summary()
