"""
FA-NN architecture ablations (FA-NN main variant only).

Usage:
    python Evaluation/ablation_pretest.py --zones DE1,DE2 --periods 2023-07-01:2023-09-30,2023-10-01:2023-12-31 current ln_pre
Every period is an out-of-sample block: the backbone is trained on the data before the period start.
Variants:
    current  model as in Models.fann
    ln       + layer normalization after every hidden activation (Dense -> tanh -> LN, as in the earlier backtest)
    ln_pre   + layer normalization before every activation (Dense -> LN -> tanh, as in Ba et al., 2016)
    reg      + L1 kernel penalty (1e-4) in Stage 1 and soft-thresholding (1e-5) in Stage 2
    ln_reg, ln_pre_reg   combinations (ln_reg = configuration of the earlier backtest)
Design decisions use pre-test periods (2023). Runs on the test year are diagnostics only.
Results: RESULTS_DIR/ablation/[<mode>_]<variant>_<zone>_<start>_<end>.npz (mode prefix for FOMAML modes)

Done (DE2 Q4 2023): current 20.9321, ln 20.9148, reg 21.1155 (worse, DM p<0.001), ln_reg 20.9277.
Decision rule for ln_pre vs. current (fixed before the 2023 validation, 4 zones x 4 quarters):
adopt ln_pre only if its mean MAE over the 16 blocks is lower AND it is better in at least 9 of 16 blocks.
Result (2026-10-21): ln_pre mean +0.0079 EUR/MWh, better in 6 of 16 blocks, pooled DM n.s. -> current kept.
Same rule for FOMAML without pre-training (--mode fomaml_scratch): ln_pre vs. current on the same 16 blocks.
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per process; parallelism across jobs

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import FEATURE_SETS, N_QH, RESULTS_DIR, SET_TAGS

OUT_DIR = os.path.join(RESULTS_DIR, "ablation")
VARIANTS = {"current": (None, False), "ln": ("post", False), "ln_pre": ("pre", False), "reg": (None, True),
            "ln_reg": ("post", True), "ln_pre_reg": ("pre", True)}
L1, PROX = 1e-4, 1e-5


def patched_fann(layer_norm: str | None, reg: bool):
    """FastAdaptNN with optional layer normalization and weight regularization (diagnostic variants)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    import Models.fann as fann

    def build_net(n_features, seed=fann.RANDOM_SEED):
        init = keras.initializers.GlorotUniform(seed=seed)
        x = inp = keras.Input(shape=(n_features,))
        for i, h in enumerate(fann.FANN_HIDDEN_SIZES):
            x = layers.Dense(h, kernel_initializer=init, name=f"dense_{i}",
                             kernel_regularizer=regularizers.l1(L1) if reg else None)(x)
            if layer_norm == "pre":
                x = layers.LayerNormalization()(x)
            x = layers.Activation(fann.FANN_ACTIVATION)(x)
            if layer_norm == "post":
                x = layers.LayerNormalization()(x)
        out = layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="output")(x)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=fann.FANN_LR), loss=keras.losses.MeanAbsoluteError())
        return model

    class Net(fann.FastAdaptNN):
        @tf.function(reduce_retracing=True)
        def _inner_adapt(self, xs, ys):
            for _ in range(fann.FANN_INNER_STEPS):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.abs(ys - self._model(xs, training=False)))
                for v, g in zip(self._vars, tape.gradient(loss, self._vars)):
                    v.assign_sub(fann.FANN_INNER_LR * g)
                    if reg and "kernel" in v.name:
                        v.assign(tf.sign(v) * tf.maximum(0.0, tf.abs(v) - PROX))

    fann.build_net = build_net
    return Net


def run(job) -> str:
    mode, variant, zone, start, end = job
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    Net = patched_fann(*VARIANTS[variant])
    from data_loader import QHDataLoader
    from Models.ensemble import inverse_mae_average

    loader = QHDataLoader()
    test = loader.test_targets(zone, start, end)
    days = sorted(test["date"].unique())
    d2i = {d: i for i, d in enumerate(days)}
    rows, cols = test["date"].map(d2i).to_numpy(), test["qh"].to_numpy()
    y, naive = np.full((len(days), N_QH), np.nan), np.full((len(days), N_QH), np.nan)
    y[rows, cols], naive[rows, cols] = test["y"].to_numpy(), test["naive"].to_numpy()

    preds, t0 = {}, time.perf_counter()
    for fs in FEATURE_SETS:
        net = Net(zone, loader.n_features(zone, fs), fs, training_mode=mode)
        net.pretrain(loader, test_start=start)
        p = np.full_like(y, np.nan)
        for r in test.itertuples(index=False):
            w = loader.get_window(zone, r.qh, r.date, fs)
            p[d2i[r.date], r.qh] = net.adapt_and_predict(w.X_train_full, w.y_train, w.X_test_full) * w.y_std + w.y_mean + r.naive
        preds[SET_TAGS[fs]] = p
    ens, _ = inverse_mae_average(preds, y)

    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = "" if mode == "fann" else f"{mode}_"
    np.savez_compressed(os.path.join(OUT_DIR, f"{prefix}{variant}_{zone}_{start}_{end}.npz"), dates=np.array(days), y_true=y,
                        naive=naive, pred=ens, **{f"pred_{k}": v for k, v in preds.items()})
    mae = lambda a: np.nanmean(np.abs(a - y))
    return (f"{prefix}{variant:7s} {zone} {start}..{end} ens={mae(ens):.4f} naive={mae(naive):.4f} | "
            + " ".join(f"{k}={mae(v):.4f}" for k, v in preds.items()) + f" | {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", choices=list(VARIANTS))
    parser.add_argument("--zones", default="DE2")
    parser.add_argument("--mode", default="fann", choices=["fann", "fomaml", "fomaml_scratch"])
    parser.add_argument("--periods", default="2023-10-01:2023-12-31", help="comma-separated start:end blocks")
    args = parser.parse_args()
    periods = [tuple(p.split(":")) for p in args.periods.split(",")]
    jobs = [(args.mode, v, z, s, e) for (s, e), z, v in itertools.product(periods, args.zones.split(","), args.variants)]
    multiprocessing.set_start_method("spawn", force=True)
    with multiprocessing.Pool(min(len(jobs), 12)) as pool:
        for msg in pool.map(run, jobs):
            print(msg, flush=True)
