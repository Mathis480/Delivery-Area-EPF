"""
Why does the supervised backbone (FA-NN) outperform FOMAML? Diagnostics on a pre-test period
(DE2, feature set S2, calibration window before 2023-10-01, evaluation on Q4 2023).

Figure D1: (a) training paths in weight space (PCA) over the contours of the pooled training loss,
           (b) MAE after local adaptation vs. number of samples entering the (outer) gradient.
Figure D2: sensitivity of the backbone learning rate x epochs and of the inner learning rate x steps.
Metrics (printed and saved): gradient alignment and signal-to-noise ratios, size of the local adaptation.

Usage: python Evaluation/fomaml_diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from tensorflow import keras

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Evaluation"))
import plot_style
import Models.fann as fann
from config import FANN_BATCH, FANN_EPOCHS, FANN_LR, FANN_SUPPORT_K, FEATURE_SET_NEIGHBOR, FOMAML_EPOCHS, \
    FOMAML_META_BATCH, FOMAML_SCRATCH_EPOCHS, N_QH, RANDOM_SEED
from data_loader import QHDataLoader

ZONE, FS, CALIB_END, EVAL_START, EVAL_END = "DE2", FEATURE_SET_NEIGHBOR, "2023-10-01", "2023-10-01", "2023-12-31"
N_EVAL_DAYS = 12
OUT_DIR = os.path.join(ROOT, "Evaluation", "plots", "drafts")
COL = {"FA-NN backbone": "#1f77b4", "FOMAML (from backbone)": "#9467bd", "FOMAML (scratch)": "#d62728"}
plot_style.apply()


# ---------------------------------------------------------------- data
def load_data(loader):
    wins = [loader.get_window(ZONE, q, CALIB_END, FS) for q in range(N_QH)]
    X_by_qh = [w.X_train_full.astype(np.float32) for w in wins]
    y_by_qh = [w.y_train.astype(np.float32) for w in wins]
    test = loader.test_targets(ZONE, EVAL_START, EVAL_END)
    days = sorted(test["date"].unique())
    days = [days[i] for i in np.linspace(0, len(days) - 1, N_EVAL_DAYS).astype(int)]
    ev = []
    for r in test[test["date"].isin(days)].itertuples(index=False):
        w = loader.get_window(ZONE, r.qh, r.date, FS)
        ev.append((w.X_train_full.astype(np.float32), w.y_train.astype(np.float32), w.X_test_full.astype(np.float32),
                   w.y_mean, w.y_std, r.naive, r.y))
    return X_by_qh, y_by_qh, ev


def eval_mae(weights, ev, net) -> float:
    """MAE (EUR/MWh) of Stage-2 forecasts starting from `weights`."""
    net._backbone = weights
    err = [abs(net.adapt_and_predict(Xs, ys, xt) * s + m + nv - y) for Xs, ys, xt, m, s, nv, y in ev]
    return float(np.mean(err))


def flat(weights) -> np.ndarray:
    return np.concatenate([w.ravel() for w in weights])


# ---------------------------------------------------------------- training with snapshots
def train_backbone(X, y, n_features, lr=FANN_LR, epochs=FANN_EPOCHS, every=50):
    """Stage-1 training as in Models.fann with snapshots every `every` batches and after every epoch."""
    keras.utils.set_random_seed(RANDOM_SEED)
    model = fann.build_net(n_features)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.MeanAbsoluteError())
    snaps, epoch_snaps, seen = [(0, model.get_weights())], [], [0]

    class Snap(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            seen[0] += 1
            if seen[0] % every == 0:
                snaps.append((seen[0] * FANN_BATCH, model.get_weights()))

        def on_epoch_end(self, epoch, logs=None):
            epoch_snaps.append(model.get_weights())

    model.fit(X, y, epochs=epochs, batch_size=FANN_BATCH, shuffle=True, verbose=0, callbacks=[Snap()])
    snaps.append((seen[0] * FANN_BATCH, model.get_weights()))
    return model.get_weights(), snaps, epoch_snaps


class SnapFOMAML(fann.FastAdaptNN):
    """FOMAML with weight snapshots every `every` meta-updates (same algorithm as Models.fann)."""

    def run(self, X_by_qh, y_by_qh, epochs, every=15):
        n_days = len(X_by_qh[0])
        rng = np.random.default_rng(self.seed)
        opt = keras.optimizers.Adam(learning_rate=FANN_LR)
        snaps = [(0, self._model.get_weights())]
        n_updates = epochs * max(1, n_days // FOMAML_META_BATCH)
        for u in range(1, n_updates + 1):
            grads = self.meta_gradients(X_by_qh, y_by_qh, rng)
            opt.apply_gradients(zip([tf.reduce_mean([g[i] for g in grads], axis=0) for i in range(len(self._vars))],
                                    self._vars))
            if u % every == 0 or u == n_updates:
                snaps.append((u * FOMAML_META_BATCH, self._model.get_weights()))
        return self._model.get_weights(), snaps

    def meta_gradients(self, X_by_qh, y_by_qh, rng, n_tasks=FOMAML_META_BATCH):
        theta = [tf.identity(v) for v in self._vars]
        n_days, grads = len(X_by_qh[0]), []
        for _ in range(n_tasks):
            q, t = rng.integers(len(X_by_qh)), rng.integers(FANN_SUPPORT_K, n_days)
            Xq, yq = X_by_qh[q], y_by_qh[q]
            pool = np.arange(t)
            nn = pool[np.argsort(cdist(Xq[pool], Xq[t:t + 1], metric="cityblock").ravel())[:FANN_SUPPORT_K]]
            for v, tv in zip(self._vars, theta):
                v.assign(tv)
            grads.append(self._query_grad(tf.constant(Xq[nn]), tf.constant(yq[nn, None]),
                                          tf.constant(Xq[t:t + 1]), tf.constant(yq[t:t + 1, None])))
        for v, tv in zip(self._vars, theta):
            v.assign(tv)
        return grads


# ---------------------------------------------------------------- metrics
def supervised_grad(model, X, y) -> np.ndarray:
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.abs(y[:, None] - model(X, training=False)))
    return flat([g.numpy() for g in tape.gradient(loss, model.trainable_variables)])


def snr(samples: np.ndarray) -> float:
    """||E g||^2 / E||g - E g||^2 for gradient samples (rows)."""
    mean = samples.mean(axis=0)
    return float(mean @ mean / np.mean(np.sum((samples - mean) ** 2, axis=1)))


def cosine(a, b) -> float:
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main() -> None:
    loader = QHDataLoader()
    X_by_qh, y_by_qh, ev = load_data(loader)
    X, y = np.vstack(X_by_qh), np.concatenate(y_by_qh)
    n_feat = X.shape[1]
    evaluator = fann.FastAdaptNN(ZONE, n_feat, FS)
    naive_mae = float(np.mean([abs(nv - yv) for *_, nv, yv in ev]))

    # 1. Training paths
    w_bb, snaps_bb, _ = train_backbone(X, y, n_feat)
    fom = SnapFOMAML(ZONE, n_feat, FS, training_mode="fomaml")
    fom._model.set_weights(w_bb)
    _, snaps_fom = fom.run(X_by_qh, y_by_qh, FOMAML_EPOCHS)
    scr = SnapFOMAML(ZONE, n_feat, FS, training_mode="fomaml_scratch")
    _, snaps_scr = scr.run(X_by_qh, y_by_qh, FOMAML_SCRATCH_EPOCHS, every=25)
    offset = snaps_bb[-1][0]  # FOMAML from the backbone continues after the backbone budget
    paths = {"FA-NN backbone": snaps_bb,
             "FOMAML (from backbone)": [(offset + n, w) for n, w in snaps_fom],
             "FOMAML (scratch)": snaps_scr}
    curves = {k: [(n, eval_mae(w, ev, evaluator)) for n, w in v] for k, v in paths.items()}

    # 2. Gradient metrics at the backbone
    model = fann.build_net(n_feat)
    model.set_weights(w_bb)
    rng = np.random.default_rng(RANDOM_SEED)
    g_full = supervised_grad(model, X, y)
    sup = {b: np.array([supervised_grad(model, X[i], y[i]) for i in (rng.choice(len(X), b, replace=False) for _ in range(200))])
           for b in (16, 256)}
    fom_eval = SnapFOMAML(ZONE, n_feat, FS, training_mode="fomaml")
    fom_eval._model.set_weights(w_bb)
    meta = np.array([flat([g.numpy() for g in fom_eval.meta_gradients(X_by_qh, y_by_qh, rng, n_tasks=1)[0]])
                     for _ in range(3200)])
    meta16 = meta.reshape(200, 16, -1).mean(axis=1)
    shifts = []
    for Xs, ys, xt, *_ in ev[:: max(1, len(ev) // 200)]:
        evaluator._backbone = w_bb
        evaluator._model.set_weights(w_bb)
        nn = np.argsort(cdist(Xs, xt.reshape(1, -1), metric="cityblock").ravel())[:FANN_SUPPORT_K]
        evaluator._inner_adapt(tf.constant(Xs[nn]), tf.constant(ys[nn, None]))
        shifts.append(np.linalg.norm(flat(evaluator._model.get_weights()) - flat(w_bb)) / np.linalg.norm(flat(w_bb)))
    metrics = {
        "naive_mae": naive_mae,
        "cos(mean FOMAML meta-gradient, full supervised gradient)": cosine(meta16.mean(axis=0), g_full),
        "SNR supervised batch 256": snr(sup[256]), "SNR supervised batch 16": snr(sup[16]),
        "SNR FOMAML meta-batch 16": snr(meta16),
        "relative weight change by Stage 2 (median)": float(np.median(shifts)),
        "final MAE": {k: c[-1][1] for k, c in curves.items()},
    }

    # 3. Sensitivity: backbone learning rate x epochs, inner learning rate x steps
    sens_bb = {}
    for lr in (1e-4, FANN_LR, 2e-3):
        _, _, ep = train_backbone(X, y, n_feat, lr=lr, epochs=12, every=10 ** 9)
        sens_bb[lr] = [eval_mae(w, ev, evaluator) for w in ep]
    sens_inner, defaults = {}, (fann.FANN_INNER_LR, fann.FANN_INNER_STEPS)
    for alpha in (0.001, fann.FANN_INNER_LR, 0.005):
        sens_inner[alpha] = []
        for steps in (0, 2, 4, 8, 12, 16):
            fann.FANN_INNER_LR, fann.FANN_INNER_STEPS = alpha, steps
            sens_inner[alpha].append((steps, eval_mae(w_bb, ev, fann.FastAdaptNN(ZONE, n_feat, FS))))
    fann.FANN_INNER_LR, fann.FANN_INNER_STEPS = defaults

    # ---------------------------------------------------------------- figures
    W = np.array([flat(w) for v in paths.values() for _, w in v])
    center = flat(w_bb)
    _, _, vt = np.linalg.svd(W - center, full_matrices=False)
    pcs = vt[:2]
    proj = {k: np.array([(flat(w) - center) @ pcs.T for _, w in v]) for k, v in paths.items()}
    allp = np.vstack(list(proj.values()))
    gx = np.linspace(allp[:, 0].min() * 1.15, allp[:, 0].max() * 1.15, 21)
    gy = np.linspace(allp[:, 1].min() * 1.15, allp[:, 1].max() * 1.15, 21)
    shapes = [w.shape for w in w_bb]
    loss = np.zeros((len(gy), len(gx)))
    sub = rng.choice(len(X), 20000, replace=False)
    for i, b in enumerate(gy):
        for j, a in enumerate(gx):
            vec = center + a * pcs[0] + b * pcs[1]
            ws, k = [], 0
            for s in shapes:
                n = int(np.prod(s)); ws.append(vec[k:k + n].reshape(s)); k += n
            model.set_weights(ws)
            loss[i, j] = float(np.mean(np.abs(y[sub, None] - model(X[sub], training=False).numpy())))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_style.TEXT_WIDTH_IN, 2.9))
    cs = ax1.contourf(gx, gy, loss, levels=14, cmap=plot_style.SEQUENTIAL_CMAP + "_r", alpha=0.85)
    for k, p in proj.items():
        ax1.plot(p[:, 0], p[:, 1], color=COL[k], marker="o", markersize=2, linewidth=1.0, label=k)
        ax1.plot(*p[-1], marker="*", markersize=9, color=COL[k], markeredgecolor="black", markeredgewidth=0.5)
    ax1.set_xlabel("PC 1"); ax1.set_ylabel("PC 2"); ax1.grid(False)
    fig.colorbar(cs, ax=ax1, fraction=0.046, pad=0.02).set_label("Calibration MAE (std.)")
    for k, c in curves.items():
        n, m = zip(*c)
        ax2.plot(np.maximum(n, 1), m, color=COL[k], marker="o", markersize=2, label=k)
    ax2.axhline(naive_mae, color="black", linestyle="--", linewidth=1.0, label="Naive")
    ax2.set_xscale("log"); ax2.set_xlabel("Samples in (outer) gradient"); ax2.set_ylabel("MAE Q4 2023 (EUR/MWh)")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"diag_D1_weight_space.{ext}"))
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_style.TEXT_WIDTH_IN, 2.6), sharey=True)
    cmap = plt.get_cmap(plot_style.SEQUENTIAL_CMAP)
    for i, (lr, m) in enumerate(sens_bb.items()):
        ax1.plot(range(1, len(m) + 1), m, marker="o", markersize=3, color=cmap(0.15 + 0.35 * i),
                 label=f"lr = {lr:g}" + (" (used)" if lr == FANN_LR else ""))
    ax1.axvline(FANN_EPOCHS, color="black", linewidth=0.6, linestyle=":")
    ax1.set_xlabel("Backbone epochs"); ax1.set_ylabel("MAE Q4 2023 (EUR/MWh)"); ax1.legend(fontsize=7)
    for i, (alpha, pts) in enumerate(sens_inner.items()):
        s, m = zip(*pts)
        ax2.plot(s, m, marker="o", markersize=3, color=cmap(0.15 + 0.35 * i),
                 label=f"α = {alpha:g}" + (" (used)" if alpha == defaults[0] else ""))
    ax2.axvline(defaults[1], color="black", linewidth=0.6, linestyle=":")
    ax2.set_xlabel("Inner steps G"); ax2.legend(fontsize=7)
    for ax in (ax1, ax2):
        ax.axhline(naive_mae, color="black", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"diag_D2_sensitivity.{ext}"))
    plt.close(fig)

    metrics["sensitivity_backbone"] = {str(k): v for k, v in sens_bb.items()}
    metrics["sensitivity_inner"] = {str(k): v for k, v in sens_inner.items()}
    metrics["curves"] = {k: c for k, c in curves.items()}
    with open(os.path.join(OUT_DIR, "diag_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict) or k == "final MAE"}, indent=1))


if __name__ == "__main__":
    main()
