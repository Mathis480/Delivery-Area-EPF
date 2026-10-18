"""
Per-QH EPF neural networks with inference-time fast adaptation (K nearest support days, SGD inner loop).

* ``WarmStartNN`` (mode ``warmstart``): pooled supervised pre-training of the backbone across all
  96 QH (Adam, MAE). Plain pre-train + fine-tune, NOT meta-learning.
* ``FOMAMLNN`` (mode ``maml``): warm-start on fold 0, then FIRST-ORDER MAML (Finn et al., 2017,
  Sec. 5.2). Per task, theta' = theta - alpha * grad L_support(theta) (MAML_INNER_STEPS SGD steps)
  and the meta-gradient is grad_{theta'} L_query(theta'); the second-order term d theta' / d theta
  is dropped. The averaged meta-gradient is applied to theta with Adam. NOT full second-order MAML.

Extensions beyond Finn et al.: warm-start initialization, kNN support selection, proximal
soft-thresholding in the inner loop, outer-gradient clipping (Antoniou et al., 2019), linear bypass.
Backbones are always retrained; checkpoints are written for reproduction only.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from Models.linear import train_lasso
from config import (
    MAML_ACTIVATION, MAML_BACKBONE_BATCH, MAML_HIDDEN_SIZES,
    MAML_INNER_LR, MAML_INNER_STEPS, MAML_L1_REG, MAML_META_EPOCHS,
    MAML_META_LR, MAML_PROX_SHRINK, MAML_BYPASS_SATURATION_M,
    MAML_SUPPORT_K, MAML_TRAINING_MODES, MAML_BILEVEL_EPOCHS, MAML_OUTER_CLIP,
    MAML_META_BATCH, MAML_N_FOLDS, MAML_WARMSTART_SCOPE,
    N_QH, LOOKBACK_DAYS, TRAINED_MODELS_DIR, FEATURE_SET_MACRO, RANDOM_SEED,
)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_maml_net(
    n_features: int,
    hidden_sizes: list[int] = MAML_HIDDEN_SIZES,
    l1_reg: float = MAML_L1_REG,
    activation: str = MAML_ACTIVATION,
    seed: int | None = RANDOM_SEED,
) -> keras.Model:
    """Constructs a compact MLP with LayerNormalization and MAE loss."""
    inp = keras.Input(shape=(n_features,), name="features")
    reg = regularizers.l1(l1_reg) if l1_reg > 0 else None
    k_init = keras.initializers.GlorotUniform(seed=seed)
    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, name=f"dense_{i}", kernel_initializer=k_init, kernel_regularizer=reg)(x)
        x = layers.Activation(activation, name=f"act_{i}")(x)
        x = layers.LayerNormalization(name=f"ln_{i}")(x)

    out = layers.Dense(1, activation="linear", name="residual_out", kernel_initializer="zeros", bias_initializer="zeros")(x)
    model = keras.Model(inputs=inp, outputs=out, name="MAML_EPFNet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=MAML_META_LR), loss=tf.keras.losses.MeanAbsoluteError())
    return model


class FastAdaptNN:
    """Shared base: network, data preparation, inner-loop adaptation and inference. Subclasses implement ``_train``."""

    training_mode: str

    def __init__(
        self,
        zone: str,
        n_features: int,
        feature_set: str = FEATURE_SET_MACRO,
        seed: int = RANDOM_SEED,
        bypass_mode: str = "nobypass",
    ):
        self.zone = zone
        self.n_features = n_features
        self.feature_set = feature_set
        self.seed = seed
        self.bypass_mode = bypass_mode

        # Reproduction-only artifact: written after training, never loaded.
        checkpoint_dir = os.path.join(TRAINED_MODELS_DIR, "maml_backbones")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            checkpoint_dir, f"backbone_{zone}_{feature_set}_{bypass_mode}_{self.training_mode}_seed{seed}.pkl"
        )

        self._working_model = build_maml_net(n_features, seed=seed)
        self._variables = self._working_model.trainable_variables
        self._backbone_weights: list[np.ndarray] | None = None

    def pretrain_backbone(self, loader, test_start: str = "2024-01-01", lookback_days: int = LOOKBACK_DAYS) -> None:
        """Loads all 96 QH training windows (target = raw spread or linear residual), trains and checkpoints."""
        windows = [
            loader.get_window(self.zone, q, test_start, feature_set=self.feature_set, lookback_days=lookback_days)
            for q in range(N_QH)
        ]
        X_by_qh = [w.X_train_full.astype(np.float32) for w in windows]
        if self.bypass_mode == "optlasso":
            y_by_qh = []
            for w in windows:
                m_l, _ = train_lasso(w.X_train_full, w.y_train)
                y_by_qh.append((w.y_train - m_l.predict(w.X_train_full)).astype(np.float32))
        else:
            y_by_qh = [w.y_train.astype(np.float32) for w in windows]

        self._train(X_by_qh, y_by_qh)

        meta = {
            "zone": self.zone, "feature_set": self.feature_set, "bypass_mode": self.bypass_mode,
            "training_mode": self.training_mode, "seed": self.seed, "n_features": self.n_features,
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump({"weights": self._backbone_weights, "meta": meta}, f)

    def _train(self, X_by_qh: list[np.ndarray], y_by_qh: list[np.ndarray]) -> None:
        raise NotImplementedError

    def _fit_warmstart(self, X: np.ndarray, y: np.ndarray) -> None:
        """Supervised pre-training on pooled samples (Adam, MAE)."""
        keras.utils.set_random_seed(self.seed)
        self._working_model.fit(X, y, epochs=MAML_META_EPOCHS, batch_size=MAML_BACKBONE_BATCH, shuffle=True, verbose=0)
        self._backbone_weights = self._working_model.get_weights()

    @tf.function(reduce_retracing=True)
    def _inner_adapt(self, xs: tf.Tensor, ys: tf.Tensor) -> None:
        """MAML_INNER_STEPS SGD steps on the support set, in place, with proximal soft-thresholding."""
        for _ in range(MAML_INNER_STEPS):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.abs(ys - self._working_model(xs, training=True)))
            grads = tape.gradient(loss, self._variables)
            for v, g in zip(self._variables, grads):
                v.assign_sub(MAML_INNER_LR * g)
                if MAML_PROX_SHRINK > 0.0 and "kernel" in v.name:
                    v.assign(tf.sign(v) * tf.maximum(0.0, tf.abs(v) - MAML_PROX_SHRINK))

    @tf.function(reduce_retracing=True)
    def _run_inner_loop(self, xs: tf.Tensor, ys: tf.Tensor, xte: tf.Tensor) -> tf.Tensor:
        """Adapts to the support set and predicts the test point."""
        self._inner_adapt(xs, ys)
        return self._working_model(xte, training=False)

    def set_backbone_weights(self, weights: list[np.ndarray]) -> None:
        """Set shared backbone weights directly (used for testing)."""
        self._backbone_weights = weights
        self._working_model.set_weights(weights)

    def adapt_and_predict(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_test: np.ndarray,
        linear_support: np.ndarray | None = None,
        linear_test: float | None = None,
    ) -> float:
        """Fast inner-loop adaptation from the trained backbone on the K nearest support days."""
        self._working_model.set_weights(self._backbone_weights)

        if self.bypass_mode == "optlasso":
            m = MAML_BYPASS_SATURATION_M
            p_lin_sup = m * np.tanh(linear_support / m)
            p_lin_test = float(m * np.tanh(linear_test / m))
            y_target = y_support.ravel() - p_lin_sup.ravel()
        else:
            p_lin_test = 0.0
            y_target = y_support.ravel()

        # K closest market regimes using L1 (cityblock) distance
        dists = cdist(X_support, X_test.reshape(1, -1), metric="cityblock").ravel()
        sel_idx = np.argsort(dists)[:MAML_SUPPORT_K]

        X_s = tf.constant(X_support[sel_idx], dtype=tf.float32)
        y_s = tf.constant(y_target[sel_idx, None], dtype=tf.float32)
        x_te = tf.constant(X_test.reshape(1, -1), dtype=tf.float32)

        pred = float(self._run_inner_loop(X_s, y_s, x_te).numpy()[0, 0])
        return pred + p_lin_test


class WarmStartNN(FastAdaptNN):
    """Pooled warm-start backbone over all training days and all QH; no meta-learning."""

    training_mode = "warmstart"

    def _train(self, X_by_qh: list[np.ndarray], y_by_qh: list[np.ndarray]) -> None:
        self._fit_warmstart(np.vstack(X_by_qh), np.concatenate(y_by_qh))


class FOMAMLNN(FastAdaptNN):
    """Warm-start initialization followed by first-order MAML meta-training (Finn et al., 2017, Sec. 5.2)."""

    training_mode = "maml"

    @tf.function(reduce_retracing=True)
    def _query_grad(self, xs: tf.Tensor, ys: tf.Tensor, xq: tf.Tensor, yq: tf.Tensor) -> list[tf.Tensor]:
        """FOMAML task gradient: adapt to support, return grad_{theta'} L_query(theta').

        First-order: nothing is propagated through the inner updates (weights are overwritten in place).
        """
        self._inner_adapt(xs, ys)
        with tf.GradientTape() as tape:
            q_loss = tf.reduce_mean(tf.abs(yq - self._working_model(xq, training=True)))
        return tape.gradient(q_loss, self._variables)

    def _train(self, X_by_qh: list[np.ndarray], y_by_qh: list[np.ndarray]) -> None:
        n_days = len(X_by_qh[0])
        fold_size = n_days // MAML_N_FOLDS

        # Stage 1: warm-start theta_0. Scope "fold0" keeps all query days (folds 1..K-1) unseen.
        if MAML_WARMSTART_SCOPE == "fold0":
            self._fit_warmstart(
                np.vstack([x[:fold_size] for x in X_by_qh]), np.concatenate([y[:fold_size] for y in y_by_qh])
            )
        else:
            self._fit_warmstart(np.vstack(X_by_qh), np.concatenate(y_by_qh))

        # Stage 2: FOMAML. Task = (QH q, query day t in fold k); support = K nearest days (L1) of QH q
        # from folds [0..k-1]. Averaged meta-gradient over MAML_META_BATCH tasks, clipped, applied via Adam.
        rng = np.random.default_rng(self.seed)
        opt = keras.optimizers.Adam(learning_rate=MAML_META_LR)
        total_meta_steps = MAML_BILEVEL_EPOCHS * max(1, n_days // MAML_META_BATCH)

        for _ in range(total_meta_steps):
            theta = self._working_model.get_weights()
            grads_list = []
            for _ in range(MAML_META_BATCH):
                q = rng.integers(len(X_by_qh))
                X_q, y_q = X_by_qh[q], y_by_qh[q]

                k = rng.integers(1, MAML_N_FOLDS)
                past_end = k * fold_size
                t = rng.integers(past_end, min((k + 1) * fold_size, n_days))
                supp_pool = np.arange(past_end)

                xq, yq = X_q[t : t + 1], y_q[t : t + 1, None]
                dists = cdist(X_q[supp_pool], xq, metric="cityblock").ravel()
                nn = supp_pool[np.argsort(dists)[:MAML_SUPPORT_K]]

                self._working_model.set_weights(theta)
                grads_list.append(self._query_grad(
                    tf.constant(X_q[nn]), tf.constant(y_q[nn, None]), tf.constant(xq), tf.constant(yq)
                ))

            self._working_model.set_weights(theta)
            avg_grads = [tf.reduce_mean([g[i] for g in grads_list], axis=0) for i in range(len(self._variables))]
            if MAML_OUTER_CLIP > 0.0:
                avg_grads, _ = tf.clip_by_global_norm(avg_grads, MAML_OUTER_CLIP)
            opt.apply_gradients(zip(avg_grads, self._variables))
        self._backbone_weights = self._working_model.get_weights()


def MAMLManager(
    zone: str,
    n_features: int,
    feature_set: str = FEATURE_SET_MACRO,
    seed: int = RANDOM_SEED,
    bypass_mode: str = "nobypass",
    training_mode: str | None = None,
) -> FastAdaptNN:
    """Factory: returns ``WarmStartNN`` (``warmstart``) or ``FOMAMLNN`` (``maml``)."""
    mode = (training_mode or MAML_TRAINING_MODES[0]).lower()
    classes = {"warmstart": WarmStartNN, "maml": FOMAMLNN}
    if mode not in classes:
        raise ValueError(f"Unknown training_mode: '{mode}'. Expected 'warmstart' or 'maml'.")
    return classes[mode](zone, n_features, feature_set, seed, bypass_mode)