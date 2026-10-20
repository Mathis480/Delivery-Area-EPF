"""
Neural network with fast local adaptation (FA-NN).

Stage 1 trains a backbone once per zone and feature set on the pooled calibration samples of all 96 products.
Stage 2 adapts a copy of the backbone for every forecast with a few proximal SGD steps on the K most similar
calibration days (L1 distance) of the same product.

Training modes:
* ``fann``: supervised backbone (Adam, MAE).
* ``fomaml``: supervised backbone followed by first-order MAML (Finn et al., 2017), ablation.
* ``fomaml_scratch``: first-order MAML from random initialization, ablation.

Bypass modes: ``none`` or ``lasso`` (the network fits the residual of a saturated LASSO forecast).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import (
    CORR_THRESHOLDS, FANN_ACTIVATION, FANN_BATCH, FANN_BYPASS_SATURATION_M, FANN_EPOCHS, FANN_HIDDEN_SIZES,
    FANN_INNER_LR, FANN_INNER_STEPS, FANN_L1_REG, FANN_LR, FANN_PROX_SHRINK, FANN_SUPPORT_K, FOMAML_EPOCHS,
    FOMAML_META_BATCH, FOMAML_SCRATCH_EPOCHS, N_QH, RANDOM_SEED, TEST_START,
)
from Models.linear import train_lasso

TRAINING_MODES = ("fann", "fomaml", "fomaml_scratch")
BYPASS_MODES = ("none", "lasso")


def build_net(n_features: int, seed: int = RANDOM_SEED) -> keras.Model:
    """MLP with tanh, layer normalization, L1-regularized kernels and a zero-initialized linear output."""
    init = keras.initializers.GlorotUniform(seed=seed)
    x = inp = keras.Input(shape=(n_features,))
    for i, h in enumerate(FANN_HIDDEN_SIZES):
        x = layers.Dense(h, kernel_initializer=init, kernel_regularizer=regularizers.l1(FANN_L1_REG), name=f"dense_{i}")(x)
        x = layers.Activation(FANN_ACTIVATION)(x)
        x = layers.LayerNormalization()(x)
    out = layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="output")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FANN_LR), loss=keras.losses.MeanAbsoluteError())
    return model


def saturate(p: np.ndarray) -> np.ndarray:
    """Smooth saturation of the LASSO bypass: m * tanh(p / m)."""
    return FANN_BYPASS_SATURATION_M * np.tanh(p / FANN_BYPASS_SATURATION_M)


class FastAdaptNN:
    """FA-NN for one zone and feature set."""

    def __init__(self, zone: str, n_features: int, feature_set: str, training_mode: str = "fann",
                 bypass_mode: str = "none", seed: int = RANDOM_SEED):
        if training_mode not in TRAINING_MODES or bypass_mode not in BYPASS_MODES:
            raise ValueError(f"Unknown mode: {training_mode}/{bypass_mode}")
        self.zone, self.feature_set, self.seed = zone, feature_set, seed
        self.training_mode, self.bypass_mode = training_mode, bypass_mode
        self._model = build_net(n_features, seed=seed)
        self._vars = self._model.trainable_variables
        self._backbone: list[np.ndarray] | None = None

    def pretrain(self, loader, test_start: str = TEST_START) -> None:
        """Stage 1 on the calibration window preceding the first test day (target: spread or LASSO residual)."""
        corr = self.feature_set in CORR_THRESHOLDS  # same LASSO inputs as the LASSO benchmark
        windows = [loader.get_window(self.zone, q, test_start, self.feature_set, corr_filter=corr) for q in range(N_QH)]
        X_by_qh = [w.X_train_full.astype(np.float32) for w in windows]
        if self.bypass_mode == "lasso":
            y_by_qh = [(w.y_train - saturate(train_lasso(w.X_train, w.y_train).predict(w.X_train))).astype(np.float32)
                       for w in windows]
        else:
            y_by_qh = [w.y_train.astype(np.float32) for w in windows]

        if self.training_mode != "fomaml_scratch":
            keras.utils.set_random_seed(self.seed)
            self._model.fit(np.vstack(X_by_qh), np.concatenate(y_by_qh), epochs=FANN_EPOCHS,
                            batch_size=FANN_BATCH, shuffle=True, verbose=0)
        if self.training_mode != "fann":
            epochs = FOMAML_SCRATCH_EPOCHS if self.training_mode == "fomaml_scratch" else FOMAML_EPOCHS
            self._fomaml(X_by_qh, y_by_qh, epochs)
        self._backbone = self._model.get_weights()

    @tf.function(reduce_retracing=True)
    def _inner_adapt(self, xs: tf.Tensor, ys: tf.Tensor) -> None:
        """Stage 2: FANN_INNER_STEPS proximal SGD steps on the support set (in place)."""
        for _ in range(FANN_INNER_STEPS):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.abs(ys - self._model(xs, training=False)))
            for v, g in zip(self._vars, tape.gradient(loss, self._vars)):
                v.assign_sub(FANN_INNER_LR * g)
                if "kernel" in v.name:
                    v.assign(tf.sign(v) * tf.maximum(0.0, tf.abs(v) - FANN_PROX_SHRINK))

    @tf.function(reduce_retracing=True)
    def _query_grad(self, xs, ys, xq, yq) -> list[tf.Tensor]:
        """First-order meta-gradient: adapt on the support set, gradient of the query loss at the adapted weights."""
        self._inner_adapt(xs, ys)
        with tf.GradientTape() as tape:
            q_loss = tf.reduce_mean(tf.abs(yq - self._model(xq, training=False))) + tf.add_n(self._model.losses)
        return tape.gradient(q_loss, self._vars)

    def _fomaml(self, X_by_qh: list[np.ndarray], y_by_qh: list[np.ndarray], epochs: int) -> None:
        """FOMAML: tasks = (product, query day) sampled i.i.d.; support = K nearest earlier days."""
        n_days = len(X_by_qh[0])
        rng = np.random.default_rng(self.seed)
        opt = keras.optimizers.Adam(learning_rate=FANN_LR)
        for _ in range(epochs * max(1, n_days // FOMAML_META_BATCH)):
            theta = [tf.identity(v) for v in self._vars]
            grads = []
            for _ in range(FOMAML_META_BATCH):
                q = rng.integers(len(X_by_qh))
                t = rng.integers(FANN_SUPPORT_K, n_days)
                X_q, y_q = X_by_qh[q], y_by_qh[q]
                pool = np.arange(t)
                nn = pool[np.argsort(cdist(X_q[pool], X_q[t: t + 1], metric="cityblock").ravel())[:FANN_SUPPORT_K]]
                for v, t_val in zip(self._vars, theta):
                    v.assign(t_val)
                grads.append(self._query_grad(tf.constant(X_q[nn]), tf.constant(y_q[nn, None]),
                                              tf.constant(X_q[t: t + 1]), tf.constant(y_q[t: t + 1, None])))
            for v, t_val in zip(self._vars, theta):
                v.assign(t_val)
            opt.apply_gradients(zip([tf.reduce_mean([g[i] for g in grads], axis=0) for i in range(len(self._vars))],
                                    self._vars))

    def adapt_and_predict(self, X_support: np.ndarray, y_support: np.ndarray, X_test: np.ndarray,
                          lasso_support: np.ndarray | None = None, lasso_test: float | None = None) -> float:
        """Stage 2: adapt the backbone on the K nearest calibration days and forecast the test day."""
        self._model.set_weights(self._backbone)
        y_target, offset = y_support.ravel(), 0.0
        if self.bypass_mode == "lasso":
            y_target = y_target - saturate(lasso_support).ravel()
            offset = float(saturate(np.asarray(lasso_test)))

        nn = np.argsort(cdist(X_support, X_test.reshape(1, -1), metric="cityblock").ravel())[:FANN_SUPPORT_K]
        self._inner_adapt(tf.constant(X_support[nn], dtype=tf.float32), tf.constant(y_target[nn, None], dtype=tf.float32))
        return float(self._model(tf.constant(X_test.reshape(1, -1), dtype=tf.float32), training=False).numpy()[0, 0]) + offset
