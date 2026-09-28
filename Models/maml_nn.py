"""
MAML-NN with Generalized Negative Correlation Learning (GNCL) for per-QH EPF Forecasting.
Implements Model-Agnostic Meta-Learning (Finn et al., 2017) with cooperative GNCL multi-set
backbone pre-training (Buschjäger, Pfahler & Morik, 2020) and proximal soft-thresholding.
"""
from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Strictly CPU execution to avoid multi-process GPU conflicts
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge

from config import (
    MAML_ACTIVATION,
    MAML_BACKBONE_BATCH,
    MAML_DROPOUT,
    MAML_HIDDEN_SIZES,
    MAML_INNER_LR,
    MAML_INNER_STEPS,
    MAML_L1_REG,
    MAML_META_EPOCHS,
    MAML_META_LR,
    MAML_PROX_SHRINK,
    MAML_CLIP_RESIDUAL,
    MAML_USE_LINEAR_BYPASS,
    MAML_BYPASS_SATURATION_M,
    MAML_SUPPORT_SELECTION,
    MAML_SUPPORT_K,
    MAML_SUPPORT_WEIGHTING,
    MAML_KERNEL_TAU,
    N_QH,
    TRAINED_MODELS_DIR,
    FEATURE_SET_ALL,
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
)

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)


# Feature sets definition
FEATURE_SETS_ALL_FOUR = [
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
]


def build_maml_net(
    n_features: int,
    hidden_sizes: list[int] = MAML_HIDDEN_SIZES,
    dropout: float = MAML_DROPOUT,
    l1_reg: float = MAML_L1_REG,
    activation: str = MAML_ACTIVATION,
    seed: Optional[int] = None,
) -> keras.Model:
    """Constructs a compact residual MLP with LayerNormalization, L1 penalty, and MAE loss."""
    inp = keras.Input(shape=(n_features,), name="features")
    reg = regularizers.l1(l1_reg) if l1_reg > 0 else None
    k_init = keras.initializers.GlorotUniform(seed=seed) if seed is not None else "glorot_uniform"
    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, name=f"dense_{i}", kernel_initializer=k_init, kernel_regularizer=reg)(x)
        if activation == "gelu":
            x = layers.Activation("gelu", name=f"gelu_{i}")(x)
        elif activation == "leaky_relu":
            x = layers.LeakyReLU(negative_slope=0.01, name=f"lrelu_{i}")(x)
        else:
            x = layers.Activation(activation, name=f"act_{i}")(x)
        x = layers.LayerNormalization(name=f"ln_{i}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = layers.Dense(1, activation="linear", name="residual_out", kernel_initializer="zeros", bias_initializer="zeros")(x)
    model = keras.Model(inputs=inp, outputs=out, name="MAML_EPFNet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=MAML_META_LR), loss=tf.keras.losses.MeanAbsoluteError())
    return model


class MAMLManager:
    """Manages the shared MAML residual backbone and fast per-QH inner-loop adaptation."""

    def __init__(
        self,
        zone: str,
        n_features: int,
        n_qh: int = N_QH,
        feature_set: str = FEATURE_SET_ALL,
        hidden_sizes: list[int] = MAML_HIDDEN_SIZES,
        dropout: float = MAML_DROPOUT,
        l1_reg: float = MAML_L1_REG,
        activation: str = MAML_ACTIVATION,
        seed: int = 42,
        cache_path: str | None = None,
    ):
        self.zone = zone
        self.n_features = n_features
        self.n_qh = n_qh
        self.feature_set = feature_set
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.l1_reg = l1_reg
        self.activation = activation
        self.seed = seed
        self.cache_path = cache_path or os.path.join(TRAINED_MODELS_DIR, "maml_cache")
        os.makedirs(self.cache_path, exist_ok=True)

        self._backbone_weights: list[np.ndarray] | None = None
        self._working_model = build_maml_net(n_features, hidden_sizes, dropout, l1_reg, activation, seed=seed)
        self._variables = self._working_model.trainable_variables

    def pretrain_backbone(
        self,
        loader,
        test_start: str = "2024-01-01",
        quarter_tag: Optional[str] = None,
        lookback_days: int = 822,
        val_days: int = 30,
        meta_epochs: int = MAML_META_EPOCHS,
        verbose: bool = True,
        force_retrain: bool = False,
    ) -> None:
        """Loads or trains the shared MAML backbone across QH positions with seed-specific checkpoint."""
        q_tag = f"_{quarter_tag}" if quarter_tag else ""
        arch_tag = f"_h{'-'.join(map(str, self.hidden_sizes))}_d{int(self.dropout*100)}_l1{self.l1_reg:.0e}{q_tag}_seed{self.seed}"
        checkpoint = os.path.join(self.cache_path, f"backbone_{self.zone}_{self.feature_set}{arch_tag}.pkl")
        if os.path.exists(checkpoint) and not force_retrain:
            self._backbone_weights = pickle.load(open(checkpoint, "rb"))
            self._working_model.set_weights(self._backbone_weights)
            if verbose:
                print(f"[{self.zone} {self.feature_set} seed={self.seed}] Loaded cached backbone: {os.path.basename(checkpoint)}")
            return

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        windows = [loader.get_window(self.zone, q, test_start, feature_set=self.feature_set, lookback_days=lookback_days, val_days=val_days) for q in range(self.n_qh)]
        valid_w = [w for w in windows if w is not None and w.X_train_full is not None]
        X_all = np.vstack([w.X_train_full for w in valid_w]).astype(np.float32)
        r_all = np.concatenate([w.y_train - Ridge(alpha=100.0).fit(w.X_train_full, w.y_train).predict(w.X_train_full) for w in valid_w]).astype(np.float32)

        self._working_model.fit(X_all, r_all, epochs=meta_epochs, batch_size=MAML_BACKBONE_BATCH, shuffle=True, verbose=0)
        self._backbone_weights = self._working_model.get_weights()
        with open(checkpoint, "wb") as f:
            pickle.dump(self._backbone_weights, f)
        if verbose:
            print(f"[{self.zone} {self.feature_set} seed={self.seed}] Pretrained and saved backbone: {os.path.basename(checkpoint)}")

    def set_backbone_weights(self, weights: list[np.ndarray]) -> None:
        """Dynamically update shared backbone weights (e.g. for quarterly rolling updates)."""
        self._backbone_weights = weights
        self._working_model.set_weights(weights)

    @tf.function(reduce_retracing=True)
    def _adapt_and_predict_graph(self, xs: tf.Tensor, rs: tf.Tensor, xte: tf.Tensor, lr: tf.Tensor, prox_shrink: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        for _ in range(MAML_INNER_STEPS):
            with tf.GradientTape() as tape:
                pred = self._working_model(xs, training=True)
                loss = tf.reduce_sum(weights * tf.abs(rs - pred))
            grads = tape.gradient(loss, self._variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            for v, g in zip(self._variables, grads):
                if g is not None:
                    v.assign_sub(lr * g)
                    if prox_shrink > 0.0:
                        v.assign(tf.sign(v) * tf.maximum(0.0, tf.abs(v) - prox_shrink))
        return self._working_model(xte, training=False)

    def adapt_and_predict(
        self,
        qh_idx: int,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_test: np.ndarray,
        linear_coef: Optional[np.ndarray] = None,
        linear_intercept: float = 0.0,
        use_linear_bypass: bool = MAML_USE_LINEAR_BYPASS,
        inner_steps: int = MAML_INNER_STEPS,
        inner_lr: float = MAML_INNER_LR,
        prox_shrink: float = MAML_PROX_SHRINK,
        clip_residual: Optional[float] = MAML_CLIP_RESIDUAL,
        bypass_saturation_m: Optional[float] = MAML_BYPASS_SATURATION_M,
        selection_mode: str = MAML_SUPPORT_SELECTION,
        support_k: int = MAML_SUPPORT_K,
        return_residual: bool = False,
    ) -> float:
        """Fast compiled inner-loop adaptation starting from warm-start backbone."""
        if self._backbone_weights is None:
            raise RuntimeError(f"[MAML {self.zone}] Shared backbone not initialized.")

        # 1. Reset model to shared backbone
        self._working_model.set_weights(self._backbone_weights)

        # 2. Compute residual on support set (optional linear bypass with consistent saturation)
        if use_linear_bypass and linear_coef is not None:
            coef_vec = np.asarray(linear_coef, dtype=np.float32).ravel()
            p_lin_support = (X_support @ coef_vec + linear_intercept).astype(np.float32)
            if bypass_saturation_m is not None and bypass_saturation_m > 0:
                p_lin_support_safe = (bypass_saturation_m * np.tanh(p_lin_support / bypass_saturation_m)).astype(np.float32)
                r_support = (y_support.ravel() - p_lin_support_safe).astype(np.float32)[:, None]
            else:
                r_support = (y_support.ravel() - p_lin_support).astype(np.float32)[:, None]
        else:
            r_support = y_support.ravel().astype(np.float32)[:, None]

        # 3. Support set selection (L1 regime nearest neighbors or trailing calendar days)
        if selection_mode == "regime_l1" and len(X_support) > support_k:
            dists = cdist(X_support, X_test.reshape(1, -1), metric="cityblock").ravel()
            sel_idx = np.argsort(dists)[:support_k]
            X_s, r_s = tf.cast(X_support[sel_idx], tf.float32), tf.cast(r_support[sel_idx], tf.float32)
            if MAML_SUPPORT_WEIGHTING == "soft_kernel":
                d_sel = dists[sel_idx]
                scale = max(float(np.std(d_sel)), 1e-4) * MAML_KERNEL_TAU
                w_k = np.exp(-(d_sel - d_sel[0]) / scale)
                w_k = (w_k / np.sum(w_k)).astype(np.float32)[:, None]
                weights_s = tf.cast(w_k, tf.float32)
            else:
                weights_s = tf.ones((support_k, 1), dtype=tf.float32) / float(support_k)
        else:
            k = min(support_k, len(X_support))
            X_s, r_s = tf.cast(X_support[-k:], tf.float32), tf.cast(r_support[-k:], tf.float32)
            weights_s = tf.ones((k, 1), dtype=tf.float32) / float(k)

        # 4. Fast compiled inner adaptation and prediction in single TF graph
        lr_t, prox_t = tf.constant(inner_lr, tf.float32), tf.constant(prox_shrink, tf.float32)
        x_te = tf.cast(X_test.reshape(1, -1), tf.float32)
        r_pred = float(self._adapt_and_predict_graph(X_s, r_s, x_te, lr_t, prox_t, weights_s).numpy().ravel()[0])
        if clip_residual is not None and clip_residual > 0:
            r_pred = float(np.clip(r_pred, -clip_residual, clip_residual))

        if return_residual or (not use_linear_bypass) or linear_coef is None:
            return r_pred

        # 6. Optional linear bypass with smooth tanh saturation
        p_lin_test = float((X_test.ravel() @ coef_vec) + linear_intercept)
        if bypass_saturation_m is not None and bypass_saturation_m > 0:
            p_lin_safe = float(bypass_saturation_m * np.tanh(p_lin_test / bypass_saturation_m))
        else:
            p_lin_safe = p_lin_test

        return float(p_lin_safe + r_pred)


