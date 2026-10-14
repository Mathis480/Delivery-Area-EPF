"""
MAML-NN (Model-Agnostic Meta-Learning Neural Network) for per-QH EPF Forecasting.
Implements Model-Agnostic Meta-Learning (Finn et al., 2017) with warm-start backbone
pre-training, optional linear bypass residual learning, and fast SGD inner-loop adaptation.
"""
from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

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
    MAML_ACTIVATION, MAML_BACKBONE_BATCH, MAML_DROPOUT, MAML_HIDDEN_SIZES,
    MAML_INNER_LR, MAML_INNER_STEPS, MAML_L1_REG, MAML_META_EPOCHS,
    MAML_META_LR, MAML_PROX_SHRINK, MAML_BYPASS_SATURATION_M,
    MAML_SUPPORT_K, N_QH, TRAINED_MODELS_DIR, FEATURE_SET_MACRO, RANDOM_SEED,
)

warnings.filterwarnings("ignore")
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_maml_net(
    n_features: int,
    hidden_sizes: list[int] = MAML_HIDDEN_SIZES,
    dropout: float = MAML_DROPOUT,
    l1_reg: float = MAML_L1_REG,
    activation: str = MAML_ACTIVATION,
    seed: Optional[int] = RANDOM_SEED,
) -> keras.Model:
    """Constructs a compact residual MLP with LayerNormalization and MAE loss."""
    inp = keras.Input(shape=(n_features,), name="features")
    reg = regularizers.l1(l1_reg) if l1_reg > 0 else None
    k_init = keras.initializers.GlorotUniform(seed=seed)
    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, name=f"dense_{i}", kernel_initializer=k_init, kernel_regularizer=reg)(x)
        x = layers.Activation(activation, name=f"act_{i}")(x)
        x = layers.LayerNormalization(name=f"ln_{i}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = layers.Dense(1, activation="linear", name="residual_out", kernel_initializer="zeros", bias_initializer="zeros")(x)
    model = keras.Model(inputs=inp, outputs=out, name="MAML_EPFNet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=MAML_META_LR), loss=tf.keras.losses.MeanAbsoluteError())
    return model


class MAMLManager:
    """Manages MAML backbone pretraining, checkpointing, and fast inner adaptation."""

    def __init__(
        self,
        zone: str,
        n_features: int,
        feature_set: str = FEATURE_SET_MACRO,
        seed: int = RANDOM_SEED,
        bypass_mode: str = "nobypass",
    ):
        if bypass_mode not in ("nobypass", "optlasso"):
            raise ValueError(f"Unknown bypass_mode: '{bypass_mode}'. Expected 'nobypass' or 'optlasso'.")

        self.zone = zone
        self.n_features = n_features
        self.feature_set = feature_set
        self.seed = seed
        self.bypass_mode = bypass_mode

        self.cache_dir = os.path.join(TRAINED_MODELS_DIR, "maml_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            self.cache_dir, f"backbone_{zone}_{feature_set}_{bypass_mode}_seed{seed}.pkl"
        )

        self._working_model = build_maml_net(n_features, seed=seed)
        self._variables = self._working_model.trainable_variables
        self._backbone_weights: list[np.ndarray] | None = None

    def pretrain_backbone(self, loader, test_start: str = "2024-01-01", lookback_days: int = 822) -> None:
        """Trains the shared MAML backbone across all 96 QH positions on historical data."""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self._working_model = build_maml_net(self.n_features, seed=self.seed)
        self._variables = self._working_model.trainable_variables

        windows = [
            loader.get_window(self.zone, q, test_start, feature_set=self.feature_set, lookback_days=lookback_days)
            for q in range(N_QH)
        ]
        valid_w = [w for w in windows if w is not None and w.X_train_full is not None]
        X_all = np.vstack([w.X_train_full for w in valid_w]).astype(np.float32)

        # Pretraining target: raw spread or linear residual
        if self.bypass_mode == "optlasso":
            r_list = []
            for w in valid_w:
                m_l, _ = train_lasso(w.X_train_full, w.y_train)
                r_list.append(w.y_train - m_l.predict(w.X_train_full))
            y_all = np.concatenate(r_list).astype(np.float32)
        else:
            y_all = np.concatenate([w.y_train for w in valid_w]).astype(np.float32)

        self._working_model.fit(X_all, y_all, epochs=MAML_META_EPOCHS, batch_size=MAML_BACKBONE_BATCH, shuffle=True, verbose=0)
        self._backbone_weights = self._working_model.get_weights()
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(self._backbone_weights, f)

    def set_backbone_weights(self, weights: list[np.ndarray]) -> None:
        """Set shared backbone weights directly (used for warm-start and testing)."""
        self._backbone_weights = weights
        self._working_model.set_weights(weights)

    def adapt_and_predict(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_test: np.ndarray,
        linear_support: Optional[np.ndarray] = None,
        linear_test: Optional[float] = None,
    ) -> float:
        """Executes fast inner-loop adaptation starting from warm-start backbone."""
        if self._backbone_weights is None:
            raise RuntimeError(f"[MAML {self.zone}] Shared backbone not initialized.")

        self._working_model.set_weights(self._backbone_weights)

        # 1. Switchable linear bypass handling
        if self.bypass_mode == "optlasso":
            if linear_support is None or linear_test is None:
                raise ValueError("Bypass mode 'optlasso' requires linear_support and linear_test.")
            m = MAML_BYPASS_SATURATION_M
            p_lin_sup = m * np.tanh(linear_support / m) if m else linear_support
            p_lin_test = float(m * np.tanh(linear_test / m) if m else linear_test)
            y_target = y_support.ravel() - p_lin_sup.ravel()
        else:
            p_lin_test = 0.0
            y_target = y_support.ravel()

        # 2. Select K closest market regimes using L1 (cityblock) distance
        dists = cdist(X_support, X_test.reshape(1, -1), metric="cityblock").ravel()
        sel_idx = np.argsort(dists)[:MAML_SUPPORT_K]

        X_s = tf.constant(X_support[sel_idx], dtype=tf.float32)
        y_s = tf.constant(y_target[sel_idx, None], dtype=tf.float32)
        x_te = tf.constant(X_test.reshape(1, -1), dtype=tf.float32)

        # 3. Fast adaptation via compiled TF graph
        pred = float(self._run_inner_loop(X_s, y_s, x_te).numpy()[0, 0])
        return float(pred + p_lin_test)

    @tf.function(reduce_retracing=True)
    def _run_inner_loop(self, xs: tf.Tensor, rs: tf.Tensor, xte: tf.Tensor) -> tf.Tensor:
        """Executes 8 inner SGD steps with gradient clipping and proximal soft-thresholding."""
        for _ in range(MAML_INNER_STEPS):
            with tf.GradientTape() as tape:
                pred = self._working_model(xs, training=True)
                loss = tf.reduce_mean(tf.abs(rs - pred))

            grads = tape.gradient(loss, self._variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)

            for v, g in zip(self._variables, grads):
                if g is not None:
                    v.assign_sub(MAML_INNER_LR * g)
                    if MAML_PROX_SHRINK > 0.0 and "kernel" in v.name:
                        v.assign(tf.sign(v) * tf.maximum(0.0, tf.abs(v) - MAML_PROX_SHRINK))

        return self._working_model(xte, training=False)