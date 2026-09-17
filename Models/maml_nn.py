"""
MAML-NN with Linear Bypass for per-QH EPF forecasting (TensorFlow/Keras).

Architecture
------------
The network is a 2-layer MLP with a *linear bypass* connection:

    y_hat = MLP(x) + W_bypass @ x + b_bypass

where (W_bypass, b_bypass) is initialized from linear model weights (LR or LASSO).
The nonlinear MLP path starts with zero weights so the network starts with the
linear foundation and learns nonlinear residuals on top.

Warm-Start Strategy (two phases):
  Phase 1 — Shared Backbone Pre-training:
    A single shared model is trained on data from all 96 QH positions
    simultaneously for MAML_META_EPOCHS epochs.
  Phase 2 — QH-Specific Adaptation (daily backtest):
    Each day T, for QH position q:
      - Start from the shared backbone weights (or cached weights from day T-1).
      - Run MAML_INNER_STEPS gradient steps on (X_train_q, y_train_q).
      - Predict on X_test_q.
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    MAML_ADAPT_EPOCHS,
    MAML_BACKBONE_BATCH,
    MAML_DROPOUT,
    MAML_HIDDEN_SIZES,
    MAML_INNER_LR,
    MAML_INNER_STEPS,
    MAML_META_EPOCHS,
    MAML_META_LR,
    TRAINED_MODELS_DIR,
)

warnings.filterwarnings("ignore")
tf.random.set_seed(42)


def build_epf_net(
    n_features: int,
    hidden_sizes: list[int] = MAML_HIDDEN_SIZES,
    dropout: float = MAML_DROPOUT,
) -> keras.Model:
    """
    Constructs a 2-layer MLP with linear bypass connection.
    Linear path is named 'linear_path', MLP final output is named 'nn_path'.
    """
    inp = keras.Input(shape=(n_features,), name="features")

    # Linear bypass path
    linear_path = layers.Dense(1, activation="linear", name="linear_path")(inp)

    # Deep nonlinear residual path
    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, name=f"dense_{i}")(x)
        x = layers.Activation("gelu", name=f"gelu_{i}")(x)
        x = layers.LayerNormalization(name=f"ln_{i}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    # Final projection initialized with zeros so network starts as pure linear
    nn_path = layers.Dense(
        1,
        activation="linear",
        name="nn_path",
        kernel_initializer="zeros",
        bias_initializer="zeros",
    )(x)

    out = layers.Add(name="price")([linear_path, nn_path])
    model = keras.Model(inputs=inp, outputs=out, name="EPFNet")
    return model


@dataclass
class QHMAMLState:
    """State for a single QH model (cached weights)."""
    weights: list[np.ndarray] | None = None
    best_val_loss: float = float("inf")


class MAMLManager:
    """
    Manages 96 QH-specific MAML models with linear bypass and warm-start.
    """

    def __init__(
        self,
        n_features: int,
        n_qh: int = 96,
        hidden_sizes: list[int] = MAML_HIDDEN_SIZES,
        dropout: float = MAML_DROPOUT,
        cache_path: str | None = None,
    ):
        self.n_features = n_features
        self.n_qh = n_qh
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.cache_path = cache_path or os.path.join(TRAINED_MODELS_DIR, "maml_cache")
        os.makedirs(self.cache_path, exist_ok=True)

        self._backbone_weights: list[np.ndarray] | None = None
        self._qh_state: list[QHMAMLState] = [QHMAMLState() for _ in range(n_qh)]

        # Pre-compile one working model for efficiency
        self._working_model = build_epf_net(n_features, hidden_sizes, dropout)
        self._loss_fn = tf.keras.losses.Huber(delta=1.0)

    def pretrain_backbone(
        self,
        qh_windows: list[tuple[np.ndarray, np.ndarray]],
        bypass_coef: np.ndarray,
        bypass_intercept: float,
        epochs: int = MAML_META_EPOCHS,
        batch_size: int = MAML_BACKBONE_BATCH,
        verbose: bool = True,
    ) -> None:
        """
        Train the shared backbone on data from all 96 QH positions.
        The linear bypass is initialized with linear model (LR or LASSO) weights.
        """
        checkpoint = os.path.join(self.cache_path, "backbone_weights.pkl")
        if os.path.exists(checkpoint):
            if verbose:
                print(f"[MAML] Loading backbone weights from {checkpoint}")
            with open(checkpoint, "rb") as f:
                self._backbone_weights = pickle.load(f)
            self._working_model.set_weights(self._backbone_weights)
            return

        if verbose:
            print(f"[MAML] Pre-training shared backbone on {len(qh_windows)} QH sets "
                  f"for {epochs} epochs ...")

        # Initialize linear bypass with provided linear weights
        linear_layer = self._working_model.get_layer("linear_path")
        coef = np.asarray(bypass_coef, dtype=np.float32).reshape(-1, 1)
        intercept = np.asarray([bypass_intercept], dtype=np.float32)
        linear_layer.set_weights([coef, intercept])

        # Concatenate all available QH data
        X_all = np.vstack([w[0] for w in qh_windows if len(w[0]) > 0]).astype(np.float32)
        y_all = np.concatenate([w[1] for w in qh_windows if len(w[1]) > 0]).astype(np.float32)

        opt = keras.optimizers.Adam(learning_rate=MAML_META_LR)
        self._working_model.compile(optimizer=opt, loss=self._loss_fn)

        self._working_model.fit(
            X_all,
            y_all,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1 if verbose else 0,
        )

        self._backbone_weights = self._working_model.get_weights()
        with open(checkpoint, "wb") as f:
            pickle.dump(self._backbone_weights, f)
        if verbose:
            print(f"[MAML] Backbone weights saved to {checkpoint}")

    def adapt_and_predict(
        self,
        qh_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        inner_steps: int = MAML_INNER_STEPS,
        inner_lr: float = MAML_INNER_LR,
    ) -> float:
        """
        Fast inner-loop adaptation on (X_train, y_train) starting from
        warm-start backbone or previous day's QH state.
        """
        if self._backbone_weights is None:
            raise RuntimeError("Call pretrain_backbone() before adapt_and_predict().")

        # Restore weights: start from cached QH state or backbone
        cached = self._qh_state[qh_idx].weights
        init_weights = cached if cached is not None else self._backbone_weights
        self._working_model.set_weights(init_weights)

        # Inner-loop adaptation
        X_tr = tf.cast(X_train, tf.float32)
        y_tr = tf.cast(y_train, tf.float32)[:, None]
        X_v = tf.cast(X_val, tf.float32)
        y_v = tf.cast(y_val, tf.float32)[:, None]

        opt = keras.optimizers.SGD(learning_rate=inner_lr, momentum=0.9)

        best_weights = None
        best_val = float("inf")

        for _ in range(inner_steps * MAML_ADAPT_EPOCHS):
            with tf.GradientTape() as tape:
                preds = self._working_model(X_tr, training=True)
                loss = self._loss_fn(y_tr, preds)
            grads = tape.gradient(loss, self._working_model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            opt.apply_gradients(zip(grads, self._working_model.trainable_variables))

            val_loss = float(self._loss_fn(y_v, self._working_model(X_v, training=False)))
            if val_loss < best_val:
                best_val = val_loss
                best_weights = self._working_model.get_weights()

        if best_weights is not None:
            self._working_model.set_weights(best_weights)
            self._qh_state[qh_idx].weights = best_weights

        # Predict
        X_te = tf.cast(X_test, tf.float32)
        pred = self._working_model(X_te, training=False).numpy().ravel()
        return float(pred[0]) if len(pred) > 0 else 0.0

    def save_qh_states(self, path: str | None = None):
        p = path or os.path.join(self.cache_path, "qh_states.pkl")
        with open(p, "wb") as f:
            pickle.dump(self._qh_state, f)

    def load_qh_states(self, path: str | None = None):
        p = path or os.path.join(self.cache_path, "qh_states.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                self._qh_state = pickle.load(f)
