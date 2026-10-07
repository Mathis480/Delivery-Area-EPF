"""
MAML-NN (Model-Agnostic Meta-Learning Neural Network) for per-QH EPF Forecasting.
Implements Model-Agnostic Meta-Learning (Finn et al., 2017) with warm-start backbone
pre-training, optional linear bypass residual learning, regime-based nearest-neighbor
support set sampling, and proximal soft-thresholding.
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
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge, Lasso, LassoLarsCV

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
    MAML_SUPPORT_SELECTION,
    MAML_SUPPORT_K,
    MAML_SUPPORT_WEIGHTING,
    MAML_KERNEL_TAU,
    N_QH,
    TRAINED_MODELS_DIR,
    FEATURE_SET_MACRO,
)

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)


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
    """Manages the shared MAML residual backbone and inner-loop adaptation."""

    def __init__(
        self,
        zone: str,
        n_features: int,
        feature_set: str = FEATURE_SET_MACRO,
        seed: int = 42,
        bypass_mode: Optional[str] = None,
    ):
        self.zone = zone
        self.n_features = n_features
        self.feature_set = feature_set
        self.seed = seed
        self.n_qh = N_QH
        self.use_linear_bypass = MAML_USE_LINEAR_BYPASS
        if bypass_mode is not None:
            self.bypass_mode = bypass_mode
        else:
            self.bypass_mode = "optlasso" if self.use_linear_bypass else "nobypass"
        self.hidden_sizes = MAML_HIDDEN_SIZES
        self.dropout = MAML_DROPOUT
        self.l1_reg = MAML_L1_REG
        self.activation = MAML_ACTIVATION

        self.cache_path = os.path.join(TRAINED_MODELS_DIR, "maml_cache")
        os.makedirs(self.cache_path, exist_ok=True)

        self._backbone_weights: list[np.ndarray] | None = None
        self._working_model = build_maml_net(
            n_features, self.hidden_sizes, self.dropout, self.l1_reg, self.activation, seed=seed
        )
        self._variables = self._working_model.trainable_variables

    def _backbone_path(self, quarter_tag: Optional[str] = None) -> str:
        """Constructs path for saved shared backbone weights."""
        q_suffix = f"_{quarter_tag}" if quarter_tag else ""
        bp_suffix = f"_{self.bypass_mode}" if self.bypass_mode else ""
        h_str = "-".join(map(str, self.hidden_sizes))
        filename = (
            f"backbone_{self.zone}_{self.feature_set}{bp_suffix}"
            f"_h{h_str}_d{int(self.dropout * 100)}_l1{self.l1_reg:.0e}{q_suffix}_seed{self.seed}.pkl"
        )
        return os.path.join(self.cache_path, filename)

    def _compute_backbone_targets(self, valid_windows: list) -> np.ndarray:
        """Computes backbone pretraining targets (raw spreads or residuals) based on bypass_mode."""
        if self.bypass_mode == "nobypass":
            return np.concatenate([w.y_train for w in valid_windows]).astype(np.float32)

        if self.bypass_mode == "optlasso":
            r_list = []
            for w in valid_windows:
                n_tr = len(w.X_train_full)
                cv_splits = [(list(range(n_tr - 14)), list(range(n_tr - 14, n_tr)))]
                m_cv = LassoLarsCV(cv=cv_splits, max_iter=200, fit_intercept=True).fit(w.X_train_full, w.y_train)
                mean_mse = np.mean(m_cv.mse_path_, axis=1)
                se_mse = np.std(m_cv.mse_path_, axis=1, ddof=1) if m_cv.mse_path_.shape[1] > 1 else 0.0
                target = np.min(mean_mse) + se_mse
                valid = np.where(mean_mse <= target)[0]
                alpha_1se = m_cv.cv_alphas_[valid[-1]] if len(valid) > 0 else m_cv.alpha_
                m_final = Lasso(alpha=alpha_1se, max_iter=200, fit_intercept=True).fit(w.X_train_full, w.y_train)
                r_list.append(w.y_train - m_final.predict(w.X_train_full))
            return np.concatenate(r_list).astype(np.float32)

        # Fallback for legacy Ridge bypass
        return np.concatenate([
            w.y_train - Ridge(alpha=100.0).fit(w.X_train_full, w.y_train).predict(w.X_train_full)
            for w in valid_windows
        ]).astype(np.float32)

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
        backbone_path = self._backbone_path(quarter_tag)
        if os.path.exists(backbone_path) and not force_retrain:
            with open(backbone_path, "rb") as f:
                self._backbone_weights = pickle.load(f)
            self._working_model.set_weights(self._backbone_weights)
            if verbose:
                print(f"[{self.zone} {self.feature_set} seed={self.seed}] Loaded cached backbone: {os.path.basename(backbone_path)}")
            return

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self._working_model = build_maml_net(
            self.n_features, self.hidden_sizes, self.dropout, self.l1_reg, self.activation, seed=self.seed
        )
        self._variables = self._working_model.trainable_variables

        windows = [loader.get_window(self.zone, q, test_start, feature_set=self.feature_set, lookback_days=lookback_days, val_days=val_days) for q in range(self.n_qh)]
        valid_w = [w for w in windows if w is not None and w.X_train_full is not None]
        X_all = np.vstack([w.X_train_full for w in valid_w]).astype(np.float32)
        r_all = self._compute_backbone_targets(valid_w)

        self._working_model.fit(X_all, r_all, epochs=meta_epochs, batch_size=MAML_BACKBONE_BATCH, shuffle=True, verbose=0)
        self._backbone_weights = self._working_model.get_weights()
        with open(backbone_path, "wb") as f:
            pickle.dump(self._backbone_weights, f)
        if verbose:
            print(f"[{self.zone} {self.feature_set} seed={self.seed}] Pretrained and saved backbone: {os.path.basename(backbone_path)}")

    def set_backbone_weights(self, weights: list[np.ndarray]) -> None:
        """Dynamically update shared backbone weights (e.g. for quarterly rolling updates)."""
        self._backbone_weights = weights
        self._working_model.set_weights(weights)

    def _select_support_set(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_test: np.ndarray,
        support_k: int = MAML_SUPPORT_K,
        selection_mode: str = MAML_SUPPORT_SELECTION,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Selects K regime-nearest historical days and computes soft-kernel weights."""
        y_vec = y_support.ravel().astype(np.float32)[:, None]
        if selection_mode == "regime_l1" and len(X_support) > support_k:
            dists = cdist(X_support, X_test.reshape(1, -1), metric="cityblock").ravel()
            sel_idx = np.argsort(dists)[:support_k]
            X_s = tf.cast(X_support[sel_idx], tf.float32)
            y_s = tf.cast(y_vec[sel_idx], tf.float32)
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
            X_s = tf.cast(X_support[-k:], tf.float32)
            y_s = tf.cast(y_vec[-k:], tf.float32)
            weights_s = tf.ones((k, 1), dtype=tf.float32) / float(k)
        return X_s, y_s, weights_s

    def adapt_and_predict(
        self,
        X_support: np.ndarray,
        y_support: np.ndarray,
        X_test: np.ndarray,
    ) -> float:
        """Fast compiled inner-loop adaptation starting from warm-start backbone.

        Resets model to shared backbone, selects regime-nearest support days,
        and executes compiled gradient adaptation.
        """
        if self._backbone_weights is None:
            raise RuntimeError(f"[MAML {self.zone}] Shared backbone not initialized.")

        # 1. Reset model to shared backbone
        self._working_model.set_weights(self._backbone_weights)

        # 2. Select K regime-nearest days and soft-kernel weights
        X_s, y_s, weights_s = self._select_support_set(X_support, y_support, X_test)

        # 3. Fast compiled inner adaptation and prediction in single TF graph
        lr_t = tf.constant(MAML_INNER_LR, tf.float32)
        prox_t = tf.constant(MAML_PROX_SHRINK, tf.float32)
        x_te = tf.cast(X_test.reshape(1, -1), tf.float32)
        pred = float(self._run_inner_loop_graph(X_s, y_s, x_te, lr_t, prox_t, weights_s).numpy().ravel()[0])

        # 4. Optional residual clipping if configured
        if MAML_CLIP_RESIDUAL is not None and MAML_CLIP_RESIDUAL > 0:
            pred = float(np.clip(pred, -MAML_CLIP_RESIDUAL, MAML_CLIP_RESIDUAL))

        return pred

    @tf.function(reduce_retracing=True)
    def _run_inner_loop_graph(
        self,
        xs: tf.Tensor,
        rs: tf.Tensor,
        xte: tf.Tensor,
        lr: tf.Tensor,
        prox_shrink: tf.Tensor,
        weights: tf.Tensor,
    ) -> tf.Tensor:
        """Compiled TensorFlow graph executing inner-loop SGD adaptation and out-of-sample prediction."""
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
