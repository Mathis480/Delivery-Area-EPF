#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-Scale Hyperparameter & Architecture Optimizer for Summer EPF Dynamics (June 2024 - DE2 Amprion).

Systematically evaluates small, medium, and large deep meta-learning architectures with
Buschjäger et al. (2020) Generalized Negative Correlation Learning (GNCL) across all 4 feature sets:
  - Small architectures: [32], [32, 16], [48, 24], [64, 32]
  - Medium architectures: [64, 64], [96, 48], [48, 32, 16], [64, 48, 24]
  - Large architectures: [128, 64], [96, 64, 32], [128, 64, 32], [128, 96, 48]
  - Buschjäger GNCL Objective: Ambiguity difference penalty vs. convex ensemble loss
  - Full RAM Pre-Caching: Windows and LASSO bypasses cached in RAM for ultra-fast Bayesian trials (TPE)
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Force CPU execution & multithreaded BLAS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import (
    RESULTS_DIR,
    LOOKBACK_DAYS,
    VAL_DAYS,
    N_QH,
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
    MAML_BACKBONE_BATCH,
    MAML_META_EPOCHS,
    MAML_META_LR,
)
from data_loader import QHDataLoader
from Models.linear import train_lasso

# Output directories
HPO_DIR = os.path.join(RESULTS_DIR, "maml_large_scale_hpo_june2024")
os.makedirs(HPO_DIR, exist_ok=True)
CACHE_DIR = os.path.join(HPO_DIR, "backbone_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRIALS_CSV = os.path.join(HPO_DIR, "trials.csv")
BEST_JSON = os.path.join(HPO_DIR, "best_config.json")
SUMMARY_MD = os.path.join(HPO_DIR, "hpo_summary.md")

FEATURE_SETS = [
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
]

SET_TAGS = {
    FEATURE_SET_MACRO: "s1",
    FEATURE_SET_NEIGHBOR: "s2",
    FEATURE_SET_FUNDAMENTAL: "s3",
    FEATURE_SET_BALANCE: "s4",
}


def _backtransform(y_pred_scaled: float, y_mean: float, y_std: float, benchmark_eur: float) -> float:
    return float(y_pred_scaled * y_std + y_mean + benchmark_eur)


def dm_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray):
    """Diebold-Mariano test with asymptotic standard error."""
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d < 1e-12:
        return 0.0, 1.0
    stat = mean_d / np.sqrt(var_d / len(d))
    p_val = 2 * (1 - stats.norm.cdf(np.abs(stat)))
    return float(stat), float(p_val)


def build_maml_net(
    n_features: int,
    hidden_sizes: list[int],
    dropout: float = 0.10,
    l1_reg: float = 1e-4,
    activation: str = "gelu",
    seed: int = 42,
) -> keras.Model:
    """Builds MLP model for residual EPF learning."""
    inp = keras.Input(shape=(n_features,), name="features")
    reg = regularizers.l1(l1_reg) if l1_reg > 0 else None
    k_init = keras.initializers.GlorotUniform(seed=seed)

    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, name=f"dense_{i}", kernel_initializer=k_init, kernel_regularizer=reg)(x)
        if activation == "gelu":
            x = layers.Activation("gelu", name=f"gelu_{i}")(x)
        elif activation == "leaky_relu":
            x = layers.LeakyReLU(negative_slope=0.01, name=f"lrelu_{i}")(x)
        elif activation == "elu":
            x = layers.ELU(name=f"elu_{i}")(x)
        elif activation == "tanh":
            x = layers.Activation("tanh", name=f"tanh_{i}")(x)
        else:
            x = layers.Activation("relu", name=f"relu_{i}")(x)

        x = layers.LayerNormalization(name=f"ln_{i}")(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = layers.Dense(1, activation="linear", name="residual_out", kernel_initializer="zeros", bias_initializer="zeros")(x)
    model = keras.Model(inputs=inp, outputs=out, name="MAML_EPFNet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=MAML_META_LR), loss=tf.keras.losses.MeanAbsoluteError())
    return model


class SummerEvaluationContext:
    """
    Pre-caches all June 2024 QH windows, true prices, and LASSO models in RAM
    for ultra-fast parallel Bayesian evaluations.
    """
    def __init__(self, zone: str = "DE2", month: str = "2024-06"):
        self.zone = zone
        self.month = month
        self.loader = QHDataLoader()

        in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
        if not os.path.exists(in_npz):
            raise FileNotFoundError(f"Missing base annual run: {in_npz}")
        base_data = np.load(in_npz)

        dates_all = pd.to_datetime(base_data["dates"])
        start_date = f"{month}-01"
        end_date = f"{month}-30"
        mask = (dates_all >= start_date) & (dates_all <= end_date)

        self.dates_arr = base_data["dates"][mask]
        self.qh_arr = base_data["qh_idx"][mask]
        self.y_true = base_data["y_true"][mask]
        self.bm = base_data["benchmark"][mask]
        self.csvr_ens_wn = base_data["pred_csvr_ens_wn"][mask]

        self.n_eval = len(self.y_true)
        self.mae_naive = float(np.mean(np.abs(self.y_true - self.bm)))
        self.mae_csvr = float(np.mean(np.abs(self.y_true - self.csvr_ens_wn)))
        self.rmae_csvr = self.mae_csvr / self.mae_naive

        print(f"\n[SummerContext {zone}] Initialized for {month} (N={self.n_eval} quarter-hours):")
        print(f"  Naive Benchmark MAE:       {self.mae_naive:.4f} EUR/MWh (1.0000)")
        print(f"  cSVR Ensemble (+Naive) MAE: {self.mae_csvr:.4f} EUR/MWh ({self.rmae_csvr:.4f})")

        cache_context_file = os.path.join(CACHE_DIR, f"context_cache_{zone}_{month}.pkl")
        if os.path.exists(cache_context_file):
            print(f"[SummerContext {zone}] Loading pre-cached windows and models from {cache_context_file}...")
            with open(cache_context_file, "rb") as f:
                data = pickle.load(f)
            self.window_cache = data["window_cache"]
            self.lasso_cache = data["lasso_cache"]
            self.X_arrays = data["X_arrays"]
            self.r_arr = data["r_arr"]
            self.N_samples = data["N_samples"]
            print(f"[SummerContext {zone}] Loaded cache successfully.")
            return

        # Pre-cache windows and LASSO fits in RAM
        print(f"[SummerContext {zone}] Pre-caching all 4 feature set windows and LASSO models in RAM...")
        t0 = time.time()
        self.window_cache = {s: [] for s in FEATURE_SETS}
        self.lasso_cache = {s: [] for s in FEATURE_SETS}

        for i in range(self.n_eval):
            d_str = str(self.dates_arr[i])
            q = int(self.qh_arr[i])
            for s in FEATURE_SETS:
                w = self.loader.get_window(self.zone, q, d_str, feature_set=s, lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS, filter_zero_var=False)
                self.window_cache[s].append(w)
                if w is not None and w.X_trainval is not None:
                    try:
                        m_lasso, _ = train_lasso(w.X_trainval, w.y_trainval)
                        self.lasso_cache[s].append((m_lasso.coef_, float(m_lasso.intercept_)))
                    except Exception:
                        fallback = Ridge(alpha=100.0).fit(w.X_trainval, w.y_trainval)
                        self.lasso_cache[s].append((fallback.coef_, float(fallback.intercept_)))
                else:
                    self.lasso_cache[s].append((None, 0.0))

        print(f"[SummerContext {zone}] RAM pre-caching completed in {time.time() - t0:.1f}s.")

        # Pre-gather pooled training data for backbone GNCL training (test_start = 2024-06-01)
        print(f"[SummerContext {zone}] Pre-gathering pooled historical data for GNCL backbone pre-training...")
        self.X_pools = {s: [] for s in FEATURE_SETS}
        self.r_pools = []
        for q in range(N_QH):
            windows = {s: self.loader.get_window(self.zone, q, "2024-06-01", feature_set=s, lookback_days=LOOKBACK_DAYS, val_days=VAL_DAYS) for s in FEATURE_SETS}
            if any(w is None or w.X_train_full is None for w in windows.values()):
                continue
            res_list = [windows[s].y_train - Ridge(alpha=100.0).fit(windows[s].X_train_full, windows[s].y_train).predict(windows[s].X_train_full).astype(np.float32) for s in FEATURE_SETS]
            for s in FEATURE_SETS:
                self.X_pools[s].append(windows[s].X_train_full.astype(np.float32))
            self.r_pools.append(np.mean(res_list, axis=0).astype(np.float32))

        self.X_arrays = {s: np.vstack(self.X_pools[s]) for s in FEATURE_SETS}
        self.r_arr = np.concatenate(self.r_pools)
        self.N_samples = len(self.r_arr)
        print(f"[SummerContext {zone}] Pooled {self.N_samples} observations across 96 QH.")

        cache_data = {
            "window_cache": self.window_cache,
            "lasso_cache": self.lasso_cache,
            "X_arrays": self.X_arrays,
            "r_arr": self.r_arr,
            "N_samples": self.N_samples,
        }
        with open(cache_context_file, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"[SummerContext {zone}] Saved context cache to disk ({cache_context_file}).")


def pretrain_gncl_backbones(
    ctx: SummerEvaluationContext,
    hidden_sizes: list[int],
    dropout: float,
    l1_reg: float,
    activation: str,
    ncl_mode: str,
    lambda_ncl: float,
    ambiguity_type: str = "squared",
    meta_epochs: int = 6,
    batch_size: int = MAML_BACKBONE_BATCH,
    meta_lr: float = MAML_META_LR,
) -> dict[str, list[np.ndarray]]:
    """
    Pre-trains shared GNCL backbones with Buschjäger et al. (2020) objective function.
    Supports:
      - 'difference': MAE_i - lambda_ncl * div(p_i, P_ens) (Ambiguity diversity penalty)
      - 'convex': (1 - lambda_ncl) * MAE_mean + lambda_ncl * MAE(P_ens, y) (Joint ensemble loss)
    """
    key_str = f"{ctx.zone}_h{'-'.join(map(str, hidden_sizes))}_d{int(dropout*100)}_l1{l1_reg:.0e}_{activation}_{ncl_mode}_{ambiguity_type}_lam{int(lambda_ncl*100)}"
    cache_file = os.path.join(CACHE_DIR, f"gncl_{hashlib.md5(key_str.encode()).hexdigest()}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    models = {
        s: build_maml_net(ctx.X_arrays[s].shape[1], hidden_sizes=hidden_sizes, dropout=dropout, l1_reg=l1_reg, activation=activation)
        for s in FEATURE_SETS
    }
    optimizers = {s: tf.keras.optimizers.Adam(learning_rate=meta_lr) for s in FEATURE_SETS}

    @tf.function
    def train_step_difference(x1, x2, x3, x4, y_true):
        with tf.GradientTape(persistent=True) as tape:
            p1 = models[FEATURE_SET_MACRO](x1, training=True)
            p2 = models[FEATURE_SET_NEIGHBOR](x2, training=True)
            p3 = models[FEATURE_SET_FUNDAMENTAL](x3, training=True)
            p4 = models[FEATURE_SET_BALANCE](x4, training=True)

            P_ens = (p1 + p2 + p3 + p4) / 4.0

            mae1 = tf.reduce_mean(tf.abs(y_true - p1))
            mae2 = tf.reduce_mean(tf.abs(y_true - p2))
            mae3 = tf.reduce_mean(tf.abs(y_true - p3))
            mae4 = tf.reduce_mean(tf.abs(y_true - p4))

            # Buschjäger difference penalty: (p_i - P_ens)^2 or |p_i - P_ens|
            if ambiguity_type == "abs":
                amb1 = tf.reduce_mean(tf.abs(p1 - P_ens))
                amb2 = tf.reduce_mean(tf.abs(p2 - P_ens))
                amb3 = tf.reduce_mean(tf.abs(p3 - P_ens))
                amb4 = tf.reduce_mean(tf.abs(p4 - P_ens))
            else:
                amb1 = tf.reduce_mean(tf.square(p1 - P_ens))
                amb2 = tf.reduce_mean(tf.square(p2 - P_ens))
                amb3 = tf.reduce_mean(tf.square(p3 - P_ens))
                amb4 = tf.reduce_mean(tf.square(p4 - P_ens))

            loss1 = mae1 - lambda_ncl * amb1
            loss2 = mae2 - lambda_ncl * amb2
            loss3 = mae3 - lambda_ncl * amb3
            loss4 = mae4 - lambda_ncl * amb4

        for s, l, m in [
            (FEATURE_SET_MACRO, loss1, models[FEATURE_SET_MACRO]),
            (FEATURE_SET_NEIGHBOR, loss2, models[FEATURE_SET_NEIGHBOR]),
            (FEATURE_SET_FUNDAMENTAL, loss3, models[FEATURE_SET_FUNDAMENTAL]),
            (FEATURE_SET_BALANCE, loss4, models[FEATURE_SET_BALANCE]),
        ]:
            grads = tape.gradient(l, m.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizers[s].apply_gradients(zip(grads, m.trainable_variables))
        del tape
        return mae1, mae2, mae3, mae4

    @tf.function
    def train_step_convex(x1, x2, x3, x4, y_true):
        with tf.GradientTape(persistent=True) as tape:
            p1 = models[FEATURE_SET_MACRO](x1, training=True)
            p2 = models[FEATURE_SET_NEIGHBOR](x2, training=True)
            p3 = models[FEATURE_SET_FUNDAMENTAL](x3, training=True)
            p4 = models[FEATURE_SET_BALANCE](x4, training=True)

            P_ens = (p1 + p2 + p3 + p4) / 4.0
            loss_ens = tf.reduce_mean(tf.abs(y_true - P_ens))

            maes = [tf.reduce_mean(tf.abs(y_true - p)) for p in [p1, p2, p3, p4]]
            total_loss = lambda_ncl * loss_ens + (1.0 - lambda_ncl) * (tf.add_n(maes) / 4.0)

        for s in FEATURE_SETS:
            grads = tape.gradient(total_loss, models[s].trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizers[s].apply_gradients(zip(grads, models[s].trainable_variables))
        del tape
        return maes[0], maes[1], maes[2], maes[3]

    indices = np.arange(ctx.N_samples)
    for epoch in range(meta_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, ctx.N_samples, batch_size):
            b_idx = indices[start_idx : start_idx + batch_size]
            x1 = tf.constant(ctx.X_arrays[FEATURE_SET_MACRO][b_idx])
            x2 = tf.constant(ctx.X_arrays[FEATURE_SET_NEIGHBOR][b_idx])
            x3 = tf.constant(ctx.X_arrays[FEATURE_SET_FUNDAMENTAL][b_idx])
            x4 = tf.constant(ctx.X_arrays[FEATURE_SET_BALANCE][b_idx])
            yt = tf.constant(ctx.r_arr[b_idx, None])

            if ncl_mode == "difference":
                train_step_difference(x1, x2, x3, x4, yt)
            else:
                train_step_convex(x1, x2, x3, x4, yt)

    weights = {s: models[s].get_weights() for s in FEATURE_SETS}
    tf.keras.backend.clear_session()
    with open(cache_file, "wb") as f:
        pickle.dump(weights, f)
    return weights


def evaluate_trial(params: dict, ctx: SummerEvaluationContext, trial_idx: int) -> dict:
    t_start = time.time()

    hidden_sizes = list(params["hidden_sizes"])
    dropout = float(params["dropout"])
    l1_reg = float(params["l1_reg"])
    activation = str(params["activation"])
    ncl_mode = str(params["ncl_mode"])
    lambda_ncl = float(params["lambda_ncl"])
    ambiguity_type = str(params.get("ambiguity_type", "squared"))
    inner_steps = int(params["inner_steps"])
    inner_lr = float(params["inner_lr"])
    prox_shrink = float(params["prox_shrink"])
    support_k = int(params["support_k"])
    selection_mode = str(params["selection_mode"])
    use_linear_bypass = bool(params["use_linear_bypass"])
    bypass_saturation_m = float(params["bypass_saturation_m"]) if params["bypass_saturation_m"] is not None else None
    ensemble_power = float(params["ensemble_power"])
    calib_window = int(params.get("calib_window", 14))
    qh_adaptive = bool(params.get("qh_adaptive", False))

    # 1. Pretrain or load cached GNCL backbones
    backbone_weights = pretrain_gncl_backbones(
        ctx=ctx,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        l1_reg=l1_reg,
        activation=activation,
        ncl_mode=ncl_mode,
        lambda_ncl=lambda_ncl,
        ambiguity_type=ambiguity_type,
    )

    # 2. Evaluate all 4 sets with compiled graph execution
    def eval_set(s: str) -> np.ndarray:
        model = build_maml_net(ctx.X_arrays[s].shape[1], hidden_sizes=hidden_sizes, dropout=dropout, l1_reg=l1_reg, activation=activation)
        b_w = backbone_weights[s]
        variables = model.trainable_variables
        lr_t = tf.constant(inner_lr, tf.float32)

        @tf.function(reduce_retracing=True)
        def _adapt_and_predict_graph(xs, rs, xte):
            for _ in range(inner_steps):
                with tf.GradientTape() as tape:
                    pred = model(xs, training=True)
                    loss = tf.reduce_mean(tf.abs(rs - pred))
                grads = tape.gradient(loss, variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                for v, g in zip(variables, grads):
                    if g is not None:
                        v.assign_sub(lr_t * g)
                        if prox_shrink > 0.0:
                            v.assign(tf.sign(v) * tf.maximum(0.0, tf.abs(v) - prox_shrink))
            return model(xte, training=False)

        preds = []
        for i in range(ctx.n_eval):
            w = ctx.window_cache[s][i]
            bench_eur = float(ctx.bm[i])
            if w is not None and w.X_val_full is not None:
                lin_c, lin_i = ctx.lasso_cache[s][i]
                try:
                    model.set_weights(b_w)

                    if use_linear_bypass and lin_c is not None:
                        coef_vec = np.asarray(lin_c, dtype=np.float32).ravel()
                        p_lin_support = (w.X_val_full @ coef_vec + lin_i).astype(np.float32)
                        r_support = (w.y_val.ravel() - p_lin_support).astype(np.float32)[:, None]
                    else:
                        r_support = w.y_val.ravel().astype(np.float32)[:, None]

                    if selection_mode == "regime_l1" and len(w.X_val_full) > support_k:
                        dists = cdist(w.X_val_full, w.X_test_full.reshape(1, -1), metric="cityblock").ravel()
                        sel_idx = np.argsort(dists)[:support_k]
                        X_s, r_s = tf.cast(w.X_val_full[sel_idx], tf.float32), tf.cast(r_support[sel_idx], tf.float32)
                    else:
                        k = min(support_k, len(w.X_val_full))
                        X_s, r_s = tf.cast(w.X_val_full[-k:], tf.float32), tf.cast(r_support[-k:], tf.float32)

                    x_te = tf.cast(w.X_test_full.reshape(1, -1), tf.float32)
                    r_pred = float(_adapt_and_predict_graph(X_s, r_s, x_te).numpy().ravel()[0])

                    if not use_linear_bypass or lin_c is None:
                        p_scaled = r_pred
                    else:
                        p_lin_test = float((w.X_test_full.ravel() @ coef_vec) + lin_i)
                        if bypass_saturation_m is not None and bypass_saturation_m > 0:
                            p_lin_safe = float(bypass_saturation_m * np.tanh(p_lin_test / bypass_saturation_m))
                        else:
                            p_lin_safe = p_lin_test
                        p_scaled = float(p_lin_safe + r_pred)

                    preds.append(_backtransform(p_scaled, w.y_mean, w.y_std, bench_eur))
                except Exception:
                    preds.append(bench_eur)
            else:
                preds.append(bench_eur)

        tf.keras.backend.clear_session()
        return np.array(preds, dtype=np.float32)

    # Evaluate all 4 feature sets
    set_preds = {s: eval_set(s) for s in FEATURE_SETS}

    # 3. Individual MAEs
    maes = {s: float(np.mean(np.abs(ctx.y_true - set_preds[s]))) for s in FEATURE_SETS}

    # 4. Multi-Set Ensembling (Rolling Inverse-MAE with power)
    n_days = 30
    calib = calib_window
    y_2d = ctx.y_true.reshape(n_days, N_QH)
    bm_2d = ctx.bm.reshape(n_days, N_QH)
    preds_2d = {s: set_preds[s].reshape(n_days, N_QH) for s in FEATURE_SETS}
    all_candidates = {**preds_2d, "naive": bm_2d}

    pred_ens_2d = np.zeros((n_days, N_QH), dtype=np.float64)
    model_keys = list(all_candidates.keys())

    for d in range(n_days):
        if d < calib:
            w = np.ones(len(model_keys)) / len(model_keys)
            for q in range(N_QH):
                for i_m, k in enumerate(model_keys):
                    pred_ens_2d[d, q] += w[i_m] * all_candidates[k][d, q]
        else:
            if not qh_adaptive:
                cal_true = y_2d[d - calib : d]
                cal_maes = []
                for k in model_keys:
                    c_pred = all_candidates[k][d - calib : d]
                    cal_maes.append(max(float(np.mean(np.abs(cal_true - c_pred))), 1e-4))
                inv_maes = 1.0 / (np.array(cal_maes) ** ensemble_power)
                w = inv_maes / np.sum(inv_maes)

                for q in range(N_QH):
                    for i_m, k in enumerate(model_keys):
                        pred_ens_2d[d, q] += w[i_m] * all_candidates[k][d, q]
            else:
                for q in range(N_QH):
                    cal_true_q = y_2d[d - calib : d, q]
                    cal_maes_q = []
                    for k in model_keys:
                        c_pred_q = all_candidates[k][d - calib : d, q]
                        cal_maes_q.append(max(float(np.mean(np.abs(cal_true_q - c_pred_q))), 1e-4))
                    inv_maes_q = 1.0 / (np.array(cal_maes_q) ** ensemble_power)
                    w_q = inv_maes_q / np.sum(inv_maes_q)

                    for i_m, k in enumerate(model_keys):
                        pred_ens_2d[d, q] += w_q[i_m] * all_candidates[k][d, q]

    mae_ens = float(np.mean(np.abs(ctx.y_true - pred_ens_2d.ravel())))
    rmae_ens = mae_ens / ctx.mae_naive

    # Pure MAML 4 sets (without naive)
    p_pure_mean = np.mean([set_preds[s] for s in FEATURE_SETS], axis=0)
    mae_pure = float(np.mean(np.abs(ctx.y_true - p_pure_mean)))

    # DM test vs cSVR and vs Naive
    dm_csvr, p_csvr = dm_test(ctx.y_true, pred_ens_2d.ravel(), ctx.csvr_ens_wn)
    dm_naive, p_naive = dm_test(ctx.y_true, pred_ens_2d.ravel(), ctx.bm)

    elapsed = time.time() - t_start
    beats_csvr = mae_ens < ctx.mae_csvr

    result_row = {
        "trial": trial_idx,
        "beats_csvr": beats_csvr,
        "mae_ensemble": mae_ens,
        "rmae_ensemble": rmae_ens,
        "diff_vs_csvr": mae_ens - ctx.mae_csvr,
        "dm_vs_csvr": dm_csvr,
        "p_vs_csvr": p_csvr,
        "dm_vs_naive": dm_naive,
        "mae_pure_mean": mae_pure,
        "mae_s1": maes[FEATURE_SET_MACRO],
        "mae_s2": maes[FEATURE_SET_NEIGHBOR],
        "mae_s3": maes[FEATURE_SET_FUNDAMENTAL],
        "mae_s4": maes[FEATURE_SET_BALANCE],
        "hidden_sizes": str(hidden_sizes),
        "activation": activation,
        "dropout": dropout,
        "l1_reg": l1_reg,
        "ncl_mode": ncl_mode,
        "ambiguity_type": ambiguity_type,
        "lambda_ncl": lambda_ncl,
        "inner_lr": inner_lr,
        "inner_steps": inner_steps,
        "prox_shrink": prox_shrink,
        "support_k": support_k,
        "selection_mode": selection_mode,
        "use_linear_bypass": use_linear_bypass,
        "bypass_saturation_m": bypass_saturation_m,
        "ensemble_power": ensemble_power,
        "calib_window": calib_window,
        "qh_adaptive": qh_adaptive,
        "elapsed_sec": round(elapsed, 1),
    }

    # Append to CSV
    df_row = pd.DataFrame([result_row])
    header = not os.path.exists(TRIALS_CSV)
    df_row.to_csv(TRIALS_CSV, mode="a", header=header, index=False)

    status_str = f"🏆 BEATS cSVR! ({mae_ens:.4f} < {ctx.mae_csvr:.4f}, DM={dm_csvr:.2f})" if beats_csvr else f"MAE={mae_ens:.4f} (cSVR={ctx.mae_csvr:.4f})"
    print(f"[Trial {trial_idx:3d}] {status_str} | rMAE={rmae_ens:.4f} | Arch={hidden_sizes} {activation} | GNCL({ncl_mode}, {ambiguity_type}, lam={lambda_ncl}) | K={support_k}, LR={inner_lr:.4f}, M={bypass_saturation_m} ({elapsed:.1f}s)")

    return {
        "loss": mae_ens,
        "status": STATUS_OK,
        "eval_time": elapsed,
        "metrics": result_row,
    }


def main():
    parser = argparse.ArgumentParser(description="Large-Scale Summer HPO (June 2024 DE2)")
    parser.add_argument("--zone", type=str, default="DE2", help="Delivery area (default: DE2)")
    parser.add_argument("--month", type=str, default="2024-06", help="Test month (default: 2024-06)")
    parser.add_argument("--max_trials", type=int, default=2500, help="Maximum number of Bayesian trials")
    parser.add_argument("--timeout_hours", type=float, default=13.8, help="Hard timeout budget in hours (default: 13.8)")
    args = parser.parse_args()

    max_seconds = int(args.timeout_hours * 3600)
    start_global = time.time()

    ctx = SummerEvaluationContext(zone=args.zone, month=args.month)

    # Search Space covering Small, Medium, and Large architectures
    space = {
        "hidden_sizes": hp.choice("hidden_sizes", [
            # Small Architectures
            (32,),
            (48,),
            (32, 16),
            (48, 24),
            (64, 32),
            # Medium Architectures
            (64, 64),
            (96, 48),
            (80, 40),
            (48, 32, 16),
            (64, 48, 24),
            # Large Architectures
            (128, 64),
            (128, 128),
            (160, 80),
            (96, 64, 32),
            (128, 64, 32),
            (128, 96, 48),
            (192, 96, 48),
        ]),
        "activation": hp.choice("activation", ["gelu", "relu", "leaky_relu", "elu", "tanh"]),
        "dropout": hp.choice("dropout", [0.0, 0.05, 0.10, 0.15, 0.20]),
        "l1_reg": hp.choice("l1_reg", [0.0, 1e-5, 5e-5, 1e-4, 5e-4]),
        "ncl_mode": hp.choice("ncl_mode", ["difference", "convex"]),
        "ambiguity_type": hp.choice("ambiguity_type", ["squared", "abs"]),
        "lambda_ncl": hp.choice("lambda_ncl", [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]),
        "inner_lr": hp.loguniform("inner_lr", np.log(0.0005), np.log(0.010)),
        "inner_steps": hp.choice("inner_steps", [1, 2, 3, 5, 8, 10]),
        "prox_shrink": hp.choice("prox_shrink", [0.0, 1e-5, 5e-5, 1e-4]),
        "support_k": hp.choice("support_k", [5, 7, 10, 14, 21, 28]),
        "selection_mode": hp.choice("selection_mode", ["regime_l1", "chronological"]),
        "use_linear_bypass": hp.choice("use_linear_bypass", [True, False]),
        "bypass_saturation_m": hp.choice("bypass_saturation_m", [0.06, 0.08, 0.10, 0.15, 0.25, 0.40, None]),
        "ensemble_power": hp.choice("ensemble_power", [1.0, 2.0, 3.0, 4.0]),
        "calib_window": hp.choice("calib_window", [7, 14, 21, 28]),
        "qh_adaptive": hp.choice("qh_adaptive", [False, True]),
    }

    # Resume from existing trials if available
    trial_counter = 0
    best_loss = float("inf")
    best_params = None

    if os.path.exists(TRIALS_CSV):
        try:
            df_existing = pd.read_csv(TRIALS_CSV)
            trial_counter = len(df_existing)
            if trial_counter > 0 and "mae_ensemble" in df_existing.columns:
                best_loss = float(df_existing["mae_ensemble"].min())
                best_idx = df_existing["mae_ensemble"].idxmin()
                best_params = df_existing.iloc[best_idx].to_dict()
                print(f"[Summer HPO] Resuming from {trial_counter} existing trials. Current best MAE: {best_loss:.4f} EUR/MWh")
        except Exception:
            pass

    trials = Trials()

    def objective(p):
        nonlocal trial_counter, best_loss, best_params

        elapsed_total = time.time() - start_global
        if elapsed_total > max_seconds:
            print(f"\n[Summer HPO] Timeout budget of {args.timeout_hours}h ({max_seconds}s) reached. Gracefully stopping.")
            sys.exit(0)

        trial_counter += 1
        res = evaluate_trial(p, ctx, trial_counter)

        if res["loss"] < best_loss:
            best_loss = res["loss"]
            best_params = res["metrics"]
            with open(BEST_JSON, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"  >>> ⭐ NEW GLOBAL BEST CHAMPION: MAE = {best_loss:.4f} EUR/MWh (vs cSVR {ctx.mae_csvr:.4f}) <<<")

        return res

    print(f"\n[Summer HPO] Starting up to {args.max_trials} Bayesian Optimization trials with TPE (Budget: {args.timeout_hours}h)...")
    try:
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.max_trials,
            trials=trials,
            rstate=np.random.default_rng(42),
        )
    except SystemExit:
        pass
    except Exception as e:
        print(f"[Summer HPO Error] {e}")

    total_time = time.time() - start_global
    print(f"\n================================================================================")
    print(f"       SUMMER HPO FINISHED IN {total_time/3600:.2f} h ({total_time:.1f}s)       ")
    print(f"================================================================================")
    print(f"Total Trials Evaluated: {trial_counter}")
    if os.path.exists(BEST_JSON):
        with open(BEST_JSON, "r") as f:
            best = json.load(f)
        print(f"\nBest Configuration Found:")
        for k, v in best.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
