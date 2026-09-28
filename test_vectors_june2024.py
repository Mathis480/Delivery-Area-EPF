#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimental Vector Exploration on June 2024 (DE2 Amprion):
Systematically tests:
  1. Support Pool Window: 56 days vs 365 days vs Full 822 days (like cSVR).
  2. Support Selection: Feature cityblock vs Feature + Benchmark distance (cSVR-style).
  3. Bypass Architecture: LASSO (current) vs Ridge vs Consistent Saturation vs No Bypass.
  4. Ensemble Power: p=1.0 vs p=4.0 vs p=8.0 (+Naive).
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge, Lasso

from config import (
    RESULTS_DIR,
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
    MAML_BACKBONE_BATCH,
    MAML_META_LR,
)
from Models.ensemble import compute_rolling_intelligent_ensemble
from data_loader import QHDataLoader

FEATURE_SETS = [
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
]

def build_maml_net(n_features: int, hidden_sizes=[128, 64], activation="tanh", seed=42):
    inp = keras.Input(shape=(n_features,), name="features")
    reg = regularizers.l1(1e-4)
    k_init = keras.initializers.GlorotUniform(seed=seed)
    x = inp
    for i, h in enumerate(hidden_sizes):
        x = layers.Dense(h, name=f"dense_{i}", kernel_initializer=k_init, kernel_regularizer=reg)(x)
        x = layers.Activation(activation, name=f"act_{i}")(x)
        x = layers.LayerNormalization(name=f"ln_{i}")(x)
    out = layers.Dense(1, activation="linear", name="residual_out", kernel_initializer="zeros", bias_initializer="zeros")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=MAML_META_LR), loss=tf.keras.losses.MeanAbsoluteError())
    return model

def pretrain_backbone(loader, zone, feature_set, hidden_sizes=[128, 64], activation="tanh"):
    windows = [loader.get_window(zone, q, "2024-01-01", feature_set=feature_set, lookback_days=822, val_days=30) for q in range(96)]
    valid_w = [w for w in windows if w is not None and w.X_train_full is not None]
    X_all = np.vstack([w.X_train_full for w in valid_w]).astype(np.float32)
    r_all = np.concatenate([w.y_train - Ridge(alpha=100.0).fit(w.X_train_full, w.y_train).predict(w.X_train_full) for w in valid_w]).astype(np.float32)
    model = build_maml_net(X_all.shape[1], hidden_sizes=hidden_sizes, activation=activation)
    model.fit(X_all, r_all, epochs=6, batch_size=MAML_BACKBONE_BATCH, shuffle=True, verbose=0)
    return model.get_weights()

def run_experiment(
    desc: str,
    loader: QHDataLoader,
    backbones: dict,
    pool_mode: str = "val_56",       # "val_56", "full_822"
    use_bm_in_support: bool = False, # add |bm_i - bm_test| to distance
    bm_weight: float = 1.0,
    bypass_type: str = "lasso_m015", # "lasso_m015", "lasso_consistent", "ridge_m015", "none"
    inner_steps: int = 8,
    inner_lr: float = 0.0025,
    support_k: int = 28,
    tau: float = 4.0,
    bypass_m: float = 0.15,
):
    print(f"\n--- Running: {desc} ---")
    t0 = time.time()
    zone = "DE2"
    month = "2024-06"
    in_npz = os.path.join(RESULTS_DIR, "annual_run_2024", f"results_{zone}_full4sets_2024.npz")
    base_data = np.load(in_npz)
    dates_all = pd.to_datetime(base_data["dates"])
    mask = (dates_all >= f"{month}-01") & (dates_all <= f"{month}-30")
    dates_eval = base_data["dates"][mask]
    qh_eval = base_data["qh_idx"][mask]
    y_true = base_data["y_true"][mask]
    bm = base_data["benchmark"][mask]
    n_eval = len(y_true)

    preds_eur = {}
    preds_2d = {}
    n_days, n_qh = 30, 96

    for s in FEATURE_SETS:
        model = build_maml_net(loader.feature_names(zone, s).__len__(), hidden_sizes=[128, 64], activation="tanh")
        b_weights = backbones[s]
        model.set_weights(b_weights)
        variables = model.trainable_variables
        lr_t = tf.constant(inner_lr, tf.float32)

        @tf.function(reduce_retracing=True)
        def _adapt_step(xs, rs, xte, w_k):
            for _ in range(inner_steps):
                with tf.GradientTape() as tape:
                    p = model(xs, training=True)
                    loss = tf.reduce_sum(w_k * tf.abs(rs - p))
                grads = tape.gradient(loss, variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                for v, g in zip(variables, grads):
                    if g is not None:
                        v.assign_sub(lr_t * g)
            return model(xte, training=False)

        pred_list = []
        for i in range(n_eval):
            d_str = str(dates_eval[i])
            q = int(qh_eval[i])
            bench_eur = float(bm[i])

            # Get window with pool_days
            pool_days = 822 if pool_mode == "full_822" else 56
            w = loader.get_window(zone, q, d_str, feature_set=s, lookback_days=822, val_days=pool_days, filter_zero_var=False)
            if w is None or w.X_val_full is None:
                pred_list.append(bench_eur)
                continue

            model.set_weights(b_weights)

            # Determine support pool
            if pool_mode == "full_822":
                X_pool = w.X_trainval
                y_pool = w.y_trainval
                bm_pool = w.bm_trainval
            else:
                X_pool = w.X_val_full
                y_pool = w.y_val
                # val bm
                n_v = len(w.y_val)
                bm_pool = w.bm_trainval[-n_v:] if w.bm_trainval is not None else None

            # Linear bypass fit
            if bypass_type.startswith("lasso"):
                # Fast Lasso
                m_lin = Lasso(alpha=0.01, max_iter=200).fit(w.X_trainval, w.y_trainval)
                coef_vec = m_lin.coef_.astype(np.float32)
                intercept = float(m_lin.intercept_)
            elif bypass_type.startswith("ridge"):
                m_lin = Ridge(alpha=100.0).fit(w.X_trainval, w.y_trainval)
                coef_vec = m_lin.coef_.astype(np.float32)
                intercept = float(m_lin.intercept_)
            else:
                coef_vec = None
                intercept = 0.0

            # Residual computation
            if coef_vec is not None:
                p_lin_sup = (X_pool @ coef_vec + intercept).astype(np.float32)
                if bypass_type == "lasso_consistent":
                    p_lin_sup_safe = (bypass_m * np.tanh(p_lin_sup / bypass_m)).astype(np.float32)
                    r_sup = (y_pool.ravel() - p_lin_sup_safe).astype(np.float32)[:, None]
                else:
                    r_sup = (y_pool.ravel() - p_lin_sup).astype(np.float32)[:, None]
            else:
                r_sup = y_pool.ravel().astype(np.float32)[:, None]

            # Distances
            dists = cdist(X_pool, w.X_test_full.reshape(1, -1), metric="cityblock").ravel()
            if use_bm_in_support and bm_pool is not None:
                # Add scaled benchmark distance
                bm_std = float(np.std(bm_pool)) if np.std(bm_pool) > 1e-4 else 1.0
                d_bm = np.abs(bm_pool - bench_eur) / bm_std
                dists = dists + bm_weight * d_bm

            sel_idx = np.argsort(dists)[:support_k]
            d_sel = dists[sel_idx]
            scale = max(float(np.std(d_sel)), 1e-4) * tau
            w_k = np.exp(-(d_sel - d_sel[0]) / scale)
            w_k = (w_k / np.sum(w_k)).astype(np.float32)[:, None]

            xs = tf.cast(X_pool[sel_idx], tf.float32)
            rs = tf.cast(r_sup[sel_idx], tf.float32)
            xte = tf.cast(w.X_test_full.reshape(1, -1), tf.float32)
            wt = tf.cast(w_k, tf.float32)

            r_p = float(_adapt_step(xs, rs, xte, wt).numpy().ravel()[0])
            r_p = float(np.clip(r_p, -0.5, 0.5))

            if coef_vec is not None:
                p_lin = float((w.X_test_full.ravel() @ coef_vec) + intercept)
                p_lin_safe = float(bypass_m * np.tanh(p_lin / bypass_m)) if (bypass_m is not None and bypass_m > 0) else p_lin
            else:
                p_lin_safe = 0.0

            p_scaled = p_lin_safe + r_p
            p_final_eur = p_scaled * w.y_std + w.y_mean + bench_eur
            pred_list.append(p_final_eur)

        preds_eur[s] = np.array(pred_list, dtype=np.float64)
        preds_2d[s] = preds_eur[s].reshape(n_days, n_qh)
        mae_s = float(np.mean(np.abs(y_true - preds_eur[s])))
        print(f"  {s}: MAE = {mae_s:.4f} EUR/MWh")
        tf.keras.backend.clear_session()

    # Ensembles
    y_true_2d = y_true.reshape(n_days, n_qh)
    bm_2d = bm.reshape(n_days, n_qh)
    mae_naive = float(np.mean(np.abs(y_true - bm)))

    # Pure Ensemble p=1
    p_ens_p1, _, m_p1 = compute_rolling_intelligent_ensemble(
        {f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, y_true_2d, bm_2d, calib_window=14, power=1.0
    )
    # Pure Ensemble p=4
    p_ens_p4, _, m_p4 = compute_rolling_intelligent_ensemble(
        {f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, y_true_2d, bm_2d, calib_window=14, power=4.0
    )
    # +Naive Ensemble p=4
    p_ens_wn, _, m_wn = compute_rolling_intelligent_ensemble(
        {**{f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, "naive": bm_2d}, y_true_2d, bm_2d, calib_window=14, power=4.0
    )

    elapsed = time.time() - t0
    print(f"  => Pure Ens (p=1): {m_p1['mae_ensemble']:.4f} (rMAE: {m_p1['rmae_ensemble']:.4f})")
    print(f"  => Pure Ens (p=4): {m_p4['mae_ensemble']:.4f} (rMAE: {m_p4['rmae_ensemble']:.4f})")
    print(f"  => +Naive Ens(p=4): {m_wn['mae_ensemble']:.4f} (rMAE: {m_wn['rmae_ensemble']:.4f}) [Elapsed: {elapsed:.1f}s]")

    return {
        "desc": desc,
        "mae_p1": m_p1["mae_ensemble"],
        "rmae_p1": m_p1["rmae_ensemble"],
        "mae_p4": m_p4["mae_ensemble"],
        "rmae_p4": m_p4["rmae_ensemble"],
        "mae_wn": m_wn["mae_ensemble"],
        "rmae_wn": m_wn["rmae_ensemble"],
        "elapsed": elapsed,
    }

if __name__ == "__main__":
    loader = QHDataLoader()
    zone = "DE2"

    print("Pretraining backbones [128, 64]...")
    backbones = {}
    for s in FEATURE_SETS:
        backbones[s] = pretrain_backbone(loader, zone, s, hidden_sizes=[128, 64], activation="tanh")

    results = []

    # Config 1: Current Baseline (pool=56, lasso_m015, no bm)
    r1 = run_experiment("1. Baseline (Pool=56d, Lasso m=0.15)", loader, backbones, pool_mode="val_56", bypass_type="lasso_m015")
    results.append(r1)

    # Config 2: Full 822-day Lookback (Pool=822d like cSVR!)
    r2 = run_experiment("2. Full History Pool (822d like cSVR)", loader, backbones, pool_mode="full_822", bypass_type="lasso_m015")
    results.append(r2)

    # Config 3: Benchmark Distance in Support Selection (Pool=822d + BM distance)
    r3 = run_experiment("3. Full Pool + BM-Conditioned Support Selection", loader, backbones, pool_mode="full_822", use_bm_in_support=True, bm_weight=1.0, bypass_type="lasso_m015")
    results.append(r3)

    # Config 4: Consistent Saturation (Pool=822d, support & test saturated)
    r4 = run_experiment("4. Consistent Saturation (Support & Test)", loader, backbones, pool_mode="full_822", bypass_type="lasso_consistent")
    results.append(r4)

    # Config 5: Ridge Bypass (alpha=100.0 instead of Lasso)
    r5 = run_experiment("5. Ridge Bypass (alpha=100, smooth)", loader, backbones, pool_mode="full_822", bypass_type="ridge_m015")
    results.append(r5)

    # Config 6: No Bypass (Direct MAML, pure neural network)
    r6 = run_experiment("6. No Bypass (Pure MAML Neural Network)", loader, backbones, pool_mode="full_822", bypass_type="none")
    results.append(r6)

    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL SUMMARY: VECTOR EXPLORATION JUNE 2024 DE2")
    print("="*80)
    print(df[["desc", "mae_p1", "rmae_p1", "mae_p4", "mae_wn"]].to_string(index=False))
    df.to_csv("Results/vector_exploration_june2024_de2.csv", index=False)
