#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2x2 Factorial Cross-Test for June 2024 on DE2 (Amprion):
  - Factor 1: GNCL (Generalized Negative Correlation Learning: Off vs On)
  - Factor 2: Linear Bypass Saturation m (0.15 vs 0.70)
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# Suppress TF logs & use CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.spatial.distance import cdist

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
from optimize_maml_large_scale_june2024 import SummerEvaluationContext, build_maml_net, pretrain_gncl_backbones

FEATURE_SETS = [
    FEATURE_SET_MACRO,
    FEATURE_SET_NEIGHBOR,
    FEATURE_SET_FUNDAMENTAL,
    FEATURE_SET_BALANCE,
]


def run_single_eval(
    ctx: SummerEvaluationContext,
    use_gncl: bool,
    lambda_ncl: float,
    bypass_m: float,
    hidden_sizes: list[int] = [128, 64],
    activation: str = "tanh",
    inner_steps: int = 8,
    inner_lr: float = 0.0025,
    support_k: int = 28,
    tau: float = 4.0,
) -> dict:
    t0 = time.time()
    
    # 1. Backbones
    if use_gncl:
        backbones = pretrain_gncl_backbones(
            ctx=ctx,
            hidden_sizes=hidden_sizes,
            dropout=0.0,
            l1_reg=1e-4,
            activation=activation,
            ncl_mode="difference",
            lambda_ncl=lambda_ncl,
            ambiguity_type="squared",
            meta_epochs=6,
        )
    else:
        # Train or load independent backbones (lambda=0)
        backbones = pretrain_gncl_backbones(
            ctx=ctx,
            hidden_sizes=hidden_sizes,
            dropout=0.0,
            l1_reg=1e-4,
            activation=activation,
            ncl_mode="difference",
            lambda_ncl=0.0,
            ambiguity_type="squared",
            meta_epochs=6,
        )

    # 2. Prediction loop per feature set
    preds_eur = {}
    preds_2d = {}

    # Target shape: (30 days, 96 QH)
    n_days = 30
    n_qh = 96

    for s in FEATURE_SETS:
        model = build_maml_net(ctx.X_arrays[s].shape[1], hidden_sizes=hidden_sizes, dropout=0.0, l1_reg=1e-4, activation=activation)
        model.set_weights(backbones[s])
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
        b_weights = backbones[s]

        for i in range(ctx.n_eval):
            w = ctx.window_cache[s][i]
            coef_vec, intercept = ctx.lasso_cache[s][i]
            bench_eur = float(ctx.bm[i])

            if w is None or w.X_val_full is None:
                pred_list.append(bench_eur)
                continue

            # Reset model to backbone
            model.set_weights(b_weights)

            # Compute residual on support set
            if coef_vec is not None:
                p_lin_sup = (w.X_val_full @ coef_vec + intercept).astype(np.float32)
                r_sup = (w.y_val.ravel() - p_lin_sup).astype(np.float32)[:, None]
            else:
                r_sup = w.y_val.ravel().astype(np.float32)[:, None]

            # Soft-kernel selection
            dists = cdist(w.X_val_full, w.X_test_full.reshape(1, -1), metric="cityblock").ravel()
            sel_idx = np.argsort(dists)[:support_k]
            d_sel = dists[sel_idx]
            scale = max(float(np.std(d_sel)), 1e-4) * tau
            w_k = np.exp(-(d_sel - d_sel[0]) / scale)
            w_k = (w_k / np.sum(w_k)).astype(np.float32)[:, None]

            xs = tf.cast(w.X_val_full[sel_idx], tf.float32)
            rs = tf.cast(r_sup[sel_idx], tf.float32)
            xte = tf.cast(w.X_test_full.reshape(1, -1), tf.float32)
            wt = tf.cast(w_k, tf.float32)

            r_p = float(_adapt_step(xs, rs, xte, wt).numpy().ravel()[0])
            r_p = float(np.clip(r_p, -0.5, 0.5))

            # Linear bypass with saturation
            if coef_vec is not None:
                p_lin = float((w.X_test_full.ravel() @ coef_vec) + intercept)
                if bypass_m is not None and bypass_m > 0:
                    p_lin_safe = float(bypass_m * np.tanh(p_lin / bypass_m))
                else:
                    p_lin_safe = p_lin
            else:
                p_lin_safe = 0.0

            p_scaled = p_lin_safe + r_p
            p_final_eur = p_scaled * w.y_std + w.y_mean + bench_eur
            pred_list.append(p_final_eur)

        preds_eur[s] = np.array(pred_list, dtype=np.float64)
        preds_2d[s] = preds_eur[s].reshape(n_days, n_qh)
        tf.keras.backend.clear_session()

    # 3. Compute Ensembles
    y_true_2d = ctx.y_true.reshape(n_days, n_qh)
    bm_2d = ctx.bm.reshape(n_days, n_qh)

    p_ens_pure_2d, _, m_pure = compute_rolling_intelligent_ensemble(
        {f"m_{s}": preds_2d[s] for s in FEATURE_SETS},
        y_true_2d,
        bm_2d,
        calib_window=14,
        power=1.0,
    )
    p_ens_wn_2d, _, m_wn = compute_rolling_intelligent_ensemble(
        {**{f"m_{s}": preds_2d[s] for s in FEATURE_SETS}, "naive": bm_2d},
        y_true_2d,
        bm_2d,
        calib_window=14,
        power=1.0,
    )

    # 4. Error correlation between models
    err_mat = np.column_stack([np.abs(ctx.y_true - preds_eur[s]) for s in FEATURE_SETS])
    corr_mat = np.corrcoef(err_mat.T)
    # Average off-diagonal correlation
    mask_off = ~np.eye(4, dtype=bool)
    avg_corr = float(np.mean(corr_mat[mask_off]))

    maes_single = {s: float(np.mean(np.abs(ctx.y_true - preds_eur[s]))) for s in FEATURE_SETS}

    elapsed = time.time() - t0
    return {
        "use_gncl": use_gncl,
        "lambda_ncl": lambda_ncl,
        "bypass_m": bypass_m,
        "mae_s1": maes_single[FEATURE_SET_MACRO],
        "mae_s2": maes_single[FEATURE_SET_NEIGHBOR],
        "mae_s3": maes_single[FEATURE_SET_FUNDAMENTAL],
        "mae_s4": maes_single[FEATURE_SET_BALANCE],
        "best_single_mae": min(maes_single.values()),
        "mae_ens_pure": m_pure["mae_ensemble"],
        "rmae_ens_pure": m_pure["rmae_ensemble"],
        "mae_ens_wn": m_wn["mae_ensemble"],
        "rmae_ens_wn": m_wn["rmae_ensemble"],
        "avg_error_corr": avg_corr,
        "elapsed_s": elapsed,
    }


def main():
    print("=================================================================")
    print("  STARTING 2x2 FACTORIAL CROSS-TEST ON JUNE 2024 (DE2 AMPRION)")
    print("=================================================================")
    ctx = SummerEvaluationContext(zone="DE2", month="2024-06")

    configs = [
        {"name": "1. No GNCL + Bypass m=0.15 (Base)", "use_gncl": False, "lam": 0.0, "m": 0.15},
        {"name": "2. No GNCL + Bypass m=0.70",        "use_gncl": False, "lam": 0.0, "m": 0.70},
        {"name": "3. With GNCL + Bypass m=0.15",      "use_gncl": True,  "lam": 0.10, "m": 0.15},
        {"name": "4. With GNCL + Bypass m=0.70",      "use_gncl": True,  "lam": 0.10, "m": 0.70},
    ]

    results = []
    for cfg in configs:
        print(f"\n--- Running Configuration: {cfg['name']} ---")
        res = run_single_eval(
            ctx=ctx,
            use_gncl=cfg["use_gncl"],
            lambda_ncl=cfg["lam"],
            bypass_m=cfg["m"],
        )
        res["Config"] = cfg["name"]
        results.append(res)
        print(f"  MAE Ens (Pure): {res['mae_ens_pure']:.4f} EUR/MWh (rMAE={res['rmae_ens_pure']:.4f})")
        print(f"  MAE Ens (+WN):  {res['mae_ens_wn']:.4f} EUR/MWh (rMAE={res['rmae_ens_wn']:.4f})")
        print(f"  Avg Error Corr: {res['avg_error_corr']:.4f}")
        print(f"  Finished in {res['elapsed_s']:.1f}s")

    df = pd.DataFrame(results)
    
    print("\n=================================================================")
    print("              FINAL 2x2 CROSS-TEST COMPARISON TABLE              ")
    print("=================================================================")
    print(f"Naive Benchmark:     {ctx.mae_naive:.4f} EUR/MWh")
    print(f"cSVR (+Naive) Ens:   {ctx.mae_csvr:.4f} EUR/MWh ({ctx.rmae_csvr:.4f})")
    print("-----------------------------------------------------------------")
    cols = ["Config", "mae_s1", "mae_s2", "mae_s3", "mae_s4", "mae_ens_pure", "rmae_ens_pure", "mae_ens_wn", "rmae_ens_wn", "avg_error_corr"]
    print(df[cols].to_string(index=False))

    out_csv = os.path.join(RESULTS_DIR, "crosstest_june2024_de2.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
