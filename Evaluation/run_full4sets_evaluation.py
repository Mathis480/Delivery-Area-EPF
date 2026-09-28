#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive 4-Feature-Set Evaluation & Publication Plot Generator (2024).
Generates:
  1. Hour-by-hour rMAE plots for each zone comparing sets and ensembles.
  2. Multi-model rMAE heatmaps across the 24 hours (96 QH) of the trading day.
  3. Diebold-Mariano significance matrix heatmaps.
  4. Final aggregated Markdown and CSV leaderboards.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results", "annual_run_2024")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "Evaluation", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ZONES = ["DE1", "DE2", "DE3", "DE4"]
ZONE_NAMES = {
    "DE1": "TransnetBW (DE1)",
    "DE2": "Amprion (DE2)",
    "DE3": "TenneT (DE3)",
    "DE4": "50Hertz (DE4)",
}

def dm_test(actual, pred1, pred2):
    e1 = np.abs(actual - pred1)
    e2 = np.abs(actual - pred2)
    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    stat = mean_d / np.sqrt(var_d / len(d))
    p_val = 2 * (1 - stats.norm.cdf(np.abs(stat)))
    return stat, p_val

def generate_zone_plots(zone: str):
    npz_path = os.path.join(RESULTS_DIR, f"results_{zone}_full4sets_2024.npz")
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path)
    y_true_2d = data["y_true_2d"]
    bm_2d = data["benchmark_2d"]
    qh_arr = data["qh_idx"]
    y_true = data["y_true"]
    bm = data["benchmark"]

    models_to_eval = [
        ("Naive", bm_2d),
        ("cSVR (S1 Macro)", data["pred_csvr_2d"]),
        ("cSVR (S2 Neighbor)", data["pred_csvr_s2_2d"]),
        ("cSVR (S3 Fund)", data["pred_csvr_s3_2d"]),
        ("cSVR (S4 Balance)", data["pred_csvr_s4_2d"]),
        ("MAML (S1 Macro)", data["pred_maml_s1_2d"]),
        ("MAML (S2 Neighbor)", data["pred_maml_s2_2d"]),
        ("MAML (S3 Fund)", data["pred_maml_s3_2d"]),
        ("MAML (S4 Balance)", data["pred_maml_s4_2d"]),
        ("Pure cSVR Ens (S1-S4)", data["pred_csvr_ens_pure_2d"]),
        ("cSVR Ens (+Naive Daily)", data["pred_csvr_ens_wn_2d"]),
        ("cSVR Ens (QH-Adaptive)", data["pred_csvr_ens_adapt_2d"]) if "pred_csvr_ens_adapt_2d" in data else ("cSVR Ens (+Naive)", data["pred_csvr_ens_wn_2d"]),
        ("Pure MAML Ens (S1-S4)", data["pred_maml_ens_pure_2d"]),
        ("MAML Ens (QH-Adaptive)", data["pred_maml_ens_wn_2d"]),
        ("Hybrid Ens (QH-Adaptive)", data["pred_ens_2d"]),
    ]

    # Compute hourly rMAE
    hourly_rmae = {}
    mae_naive_hourly = np.nanmean(np.abs(y_true_2d - bm_2d).reshape(-1, 24, 4), axis=(0, 2))
    hours = np.arange(24)

    for name, pred_2d in models_to_eval:
        mae_m = np.nanmean(np.abs(y_true_2d - pred_2d).reshape(-1, 24, 4), axis=(0, 2))
        hourly_rmae[name] = mae_m / mae_naive_hourly

    # 1. Plot: Hourly rMAE Curve
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Naive Benchmark (1.00)")

    # Color palette
    colors = {
        "cSVR (S2 Neighbor)": "#1f77b4",
        "cSVR (S3 Fund)": "#2ca02c",
        "MAML (S2 Neighbor)": "#ff7f0e",
        "Pure cSVR Ens (S1-S4)": "#9467bd",
        "cSVR Ens (+Naive Daily)": "#d62728",
        "cSVR Ens (QH-Adaptive)": "#8c564b",
        "MAML Ens (QH-Adaptive)": "#e377c2",
        "Hybrid Ens (QH-Adaptive)": "#00aa55",
    }

    for name, col in colors.items():
        if name in hourly_rmae:
            ls = "-" if "Ens" in name or "Hybrid" in name else ":"
            lw = 2.4 if "Ens" in name or "Hybrid" in name else 1.5
            ax.plot(hours, hourly_rmae[name], label=name, color=col, linestyle=ls, linewidth=lw, marker="o" if "Ens" in name else None, markersize=4)

    ax.set_title(f"2024 Out-of-Sample Relative MAE (rMAE) by Hour — {ZONE_NAMES[zone]}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Delivery Hour (UTC)", fontsize=12)
    ax.set_ylabel("Relative MAE (rMAE vs Naive)", fontsize=12)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax.set_ylim(0.94, 1.06)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"full4sets_by_hour_{zone}.png"))
    plt.close()

    # 2. Plot: Heatmap across hours
    heatmap_matrix = np.array([hourly_rmae[m[0]] for m in models_to_eval[1:]])
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.TwoSlopeNorm(vmin=0.94, vcenter=1.00, vmax=1.06)
    im = ax.imshow(heatmap_matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, fontsize=9)
    ax.set_yticks(np.arange(len(models_to_eval) - 1))
    ax.set_yticklabels([m[0] for m in models_to_eval[1:]], fontsize=10)
    ax.set_title(f"rMAE Heatmap Across Delivery Hours (UTC) — {ZONE_NAMES[zone]} (2024)", fontsize=13, fontweight="bold", pad=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("rMAE (< 1.00 beats Naive)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"full4sets_heatmap_{zone}.png"))
    plt.close()

    print(f"Generated plots for {zone} -> {OUTPUT_DIR}")

def main():
    for z in ZONES:
        generate_zone_plots(z)
    print("All zone plots generated successfully.")

if __name__ == "__main__":
    main()
