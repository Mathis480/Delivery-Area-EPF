#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
4-Feature-Set Evaluation
  1. Hour-by-hour rMAE plots for each zone comparing sets and ensembles.
  2. Multi-model rMAE heatmaps across the 24 hours (96 QH) of the trading day.
  3. Diebold-Mariano significance matrix heatmaps.

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
import matplotlib.ticker as ticker
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

    # Strictly MAML-NN vs cSVR (Rule 5.2: No cross-architecture hybrids)
    models_to_eval = [
        ("Naive Benchmark", bm_2d),
        ("cSVR (S1 Macro)", data["pred_csvr_s1_2d"] if "pred_csvr_s1_2d" in data else data["pred_csvr_2d"]),
        ("MAML (S1 Macro)", data["pred_maml_s1_2d"]),
        ("cSVR (S2 Neighbor)", data["pred_csvr_s2_2d"]),
        ("MAML (S2 Neighbor)", data["pred_maml_s2_2d"]),
        ("cSVR (S3 Fundamentals)", data["pred_csvr_s3_2d"]),
        ("MAML (S3 Fundamentals)", data["pred_maml_s3_2d"]),
        ("cSVR (S4 Balances)", data["pred_csvr_s4_2d"]),
        ("MAML (S4 Balances)", data["pred_maml_s4_2d"]),
        ("Pure cSVR Ens (S1-S4)", data["pred_csvr_ens_pure_2d"]),
        ("Pure MAML Ens (S1-S4)", data["pred_maml_ens_pure_2d"]),
        ("cSVR Ens (+Naive)", data["pred_csvr_ens_wn_2d"]),
        ("MAML Ens (+Naive)", data["pred_maml_ens_wn_2d"]),
    ]

    # Compute 15-minute Quarter-Hour rMAE (96 intervals per day)
    qh_rmae = {}
    mae_naive_qh = np.nanmean(np.abs(y_true_2d - bm_2d), axis=0) # shape (96,)
    qhs = np.arange(96)

    for name, pred_2d in models_to_eval:
        mae_m = np.nanmean(np.abs(y_true_2d - pred_2d), axis=0)
        qh_rmae[name] = mae_m / mae_naive_qh

    # 1. Plot: 96-QH Intraday Curve
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Naive Benchmark (1.00)")

    # Color palette
    colors = {
        "cSVR (S2 Neighbor)": ("#1f77b4", ":", 1.8, None),
        "MAML (S2 Neighbor)": ("#ff7f0e", ":", 2.0, None),
        "Pure cSVR Ens (S1-S4)": ("#2ca02c", "-", 2.5, "o"),
        "Pure MAML Ens (S1-S4)": ("#d62728", "-", 2.5, "s"),
        "cSVR Ens (+Naive)": ("#9467bd", "--", 2.0, "^"),
        "MAML Ens (+Naive)": ("#e377c2", "--", 2.0, "v"),
    }

    for name, (col, ls, lw, marker) in colors.items():
        if name in qh_rmae:
            ax.plot(qhs, qh_rmae[name], label=name, color=col, linestyle=ls, linewidth=lw, marker=marker, markersize=3, markevery=4)

    ax.set_title(f"2024 Out-of-Sample Relative MAE (rMAE) by 15-Min Quarter Hour — {ZONE_NAMES[zone]}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Quarter Hour of Delivery Day (UTC, 15-min Intervals)", fontsize=11, labelpad=8)
    ax.set_ylabel("Relative MAE (rMAE vs Naive)", fontsize=11)
    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=9)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(0.93, 1.05)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"full4sets_by_hour_{zone}.png"))
    plt.close()

    # 2. Plot: 96-QH Heatmap
    heatmap_matrix = np.array([qh_rmae[m[0]] for m in models_to_eval[1:]])
    
    # rMAE <= 0.950: > 5.0% Outperformance : Blue
    # rMAE > 0.950:  Dark green to soft yellow
    # rMAE > 1.000: Soft yellow to dark red
    N = 512
    vals = np.linspace(0.92, 1.06, N)
    colors_list = []
    for v in vals:
        if v <= 0.950:
            colors_list.append((0.10, 0.45, 0.91, 1.0))
        elif v <= 1.000:
            t = (v - 0.950) / (1.000 - 0.950)
            r = 0.11 + t * (0.98 - 0.11)
            g = 0.55 + t * (0.98 - 0.55)
            b = 0.18 + t * (0.82 - 0.18)
            colors_list.append((r, g, b, 1.0))
        else:
            t = min(1.0, (v - 1.000) / (1.060 - 1.000))
            r = 0.98 - t * (0.98 - 0.78)
            g = 0.98 - t * (0.98 - 0.12)
            b = 0.82 - t * (0.82 - 0.12)
            colors_list.append((r, g, b, 1.0))

    cmap = mcolors.ListedColormap(colors_list)
    norm = mcolors.Normalize(vmin=0.92, vmax=1.06)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.grid(False)
    im = ax.imshow(heatmap_matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(0, 96, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=9)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_yticks(range(len(models_to_eval) - 1))
    ax.set_yticklabels([m[0] for m in models_to_eval[1:]], fontsize=10, fontweight="bold")
    ax.set_xlabel("Quarter Hour of Delivery Day (UTC, 15-min Intervals)", fontsize=11, labelpad=8)
    ax.set_title(f"rMAE Heatmap Across 15-Minute Quarter Hours — {ZONE_NAMES[zone]} (2024 Full Year)\n(Blue: >5.0% Outperformance | Green: 0% to 5.0% Outperformance | Red: Underperformance)", fontsize=12, fontweight="bold", pad=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0.92, 0.95, 0.98, 1.00, 1.02, 1.04, 1.06])
    cbar.set_ticklabels(["< 0.92", "0.950 (+5.0%)", "0.98", "1.000 (Naive)", "1.02", "1.04", "> 1.06"])
    cbar.set_label("rMAE (< 1.00 beats Naive | Blue: > 5.0% Outperformance)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"full4sets_heatmap_{zone}.png"))
    plt.close()

    print(f"Generated 15-min QH plots for {zone} -> {OUTPUT_DIR}")

def main():
    for z in ZONES:
        generate_zone_plots(z)
    print("All zone 15-min plots generated successfully.")

if __name__ == "__main__":
    main()
