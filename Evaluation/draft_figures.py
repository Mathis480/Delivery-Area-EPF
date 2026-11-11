"""
Result and draft figures for the paper:

Usage: python Evaluation/draft_figures.py [results_dir]   (default: RUN_DIR from config)
Model names as in the paper: cSVR = with correlation filter, FOMAML = without pre-training (post-hoc variants excluded).
Writes Figure_3.{pdf,svg} (DM chessboard) to Contribution/SVG/ and all figures (PDF + PNG) to Evaluation/plots/drafts/.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Evaluation"))
import plot_style
from config import MASTER_PARQUET, RUN_DIR, ZONES
from Models.ensemble import daily_loss, dm_test

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR
OUT_DIR = os.path.join(ROOT, "Evaluation", "plots", "drafts")
FIG_DIR = os.path.join(ROOT, "Contribution", "SVG")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
plot_style.apply()

MODEL_KEYS = {
    "Naive": "naive",
    "LASSO": "pred_lasso",
    "LASSO+Naive": "pred_lasso_n",
    "cSVR": "pred_csvr_corr",
    "FOMAML": "pred_fomaml",
    "FA-NN": "pred_fann",
}
COL, LS = plot_style.MODEL_COLORS, plot_style.MODEL_LINESTYLES


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)


def load_zone(zone: str) -> dict:
    d = np.load(os.path.join(RESULTS_DIR, f"results_{zone}.npz"))
    return {"y": d["y_true"], "dates": pd.to_datetime(d["dates"]), **{lab: d[k] for lab, k in MODEL_KEYS.items()}}


def fig_decoupling() -> None:
    """F0: absolute zonal-national price spread before (90-105 min) and within the target window (0-30 min)."""
    cols = ["Date", "VWAP_90to105", "VWAP_0to30"] + [f"{z}_VWAP_{w}" for z in ZONES for w in ("90to105", "0to30")]
    m = pd.read_parquet(MASTER_PARQUET, columns=cols)
    m = m[pd.to_datetime(m["Date"], utc=True) >= "2024-01-01"]
    cmap = plt.get_cmap(plot_style.SEQUENTIAL_CMAP)
    shades = {"90to105": cmap(0.75), "0to30": cmap(0.25)}
    fig, ax = plt.subplots(figsize=(plot_style.TEXT_WIDTH_IN, 2.6))
    data, pos, colors = [], [], []
    for i, z in enumerate(ZONES):
        for j, w in enumerate(("90to105", "0to30")):
            data.append(np.abs(m[f"{z}_VWAP_{w}"] - m[f"VWAP_{w}"]).dropna().values)
            pos.append(i * 3 + j); colors.append(shades[w])
    bp = ax.boxplot(data, positions=pos, widths=0.75, showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.0}, boxprops={"linewidth": 0.6})
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(c)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(ZONES))], [f"Control Area {z}" for z in ZONES])
    ax.set_ylabel("|Zonal − national VWAP| (EUR/MWh)")
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ["90–105 min before delivery", "0–30 min before delivery"],
              loc="upper right")
    _save(fig, "draft_F0_zonal_decoupling")


def fig_dm_chessboard(zd_all: dict) -> None:
    """F1 / Figure_3: one-sided DM p-values in epftoolbox chessboard style (Lago et al., 2021, Fig. 5)."""
    labels = list(MODEL_KEYS)
    # Exact epftoolbox color scale: [0, 0.05) dark green -> yellow, [0.05, 0.10) red -> dark red, >= 0.10 black
    red = np.concatenate([np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.5, 50)[1:]])
    green = np.concatenate([np.linspace(0.5, 1.0, 50), np.zeros(49)])
    cmap = matplotlib.colors.ListedColormap(np.column_stack([red, green, np.zeros(99)]))
    cmap.set_bad("black")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.10)

    fig = plt.figure(figsize=(5.45, 5.05))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.052], wspace=0.05, hspace=0.14)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])
    cax = fig.add_subplot(gs[:, 2])
    for idx, (ax, z) in enumerate(zip(axes.ravel(), ZONES)):
        r_idx, c_idx = divmod(idx, 2)
        zd = zd_all[z]
        loss = {lab: daily_loss(zd[lab], zd["y"]) for lab in labels}
        p = np.full((len(labels), len(labels)), np.nan)
        for i, r in enumerate(labels):
            for j, c in enumerate(labels):
                if i != j:
                    p[i, j] = dm_test(loss[r], loss[c])[1]
        im = ax.imshow(np.ma.masked_where(~(p < 0.10), p), cmap=cmap, norm=norm)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.tick_params(axis="both", which="major", length=2.5, pad=2)
        if r_idx == 1:
            ax.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="major", bottom=False)
        if c_idx == 0:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="major", left=False)
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="#333333", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        for i in range(len(labels)):
            ax.plot(i, i, marker="x", color="white", markersize=5.2, markeredgewidth=1.0)
        ax.set_title(f"Control area {z}", pad=3.5)
    cbar = fig.colorbar(im, cax=cax, ticks=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    cbar.ax.tick_params(length=2.5, pad=2)
    cbar.set_label("DM p-value", labelpad=4)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIG_DIR, f"Figure_3.{ext}"))
    _save(fig, "draft_F1_dm_chessboard")


def fig_rmae_by_hour(zd: dict, zone: str = "DE2") -> None:
    """F2: rMAE per delivery hour for the main models."""
    hours = np.r_[np.arange(96) // 4, [2, 2, 2, 2]]  # the slots behind the 96 products are second 02:xx deliveries
    naive_err = np.abs(zd["Naive"] - zd["y"])
    fig, ax = plt.subplots(figsize=(plot_style.TEXT_WIDTH_IN, 2.6))
    ax.axhline(1.0, color=COL["Naive"], linestyle=LS["Naive"], linewidth=1.0, label="Naive")
    for lab in ("LASSO", "cSVR", "FOMAML", "FA-NN"):
        err = np.abs(zd[lab] - zd["y"])
        r = [np.nanmean(err[:, hours == h]) / np.nanmean(naive_err[:, hours == h]) for h in range(24)]
        ax.plot(range(24), r, color=COL[lab], label=lab, marker="o", markersize=3)
    ax.set_xlabel("Delivery hour (local time)"); ax.set_ylabel("rMAE")
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)])
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=5)
    _save(fig, f"draft_F2_rmae_by_hour_{zone}")


def fig_cumulative(zd_all: dict) -> None:
    """F3: cumulative difference in daily MAE, scaled by the number of test days (end value = annual MAE difference)."""
    fig, axes = plt.subplots(2, 2, figsize=(plot_style.TEXT_WIDTH_IN, 4.2), sharex=True, sharey=True)
    for ax, z in zip(axes.ravel(), ZONES):
        zd = zd_all[z]
        l_fa = daily_loss(zd["FA-NN"], zd["y"])
        n_days = np.sum(~np.isnan(l_fa))
        for ref in ("Naive", "cSVR"):
            diff = np.nancumsum(daily_loss(zd[ref], zd["y"]) - l_fa) / n_days
            ax.plot(zd["dates"], diff, color=COL[ref], label=f"{ref} − FA-NN")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Control Area {z}")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.supylabel("Cumulative difference in daily MAE / 366 (EUR/MWh)", fontweight="bold", fontsize=9)
    axes[0, 0].legend(loc="upper left")
    _save(fig, "draft_F3_cumulative_mae_difference")


if __name__ == "__main__":
    data = {z: load_zone(z) for z in ZONES}
    fig_decoupling()
    fig_dm_chessboard(data)
    fig_rmae_by_hour(data["DE2"])
    fig_cumulative(data)
    print(f"Draft figures written to {OUT_DIR} (results: {RESULTS_DIR})")
