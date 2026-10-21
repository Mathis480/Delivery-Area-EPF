"""
Draft figures for the results section (not yet part of the paper).

Usage: python Evaluation/draft_figures.py [results_dir]   (default: RUN_DIR from config)
Figures (PDF + PNG) are written to Evaluation/plots/drafts/.
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
os.makedirs(OUT_DIR, exist_ok=True)
plot_style.apply()

MODEL_KEYS = {
    "Naive": "naive",
    "LASSO": "pred_lasso",
    "LASSO+N": "pred_lasso_n",
    "cSVR (filter)": "pred_csvr_corr",
    "cSVR": "pred_csvr",
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
    """F1: one-sided DM p-values; colored cells = column model significantly more accurate than row model."""
    labels = list(MODEL_KEYS)
    bounds = [0, 0.001, 0.01, 0.05, 0.10]
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(plot_style.SEQUENTIAL_CMAP)(np.linspace(0.05, 0.85, 4)))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(plot_style.TEXT_WIDTH_IN, 5.6), sharex=True, sharey=True)
    for ax, z in zip(axes.ravel(), ZONES):
        zd = zd_all[z]
        loss = {lab: daily_loss(zd[lab], zd["y"]) for lab in labels}
        p = np.full((len(labels), len(labels)), np.nan)
        for i, r in enumerate(labels):
            for j, c in enumerate(labels):
                if i != j:
                    p[i, j] = dm_test(loss[r], loss[c])[1]
        im = ax.imshow(np.ma.masked_where(~(p < 0.10), p), cmap=cmap, norm=norm)
        ax.set_facecolor("white")
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(False); ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        for i in range(len(labels)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#000000"))
        ax.set_title(f"Control Area {z}")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03, ticks=bounds)
    cbar.set_label("DM p-value (column better than row)")
    _save(fig, "draft_F1_dm_chessboard")


def fig_rmae_by_hour(zd: dict, zone: str = "DE2") -> None:
    """F2: rMAE per delivery hour for the main models."""
    hours = np.arange(96) // 4
    naive_err = np.abs(zd["Naive"] - zd["y"])
    fig, ax = plt.subplots(figsize=(plot_style.TEXT_WIDTH_IN, 2.6))
    ax.axhline(1.0, color=COL["Naive"], linestyle=LS["Naive"], linewidth=1.0, label="Naive")
    for lab in ("LASSO", "cSVR", "FA-NN"):
        err = np.abs(zd[lab] - zd["y"])
        r = [np.nanmean(err[:, hours == h]) / np.nanmean(naive_err[:, hours == h]) for h in range(24)]
        ax.plot(range(24), r, color=COL[lab], label=lab, marker="o", markersize=3)
    ax.set_xlabel("Delivery hour (UTC)"); ax.set_ylabel("rMAE")
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)])
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4)
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
