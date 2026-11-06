"""
Data figures of the paper (descriptive, from the master dataset):

    python Evaluation/data_figures.py

  Contribution/SVG/Figure_1.{pdf,svg}        intraday auction price (fig:Price_Development)
  Contribution/SVG/Figure_2.{pdf,svg}        mean trade count by local product and time until delivery (fig:Trade_occurence)

Period: local delivery days DATA_START to TEST_END (October 2021 to December 2024).
"""
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
from config import DATA_START, MASTER_PARQUET, N_QH, TEST_END, TIMEZONE, ZONES

FIG_DIR = os.path.join(ROOT, "Contribution", "SVG")
WINDOWS = [f"{a}to{a + 15}" for a in range(0, 360, 15)]  # 24 windows of 15 min, 0to15 ... 345to360
os.makedirs(FIG_DIR, exist_ok=True)
plot_style.apply()


def load_master(cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(MASTER_PARQUET, columns=["Date"] + cols)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    local = df["Date"].dt.tz_convert(TIMEZONE)
    keep = (local >= pd.Timestamp(DATA_START, tz=TIMEZONE)) & (local < pd.Timestamp(TEST_END, tz=TIMEZONE) + pd.Timedelta(days=1))
    df = df[keep].reset_index(drop=True)
    local = local[keep].reset_index(drop=True)
    df["_qh_idx"] = local.dt.hour * 4 + local.dt.minute // 15  # local product, as in data_loader.py
    df["_day"] = local.dt.date
    df["_year"] = local.dt.year
    return df


def _save(fig, name: str) -> None:
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)


def fig_price() -> None:
    df = load_master(["Intraday Auction Price"])
    fig, ax = plt.subplots(figsize=(plot_style.TEXT_WIDTH_IN, 2.5))
    ax.plot(df["Date"], df["Intraday Auction Price"], color=plot_style.MODEL_COLORS["FA-NN"], lw=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("Price (EUR/MWh)")
    _save(fig, "Figure_1")
    print(f"Figure_1: {len(df):,} QH, {df['Date'].iloc[0]} to {df['Date'].iloc[-1]} (UTC), "
          f"min {df['Intraday Auction Price'].min():.2f}, max {df['Intraday Auction Price'].max():.2f}")


def fig_trades() -> None:
    cols = [f"{z}_VWAP_total_trades_{w}" for z in ZONES for w in WINDOWS]
    df = load_master(cols)
    # rows: windows from delivery backwards (0to15 first), columns: local products 00:00 ... 23:45
    mean = {z: df.groupby("_qh_idx")[[f"{z}_VWAP_total_trades_{w}" for w in WINDOWS]].mean().to_numpy().T for z in ZONES}
    assert all(m.shape == (len(WINDOWS), N_QH) for m in mean.values())
    vmin, vmax = min(m.min() for m in mean.values()), max(m.max() for m in mean.values())
    x, y = np.arange(N_QH + 1) / 4, np.arange(len(WINDOWS) + 1) * 15

    fig, axes = plt.subplots(2, 2, figsize=(plot_style.TEXT_WIDTH_IN, 3.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, z in zip(axes.ravel(), ZONES):
        mesh = ax.pcolormesh(x, y, mean[z], cmap=plot_style.SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax, rasterized=True)
        ax.axhline(60, color="#ff7f0e", ls="--", lw=1.0, label="End of cross-border trading (60 min)")
        ax.axhline(30, color="#d62728", ls="--", lw=1.0, label="End of cross-zonal trading (30 min)")
        ax.set_title(f"Control area {z}")
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_yticks([30, 60, 120, 180, 240, 300, 360])
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Delivery hour (local time)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Minutes until delivery")
    fig.colorbar(mesh, ax=axes, fraction=0.04, label="Average number of trades")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    _save(fig, "Figure_2")
    print(f"Figure_2: {df['_day'].nunique()} local days, mean trades per window max {vmax:.2f} (min {vmin:.2f})")
    for z in ZONES:
        print(f"  {z}: mean trades per product 0-360 min {mean[z].sum(axis=0).mean():.1f}")


if __name__ == "__main__":
    fig_price()
    fig_trades()
