"""
Data figures of the paper (descriptive, from the master dataset):

    python Evaluation/data_figures.py

  Contribution/SVG/Figure_1.{pdf,svg}        intraday auction price with initial calibration window and test year (fig:Price_Development)
  Contribution/SVG/Figure_2.{pdf,svg}        mean trade count by local product and time until delivery (fig:Trade_occurence)
  Contribution/SVG/Figure_target_window.{pdf,svg}  (a) MAE between consecutive zonal 15-min windows, (b) zonal-national
                                             price deviation in the naive and target windows, test year (fig:TargetWindow)

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
from config import DATA_START, MASTER_PARQUET, N_QH, TEST_END, TEST_START, TIMEZONE, ZONES

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
    test_start = pd.Timestamp(TEST_START, tz=TIMEZONE).tz_convert("UTC")
    test_end = (pd.Timestamp(TEST_END, tz=TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")
    fig, ax = plt.subplots(figsize=(plot_style.TEXT_WIDTH_IN, 2.3))
    ax.plot(df["Date"], df["Intraday Auction Price"], color="#4d4d4d", lw=0.25)
    ax.axvspan(test_start, test_end, color="#cfe2f3", lw=0, zorder=0)
    ax.axvline(test_start, color="#d62728", ls="--", lw=1.0)
    ax.text(df["Date"].iloc[0] + (test_start - df["Date"].iloc[0]) / 2, 0.95, "Initial calibration window (822 days)",
            transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.text(test_start + (test_end - test_start) / 2, 0.95, "Test year", transform=ax.get_xaxis_transform(),
            ha="center", va="top")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
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


def fig_target_window() -> None:
    """(a) MAE between consecutive 15-min zonal VWAPs by time to delivery (both windows traded); (b) |zonal - national|
    VWAP in the naive (105-90 min) and target (30-0 min) windows. Test year (local delivery days). The national VWAP
    (all German trades) is used here for description only."""
    ends = list(range(0, 345, 15))  # window end in minutes before delivery, compared with the window ending 15 min earlier
    cols = ["VWAP_0to30", "VWAP_90to105"] + [f"{z}_VWAP_{c}{a}to{a + 15}" for z in ZONES for a in range(0, 360, 15)
                                             for c in ("", "total_volume_")] + [f"{z}_VWAP_{w}" for z in ZONES for w in ("0to30", "90to105")]
    df = load_master(list(dict.fromkeys(cols)))
    df = df[df["Date"].dt.tz_convert(TIMEZONE) >= pd.Timestamp(TEST_START, tz=TIMEZONE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(plot_style.TEXT_WIDTH_IN, 2.7), layout="constrained",
                               gridspec_kw={"width_ratios": [1.5, 1]})
    for x0, x1, color in ((0, 30, "#d9d9d9"), (30, 60, "#f0f0f0")):  # zonal, national (unshaded: cross-border)
        a.axvspan(x0, x1, color=color, lw=0, zorder=0)
    a.axvline(90, color="#000000", ls="--", lw=1.0)
    styles = dict(zip(ZONES, (("#08306b", "o"), ("#2171b5", "s"), ("#6baed6", "^"), ("#9ecae1", "D"))))  # one hue
    for z in ZONES:
        mae = []
        for e in ends:
            cur, prev = f"{z}_VWAP_{e}to{e + 15}", f"{z}_VWAP_{e + 15}to{e + 30}"
            traded = (df[f"{z}_VWAP_total_volume_{e}to{e + 15}"] > 0) & (df[f"{z}_VWAP_total_volume_{e + 15}to{e + 30}"] > 0)
            mae.append(float(np.abs(df[cur] - df[prev])[traded].mean()))
        a.plot(np.array(ends) + 7.5, mae, color=styles[z][0], marker=styles[z][1], ms=3, lw=1.0, label=z)
        print(f"  {z}: MAE vs. preceding window, window ending 180/90/30/15/0 min: "
              + "/".join(f"{mae[ends.index(e)]:.2f}" for e in (180, 90, 30, 15, 0)))
    a.invert_xaxis()
    a.set_xlim(352, -2)
    a.set_xticks([0, 30, 60, 90, 180, 270, 345])
    a.set_xlabel("Minutes before delivery")
    a.set_ylabel("MAE vs. preceding window (EUR/MWh)")
    a.set_title("(a) Price change per 15 minutes")
    a.legend(loc="center left", fontsize=8)
    data, pos = [], []
    for i, z in enumerate(ZONES):
        for j, w in enumerate(("VWAP_90to105", "VWAP_0to30")):
            data.append(np.abs(df[f"{z}_{w}"] - df[w]).to_numpy())
            pos.append(3 * i + j)
    bp = b.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True,
                   medianprops={"color": "#000000"})
    for k, box in enumerate(bp["boxes"]):
        box.set_facecolor("#d9d9d9" if k % 2 == 0 else "#4d4d4d")
    b.set_xticks([3 * i + 0.5 for i in range(len(ZONES))], ZONES)
    b.set_ylabel("|Zonal $-$ national price| (EUR/MWh)")
    b.set_title("(b) Zonal deviation")
    from matplotlib.patches import Patch
    b.legend([Patch(fc="#d9d9d9", ec="#000000"), Patch(fc="#4d4d4d", ec="#000000")],
             ["Naive window (105–90 min)", "Target window (30–0 min)"], loc="upper right", fontsize=8)
    _save(fig, "Figure_target_window")
    print("Figure_target_window: median |zonal - national| naive/target window: "
          + ", ".join(f"{z} {np.median(data[2 * i]):.2f}/{np.median(data[2 * i + 1]):.2f}" for i, z in enumerate(ZONES)))


if __name__ == "__main__":
    fig_price()
    fig_trades()
    fig_target_window()
