"""
Data figures and table of the paper (descriptive, from the master dataset):

    python Evaluation/data_figures.py

  Contribution/SVG/Figure_1.{pdf,svg}        intraday auction price (fig:Price_Development)
  Contribution/SVG/Figure_2.{pdf,svg}        mean trade count by local product and time until delivery (fig:Trade_occurence)
  Contribution/tables/trade_balances.tex     yearly inter-zonal trade balances 360-30 min before delivery

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
TAB_DIR = os.path.join(ROOT, "Contribution", "tables")
WINDOWS = [f"{a}to{a + 15}" for a in range(0, 360, 15)]  # 24 windows of 15 min, 0to15 ... 345to360
BALANCE_WINDOWS = WINDOWS[2:]                            # 360 to 30 min before delivery
PAIRS = [(a, b) for i, a in enumerate(ZONES) for b in ZONES[i + 1:]]
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
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
        ax.axhline(30, color="#d62728", ls="--", lw=1.0, label="End of national trading (30 min)")
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


def _fmt(v: float) -> str:
    return f"{v:,.2f}"


def tab_trade_balances() -> None:
    cols = [f"{a}_id_balance_to_{b}_{w}" for a in ZONES for b in ZONES if a != b for w in BALANCE_WINDOWS]
    cols += [f"{z}_id_balance_{k}{w}" for z in ZONES for k in ("", "ext_") for w in BALANCE_WINDOWS]
    df = load_master(cols)
    total = lambda col: df.groupby("_year")[[f"{col}_{w}" for w in BALANCE_WINDOWS]].sum().sum(axis=1)
    years = sorted(df["_year"].unique())
    n_days = df.groupby("_year")["_day"].nunique()

    bilateral = {(a, b): total(f"{a}_id_balance_to_{b}") for a, b in PAIRS}
    external = {z: total(f"{z}_id_balance_ext") for z in ZONES}
    net = {z: total(f"{z}_id_balance") for z in ZONES}
    for a, b in PAIRS:  # each bilateral trade enters both areas with opposite sign
        assert np.allclose(bilateral[a, b], -total(f"{b}_id_balance_to_{a}"), atol=1e-6)
    for z in ZONES:  # net balance = bilateral + external
        bil = sum(bilateral[a, b] * (1 if a == z else -1) for a, b in PAIRS if z in (a, b))
        assert np.allclose(net[z], bil + external[z], atol=1e-6)

    row = lambda label, s: f"    {label} & " + " & ".join(_fmt(s[y]) for y in years) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \caption{Continuous intraday trade balances of the control areas (MWh, trades 360 to 30 minutes before "
        r"delivery, local delivery days). Positive values denote net imports of the first-named control area, negative "
        r"values net exports. External: single-leg XBID trades with counterparties outside the German order book. "
        r"Net balance: bilateral plus external.}",
        r"  \label{tab:trade_balances_formatted}",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \begin{tabular}{l" + "r" * len(years) + "}",
        r"    \toprule",
        r"    \textbf{Control area} & " + " & ".join(rf"\textbf{{{y}}}" for y in years) + r" \\",
        r"    & " + " & ".join(rf"($n = {n_days[y]}$\,d)" for y in years) + r" \\",
        r"    \midrule",
        rf"    \multicolumn{{{len(years) + 1}}}{{l}}{{\textit{{Bilateral balances}}}} \\[0.05cm]",
        *[row(f"{a} vs. {b}", bilateral[a, b]) for a, b in PAIRS],
        r"    \midrule",
        rf"    \multicolumn{{{len(years) + 1}}}{{l}}{{\textit{{External balances (XBID)}}}} \\[0.05cm]",
        *[row(z, external[z]) for z in ZONES],
        r"    \midrule",
        rf"    \multicolumn{{{len(years) + 1}}}{{l}}{{\textit{{Net balance}}}} \\[0.05cm]",
        *[row(rf"\textbf{{{z}}}", net[z]) for z in ZONES],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(TAB_DIR, "trade_balances.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("trade_balances.tex: days per year", n_days.to_dict())


if __name__ == "__main__":
    fig_price()
    fig_trades()
    tab_trade_balances()
