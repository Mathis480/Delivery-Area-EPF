"""
Aggregates raw EPEX SPOT continuous trades (2021-2024) into 15-minute lead-time windows per delivery quarter-hour
for the delivery areas DE1-DE4 and Germany (DE), in UTC: VWAP, volume, trade count, net / bilateral / external
balances and intra-area volume. See README.md for the schema.

Usage: python calculate_ID_DEx_UTC.py   (reads <year>/Continuous_Trades*.csv, writes ID_DelArea_<area>_<year>.csv)
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = ["Intraday_Quarter_Hour_Power", "XBID_Quarter_Hour_Power"]
ZONES = ["DE1", "DE2", "DE3", "DE4"]
# Lead-time windows [from, to) in minutes before delivery, from 345-360 down to 0-15, plus the target window 0-30
WINDOWS = [(f"{a}to{a + 15}", a, a + 15) for a in range(345, -1, -15)] + [("0to30", 0, 30)]


def classify(area: str, side: str, target: str, n_legs: int) -> str:
    """Two-leg trades from the counterparty leg (intra/export/import); single-leg XBID trades from the own leg."""
    if n_legs == 1:
        return "ext_import" if side == "BUY" else "ext_export"
    if (str(area).startswith("DE") if target == "DE" else area == target):
        return "intra"
    return "export" if side == "BUY" else "import"  # counterparty buys -> target area exports


def net(df: pd.DataFrame, mask: pd.Series) -> float:
    sub = df[mask]
    return np.round(np.where(sub["TradeTyp"].str.endswith("export"), -sub["Volume"], sub["Volume"]).sum(), 3)


def window_metrics(df: pd.DataFrame, w: str, prev_vwap: float) -> dict:
    """Metrics of one window; an empty window takes the VWAP of the preceding window and zero volumes."""
    keys = ["VWAP", "VWAP_total_volume", "VWAP_total_trades", "id_balance"] + [f"id_balance_to_{z}" for z in ZONES] \
        + ["id_balance_ext", "id_trade_intra"]
    if df.empty:
        return dict(zip([f"{k}_{w}" for k in keys], [prev_vwap] + [0] * (len(keys) - 1)))
    vol, typ = df["Volume"].sum(), df["TradeTyp"]
    cross = typ.isin(["import", "export"])
    values = [np.round((df["Price"] * df["Volume"]).sum() / vol, 2) if vol else np.nan, np.round(vol, 2), len(df)]
    values += [np.nan] * 7 if vol == 0 else [
        net(df, typ != "intra"), *[net(df, cross & (df["DeliveryArea"] == z)) for z in ZONES],
        net(df, typ.str.startswith("ext")), np.round(df.loc[typ == "intra", "Volume"].sum(), 3)]
    return dict(zip([f"{k}_{w}" for k in keys], values))


def process_file(args: tuple) -> pd.DataFrame | None:
    path, target = args
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Product"].isin(PRODUCTS) & (df["SelfTrade"].astype(str).str.strip() == "N")]  # no wash trades
    in_target = df["DeliveryArea"].astype(str).str.startswith("DE") if target == "DE" else df["DeliveryArea"] == target
    df = df[df["TradeId"].isin(df.loc[in_target, "TradeId"])].copy()
    if df.empty:
        return None
    n_legs = df.groupby("TradeId")["TradeId"].transform("size")
    if (n_legs > 2).any():
        raise ValueError(f"TradeIds with more than two legs in {path}")
    df["TradeTyp"] = [classify(a, s, target, n) for a, s, n in zip(df["DeliveryArea"], df["Side"], n_legs)]
    # One record per trade; inter-area legs ('export'/'import') sort before 'intra'
    df = df.sort_values(["TradeId", "TradeTyp"]).drop_duplicates("TradeId", keep="first")
    df["DeliveryStart"] = pd.to_datetime(df["DeliveryStart"], utc=True)
    df["ExecutionTime"] = pd.to_datetime(df["ExecutionTime"], utc=True)
    df["TimeDiff"] = (df["DeliveryStart"] - df["ExecutionTime"]).dt.total_seconds() / 60
    df = df.sort_values(["DeliveryStart", "ExecutionTime"]).reset_index(drop=True)

    records = []
    for start, g in df.groupby("DeliveryStart", sort=True):
        record, vwap = {"DeliveryStart": start}, np.nan
        for w, a, b in WINDOWS:
            record.update(window_metrics(g[(g["TimeDiff"] >= a) & (g["TimeDiff"] < b)], w, vwap))
            vwap = record[f"VWAP_{w}"]
        records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    for year in (2021, 2022, 2023, 2024):
        files = sorted(glob.glob(os.path.join(HERE, str(year), "Continuous_Trades*.csv")))
        for area in ZONES + ["DE"]:
            with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
                parts = list(tqdm.tqdm(ex.map(process_file, [(f, area) for f in files]), total=len(files),
                                       desc=f"{area}_{year}"))
            out = pd.concat([p for p in parts if p is not None], ignore_index=True).sort_values("DeliveryStart")
            out.to_csv(os.path.join(HERE, f"ID_DelArea_{area}_{year}.csv"), index=False)
