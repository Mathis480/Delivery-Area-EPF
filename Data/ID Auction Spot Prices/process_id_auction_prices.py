"""
Process EPEX Intraday Auction Spot Prices (Germany/Luxembourg, 2021-2024).

Combines historical German 15-minute call auctions (2021 to June 13, 2024) and
the succeeding Pan-European Intraday Auction 1 (IDA1, June 14, 2024 onwards)
into a seamless, continuous 15-minute time series in Pure UTC (+00:00 / Z).
"""

import glob
import os
import re
import pandas as pd


def process_id_auction_spot_prices(data_dir: str = None) -> pd.DataFrame:
    """
    Reads all raw Intraday Auction CSVs (15-call DE + Pan-European IDA1),
    transforms wide quarter-hourly columns (Hour X Q1..Q4) into a long format,
    maps German wall-clock time (Europe/Berlin) to Pure UTC, and merges them.

    Returns:
        pd.DataFrame with columns:
            - 'Date': pd.Timestamp with UTC timezone (+00:00)
            - 'Intraday_Auction_Price': float (EUR/MWh)
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    # Identify files for 15-call DE and IDA1
    call_files = sorted(glob.glob(os.path.join(data_dir, "intraday_auction_spot_prices_15-call_*.csv")))
    ida1_files = sorted(glob.glob(os.path.join(data_dir, "pan-european_prices_*_IDA1_*.csv")))
    all_files = call_files + ida1_files

    if not all_files:
        raise FileNotFoundError(f"No Intraday Auction CSV files found in {data_dir}")

    dfs = []
    for file_path in all_files:
        df = pd.read_csv(file_path, comment="#")
        quarter_cols = [c for c in df.columns if c.startswith("Hour ")]

        melted = pd.melt(
            df[["Delivery day"] + quarter_cols],
            id_vars=["Delivery day"],
            value_vars=quarter_cols,
            var_name="Quarter_Col",
            value_name="Intraday_Auction_Price"
        )
        melted["Intraday_Auction_Price"] = pd.to_numeric(melted["Intraday_Auction_Price"], errors="coerce")
        melted = melted.dropna(subset=["Intraday_Auction_Price"])
        dfs.append(melted)

    combined = pd.concat(dfs, ignore_index=True)

    # Regex parsing: Hour (1-24), optional suffix (A/B), and Quarter (Q1-Q4)
    pattern = r"Hour\s+(\d+)([AB])?\s+Q([1-4])"
    parsed = combined["Quarter_Col"].str.extract(pattern)
    hours = parsed[0].astype(int) - 1
    quarters = parsed[2].astype(int) - 1

    delivery_dates = pd.to_datetime(combined["Delivery day"], dayfirst=True)
    local_time = (
        delivery_dates
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(quarters * 15, unit="m")
    )
    combined["local_time"] = local_time

    # DST handling for Europe/Berlin:
    # 3B represents repeated autumnal hour (CET standard); ambiguous=False
    # All other hours use ambiguous=True
    is_dst = ~combined["Quarter_Col"].str.contains("3B")
    local_dt = combined["local_time"].dt.tz_localize(
        "Europe/Berlin",
        ambiguous=is_dst,
        nonexistent="shift_forward"
    )
    combined["Date"] = local_dt.dt.tz_convert("UTC")

    result = (
        combined[["Date", "Intraday_Auction_Price"]]
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    return result


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Processing EPEX Intraday Auction Spot Prices in {current_dir}...")
    df_id = process_id_auction_spot_prices(current_dir)
    print(f"Successfully processed {len(df_id):,} quarter-hours.")
    print(f"Date range: {df_id['Date'].iloc[0]} to {df_id['Date'].iloc[-1]}")

    output_file = os.path.join(current_dir, "id_auction_spot_prices_15min_utc_2021_2024.csv")
    df_id.to_csv(output_file, index=False)
    print(f"Saved processed 15-minute UTC series to {output_file}")
