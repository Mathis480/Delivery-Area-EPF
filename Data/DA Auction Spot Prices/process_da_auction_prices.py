"""
Process EPEX Day-Ahead Auction Spot Prices (Germany/Luxembourg, 2021-2024).

Transforms wide-format hourly Day-Ahead auction price CSV files into a uniform,
continuous 15-minute time series indexed in Pure UTC (+00:00 / Z).
"""

import glob
import os
import re
import pandas as pd


def process_da_auction_spot_prices(data_dir: str = None) -> pd.DataFrame:
    """
    Reads all raw Day-Ahead auction price CSVs from data_dir, transforms wide
    hourly columns into a long format, maps German wall-clock time (Europe/Berlin)
    to Pure UTC, and expands hourly prices to 15-minute intervals.

    Returns:
        pd.DataFrame with columns:
            - 'Date': pd.Timestamp with UTC timezone (+00:00)
            - 'Dayahead_Auction_hourly': float (EUR/MWh)
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    file_pattern = os.path.join(data_dir, "auction_spot_prices_germany_luxembourg_*.csv")
    files = sorted(glob.glob(file_pattern))

    if not files:
        raise FileNotFoundError(f"No Day-Ahead auction CSV files found matching {file_pattern}")

    dfs = []
    for file_path in files:
        # Read raw CSV (skip comment rows starting with '#')
        df = pd.read_csv(file_path, comment="#")
        hour_cols = [c for c in df.columns if c.startswith("Hour ")]
        
        # Melt wide columns (Hour 1 .. Hour 24 + Hour 3A/3B) into long format
        melted = pd.melt(
            df[["Delivery day"] + hour_cols],
            id_vars=["Delivery day"],
            value_vars=hour_cols,
            var_name="Hour_Col",
            value_name="Dayahead_Auction_hourly"
        )
        melted["Dayahead_Auction_hourly"] = pd.to_numeric(melted["Dayahead_Auction_hourly"], errors="coerce")
        melted = melted.dropna(subset=["Dayahead_Auction_hourly"])
        dfs.append(melted)

    combined = pd.concat(dfs, ignore_index=True)

    # Extract hour number (1-24) and optional suffix ('A'/'B' for DST switch in October)
    pattern = r"Hour\s+(\d+)([AB])?"
    parsed = combined["Hour_Col"].str.extract(pattern)
    hours = parsed[0].astype(int) - 1  # 0-indexed hour (Hour 1 -> 0, Hour 24 -> 23)

    delivery_dates = pd.to_datetime(combined["Delivery day"], dayfirst=True)
    combined["local_time"] = delivery_dates + pd.to_timedelta(hours, unit="h")

    # Expand hourly contract prices into 4 x 15-minute delivery intervals (Q1..Q4)
    q_offsets = [pd.Timedelta(minutes=m) for m in [0, 15, 30, 45]]
    expanded = pd.concat(
        [combined.assign(local_time=combined["local_time"] + offset) for offset in q_offsets],
        ignore_index=True
    )

    # Daylight Saving Time (DST) Handling for Europe/Berlin:
    # 'Hour 3B' represents the repeated 02:00-03:00 hour in late October (winter time / standard CET).
    # Setting ambiguous=False assigns it to the second occurrence.
    # Standard hours and 'Hour 3A' use ambiguous=True (first occurrence / summer CEST).
    is_dst = ~expanded["Hour_Col"].str.contains("3B")

    local_dt = expanded["local_time"].dt.tz_localize(
        "Europe/Berlin",
        ambiguous=is_dst,
        nonexistent="shift_forward"
    )
    expanded["Date"] = local_dt.dt.tz_convert("UTC")

    result = (
        expanded[["Date", "Dayahead_Auction_hourly"]]
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    return result


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Processing EPEX Day-Ahead Auction Spot Prices in {current_dir}...")
    df_da = process_da_auction_spot_prices(current_dir)
    print(f"Successfully processed {len(df_da):,} quarter-hours.")
    print(f"Date range: {df_da['Date'].iloc[0]} to {df_da['Date'].iloc[-1]}")
    
    output_file = os.path.join(current_dir, "da_auction_spot_prices_15min_utc_2021_2024.csv")
    df_da.to_csv(output_file, index=False)
    print(f"Saved processed 15-minute UTC series to {output_file}")
