"""
Create Master ML Dataset for German Bidding Zone and TSO Control Areas (2021-2024).

This pipeline assembles, cleans, and merges all market and fundamental data streams
onto a uniform 15-minute Pure UTC (+00:00 / Z) time grid:
  1. Date Spine: 15-minute continuous UTC intervals (2021-09-30 22:00:00 to 2024-12-31 22:45:00 UTC)
  2. Activated Balancing Power (SRL / aFRR & MRL / mFRR) from Netztransparenz
  3. Energy-Charts Fundamentals (Load, Solar, Wind Onshore, Wind Offshore for DE & 4 TSOs)
  4. EPEX Day-Ahead Auction Spot Prices (DE/LU)
  5. EPEX Intraday Auction Spot Prices (15-Call DE & Pan-European IDA1)
  6. EPEX Intraday Continuous Transactions (VWAP, Volumes, Trades, Balances for DE & DE1-DE4)
  7. Calendar & Wall-Clock Delivery Dummies (Weekday 1-7, Hour 0-23, Quarter 1-4)
  8. Multi-Day Time-Lagged Features (24h, 48h, 168h lags)
  9. Canonical Column Ordering according to features.csv
"""

import glob
import os
import sys
import time
import pandas as pd
import numpy as np


# ==============================================================================
# CONFIGURATION & CANONICAL PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
FEATURES_CSV = os.path.join(BASE_DIR, "features.csv")
DEFAULT_OUTPUT_CSV = os.path.join(DATA_DIR, "master_dataset_2021_2024.csv")

# TSO to EPEX Delivery Area Mapping (per AGENTS.md)
TSO_TO_DELIVERY_AREA = {
    "TransnetBW": "DE1",
    "Amprion": "DE2",
    "TenneT": "DE3",
    "TenneT TSO": "DE3",
    "50Hertz": "DE4",
    "Deutschland": "DE",
}

# Pure UTC Date Spine Range
# 2021-09-30 22:00:00 UTC = 2021-10-01 00:00 CEST (start of delivery in Q4 2021)
# 2024-12-31 22:45:00 UTC = 2024-12-31 23:45 CET  (last delivery interval start in 2024)
START_UTC = "2021-09-30 22:00:00"
END_UTC = "2024-12-31 22:45:00"
# 7-day look-ahead buffer for multi-day lags (672 quarter-hours prior to START_UTC)
BUFFER_START_UTC = "2021-09-23 22:00:00"
FREQ = "15min"



# ==============================================================================
# 1. DATE SPINE CREATION
# ==============================================================================
def create_utc_date_spine(start: str = START_UTC, end: str = END_UTC, freq: str = FREQ) -> pd.DataFrame:
    """Generates the uniform 15-minute Pure UTC master index DataFrame."""
    date_range = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    df = pd.DataFrame(date_range, columns=["Date"])
    print(f"[1/8] Created Pure UTC date spine: {len(df):,} quarter-hours ({df['Date'].iloc[0]} to {df['Date'].iloc[-1]})")
    return df


# ==============================================================================
# 2. BALANCING ENERGY LOADER (SRL / aFRR & MRL / mFRR)
# ==============================================================================
def load_balancing_energy(base_dir: str) -> pd.DataFrame:
    """
    Loads operational balancing energy files (SRL and MRL), parses timestamps to Pure UTC,
    and maps TSO names to canonical EPEX delivery area codes (DE1-DE4, DE).
    """
    be_dir = os.path.join(base_dir, "Balancing Energy")
    res_dfs = []

    for res_type in ["SRL", "MRL"]:
        files = sorted(glob.glob(os.path.join(be_dir, res_type, "*.csv")))
        if not files:
            print(f"Warning: No {res_type} balancing files found in {os.path.join(be_dir, res_type)}")
            continue

        type_dfs = []
        for file_path in files:
            df = pd.read_csv(file_path, sep=";", decimal=".")
            dt_str = df["Datum"].astype(str).str.replace(".", "-", regex=False) + " " + df["von"].astype(str)
            df["Date"] = pd.to_datetime(dt_str, format="%d-%m-%Y %H:%M", utc=True)

            drop_cols = [c for c in ["Datum", "Zeitzone", "von", "bis", "Einheit", "MOL-Abweichung"] if c in df.columns]
            df = df.drop(columns=drop_cols)

            rename_map = {}
            for col in df.columns:
                if col == "Date":
                    continue
                is_pos = "(Positiv)" in col
                sign = "positive" if is_pos else "negative"
                area_raw = col.replace(" (Positiv)", "").replace(" (Negativ)", "").strip()
                zone = TSO_TO_DELIVERY_AREA.get(area_raw, area_raw)
                rename_map[col] = f"{zone}_{res_type}_{sign}"
            df = df.rename(columns=rename_map)
            type_dfs.append(df)

        combined_type = pd.concat(type_dfs, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"])
        res_dfs.append(combined_type)

    if not res_dfs:
        return pd.DataFrame(columns=["Date"])

    result = res_dfs[0]
    for other in res_dfs[1:]:
        result = result.merge(other, on="Date", how="outer")

    print(f"[2/8] Loaded Balancing Energy: {len(result.columns) - 1} reserve activation features.")
    return result


# ==============================================================================
# 3. FUNDAMENTALS LOADER (LOAD, SOLAR, ONSHORE, OFFSHORE)
# ==============================================================================
def load_fundamentals(base_dir: str) -> pd.DataFrame:
    """
    Loads Energy-Charts fundamentals across all 4 categories and maps control areas
    to canonical EPEX delivery area codes (DE1-DE4, DE).
    Parses Pure UTC timestamps, computes actual vs Day-Ahead forecast errors,
    and assigns canonical feature aliases.
    """
    fund_dir = os.path.join(base_dir, "Fundamentals")
    cat_dirs = ["LOAD", "SOLAR", "ONSHORE", "OFFSHORE"]
    all_cat_dfs = []

    for cat_dir in cat_dirs:
        dir_path = os.path.join(fund_dir, cat_dir)
        files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))

        reg_groups = {}
        for f in files:
            fname = os.path.basename(f)
            if "in_the_control_area_of_" in fname:
                reg = fname.split("in_the_control_area_of_")[1].split("_in_")[0]
            elif "in_Germany_" in fname:
                reg = "DE"
            else:
                continue
            reg_groups.setdefault(reg, []).append(f)

        for reg, f_list in reg_groups.items():
            zone = TSO_TO_DELIVERY_AREA.get(reg, reg)
            reg_dfs = []
            for f in f_list:
                df = pd.read_csv(f, skiprows=[1])  # Skip unit row ',Power (MW),Power (MW)'
                date_col = [c for c in df.columns if "Date" in c][0]
                df["Date"] = pd.to_datetime(df[date_col], utc=True)

                actual_col = [c for c in df.columns if c not in ["Date", date_col] and "forecast" not in c.lower()][0]
                forecast_col = [c for c in df.columns if "forecast" in c.lower()][0]

                actual_series = pd.to_numeric(df[actual_col], errors="coerce")
                forecast_series = pd.to_numeric(df[forecast_col], errors="coerce")
                diff_series = forecast_series - actual_series

                cat_name = "wind_onshore" if cat_dir == "ONSHORE" else ("wind_offshore" if cat_dir == "OFFSHORE" else cat_dir.lower())
                prefix = f"{zone}_{cat_name}"

                reg_dfs.append(pd.DataFrame({
                    "Date": df["Date"],
                    f"{prefix}_actual": actual_series,
                    f"{prefix}_forecast": forecast_series,
                    f"{prefix}_diff": diff_series
                }))

            combined_reg = pd.concat(reg_dfs, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"])
            all_cat_dfs.append(combined_reg)

    # Load Cross-Border Flows (Germany)
    cb_dir = os.path.join(fund_dir, "CROSS_BORDER")
    if os.path.isdir(cb_dir):
        cb_files = sorted(glob.glob(os.path.join(cb_dir, "*.csv")))
        cb_dfs = []
        for f in cb_files:
            df = pd.read_csv(f, skiprows=[1])
            date_col = [c for c in df.columns if "Date" in c][0]
            df["Date"] = pd.to_datetime(df[date_col], utc=True)
            val_col = [c for c in df.columns if c not in ["Date", date_col]][0]
            cb_dfs.append(pd.DataFrame({
                "Date": df["Date"],
                "DE_cross_border_trading": pd.to_numeric(df[val_col], errors="coerce")
            }))
        if cb_dfs:
            combined_cb = pd.concat(cb_dfs, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"])
            all_cat_dfs.append(combined_cb)

    if not all_cat_dfs:
        return pd.DataFrame(columns=["Date"])

    result = all_cat_dfs[0]
    for other in all_cat_dfs[1:]:
        result = result.merge(other, on="Date", how="outer")

    result = result.sort_values("Date")

    # Canonical aliases for legacy model alignment
    if "DE_load_actual" in result.columns:
        result["Load"] = result["DE_load_actual"]
    if "DE_solar_diff" in result.columns:
        result["prediction_error_solar"] = result["DE_solar_diff"]
    if "DE_wind_onshore_diff" in result.columns:
        result["prediction_error_onshore"] = result["DE_wind_onshore_diff"]
    if "DE_wind_offshore_diff" in result.columns:
        result["prediction_error_offshore"] = result["DE_wind_offshore_diff"]
    if "DE_cross_border_trading" in result.columns:
        result["cross_border_trading"] = result["DE_cross_border_trading"]

    print(f"[3/8] Loaded Fundamentals: {len(result.columns) - 1} fundamental generation & demand features.")
    return result


# ==============================================================================
# 4. DAY-AHEAD & INTRADAY AUCTION SPOT PRICES LOADER
# ==============================================================================
def load_auction_spot_prices(base_dir: str) -> pd.DataFrame:
    """Loads Day-Ahead and Intraday Auction prices in continuous 15-minute Pure UTC."""
    # Day-Ahead
    da_script_dir = os.path.join(base_dir, "DA Auction Spot Prices")
    sys.path.insert(0, da_script_dir)
    try:
        from process_da_auction_prices import process_da_auction_spot_prices
        da_df = process_da_auction_spot_prices(da_script_dir)
    except Exception as e:
        print(f"Warning: Could not run process_da_auction_prices directly ({e}), loading pre-generated CSV...")
        da_csv = os.path.join(da_script_dir, "da_auction_spot_prices_15min_utc_2021_2024.csv")
        da_df = pd.read_csv(da_csv)
        da_df["Date"] = pd.to_datetime(da_df["Date"], utc=True)

    # Intraday Auction
    id_script_dir = os.path.join(base_dir, "ID Auction Spot Prices")
    sys.path.insert(0, id_script_dir)
    try:
        from process_id_auction_prices import process_id_auction_spot_prices
        id_df = process_id_auction_spot_prices(id_script_dir)
    except Exception as e:
        print(f"Warning: Could not run process_id_auction_prices directly ({e}), loading pre-generated CSV...")
        id_csv = os.path.join(id_script_dir, "id_auction_spot_prices_15min_utc_2021_2024.csv")
        id_df = pd.read_csv(id_csv)
        id_df["Date"] = pd.to_datetime(id_df["Date"], utc=True)

    # Standardize column name to 'Intraday Auction Price'
    if "Intraday_Auction_Price" in id_df.columns:
        id_df = id_df.rename(columns={"Intraday_Auction_Price": "Intraday Auction Price"})

    auctions_df = da_df.merge(id_df, on="Date", how="outer").sort_values("Date")
    print(f"[4/8] Loaded Auction Spot Prices: Dayahead_Auction_hourly and Intraday Auction Price.")
    return auctions_df


# ==============================================================================
# 5. EPEX INTRADAY CONTINUOUS TRANSACTIONS LOADER
# ==============================================================================
def load_continuous_intraday(base_dir: str) -> pd.DataFrame:
    """
    Loads regional and national EPEX Intraday Continuous transaction metrics:
    rolling VWAP, traded volumes, trade counts, and cross-border/zonal flow balances.
    Strictly adheres to 'VWAP_' nomenclature.
    """
    cont_dir = os.path.join(base_dir, "EPEX Intraday Continuous")
    combined_cont = None

    for area in ["DE1", "DE2", "DE3", "DE4", "DE"]:
        files = sorted(glob.glob(os.path.join(cont_dir, f"ID_DelArea_{area}_*.csv")))
        if not files:
            print(f"Warning: No continuous intraday files found for {area} in {cont_dir}")
            continue

        area_dfs = []
        for file_path in files:
            df = pd.read_csv(file_path)

            if area == "DE":
                # National area: keep only national price columns without sub-components
                keep_cols = [
                    c for c in df.columns
                    if c.startswith("VWAP_") and not any(x in c for x in ["total_volume", "total_trades", "upper", "lower"])
                ]
            else:
                # Regional zones DE1-DE4: keep all VWAP, volume, trades, and inter-zonal balances
                keep_cols = [
                    c for c in df.columns
                    if c != "DeliveryStart"
                    and not any(x in c for x in ["upper", "lower"])
                    and not c.startswith(f"id_balance_to_{area}_")
                ]

            df_sub = df[["DeliveryStart"] + keep_cols].copy()
            df_sub["Date"] = pd.to_datetime(df_sub["DeliveryStart"], utc=True)
            df_sub = df_sub.drop(columns=["DeliveryStart"])

            for c in keep_cols:
                df_sub[c] = pd.to_numeric(df_sub[c], errors="coerce")
            area_dfs.append(df_sub)

        area_df = pd.concat(area_dfs, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"])

        # Regional prefix for DE1-DE4; DE national remains unprefixed (e.g. VWAP_90to105)
        if area != "DE":
            rename_dict = {col: f"{area}_{col}" for col in area_df.columns if col != "Date"}
            area_df = area_df.rename(columns=rename_dict)

        if combined_cont is None:
            combined_cont = area_df
        else:
            combined_cont = combined_cont.merge(area_df, on="Date", how="outer")

    print(f"[5/8] Loaded EPEX Intraday Continuous: {len(combined_cont.columns) - 1} rolling VWAP, volume & balance metrics.")
    return combined_cont


# ==============================================================================
# 6. CALENDAR, HOUR & QUARTER DUMMIES
# ==============================================================================
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates weekday dummies (Weekday_1 to Weekday_7), hourly delivery indicators (Hour_0 to Hour_23),
    and quarter-hour indicators (Quarter_1 to Quarter_4) based on German wall-clock time (Europe/Berlin).
    """
    local_date = df["Date"].dt.tz_convert("Europe/Berlin")
    weekday = local_date.dt.dayofweek + 1   # 1 = Monday, 7 = Sunday
    hour = local_date.dt.hour               # 0 to 23
    minute = local_date.dt.minute           # 0, 15, 30, 45
    quarter = (minute // 15) + 1           # 1 = :00, 2 = :15, 3 = :30, 4 = :45

    dummies = {}

    # Weekday dummies (Weekday_1 to Weekday_7)
    for i in range(1, 8):
        dummies[f"Weekday_{i}"] = (weekday == i).astype(int)

    # Hour dummies (Hour_0 to Hour_23)
    for h in range(24):
        dummies[f"Hour_{h}"] = (hour == h).astype(int)

    # Quarter dummies (Quarter_1 to Quarter_4)
    for q in range(1, 5):
        dummies[f"Quarter_{q}"] = (quarter == q).astype(int)

    dummy_df = pd.DataFrame(dummies, index=df.index)
    df = pd.concat([df, dummy_df], axis=1)
    print(f"[6/9] Added calendar, hourly, and quarter-hour dummies ({len(dummy_df.columns)} features).")
    return df


# ==============================================================================
# 7. HIERARCHICAL MISSING VALUE IMPUTATION (VARIANTE B)
# ==============================================================================
def impute_price_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in continuous rolling price trajectories (VWAP)
    using hierarchical domain-grounded fallbacks (Variante B per user instruction):
      - Step 1 (National DE Benchmark Trajectory):
          * Leftmost window (VWAP_345to360): Prio 0 (own trade) -> Prio 1 ('Intraday Auction Price') -> Prio 2 ('Dayahead_Auction_hourly')
          * Subsequent windows (330to345 ... 0to15): horizontal forward-fill from preceding window
          * Target VWAP_0to30: fallback to VWAP_0to15 / VWAP_15to30
      - Step 2 (Regional Delivery Areas DE1-DE4):
          * Prio 0: Own observed regional VWAP in window w
          * Prio 1: Contemporaneous national volume-weighted benchmark in the SAME window w (VWAP_{w})
      - Step 3 (Non-Price Continuous Trading Features):
          * Missing volumes, trade counts, or flow balances are filled with 0 (no trades observed).
    """
    windows_15m = [
        "345to360", "330to345", "315to330", "300to315",
        "285to300", "270to285", "255to270", "240to255",
        "225to240", "210to225", "195to210", "180to195",
        "165to180", "150to165", "135to150", "120to135",
        "105to120", "90to105", "75to90", "60to75",
        "45to60", "30to45", "15to30", "0to15"
    ]

    # 1. National DE Trajectory
    if "VWAP_345to360" in df.columns:
        if "Intraday Auction Price" in df.columns:
            df["VWAP_345to360"] = df["VWAP_345to360"].fillna(df["Intraday Auction Price"])
        if "Dayahead_Auction_hourly" in df.columns:
            df["VWAP_345to360"] = df["VWAP_345to360"].fillna(df["Dayahead_Auction_hourly"])

    for i in range(1, len(windows_15m)):
        curr_w = windows_15m[i]
        prev_w = windows_15m[i - 1]
        curr_col = f"VWAP_{curr_w}"
        prev_col = f"VWAP_{prev_w}"
        if curr_col in df.columns and prev_col in df.columns:
            df[curr_col] = df[curr_col].fillna(df[prev_col])

    if "VWAP_0to30" in df.columns:
        if "VWAP_0to15" in df.columns:
            df["VWAP_0to30"] = df["VWAP_0to30"].fillna(df["VWAP_0to15"])
        if "VWAP_15to30" in df.columns:
            df["VWAP_0to30"] = df["VWAP_0to30"].fillna(df["VWAP_15to30"])

    # 2. Regional Delivery Areas DE1-DE4
    all_windows = windows_15m + (["0to30"] if "VWAP_0to30" in df.columns else [])
    for area in ["DE1", "DE2", "DE3", "DE4"]:
        for w in all_windows:
            reg_col = f"{area}_VWAP_{w}"
            nat_col = f"VWAP_{w}"
            if reg_col in df.columns and nat_col in df.columns:
                df[reg_col] = df[reg_col].fillna(df[nat_col])

    # 3. Non-price trading features: fill NaN with 0 (no trades / 0 MW / 0 EUR)
    non_price_cols = [
        c for c in df.columns
        if any(k in c for k in ["total_volume", "total_trades", "balance_to_"])
    ]
    for c in non_price_cols:
        df[c] = df[c].fillna(0)

    print(f"[7/9] Imputed continuous price trajectories (Variante B) and non-price features.")
    return df


# ==============================================================================
# 8. MULTI-DAY LAGS
# ==============================================================================
def add_multi_day_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes multi-day time lags (24h = 96, 48h = 192, 168h = 672 steps)
    for key spot auction prices and benchmark VWAPs.
    """
    lag_shifts = {
        "lag_24h": 96,    # d-1 (24 hours)
        "lag_48h": 192,   # d-2 (48 hours)
        "lag_168h": 672   # d-7 (7 days / 168 hours)
    }

    potential_lag_targets = [
        "VWAP_90to105",
        "Dayahead_Auction_hourly",
        "Intraday Auction Price",
        "DE1_VWAP_90to105",
        "DE2_VWAP_90to105",
        "DE3_VWAP_90to105",
        "DE4_VWAP_90to105"
    ]
    cols_to_lag = [c for c in potential_lag_targets if c in df.columns]

    lagged_dict = {}
    for col in cols_to_lag:
        for lag_name, shift_val in lag_shifts.items():
            lagged_dict[f"{col}_{lag_name}"] = df[col].shift(shift_val)

    lag_df = pd.DataFrame(lagged_dict, index=df.index)
    df = pd.concat([df, lag_df], axis=1)

    print(f"[8/9] Generated {len(lag_df.columns)} multi-day lag features.")
    return df


# ==============================================================================
# 9. CANONICAL ORDERING VIA FEATURES.CSV & EXPORT
# ==============================================================================
def reorder_and_export(df: pd.DataFrame, features_csv_path: str, output_csv_path: str) -> None:
    """Sorts columns according to the version-controlled features catalog and exports CSV."""
    if os.path.exists(features_csv_path):
        f_df = pd.read_csv(features_csv_path)
        canonical_order = f_df["name"].dropna().astype(str).str.strip().tolist()

        matched_cols = [c for c in canonical_order if c in df.columns]
        unmatched_cols = [c for c in df.columns if c not in matched_cols]

        df = df[matched_cols + unmatched_cols]
        print(f"[9/9] Successfully aligned columns with {features_csv_path} ({len(matched_cols)} matched, {len(unmatched_cols)} appended).")
    else:
        print(f"[9/9] features.csv not found at {features_csv_path}, preserving original column order.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    print(f"Exporting master dataset to '{output_csv_path}' ...")
    df.to_csv(output_csv_path, index=False)
    file_size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
    print(f"Export complete! File size: {file_size_mb:.2f} MB, Total rows: {len(df):,}, Total columns: {len(df.columns)}")


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def build_master_dataset(base_dir: str = BASE_DIR, output_path: str = DEFAULT_OUTPUT_CSV) -> pd.DataFrame:
    """Executes the end-to-end master dataset generation workflow."""
    start_time = time.time()
    print("=" * 80)
    print("STARTING MASTER DATASET CREATION PIPELINE (PURE UTC)")
    print(f"Base Directory: {base_dir}")
    print("=" * 80)

    # 1. Date spine (initialized with 7-day lookback buffer for multi-day lags)
    master_df = create_utc_date_spine(start=BUFFER_START_UTC, end=END_UTC)

    # 2. Balancing Energy
    be_df = load_balancing_energy(os.path.join(base_dir, "Data"))
    master_df = master_df.merge(be_df, on="Date", how="left")

    # 3. Fundamentals
    fund_df = load_fundamentals(os.path.join(base_dir, "Data"))
    master_df = master_df.merge(fund_df, on="Date", how="left")

    # 4. Auction Spot Prices (DA and ID)
    auctions_df = load_auction_spot_prices(os.path.join(base_dir, "Data"))
    master_df = master_df.merge(auctions_df, on="Date", how="left")

    # 5. EPEX Intraday Continuous
    cont_df = load_continuous_intraday(os.path.join(base_dir, "Data"))
    master_df = master_df.merge(cont_df, on="Date", how="left")

    # 6. Calendar, Hour & Quarter features
    master_df = add_calendar_features(master_df)

    # 7. Impute continuous price trajectories & non-price features
    master_df = impute_price_trajectories(master_df)

    # 8. Multi-day lags (calculated over buffered spine so initial rows are populated)
    master_df = add_multi_day_lags(master_df)

    # Trim to canonical delivery horizon [START_UTC, END_UTC]
    master_df = master_df[(master_df["Date"] >= START_UTC) & (master_df["Date"] <= END_UTC)].copy()
    print(f"Trimmed lag buffer: Final master dataset covers {len(master_df):,} quarter-hours ({master_df['Date'].iloc[0]} to {master_df['Date'].iloc[-1]})")

    # 9. Canonical ordering & export
    reorder_and_export(master_df, FEATURES_CSV, output_path)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 80)
    return master_df


if __name__ == "__main__":
    out = DEFAULT_OUTPUT_CSV if len(sys.argv) < 2 else sys.argv[1]
    build_master_dataset(BASE_DIR, out)
