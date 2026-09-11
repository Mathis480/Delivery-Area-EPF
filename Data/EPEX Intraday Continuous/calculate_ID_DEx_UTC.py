"""
This script processes raw EPEX Intraday Continuous trade and index CSV files (2021-2024)
for Germany (national market area 'DE') as well as individual transmission system 
operator (TSO) control zones ('DE1' TransnetBW, 'DE2' Amprion, 'DE3' TenneT, 'DE4' 50Hertz).

Content:
-------------
1. Pure UTC Time Handling
2. Lead-Time Window Aggregation:
   - Calculates rolling Volume-Weighted Average Prices (VWAP), traded volumes, and trade counts 
     across 25 discrete time intervals prior to contract delivery (e.g., 345-360m down to 0-15m + label col 0-30) -> 25 columns.
3. Cross-Border & Zonal Flow Balances:
   - Identifies whether a trade is internal ('self'), an import, or an export.
   - Computes net trading balances overall and specifically against each German TSO zone (DE1-DE4).
4. Robust Filtering & Quality Control:
   - Filters for 15-minute quarter-hour contracts (ignores hourly and half-hourly products).
   - Eliminates self-trades ('SelfTrade' == 'N').
   - Deduplicates trade records, so buyer and seller legs are not double-counted. Trades with only one trade side are removed.
5. Parallel Processing:
   - ProcessPoolExecutor across multi-core CPUs for batch file processing.
   - Includes an interactive step debugger mode for auditing individual trades and verifying data.
"""

import os
import warnings
import pandas as pd
import datetime as dt
import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ==============================================================================
# DEBUG / STEPPER CONFIGURATION
# Set DEBUG_MODE = True to enable interactive step-by-step terminal inspection.
# Set DEBUG_MODE = False for fast, parallel production execution.
# ==============================================================================
DEBUG_MODE = False
DEBUG_ONLY_NON_EMPTY = False     # If True, stepper only pauses on intervals with >= 1 trade
DEBUG_FILTER_DELIVERY = None     # Filter by specific timestamp (e.g., "2021-10-01 04:00:00") or None
# ==============================================================================
SKIP_DELIVERY_START = True       # Global flag used to skip remaining intervals of current contract

# Lead-time window names: ordered chronologically from earliest (345to360m prior) down to gate closure (0to15m, 0to30m)
LEAD_TIME_WINDOW_NAMES = [
    '0to30', '0to15', '15to30', '30to45', '45to60', '60to75', '75to90', '90to105',
    '105to120', '120to135', '135to150', '150to165', '165to180', '180to195', '195to210',
    '210to225', '225to240', '240to255', '255to270', '270to285', '285to300', '300to315',
    '315to330', '330to345', '345to360'
][::-1]

# Corresponding lead-time intervals [min_lead_time_before_delivery, max_lead_time_before_delivery)
LEAD_TIME_INTERVALS = [
    (0, 30), (0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 105),
    (105, 120), (120, 135), (135, 150), (150, 165), (165, 180), (180, 195), (195, 210),
    (210, 225), (225, 240), (240, 255), (255, 270), (270, 285), (285, 300), (300, 315),
    (315, 330), (330, 345), (345, 360)
][::-1]

def calculate_vwap(df_subset: pd.DataFrame) -> float:
    """
    Calculate the Volume-Weighted Average Price (VWAP) for a subset of transactions.
    Returns NaN if total volume is zero or no trades exist.
    """
    vol = df_subset['Volume'].sum()
    if vol == 0:
        return np.nan
    else:
        return np.round((df_subset['Price'] * df_subset['Volume']).sum() / vol, 2)


def calculate_total_volume(df_subset: pd.DataFrame) -> float:
    """Calculate the total traded volume in MWh for a subset of transactions."""
    try:
        result = np.round(df_subset['Volume'].sum(), 2)
    except Exception:
        result = np.nan
    return result


def calculate_total_trades(df_subset: pd.DataFrame) -> float:
    """Count the total number of executed trades for a subset of transactions."""
    try:
        result = df_subset['Volume'].count() # Here, just Volume has been taken (could be anything)
    except Exception:
        result = np.nan
    return result


def calculate_idbalance(df_subset: pd.DataFrame) -> float:
    """
    Calculate the net commercial balance across all cross-border trades.
    
    Convention:
    - Imports add positive volume to the local area balance (+Volume).
    - Exports subtract volume from the local area balance (-Volume).
    """
    if df_subset['Volume'].sum() == 0:
        return np.nan
    else:
        temp_df_balance = df_subset[(df_subset['TradeTyp'] == 'export') | (df_subset['TradeTyp'] == 'import')].copy()
        temp_df_balance['Volume'] = np.where(temp_df_balance['TradeTyp'] == 'export', -1 * temp_df_balance['Volume'], temp_df_balance['Volume'])
        result = np.round(temp_df_balance['Volume'].sum(), 3)
    return result


def calculate_idbalance_to_zone(df_subset: pd.DataFrame, zone_code: str) -> float:
    """
    Calculate the net commercial trade balance specifically with a designated German TSO zone.
    
    Parameters:
    -----------
    df_subset : pd.DataFrame
        Trade records for the window.
    zone_code : str
        Target counterpart area identifier (e.g., 'DE1', 'DE2', 'DE3', 'DE4').
    """
    if df_subset['Volume'].sum() == 0:
        return np.nan
    else:
        temp_df_balance = df_subset[(df_subset['TradeTyp'] == 'export') | (df_subset['TradeTyp'] == 'import')].copy()
        temp_df_balance = temp_df_balance[temp_df_balance['DeliveryArea'] == zone_code]
        temp_df_balance['Volume'] = np.where(temp_df_balance['TradeTyp'] == 'export', -1 * temp_df_balance['Volume'], temp_df_balance['Volume'])
        result = np.round(temp_df_balance['Volume'].sum(), 3)
    return result


def calculate_idbalance_to_self(df_subset: pd.DataFrame) -> float:
    """Calculate the total volume of purely internal trades within the same delivery zone."""
    if df_subset['Volume'].sum() == 0:
        return np.nan
    else:
        temp_df_balance = df_subset[df_subset['TradeTyp'] == 'self'].copy()
        result = np.round(temp_df_balance['Volume'].sum(), 3)
    return result


def calculate_metrics(
    df: pd.DataFrame, 
    delivery_start: pd.Timestamp, 
    window_metrics_list: list, 
    window_idx: int, 
    interval_name: str, 
    window_names: list
) -> list:
    """
    Compute summary metrics for a given lead-time window.
    
    If no trades occurred in the window:
    - VWAP is forward-filled from the immediately preceding (earlier) window in window_metrics_list.
    - Traded volumes, counts, and flow balances are set to 0.
    """
    temp_list = []
    if df.empty:
        if len(window_metrics_list) == 0:
            # First window (345-360m prior) has no prior trade to forward-fill from -> NaN
            temp_list.append({
                'VWAP_' + interval_name: np.nan,
                'VWAP_total_volume_' + interval_name: 0,
                'VWAP_total_trades_' + interval_name: 0,
                'id_balance_' + interval_name: 0,
                'id_balance_to_DE1_' + interval_name: 0,
                'id_balance_to_DE2_' + interval_name: 0,
                'id_balance_to_DE3_' + interval_name: 0,
                'id_balance_to_DE4_' + interval_name: 0,
                'id_trade_to_self_' + interval_name: 0
            })
        else:
            # Forward-fill previous known VWAP from earlier lead-time window
            prev_price = window_metrics_list[-1].get(f'VWAP_{window_names[window_idx-1]}', np.nan)
            temp_list.append({
                'VWAP_' + interval_name: prev_price,
                'VWAP_total_volume_' + interval_name: 0,
                'VWAP_total_trades_' + interval_name: 0,
                'id_balance_' + interval_name: 0,
                'id_balance_to_DE1_' + interval_name: 0,
                'id_balance_to_DE2_' + interval_name: 0,
                'id_balance_to_DE3_' + interval_name: 0,
                'id_balance_to_DE4_' + interval_name: 0,
                'id_trade_to_self_' + interval_name: 0
            })
    else:
        # Trades exist: calculate exact weighted metrics
        temp_list.append({
            'VWAP_' + interval_name: calculate_vwap(df),
            'VWAP_total_volume_' + interval_name: calculate_total_volume(df),
            'VWAP_total_trades_' + interval_name: calculate_total_trades(df),
            'id_balance_' + interval_name: calculate_idbalance(df),
            'id_balance_to_DE1_' + interval_name: calculate_idbalance_to_zone(df, 'DE1'),
            'id_balance_to_DE2_' + interval_name: calculate_idbalance_to_zone(df, 'DE2'),
            'id_balance_to_DE3_' + interval_name: calculate_idbalance_to_zone(df, 'DE3'),
            'id_balance_to_DE4_' + interval_name: calculate_idbalance_to_zone(df, 'DE4'),
            'id_trade_to_self_' + interval_name: calculate_idbalance_to_self(df)
        })
    return temp_list


def get_trade_type(delivery_area: str, trade_id: int, target_area: str, side: str) -> str:
    """
    Classify a trade record into 'self', 'export', or 'import' relative to the analyzed area.
    
    Parameters:
    -----------
    delivery_area : str
        Delivery area of this trade record.
    trade_id : int
        Unique EPEX trade ID.
    target_area : str
        Target delivery area being analyzed (e.g., 'DE' or 'DE1'..'DE4').
    side : str
        Order side ('BUY' or 'SELL').
        
    Returns:
    --------
    'self'   : Counterparty is within the same target area.
    'export' : Target area sells power to an external area (Side == 'BUY' on counterpart leg).
    'import' : Target area buys power from an external area (Side == 'SELL' on counterpart leg).
    """
    if target_area == 'DE':
        is_target = str(delivery_area).startswith('DE')
    else:
        is_target = (delivery_area == target_area)
        
    if not is_target:
        if side == "BUY":
            return "export"
        elif side == "SELL":
            return "import"
    return "self"


def detect_skiprows(file_path: str) -> int:
    """
    Inspect the first two lines of a CSV to handle header metadata offsets.
    Returns 0 if standard header is on row 1, or 1 if metadata precedes headers.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            l1 = f.readline()
            l2 = f.readline()
        keywords = ['TradeId', 'DeliveryArea', 'DeliveryStart', 'Price', 'Volume', 'IndexName', 'TimeResolution']
        if any(k.lower() in l1.lower() for k in keywords):
            return 0
        if any(k.lower() in l2.lower() for k in keywords):
            return 1
    except Exception:
        pass
    return 0


def process_index_files(
    year: int, 
    target_area: str, 
    file_list: list, 
    source_dir: str, 
    output_dir: str
) -> None:
    """
    Process official EPEX Continuous Index files (e.g. ID3, ID1 indices published directly by EPEX).
    Pivots index prices and volumes by DeliveryStart and saves to CSV.
    """
    print(f"-> Processing EPEX Index files (UTC) for {target_area} {year}...")
    dfs = []
    for filename in tqdm.tqdm(file_list, desc=f"{target_area}_{year}_index"):
        file_path = os.path.join(source_dir, filename)
        try:
            sr = detect_skiprows(file_path)
            df = pd.read_csv(file_path, skiprows=sr)
            df.columns = [c.strip() for c in df.columns]
            # Exclude auxiliary upper/lower threshold boundaries, retaining actual index values
            if 'IndexName' in df.columns and 'DeliveryStart' in df.columns:
                df = df[~df['IndexName'].astype(str).str.lower().str.contains('upper|lower')].copy()
                dfs.append(df)
        except Exception as e:
            print(f"Error reading index file {filename}: {e}")
            
    if not dfs:
        print(f"WARNING: No index data extracted for {target_area} {year}.")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['DeliveryStart'] = pd.to_datetime(full_df['DeliveryStart'], utc=True)
    
    # Filter strictly for 15-minute quarter-hourly indices
    if 'TimeResolution' in full_df.columns:
        df_15 = full_df[full_df['TimeResolution'].astype(str).str.contains('15', na=False)].copy()
        if df_15.empty:
            df_15 = full_df.copy()
    else:
        df_15 = full_df.copy()
        
    piv_price = df_15.pivot_table(index='DeliveryStart', columns='IndexName', values='IndexPrice', aggfunc='mean')
    piv_vol = df_15.pivot_table(index='DeliveryStart', columns='IndexName', values='IndexVolume', aggfunc='sum')
    
    prefix = target_area.lower()
    piv_price.columns = [f"{prefix}_id_{str(col).lower()}_price" for col in piv_price.columns]
    piv_vol.columns = [f"{prefix}_id_{str(col).lower()}_volume" for col in piv_vol.columns]
    
    merged = pd.concat([piv_price, piv_vol], axis=1).reset_index()
    merged = merged.sort_values('DeliveryStart').drop_duplicates(subset=['DeliveryStart'])
    
    output_file = os.path.join(output_dir, f"ID_DelArea_{target_area}_{year}.csv")
    merged.to_csv(output_file, index=False)
    print(f"Completed index processing for {target_area}_{year}: Saved to {output_file} ({len(merged)} rows)")


def process_single_trade_file(args: tuple) -> pd.DataFrame:
    """
    Worker function executed in parallel for a single daily trade file.
    
    Processing Steps:
    1. Read CSV and strip whitespace from headers.
    2. Filter strictly for 15-minute contracts (duration == 15 min & Product name).
    3. Filter out wash/self-trades ('SelfTrade' == 'N').
    4. Isolate trades where at least one party belongs to the target delivery area.
    5. Classify trade direction ('self', 'export', 'import') and deduplicate TradeId.
    6. Compute lead-time difference in minutes: TimeDiff = DeliveryStart - ExecutionTime (UTC).
    7. Partition trades into 25 discrete lead-time windows and calculate metrics.
    """
    file_path, target_area = args
    try:  
        sr = detect_skiprows(file_path)
        df_trades = pd.read_csv(file_path, skiprows=sr, delimiter=',')

        if df_trades.empty:
            return None

        df_trades.columns = [str(c).strip() for c in df_trades.columns]

        # 1. Filter by Delivery Duration: strictly 15 minutes (900 seconds)
        if 'DeliveryStart' in df_trades.columns and 'DeliveryEnd' in df_trades.columns:
            ds_dt = pd.to_datetime(df_trades['DeliveryStart'], utc=True, errors='coerce')
            de_dt = pd.to_datetime(df_trades['DeliveryEnd'], utc=True, errors='coerce')
            duration_min = (de_dt - ds_dt).dt.total_seconds() / 60.0
            df_trades = df_trades[duration_min == 15]

        # 2. Filter by Product name: must be 15-min contract (exclude hourly/block contracts)
        if 'Product' in df_trades.columns:
            df_trades = df_trades[df_trades['Product'].isin(['Intraday_Quarter_Hour_Power', 'XBID_Quarter_Hour_Power'])]

        # 3. Filter by SelfTrade: discard internal wash trading ('N' = genuine trade)
        if 'SelfTrade' in df_trades.columns:
            df_trades['SelfTrade'] = df_trades['SelfTrade'].astype(str).str.strip()
            df_trades = df_trades[df_trades['SelfTrade'] == 'N']

        # Drop non-essential metadata columns to reduce memory consumption
        cols_to_drop = [c for c in ['RemoteTradeId', 'DeliveryEnd', 'UserDefinedBlock', 'Currency', 'OrderID', 'TradePhase', 'VolumeUnit', 'Product'] if c in df_trades.columns]
        df_trades = df_trades.drop(columns=cols_to_drop)

        # 4. Filter for trades touching the target delivery area
        if target_area == 'DE':
            trade_id_df = df_trades.loc[df_trades['DeliveryArea'].astype(str).str.startswith('DE')]
        else:
            trade_id_df = df_trades.loc[df_trades['DeliveryArea'] == target_area]

        target_trade_ids = trade_id_df['TradeId'].tolist()
        df_trades = df_trades[df_trades['TradeId'].isin(target_trade_ids)]

        if df_trades.empty:
            return None

        # 5. Classify trade type (self, import, export)
        df_trades['TradeTyp'] = df_trades.apply(
            lambda row: get_trade_type(row['DeliveryArea'], row['TradeId'], target_area, row.get('Side', 'BUY')), 
            axis=1
        )

        # Deduplicate trades: sort so cross-border legs are prioritized and retain one unique record per TradeId
        df_trades = df_trades.sort_values(by=['TradeId', 'TradeTyp'], ascending=True)
        df_trades = df_trades.drop_duplicates(subset='TradeId', keep='first')

        # 6. Parse timestamps strictly in UTC and compute lead-time (TimeDiff) in minutes
        df_trades['DeliveryStart'] = pd.to_datetime(df_trades['DeliveryStart'], utc=True)
        df_trades['ExecutionTime'] = pd.to_datetime(df_trades['ExecutionTime'], utc=True)
        df_trades['TimeDiff'] = (df_trades['DeliveryStart'] - df_trades['ExecutionTime']).dt.total_seconds() / 60.0

        df_trades = df_trades.sort_values(by=['DeliveryStart', 'ExecutionTime'])
        df_trades = df_trades.reset_index(drop=True)

        delivery_starts = df_trades['DeliveryStart'].unique()

        # 7. Partition trades into discrete lead-time bins
        df_binned_trades = {}
        for i, interval_name in enumerate(LEAD_TIME_WINDOW_NAMES):
            lead_from, lead_to = LEAD_TIME_INTERVALS[i]
            df_binned_trades[interval_name] = df_trades[
                (df_trades['TimeDiff'] >= lead_from) & (df_trades['TimeDiff'] < lead_to)
            ]

        # Aggregate metrics across all delivery quarter-hours
        delivery_records = []
        for delivery_start in delivery_starts:
            global SKIP_DELIVERY_START
            SKIP_DELIVERY_START = False

            window_metrics_list = []
            for j, interval_name in enumerate(LEAD_TIME_WINDOW_NAMES):
                df_sub = df_binned_trades[interval_name][df_binned_trades[interval_name]['DeliveryStart'] == delivery_start]
                lead_from, lead_to = LEAD_TIME_INTERVALS[j]
                
                if DEBUG_MODE:
                    run_step_debugger(df_sub, delivery_start, interval_name, lead_from, lead_to)

                metrics = calculate_metrics(df_sub, delivery_start, window_metrics_list, j, interval_name, LEAD_TIME_WINDOW_NAMES)
                window_metrics_list.extend(metrics)
                
            flat_dict = {k: v for d in window_metrics_list for k, v in d.items()}
            flat_dict['DeliveryStart'] = delivery_start
            delivery_records.append(flat_dict)

        if delivery_records:
            return pd.DataFrame(delivery_records)
            
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception as e:
        print(f"Error processing trade file {file_path}: {e}")
    return None


def process_trade_files(
    year: int, 
    target_area: str, 
    file_list: list, 
    source_dir: str, 
    output_dir: str
) -> None:
    """
    Coordinate parallel batch processing of daily continuous trade files for a given year and area.
    Concatenates individual daily results into a single clean annual CSV.
    """
    file_args = [(os.path.join(source_dir, filename), target_area) for filename in file_list]
    df_naive_list = []

    if DEBUG_MODE:
        print(f"\n-> [DEBUG MODE ACTIVE] Processing EPEX trade files sequentially for {target_area} {year}...")
        for arg in tqdm.tqdm(file_args, desc=f"{target_area}_{year}_trades"):
            try:
                res = process_single_trade_file(arg)
                if res is not None and not res.empty:
                    df_naive_list.append(res)
            except KeyboardInterrupt:
                print("\n[DEBUG] Debugging terminated by user.")
                break
    else:
        num_workers = max(1, (os.cpu_count() or 4) - 1)
        print(f"-> Processing EPEX continuous trade files (UTC) for {target_area} {year} ({len(file_list)} files across {num_workers} CPU cores)...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_trade_file, arg): arg for arg in file_args}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"{target_area}_{year}_trades"):
                res = future.result()
                if res is not None and not res.empty:
                    df_naive_list.append(res)

    if df_naive_list:
        df_naive = pd.concat(df_naive_list, ignore_index=True, axis=0)
        col = ["DeliveryStart"]
        df_naive = df_naive[col + [x for x in df_naive.columns if x not in col]]
        df_naive = df_naive.sort_values(by='DeliveryStart').drop_duplicates(subset=['DeliveryStart'])
        
        output_file = os.path.join(output_dir, f'ID_DelArea_{target_area}_{year}.csv')
        df_naive.to_csv(output_file, index=False)
        print(f"Completed trade processing for {target_area}_{year}: Saved to {output_file} ({len(df_naive)} rows)")
    else:
        print(f"No trade data calculated for {target_area} {year}.")


def calculate_id_da_utc(
    year: int, 
    target_area: str, 
    source_dir: str = None, 
    output_dir: str = None
) -> None:
    """
    Main entry point for processing EPEX data for a specific year and market/TSO area.
    Automatically detects whether the source directory contains raw trades or precalculated indices.
    """
    pd.options.mode.chained_assignment = None

    if output_dir is None:
        output_dir = '/home/mat/Schreibtisch/INREC/Data/Raw'
    os.makedirs(output_dir, exist_ok=True)

    if source_dir is None:
        source_dir = f'/media/mat/VERBATIM SD/EPEX/germany/Intraday Continuous/EOD/Historical/Transactions/Continuous_Trades-DE-{year}'

    if not os.path.exists(source_dir):
        print(f"WARNING: Directory not found for year {year}: {source_dir}")
        return

    # Filter for valid EPEX trade or index CSV files
    file_list = [
        filename for filename in os.listdir(source_dir) 
        if (filename.startswith('Continuous_Trades') or filename.startswith('Continuous_Index') or filename.startswith(f'Continuous_Trades-DE-{year}'))
        and filename.endswith('.csv') 
        and not filename.startswith('.')
    ]
    file_list.sort()

    if not file_list:
        print(f"No matching CSV files found in {source_dir} for year {year}.")
        return

    # Inspect sample row to detect file type
    first_file = os.path.join(source_dir, file_list[0])
    sr = detect_skiprows(first_file)
    df_sample = pd.read_csv(first_file, skiprows=sr, nrows=3)
    cols = [str(c).strip() for c in df_sample.columns]

    if any('IndexName' in c for c in cols):
        process_index_files(year, target_area, file_list, source_dir, output_dir)
    elif any('TradeId' in c or 'DeliveryArea' in c for c in cols):
        process_trade_files(year, target_area, file_list, source_dir, output_dir)
    else:
        print(f"WARNING: File format in {source_dir} could not be identified. Columns found: {cols}")

def run_step_debugger(
    df_sub: pd.DataFrame, 
    delivery_start: pd.Timestamp, 
    interval_name: str, 
    lead_time_from_min: int, 
    lead_time_to_min: int
) -> None:
    """
    Debugger to inspect individual trades and calculated metrics.

    df_sub : pd.DataFrame
        Subset of trades falling within the current lead-time window.
    delivery_start : pd.Timestamp
        Delivery start timestamp (UTC) of the quarter-hour contract.
    interval_name : str
        Interval name (e.g., '15to30', '0to15') for lead-time window.
    lead_time_from_min : int
        Lower lead-time boundary in minutes before delivery.
    lead_time_to_min : int
        Upper lead-time boundary in minutes before delivery.
    """
    global SKIP_DELIVERY_START

    if SKIP_DELIVERY_START:
        return

    # Skip contracts that do not match the specified debug filter
    if DEBUG_FILTER_DELIVERY is not None and str(DEBUG_FILTER_DELIVERY) not in str(delivery_start):
        return

    # Skip intervals without trading activity if configured
    if DEBUG_ONLY_NON_EMPTY and df_sub.empty:
        return

    price = calculate_vwap(df_sub)
    vol = calculate_total_volume(df_sub)
    trades_cnt = calculate_total_trades(df_sub)

    print("\n" + "=" * 85)
    print(f"DeliveryStart: {delivery_start} | Interval: '{interval_name}' ({lead_time_from_min} to {lead_time_to_min} min prior)")
    print("=" * 85)

    cols_display = [c for c in ['TradeId', 'ExecutionTime', 'Price', 'Volume', 'Side', 'DeliveryArea', 'SelfTrade', 'TradeTyp'] if c in df_sub.columns]

    print(f"\n--- CONTRIBUTING TRADES (Count: {len(df_sub)}) ---")
    if not df_sub.empty:
        print(df_sub[cols_display].to_string(index=False))
        if 'SelfTrade' in df_sub.columns:
            st_counts = df_sub['SelfTrade'].value_counts().to_dict()
            print(f"\n   -> SelfTrade Value Distribution: {st_counts}")
    else:
        print("   (No trades recorded in this interval)")

    print(f"\n--- CALCULATED METRICS FOR '{interval_name}' ---")
    print(f"   VWAP (Weighted Price): {price} EUR/MWh")
    print(f"   Total Traded Volume:  {vol} MWh")
    print(f"   Trade Count:          {trades_cnt}")
    print("=" * 85)

    try:
        ans = input(" [ENTER] = Next Step | [s] = Skip This DeliveryStart | [q] = Quit Debugger: ").strip().lower()
        if ans == 's':
            SKIP_DELIVERY_START = True
        elif ans == 'q':
            raise KeyboardInterrupt("Debugger terminated by user.")
    except (EOFError, KeyboardInterrupt):
        raise

if __name__ == '__main__':
    # Configuration for standalone execution
    years = [2022]
    areas = ['DE1', 'DE2', 'DE3', 'DE4', 'DE']
    out_dir = '/home/mat/Schreibtisch/INREC/Data/Raw'
    
    print("Starting recomputation for years 2021-2024 across all TSO zones in EPEX UTC time (Multiprocessing)...")
    for year in years:
        for area in areas:
            calculate_id_da_utc(year=year, target_area=area, output_dir=out_dir)
