#%%
"""
Pre-aggregates raw EPEX Intraday Continuous sub-second tick trades (2021-2024)
into 15-minute lead-time windows (VWAP, volume, trade count, net/bilateral/external
balances, intra-area volume)
for German TSO control areas (DE1-DE4) and national aggregate (DE) in pure UTC.

See README.md for full methodology, window definitions, and schema.
"""

import os
import warnings
import pandas as pd
import datetime as dt
import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# Lead-time windows: ordered chronologically from earliest (345to360m prior) down to gate closure (0to15m, *0to30m - label!*)
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
    vol = df_subset['Volume'].sum()
    if vol == 0:
        return np.nan
    return np.round((df_subset['Price'] * df_subset['Volume']).sum() / vol, 2)


def calculate_total_volume(df_subset: pd.DataFrame) -> float:
    try:
        return np.round(df_subset['Volume'].sum(), 2)
    except Exception:
        return np.nan


def calculate_total_trades(df_subset: pd.DataFrame) -> int:
    try:
        return len(df_subset)
    except Exception:
        return np.nan


IMPORT_TYPES = ['import', 'ext_import']
EXPORT_TYPES = ['export', 'ext_export']


def _net_balance(df_subset: pd.DataFrame, mask: pd.Series) -> float:
    temp_df = df_subset[mask]
    signed = np.where(temp_df['TradeTyp'].isin(EXPORT_TYPES), -temp_df['Volume'], temp_df['Volume'])
    return np.round(signed.sum(), 3)


def calculate_idbalance(df_subset: pd.DataFrame) -> float:
    """Net trade balance (imports positive, exports negative) in MWh: German areas plus external counterparties."""
    if df_subset['Volume'].sum() == 0:
        return np.nan
    return _net_balance(df_subset, df_subset['TradeTyp'].isin(IMPORT_TYPES + EXPORT_TYPES))


def calculate_idbalance_to_zone(df_subset: pd.DataFrame, zone_code: str) -> float:
    """Bilateral net trade balance with a specific German delivery area."""
    if df_subset['Volume'].sum() == 0:
        return np.nan
    return _net_balance(df_subset, df_subset['TradeTyp'].isin(['export', 'import']) & (df_subset['DeliveryArea'] == zone_code))


def calculate_idbalance_ext(df_subset: pd.DataFrame) -> float:
    """Net trade balance with external counterparties (single-leg XBID trades, area outside the EPEX DE order book)."""
    if df_subset['Volume'].sum() == 0:
        return np.nan
    return _net_balance(df_subset, df_subset['TradeTyp'].isin(['ext_export', 'ext_import']))


def calculate_trade_intra(df_subset: pd.DataFrame) -> float:
    """Volume traded between two parties within the same delivery area."""
    if df_subset['Volume'].sum() == 0:
        return np.nan
    temp_df = df_subset[df_subset['TradeTyp'] == 'intra']
    return np.round(temp_df['Volume'].sum(), 3)


def calculate_metrics(
    df: pd.DataFrame, 
    window_metrics_list: list, 
    window_idx: int, 
    interval_name: str, 
    window_names: list
) -> dict:
    """Compute 10 window metrics; forward-fills VWAP and zeroes volumes if no trades occurred."""
    if df.empty:
        prev_vwap = window_metrics_list[-1].get(f'VWAP_{window_names[window_idx-1]}', np.nan) if window_metrics_list else np.nan
        return {
            f'VWAP_{interval_name}': prev_vwap,
            f'VWAP_total_volume_{interval_name}': 0,
            f'VWAP_total_trades_{interval_name}': 0,
            f'id_balance_{interval_name}': 0,
            f'id_balance_to_DE1_{interval_name}': 0,
            f'id_balance_to_DE2_{interval_name}': 0,
            f'id_balance_to_DE3_{interval_name}': 0,
            f'id_balance_to_DE4_{interval_name}': 0,
            f'id_balance_ext_{interval_name}': 0,
            f'id_trade_intra_{interval_name}': 0,
        }

    return {
        f'VWAP_{interval_name}': calculate_vwap(df),
        f'VWAP_total_volume_{interval_name}': calculate_total_volume(df),
        f'VWAP_total_trades_{interval_name}': calculate_total_trades(df),
        f'id_balance_{interval_name}': calculate_idbalance(df),
        f'id_balance_to_DE1_{interval_name}': calculate_idbalance_to_zone(df, 'DE1'),
        f'id_balance_to_DE2_{interval_name}': calculate_idbalance_to_zone(df, 'DE2'),
        f'id_balance_to_DE3_{interval_name}': calculate_idbalance_to_zone(df, 'DE3'),
        f'id_balance_to_DE4_{interval_name}': calculate_idbalance_to_zone(df, 'DE4'),
        f'id_balance_ext_{interval_name}': calculate_idbalance_ext(df),
        f'id_trade_intra_{interval_name}': calculate_trade_intra(df),
    }


def classify_trade_direction(delivery_area: str, side: str, target_area: str, n_legs: int) -> str:
    """Classify a trade leg relative to the target area ('intra', 'export', 'import', 'ext_export', 'ext_import').

    Two-leg trades: both legs are in the EPEX DE order book; the leg outside the target area is the
    counterparty, so its side is inverted. Single-leg trades (XBID with RemoteTradeId): the counterparty
    is outside the EPEX DE order book and the only leg is the target area's own leg.
    """
    if n_legs == 1:
        return 'ext_import' if side == 'BUY' else 'ext_export'  # Own leg buys -> target area imports
    is_same_area = str(delivery_area).startswith('DE') if target_area == 'DE' else (delivery_area == target_area)
    if is_same_area:
        return 'intra'
    elif side == 'BUY':
        return 'export'  # Counterparty buys -> target area exports
    else:
        return 'import'  # Counterparty sells -> target area imports


def process_single_trade_file(args: tuple) -> pd.DataFrame:
    file_path, target_area = args
    try:  
        # 1. Read CSV and strip whitespace from headers.
        df_trades = pd.read_csv(file_path, comment='#', delimiter=',')

        if df_trades.empty:
            return None

        df_trades.columns = [str(c).strip() for c in df_trades.columns]

        # 2. Filter by Product: strictly 15-min contracts (quarter-hour power)
        if 'Product' in df_trades.columns:
            df_trades = df_trades[df_trades['Product'].isin(['Intraday_Quarter_Hour_Power', 'XBID_Quarter_Hour_Power'])]

        # 3. Filter by SelfTrade: discard internal wash trading ('N' = genuine trade)
        if 'SelfTrade' in df_trades.columns:
            df_trades = df_trades[df_trades['SelfTrade'].astype(str).str.strip() == 'N']

        # 3.1 Drop non-essential metadata columns
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

        # 5. Classify trade direction (two-leg trades from the counterparty leg, single-leg trades from the own leg)
        n_legs = df_trades.groupby('TradeId')['TradeId'].transform('size')
        if (n_legs > 2).any():
            raise ValueError(f"TradeIds with more than two legs in {file_path}")
        df_trades['TradeTyp'] = [
            classify_trade_direction(area, side, target_area, n)
            for area, side, n in zip(df_trades['DeliveryArea'], df_trades['Side'], n_legs)
        ]

        # 5.1 Deduplicate trades: sort so inter-area legs ('export'/'import') precede 'intra' and retain one record per TradeId
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

        # 7.1 Aggregate metrics across all delivery quarter-hours
        delivery_records = []
        for delivery_start in delivery_starts:
            record = {'DeliveryStart': delivery_start}
            window_metrics_list = []
            for j, interval_name in enumerate(LEAD_TIME_WINDOW_NAMES):
                df_sub = df_binned_trades[interval_name][df_binned_trades[interval_name]['DeliveryStart'] == delivery_start]
                metrics = calculate_metrics(df_sub, window_metrics_list, j, interval_name, LEAD_TIME_WINDOW_NAMES)
                window_metrics_list.append(metrics)
                record.update(metrics)
            delivery_records.append(record)

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
    """Parallel processing of all daily trade files for a year and export CSV."""
    file_args = [(os.path.join(source_dir, filename), target_area) for filename in file_list]
    num_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"-> Processing EPEX continuous trade files (UTC) for {target_area} {year} ({len(file_list)} files across {num_workers} CPU cores)...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm.tqdm(
            executor.map(process_single_trade_file, file_args),
            total=len(file_args),
            desc=f"{target_area}_{year}_trades"
        ))
    df_naive_list = [df for df in results if df is not None and not df.empty]

    # 3. Consolidate, sort, and save final annual time series
    if df_naive_list:
        # Concatenate all daily results into a single annual DataFrame
        df_naive = pd.concat(df_naive_list, ignore_index=True, axis=0)
        # Ensure DeliveryStart is the primary first column
        col = ["DeliveryStart"]
        df_naive = df_naive[col + [x for x in df_naive.columns if x not in col]]
        df_naive = df_naive.sort_values(by='DeliveryStart')
        
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
    pd.options.mode.chained_assignment = None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    if source_dir is None:
        source_dir = os.path.join(base_dir, f'{year}')

    if not os.path.exists(source_dir):
        print(f"Directory not found for year {year}: {source_dir}")
        return

    # Filter for valid EPEX trade CSV files
    file_list = [
        filename for filename in os.listdir(source_dir) 
        if (filename.startswith('Continuous_Trades') or filename.startswith(f'Continuous_Trades-DE-{year}'))
        and filename.endswith('.csv') 
        and not filename.startswith('.')
    ]
    file_list.sort()

    if not file_list:
        print(f"No matching CSV files found in {source_dir} for year {year}.")
        return

    process_trade_files(year, target_area, file_list, source_dir, output_dir)



#==================== MAIN =======================

if __name__ == '__main__':
    years = [2021, 2022, 2023, 2024]
    areas = ['DE1', 'DE2', 'DE3', 'DE4', 'DE']
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting recomputation for years 2021-2024 across all TSO zones in EPEX UTC time (Multiprocessing)...")
    for year in years:
        for area in areas:
            calculate_id_da_utc(year=year, target_area=area, output_dir=script_dir)
