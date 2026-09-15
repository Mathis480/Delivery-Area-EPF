#!/usr/bin/env python3
"""
validate_and_calculate_indices.py

Calculates the ID1 and ID3 indices from rolling lead-time transaction windows
(ID_DelArea_DE_{year}.csv) and directly benchmarks them against the official
EPEX Spot index files (2021-2024).

Lead-Time Window Definitions (EPEX SPOT):
- ID1: 60 to 30 minutes prior to contract delivery start (windows: 30to45, 45to60)
- ID3: 180 to 30 minutes prior to contract delivery start (windows: 30to45, 45to60, 60to75, 75to90,
       90to105, 105to120, 120to135, 135to150, 150to165, 165to180)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = '/home/mat/Dokumente/Delivery Area EPF/Data/EPEX Intraday Continuous'
YEARS = [2021, 2022, 2023, 2024]

# Lead-time windows for ID1 and ID3
WINDOWS_ID1 = ['30to45', '45to60']
WINDOWS_ID3 = [
    '30to45', '45to60', '60to75', '75to90', '90to105',
    '105to120', '120to135', '135to150', '150to165', '165to180'
]

def parse_single_index_file(filepath: str) -> pd.DataFrame:
    """
    Parses a single EPEX daily index CSV file for 15-minute contracts.
    """
    try:
        df = pd.read_csv(filepath, comment='#')
        df.columns = [c.strip() for c in df.columns]
        # Filter for 15-minute contracts
        df = df[df['TimeResolution'].astype(str).str.contains('15', na=False)]
        df = df[df['IndexName'].isin(['ID1', 'ID3', 'IDFULL'])]
        return df[['DeliveryStart', 'IndexName', 'IndexPrice', 'IndexVolume']]
    except Exception:
        return None

def extract_official_indices(year: int) -> pd.DataFrame:
    """
    Loads all official EPEX index files for a given year and pivots price and volume.
    """
    idx_dir = os.path.join(BASE_DIR, f"{year} index")
    if not os.path.exists(idx_dir):
        print(f"Warning: Directory {idx_dir} not found.")
        return pd.DataFrame()

    files = glob.glob(os.path.join(idx_dir, "*.csv"))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parse_single_index_file, files))

    valid_results = [r for r in results if r is not None and not r.empty]
    if not valid_results:
        return pd.DataFrame()

    df_all = pd.concat(valid_results, ignore_index=True)
    df_all['DeliveryStart'] = pd.to_datetime(df_all['DeliveryStart'], utc=True)

    # Pivot price and volume
    piv_p = df_all.pivot_table(index='DeliveryStart', columns='IndexName', values='IndexPrice', aggfunc='mean')
    piv_v = df_all.pivot_table(index='DeliveryStart', columns='IndexName', values='IndexVolume', aggfunc='sum')

    piv_p.columns = [f"official_{c.lower()}_price" for c in piv_p.columns]
    piv_v.columns = [f"official_{c.lower()}_volume" for c in piv_v.columns]

    official = pd.concat([piv_p, piv_v], axis=1).reset_index()
    official = official.sort_values('DeliveryStart').drop_duplicates(subset=['DeliveryStart'])
    return official

def calculate_indices_from_delarea(year: int) -> pd.DataFrame:
    """
    Calculates Volume-Weighted Average Prices (VWAP) for ID1 and ID3 from ID_DelArea_DE_{year}.csv.
    """
    calc_path = os.path.join(BASE_DIR, f"ID_DelArea_DE_{year}.csv")
    if not os.path.exists(calc_path):
        print(f"Warning: File {calc_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(calc_path)
    df['DeliveryStart'] = pd.to_datetime(df['DeliveryStart'], utc=True)

    # 1. ID1 (30 to 60 minutes prior to delivery)
    vols_id1 = [f"VWAP_total_volume_{w}" for w in WINDOWS_ID1]
    vwaps_id1 = [f"VWAP_{w}" for w in WINDOWS_ID1]
    tot_vol_id1 = df[vols_id1].sum(axis=1)
    # Avoid division by zero
    vol_sum_id1 = np.where(tot_vol_id1 == 0, np.nan, tot_vol_id1)
    price_id1 = sum(df[p] * df[v] for p, v in zip(vwaps_id1, vols_id1)) / vol_sum_id1

    # 2. ID3 (30 to 180 minutes prior to delivery)
    vols_id3 = [f"VWAP_total_volume_{w}" for w in WINDOWS_ID3]
    vwaps_id3 = [f"VWAP_{w}" for w in WINDOWS_ID3]
    tot_vol_id3 = df[vols_id3].sum(axis=1)
    vol_sum_id3 = np.where(tot_vol_id3 == 0, np.nan, tot_vol_id3)
    price_id3 = sum(df[p] * df[v] for p, v in zip(vwaps_id3, vols_id3)) / vol_sum_id3

    res = pd.DataFrame({
        'DeliveryStart': df['DeliveryStart'],
        'calc_id1_price': np.round(price_id1, 2),
        'calc_id1_volume': np.round(tot_vol_id1, 2),
        'calc_id3_price': np.round(price_id3, 2),
        'calc_id3_volume': np.round(tot_vol_id3, 2)
    })
    return res

def run_validation():
    print("=" * 80)
    print("EPEX Intraday Continuous: ID1 & ID3 Calculation & Validation (2021-2024)")
    print("=" * 80)

    all_years_dfs = []

    for year in YEARS:
        t0 = time.time()
        print(f"\n--- Processing Year {year} ---")

        # 1. Extract official index data
        official_df = extract_official_indices(year)
        if official_df.empty:
            print(f"No official index data available for {year}.")
            continue

        # Save standalone official index data as ID_Index_DE_{year}.csv
        official_out = os.path.join(BASE_DIR, f"ID_Index_DE_{year}.csv")
        official_df.to_csv(official_out, index=False)
        print(f"Official index data saved: {official_out} ({len(official_df)} rows)")

        # 2. Load calculated indices from trades
        calc_df = calculate_indices_from_delarea(year)
        if calc_df.empty:
            print(f"No calculated trades available for {year}.")
            continue

        # 3. Merge datasets on DeliveryStart
        merged = pd.merge(calc_df, official_df, on='DeliveryStart', how='inner')

        # 4. Compute differences (residuals)
        merged['diff_id1_price'] = np.round(merged['calc_id1_price'] - merged['official_id1_price'], 2)
        merged['diff_id3_price'] = np.round(merged['calc_id3_price'] - merged['official_id3_price'], 2)
        merged['diff_id1_volume'] = np.round(merged['calc_id1_volume'] - merged['official_id1_volume'], 2)
        merged['diff_id3_volume'] = np.round(merged['calc_id3_volume'] - merged['official_id3_volume'], 2)

        # Reorganize column order
        cols = [
            'DeliveryStart',
            'calc_id1_price', 'official_id1_price', 'diff_id1_price',
            'calc_id1_volume', 'official_id1_volume', 'diff_id1_volume',
            'calc_id3_price', 'official_id3_price', 'diff_id3_price',
            'calc_id3_volume', 'official_id3_volume', 'diff_id3_volume',
            'official_idfull_price', 'official_idfull_volume'
        ]
        existing_cols = [c for c in cols if c in merged.columns]
        merged = merged[existing_cols]

        out_csv = os.path.join(BASE_DIR, f"ID1_ID3_Validation_DE_{year}.csv")
        merged.to_csv(out_csv, index=False)
        print(f"Validation CSV saved: {out_csv} ({len(merged)} rows in {time.time()-t0:.2f}s)")

        # Print statistics
        valid_id1 = merged.dropna(subset=['calc_id1_price', 'official_id1_price'])
        valid_id3 = merged.dropna(subset=['calc_id3_price', 'official_id3_price'])

        mae_id1 = valid_id1['diff_id1_price'].abs().mean()
        corr_id1 = valid_id1['calc_id1_price'].corr(valid_id1['official_id1_price'])
        mae_id3 = valid_id3['diff_id3_price'].abs().mean()
        corr_id3 = valid_id3['calc_id3_price'].corr(valid_id3['official_id3_price'])

        print(f"Statistics for {year}:")
        print(f"  * ID1 (60-30m):  MAE = {mae_id1:.4f} EUR/MWh | Pearson Correlation = {corr_id1:.6f}")
        print(f"  * ID3 (180-30m): MAE = {mae_id3:.4f} EUR/MWh | Pearson Correlation = {corr_id3:.6f}")

        all_years_dfs.append(merged)

    if all_years_dfs:
        full_df = pd.concat(all_years_dfs, ignore_index=True)
        full_df = full_df.sort_values('DeliveryStart').drop_duplicates(subset=['DeliveryStart'])
        full_out = os.path.join(BASE_DIR, "ID1_ID3_Validation_DE_2021_2024.csv")
        full_df.to_csv(full_out, index=False)
        print(f"\n-> Full Multi-Year Validation CSV saved: {full_out} ({len(full_df)} rows across 2021-2024)")

        # Overall multi-year evaluation
        v1 = full_df.dropna(subset=['calc_id1_price', 'official_id1_price'])
        v3 = full_df.dropna(subset=['calc_id3_price', 'official_id3_price'])
        print("\n" + "=" * 80)
        print(f"OVERALL EVALUATION (2021-2024, {len(full_df)} quarter-hour intervals):")
        print(f"  * ID1 MAE: {v1['diff_id1_price'].abs().mean():.4f} EUR/MWh | Pearson Correlation: {v1['calc_id1_price'].corr(v1['official_id1_price']):.6f}")
        print(f"  * ID3 MAE: {v3['diff_id3_price'].abs().mean():.4f} EUR/MWh | Pearson Correlation: {v3['calc_id3_price'].corr(v3['official_id3_price']):.6f}")
        print("=" * 80)

if __name__ == '__main__':
    run_validation()
