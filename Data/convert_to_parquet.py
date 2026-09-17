"""
One-shot conversion: master_dataset_2021_2024.csv -> master_dataset.parquet

Reads the canonical master CSV from the Delivery Area EPF project and writes
it as a Snappy-compressed Parquet file for fast DuckDB-backed ad-hoc slicing.
Only needs to be run once (or after the master CSV is regenerated).
"""
import os
import time
import pandas as pd

MASTER_CSV = "/home/mat/Dokumente/Delivery Area EPF/Data/master_dataset_2021_2024.csv"
OUTPUT_PARQUET = os.path.join(os.path.dirname(__file__), "master_dataset.parquet")


def convert():
    print(f"Reading CSV: {MASTER_CSV}")
    t0 = time.time()
    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns in {time.time()-t0:.1f}s")

    # Fix German-locale numeric columns (comma decimal separator → period)
    # These are MRL columns stored as strings like '0,000' instead of '0.000'
    str_cols = df.select_dtypes(include='object').columns.tolist()
    str_cols = [c for c in str_cols if c != 'Date']
    for col in str_cols:
        try:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .replace({'nan': None, 'None': None, '': None})
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        except Exception:
            pass
    if str_cols:
        print(f'  Fixed {len(str_cols)} comma-decimal columns: {str_cols[:5]}...')

    # Ensure Date is UTC-aware (stored as UTC in master)
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC")

    print(f"Writing Parquet: {OUTPUT_PARQUET}")
    t1 = time.time()
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    size_mb = os.path.getsize(OUTPUT_PARQUET) / 1e6
    print(f"  Done in {time.time()-t1:.1f}s  →  {size_mb:.1f} MB")
    print(f"  Date range: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
    print("Conversion complete.")


if __name__ == "__main__":
    convert()
