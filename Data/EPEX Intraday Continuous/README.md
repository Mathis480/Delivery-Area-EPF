# EPEX SPOT Intraday Continuous Trades (2021–2024)

This directory contains quarter-hourly (15-minute) aggregated trade metrics and cross-zonal flow balances derived from raw tick-by-tick EPEX Intraday Continuous orders across Germany (`DE`) and its four TSO delivery areas (`DE1`–`DE4`) for 2021 through 2024.

## 1. Data Provenance and Attribution

* **Market Operator:** EPEX SPOT SE (sFTP Market Data Server)
* **Market Segment:** Intraday Continuous Trading (XBID / Local Continuous)
* **Products:** 15-minute contracts (`Intraday_Quarter_Hour_Power`, `XBID_Quarter_Hour_Power`)
* **Delivery Areas:**
  * `DE1`: TransnetBW (`DE-ENBW`)
  * `DE2`: Amprion (`DE-AMP`)
  * `DE3`: TenneT (`DE-TPS`)
  * `DE4`: 50Hertz (`DE-50HZ`)
  * `DE`: National aggregate (Germany / Luxembourg)
* **Timezone:** Strictly Pure UTC (`+00:00`)

## 2. Lead-Time Window Architecture

Raw trade executions are partitioned into 25 discrete lead-time windows prior to delivery start:
* **24 Continuous Pre-Delivery Windows:** 15-minute increments from `345to360` down to `0to15` minutes prior to delivery.
* **1 Target Window:** `0to30` minutes prior to delivery (used as forecasting target).

### Window Metrics (9 Variables per Lead-Time Window)
For each delivery quarter-hour and lead-time window $w$:
* `VWAP_{w}`: Volume-Weighted Average Price (EUR/MWh). Forward-filled horizontally if no trades occurred.
* `VWAP_total_volume_{w}`: Total traded volume (MWh). Filled with 0 if no trades occurred.
* `VWAP_total_trades_{w}`: Number of executed transactions. Filled with 0 if no trades occurred.
* `id_balance_{w}`: Net cross-border trade balance (Imports $-$ Exports).
* `id_balance_to_DE1_{w}` ... `id_balance_to_DE4_{w}`: Bilateral trade balance with specific German TSO area.
* `id_trade_to_self_{w}`: Purely internal trading volume within the same delivery zone.

Total feature dimension: 25 windows $\times$ 9 metrics $+$ `DeliveryStart` = 226 columns per annual file.

## 3. Data Processing & Quality Control (`calculate_ID_DEx_UTC.py`)

Raw tick data is pre-aggregated offline via `calculate_ID_DEx_UTC.py`:
1. **Contract Filtering:** Restricts trades to exactly 15-minute delivery duration (`DeliveryEnd - DeliveryStart == 900s`).
2. **Wash Trade Removal:** Excludes internal self-trades (`SelfTrade == 'N'`).
3. **Deduplication:** Aligns buyer and seller records to prevent double-counting of matched trades.
4. **Direction Classification:** Classifies each trade leg relative to the target area (`self`, `import`, or `export`).
5. **Parallel Execution:** Distributed via `ProcessPoolExecutor` across available CPU cores.

## 4. Directory Inventory

```
Data/EPEX Intraday Continuous/
├── 2021/ ... 2024/                 # Raw daily trade CSV files (sFTP archive)
├── 2021 index/ ... 2024 index/     # EPEX official reference index files
├── ID_DelArea_DE1_2021.csv ...     # Processed annual 15-min series (DE1, 2021-2024)
├── ID_DelArea_DE2_2021.csv ...     # Processed annual 15-min series (DE2, 2021-2024)
├── ID_DelArea_DE3_2021.csv ...     # Processed annual 15-min series (DE3, 2021-2024)
├── ID_DelArea_DE4_2021.csv ...     # Processed annual 15-min series (DE4, 2021-2024)
├── ID_DelArea_DE_2021.csv ...      # Processed annual 15-min series (DE, 2021-2024)
├── calculate_ID_DEx_UTC.py         # Offline ETL tick aggregator script
├── validate_and_calculate_indices.py # Validation script against official indices
└── README.md
```

### Pre-Aggregated 15-Minute Files (20 Files)

| File Pattern | Rows / Year | Columns | Description |
| :--- | :---: | :---: | :--- |
| `ID_DelArea_DE1_{year}.csv` | ~35,040 | 226 | TransnetBW delivery area continuous trade metrics |
| `ID_DelArea_DE2_{year}.csv` | ~35,040 | 226 | Amprion delivery area continuous trade metrics |
| `ID_DelArea_DE3_{year}.csv` | ~35,040 | 226 | TenneT delivery area continuous trade metrics |
| `ID_DelArea_DE4_{year}.csv` | ~35,040 | 226 | 50Hertz delivery area continuous trade metrics |
| `ID_DelArea_DE_{year}.csv` | ~35,040 | 226 | National German aggregate continuous trade metrics |
