# EPEX SPOT Continuous Intraday Trades (2021–2024)

Quarter-hourly trade metrics per lead-time window, aggregated from EPEX SPOT continuous trades for the delivery areas `DE1`–`DE4` and Germany (`DE`). Source: EPEX SPOT sFTP market data (not public). Time zone: UTC.

## Processing (`calculate_ID_DEx_UTC.py`)

1. Products `Intraday_Quarter_Hour_Power` and `XBID_Quarter_Hour_Power`; trades with `SelfTrade == 'N'` only.
2. A trade belongs to an area if at least one leg is in that area; each `TradeId` is counted once.
3. Direction: two-leg trades from the counterparty leg (`intra`, `import`, `export`); single-leg XBID trades (counterparty outside the EPEX DE order book) from the own leg (`ext_import`, `ext_export`).
4. Windows by time between execution and delivery start: 24 windows of 15 min (`345to360` … `0to15`) and the target window `0to30`.

## Columns (`ID_DelArea_<area>_<year>.csv`)

`DeliveryStart` plus 10 metrics for each of the 25 windows `w` (251 columns):

| Column | Content | Empty window |
| :--- | :--- | :--- |
| `VWAP_{w}` | volume-weighted average price (EUR/MWh) | VWAP of the preceding window |
| `VWAP_total_volume_{w}` | traded volume (MWh) | 0 |
| `VWAP_total_trades_{w}` | number of trades | 0 |
| `id_balance_{w}` | net balance with all counterparties outside the area (imports − exports, MWh) = bilateral + external | 0 |
| `id_balance_to_DE1_{w}` … `_DE4_{w}` | net balance with one German delivery area | 0 |
| `id_balance_ext_{w}` | net balance with external XBID counterparties | 0 |
| `id_trade_intra_{w}` | volume traded within the area | 0 |

Further imputation (national fallback, see `AGENTS.md`) happens in `create_master_dataset.ipynb`.

## Files

```text
2021/ … 2024/               raw daily trade files (Continuous_Trades*.csv)
2021 index/ … 2024 index/   EPEX index files
ID_DelArea_<area>_<year>.csv  aggregated series, area in DE1–DE4, DE
calculate_ID_DEx_UTC.py     aggregation script
```
