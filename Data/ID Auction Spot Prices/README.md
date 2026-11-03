# EPEX SPOT Intraday Auction Prices, Germany (2021–2024)

Quarter-hourly intraday auction clearing prices (EUR/MWh), auction at 15:00 CET on the day before delivery. Source: EPEX SPOT.

| Period | Auction | Files |
| :--- | :--- | :--- |
| 01.01.2021–13.06.2024 | German 15-minute call auction | `intraday_auction_spot_prices_15-call_germany_<year>.csv` |
| 14.06.2024–31.12.2024 | Pan-European intraday auction IDA1 (SIDC), DE-LU price | `pan-european_prices_germany_luxembourg_IDA1_2024.csv` |

Files have one row per delivery day with columns `Hour 1 Q1` … `Hour 24 Q4` (local time) and `Hour 3B Q1` … `Q4` for the 25-hour day.

`create_master_dataset.ipynb` joins both series, converts local time (`Europe/Berlin`) to UTC (feature `Intraday Auction Price`). The price also serves as fallback for the national window `VWAP_345to360`.
