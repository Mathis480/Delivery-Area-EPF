# EPEX SPOT Day-Ahead Auction Prices, DE-LU (2021–2024)

Hourly day-ahead clearing prices (EUR/MWh) for the bidding zone Germany/Luxembourg, auction at 12:00 CET on the day before delivery. Source: EPEX SPOT.

Files: `auction_spot_prices_germany_luxembourg_<year>.csv`, one row per delivery day with columns `Hour 1` … `Hour 24` (local time) and `Hour 3A`/`Hour 3B` for the 25-hour day.

`create_master_dataset.ipynb` converts local time (`Europe/Berlin`) to UTC and assigns each hourly price to its four quarter-hours (feature `Dayahead_Auction_hourly`).
