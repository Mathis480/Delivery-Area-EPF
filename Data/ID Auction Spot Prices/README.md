# EPEX Intraday Auction Spot Prices (Germany/Luxembourg, 2021–2024)

This directory contains 15-minute clearing prices for the German Intraday Auction market segment covering the full period from 2021 through 2024.

## 1. Market Mechanics & Structural Transition (June 2024)

Historical German intraday auction price discovery experienced a major regulatory transition in mid-2024:
1. **German 15-Minute Call Auction (Jan 1, 2021 – June 13, 2024):**
   * National quarter-hourly call auction cleared daily at 15:00 CET/CEST for delivery across all 96 quarter-hours of the subsequent calendar day.
   * Captured in files `intraday_auction_spot_prices_15-call_germany_{2021..2024}.csv`.
2. **Pan-European Intraday Auction 1 (IDA1) (June 14, 2024 – Dec 31, 2024 onwards):**
   * Pan-European market coupling under the Single Intraday Coupling (SIDC) framework. IDA1 is cleared daily at 15:00 CET for delivery across all quarter-hours of the next day.
   * Captured in `pan-european_prices_germany_luxembourg_IDA1_2024.csv`.

Together, these two data streams provide uninterrupted, consistent 15-minute intraday auction pricing across the entire multi-year training and evaluation window.

## 2. Preprocessing & Timezone Handling

* **Raw Structure:** Wide format with columns `Hour 1 Q1` through `Hour 24 Q4`, plus `Hour 3B Q1..Q4` during the autumnal Daylight Saving Time clock shift.
* **Pure UTC Conversion:** All timestamps are strictly localized to `Europe/Berlin` accounting for daylight saving transitions, and converted to Pure UTC (`+00:00` / `Z`).
* **Feature Nomenclature:** Mapped to the canonical feature name `Intraday_Auction_Price`.

## 3. Directory Inventory

* `intraday_auction_spot_prices_15-call_germany_2021.csv`
* `intraday_auction_spot_prices_15-call_germany_2022.csv`
* `intraday_auction_spot_prices_15-call_germany_2023.csv`
* `intraday_auction_spot_prices_15-call_germany_2024.csv`
* `pan-european_prices_germany_luxembourg_IDA1_2024.csv`
* `process_id_auction_prices.py`: Unified pipeline joining 15-call DE and Pan-European IDA1 into a continuous 15-minute UTC series.
