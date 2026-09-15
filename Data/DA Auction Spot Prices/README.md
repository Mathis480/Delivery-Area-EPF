# EPEX Day-Ahead Auction Spot Prices (Germany/Luxembourg, 2021–2024)

This directory contains Day-Ahead auction clearing spot prices for the bidding zone Germany/Luxembourg (DE/LU) covering 2021 through 2024.

## 1. Data Provenance and Attribution

* **Market Operator:** EPEX SPOT SE
* **Market Segment:** Day-Ahead Auction (hourly clearing, cleared daily at 12:00 CET for delivery on the subsequent day).
* **Delivery Zone:** Germany / Luxembourg (`DE-LU`).
* **Price Currency & Unit:** EUR / MWh.

## 2. Temporal Structure & Preprocessing

* **Raw Format:** Wide format per delivery day with columns `Hour 1` to `Hour 24` plus `Hour 3A` and `Hour 3B` (for the autumnal 25-hour daylight saving time clock transition).
* **Timezone Transformation:** Mapped from local German wall-clock time (`Europe/Berlin`) to **Pure UTC** (`+00:00` / `Z`), correctly resolving ambiguous fall-back hours (`Hour 3B` -> winter standard time).
* **Quarter-Hourly Interpolation / Step Expansion:** Day-Ahead hourly auction prices are expanded to uniform 15-minute intervals across all four quarters of each hour (`Q1: :00`, `Q2: :15`, `Q3: :30`, `Q4: :45`) under the feature name `Dayahead_Auction_hourly`.

## 3. Directory Inventory

* `auction_spot_prices_germany_luxembourg_2021.csv`: Raw wide EPEX Day-Ahead prices 2021.
* `auction_spot_prices_germany_luxembourg_2022.csv`: Raw wide EPEX Day-Ahead prices 2022.
* `auction_spot_prices_germany_luxembourg_2023.csv`: Raw wide EPEX Day-Ahead prices 2023.
* `auction_spot_prices_germany_luxembourg_2024.csv`: Raw wide EPEX Day-Ahead prices 2024.
* `process_da_auction_prices.py`: Autonomous processing pipeline generating the continuous 15-minute UTC series.
