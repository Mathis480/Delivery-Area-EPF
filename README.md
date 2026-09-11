# Delivery-Area-EPF

EPEX SPOT Intraday Continuous Feature Extraction Pipeline across German Transmission System Operator (TSO) Delivery Areas (`DE1` TransnetBW, `DE2` Amprion, `DE3` TenneT, `DE4` 50Hertz) and the National Market Area (`DE`).

## Overview

This repository provides high-performance data processing pipelines for EPEX Intraday Continuous 15-minute electricity contracts (2021–2024).

### Key Features
1. **Pure UTC Processing**: Completely eliminates daylight saving time (DST) discontinuities, double hours (2A/2B in autumn), and skipped hours (spring), avoiding look-ahead bias and alignment errors.
2. **Lead-Time Window Aggregation**: Calculates Volume-Weighted Average Prices (VWAP), traded volumes, and trade counts across 25 discrete lead-time intervals ($T-360$ down to $T-0$ minutes prior to delivery).
3. **Zonal & Cross-Border Flows**: Classifies transactions into internal (`self`), export, and import flows, computing net commercial balances between control zones and foreign markets.
4. **Data Quality & Self-Trade Filtering**: Explicitly filters out wash trades (`SelfTrade == 'N'`) and deduplicates transaction IDs.
5. **Parallel Execution**: Uses Python's `ProcessPoolExecutor` across multi-core processors with an interactive step debugger for auditing individual trades.

## Data Protection Notice

> **IMPORTANT**: Raw transaction records, proprietary exchange tick logs, and commercial market data files (`.csv`, `.parquet`, `.duckdb`, `.zip`) are strictly excluded via `.gitignore` and are not hosted in this public repository.
