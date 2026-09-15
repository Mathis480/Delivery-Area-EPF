# Agent Instructions & Workspace Rules

## 1. Language Policy

* **Direct Chat Communication with the User:** You may converse with the user in German (or match the language chosen by the user).
* **Project Documentation & Artifacts (Mandatory English):** All project documentation, README files, markdown reports, summaries, notes, and repository artifacts must be authored **exclusively in English**.
* **Code & Technical Nomenclature (Mandatory English):** All code, identifiers, function and variable names, classes, docstrings, comments, and commit messages must be strictly in English.

## 2. Methodology & Scientific Rigor

### 2.1 Pure UTC Time Base
* All time series data across all sources must be strictly formatted, indexed, and merged in **Pure UTC** (`+00:00` / `Z`).
* The master index column `Date` uniquely defines the **delivery start time** ($t_{\text{delivery}}$) of the contract product.

### 2.2 Look-Ahead Bias Prevention & Lead-Time Cutoff Policy
A core risk is Look-Ahead Bias (data leakage). In this project:
* **Prediction Point vs. Delivery Interval:** Predictions for a given contract maturity starting at $t_{\text{delivery}}$ are made at a defined decision point prior to delivery (**60 minutes before delivery**, where the last fully observed continuous trading window is `VWAP_90to105`).
* **Ex-Ante vs. Ex-Post Features:**
  * **Ex-Ante Features (Known at Prediction Time):**
    * Day-Ahead auction prices (`Dayahead_Auction_hourly`, cleared at D-1 12:00 CET).
    * Intraday auction prices (`Intraday Auction Price`, cleared at D-1 15:00 CET).
    * Day-Ahead fundamental forecasts (`load_forecast`, `solar_forecast`, `wind_onshore_forecast`, `wind_offshore_forecast`, published at D-1 12:00).
    * Calendar and delivery time dummies (`Weekday_1` to `Weekday_7`, `Hour_0` to `Hour_23`, and `Quarter_1` to `Quarter_4` based on Europe/Berlin local time).
    * Historical continuous trading windows that closed strictly prior to the cutoff (e.g., `VWAP_90to105` and older windows).
  * **Ex-Post / Realized Features (Leakage Hazard if Unlagged):**
    * Actual generation and demand (`load_actual`, `solar_actual`, `wind_onshore_actual`, `wind_offshore_actual`).
    * Realized forecast errors (`load_diff`, `solar_diff`, `wind_diff`).
    * Activated balancing reserves (`SRL_positive/negative`, `MRL_positive/negative`).
    * *Notice:* At decision time ($t_{\text{delivery}} - 60\text{m}$ to $90\text{m}$), the actual generation or balancing activation for the delivery interval $t_{\text{delivery}}$ has **not physically occurred yet**. Using unshifted actuals for interval $t_{\text{delivery}}$ creates catastrophic look-ahead leakage.

### 2.3 Two-Stage Data Architecture (Data Prep vs. Modeling Hand-Off)
To maintain a clean separation of concerns and scientific traceability:
1. **Stage 1 – Master Data Preparation (`create_master_dataset`):**
   * The master dataset (`master_dataset_2021_2024.csv`) aligns all variables systematically by their **delivery start timestamp** (`Date` = $t_{\text{delivery}}$).
   * **Rule:** No artificial feature lag shifts are baked into the master file creation stage. The master dataset serves as the pristine, canonical ground-truth repository.
2. **Stage 2 – Downstream Feature Engineering & Model Training:**
   * **Mandatory Shift Application:** In downstream modeling pipelines, all realized/actual features flagged in `features.csv` (under `shift_needed_2h = "x"`, covering actual demand, solar/wind generation, and balancing reserve activations) must be shifted backwards by at least **8 quarter-hours (120 minutes / 2.0 hours)** prior to delivery Start.

### 2.4 Canonical Target Variables
The canonical forecasting targets represent the volume-weighted average continuous price in the final 30 minutes prior to delivery (`0to30`):
* **Regional Delivery Targets:** `DE1_VWAP_0to30`, `DE2_VWAP_0to30`, `DE3_VWAP_0to30`, `DE4_VWAP_0to30`.
* **National Benchmark Target:** `VWAP_0to30`.

## 3. German TSO Control Zones & Delivery Area Mapping

To maintain strict cross-dataset consistency across EPEX continuous transactions, balancing reserves (SRL / aFRR, MRL / mFRR), and fundamentals (demand, solar, wind generation), the four German Transmission System Operators (TSOs) must always be mapped 1:1 to their official EPEX delivery area codes:

| Delivery Area Code | TSO / Control Area | EPEX Code | Zonal Prefix |
| :--- | :--- | :--- | :--- |
| **`DE1`** | **TransnetBW** | `DE-ENBW` | `DE1_` |
| **`DE2`** | **Amprion** | `DE-AMP` | `DE2_` |
| **`DE3`** | **TenneT** | `DE-TPS` | `DE3_` |
| **`DE4`** | **50Hertz** | `DE-50HZ` | `DE4_` |
| **`DE`** | **National Aggregate (Germany)** | `DE` | `DE_` (or unprefixed for national indices) |

* **Mandatory Rule:** All regional features across all domains must be prefixed with `DE1_`, `DE2_`, `DE3_`, or `DE4_` (e.g., `DE1_load_actual`, `DE1_SRL_positive`, `DE1_VWAP_0to30`). Raw company names (e.g., `TransnetBW`, `Amprion`) must never be used in feature names or ML model selectors.

## 4. Missing Data & NaN Imputation Principles

* **Strict No-Deletion Rule:** When encountering missing values (`NaN`), rows or delivery intervals must **NEVER** be deleted or dropped. The continuous time series grid must remain unbroken and complete across the entire evaluation horizon.
* **Hierarchical Price Imputation Protocol (Variante B):**
  * **National Benchmark Trajectory (`VWAP_{w}`):**
    1. Leftmost window (`VWAP_345to360`): Fallback to contemporaneous `Intraday Auction Price` (and `Dayahead_Auction_hourly` as backup).
    2. Subsequent windows (`330to345` to `0to15`): Horizontal forward-fill from preceding window within the same contract maturity.
    3. Target window (`VWAP_0to30`): Fallback to `VWAP_0to15` / `VWAP_15to30`.
  * **Regional Zonal Trajectories (`DEx_VWAP_{w}`):**
    * Prio 0: Own observed regional VWAP in window $w$.
    * Prio 1: Contemporaneous national volume-weighted price in the **same window $w$** (`VWAP_{w}`).
  * **Non-Price Continuous Features (Volume, Trades, Balances):** Unobserved trading activity is filled strictly with `0.0`.
* **Multi-Day Lag Initialization Buffer:** Time grids must be initialized with at least a 7-day (672 steps) historical buffer before the canonical start date (`2021-09-30 22:00 UTC`), ensuring zero initial NaNs in multi-day lag features (`lag_24h`, `lag_48h`, `lag_168h`).
* **Ban on Look-Ahead Filling:** `bfill` (backward fill) is strictly forbidden across all feature engineering pipelines.

