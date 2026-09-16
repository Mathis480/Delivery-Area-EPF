# Delivery-Area-EPF: Regional Electricity Price Forecasting Pipeline

Production-grade Machine Learning dataset and feature extraction pipeline for regional Electricity Price Forecasting (EPF) across the four German Transmission System Operator (TSO) control zones: **TransnetBW (`DE1`)**, **Amprion (`DE2`)**, **TenneT (`DE3`)**, and **50Hertz (`DE4`)**, alongside the German national benchmark (**`DE`**).

---

## 1. Overview & Master Dataset

The repository builds the canonical training dataset ([`Data/master_dataset_2021_2024.csv`](file:///home/mat/Dokumente/Delivery%20Area%20EPF/Data/master_dataset_2021_2024.csv)) for 15-minute continuous electricity contracts covering **October 1, 2021 to December 31, 2024** (Q4 2021 through 2024).

* **Observations ($N$):** 114,052 continuous quarter-hours.
* **Features ($K$):** 962 aligned cross-domain variables.
* **Missing Values:** **0 NaNs (0.00%)** via hierarchical imputation.
* **Time Standard:** Strictly **Pure UTC (`+00:00` / `Z`)** across all sources.

---

## 2. Integrated Data Streams

| Data Domain | Directory | Description & Sources |
| :--- | :--- | :--- |
| **Continuous Intraday** | `Data/EPEX Intraday Continuous/` | EPEX continuous 15-min rolling VWAPs, traded volumes, trade counts, and bilateral inter-zonal flow balances across 25 lead-time windows (`345to360` down to `0to15` and `0to30`). |
| **Day-Ahead Spot** | `Data/DA Auction Spot Prices/` | EPEX Day-Ahead hourly auction prices cleared at 12:00 D-1, mapped continuously to 15-minute UTC intervals. |
| **Intraday Spot** | `Data/ID Auction Spot Prices/` | EPEX Intraday 15-minute auction spot prices cleared at 15:00 D-1 (seamlessly combining 15-Call and Pan-European IDA1). |
| **Balancing Energy** | `Data/Balancing Energy/` | Operational activated balancing reserves (SRL / aFRR and MRL / mFRR, positive and negative) mapped to TSO zones. |
| **Fundamentals** | `Data/Fundamentals/` | Energy-Charts generation, demand, and cross-border fundamentals across 5 categories: `LOAD`, `SOLAR`, `ONSHORE`, `OFFSHORE`, and `CROSS_BORDER` (actuals, Day-Ahead forecasts, forecast errors, and net cross-border commercial trading flows). |
| **Calendar Dummies** | *Generated in-pipeline* | Granular wall-clock delivery indicators: `Weekday_1..7`, `Hour_0..23`, and `Quarter_1..4` (Europe/Berlin). |
| **Multi-Day Lags** | *Generated in-pipeline* | Historical multi-day shifts: `lag_24h` (96 steps), `lag_48h` (192 steps), and `lag_168h` (672 steps) for key price benchmarks. |

### Energy-Charts Cross-Border Electricity Trading Links

The cross-border commercial trading data (*Grenzüberschreitender Stromhandel*, net power flows in MW, 15-minute resolution, Pure UTC) is retrieved directly from the Fraunhofer ISE Energy-Charts spot market feed:
* **2021:** [Energy-Charts Spot Market DE (2021 UTC)](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=DE&legendItems=iy1&year=2021&interval=year&timezone=utc)
* **2022:** [Energy-Charts Spot Market DE (2022 UTC)](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=DE&legendItems=iy1&year=2022&interval=year&timezone=utc)
* **2023:** [Energy-Charts Spot Market DE (2023 UTC)](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=DE&legendItems=iy1&year=2023&interval=year&timezone=utc)
* **2024:** [Energy-Charts Spot Market DE (2024 UTC)](https://www.energy-charts.info/charts/price_spot_market/chart.htm?l=en&c=DE&legendItems=iy1&year=2024&interval=year&timezone=utc)

---

## 3. Scientific & Methodological Standards

1. **TSO Delivery Area Mapping:**
   * `DE1_` = TransnetBW (DE-ENBW)
   * `DE2_` = Amprion (DE-AMP)
   * `DE3_` = TenneT (DE-TPS)
   * `DE4_` = 50Hertz (DE-50HZ)
   * Raw corporate names are strictly forbidden in feature names and model selectors per [`AGENTS.md`](file:///home/mat/Dokumente/Delivery%20Area%20EPF/AGENTS.md).
2. **Canonical Forecasting Targets:**
   * Target variables represent the delivery price in the final 30 minutes prior to delivery: `DE1_VWAP_0to30`, `DE2_VWAP_0to30`, `DE3_VWAP_0to30`, `DE4_VWAP_0to30` (and national `VWAP_0to30`).
3. **Hierarchical Imputation (Variante B):**
   * Missing values in continuous rolling price trajectories are hierarchically imputed:
     * *National benchmark:* `VWAP_345to360` falls back to `Intraday Auction Price`; subsequent windows forward-fill from the preceding window within the same contract maturity.
     * *Regional zones:* Prio 0 (own trade) $\to$ Prio 1 (contemporaneous national volume-weighted price in the same window `VWAP_{w}`).
     * *Non-price metrics:* Volumes, trades, and flows are filled with `0.0`.
4. **Two-Stage Architecture & Look-Ahead Prevention:**
   * **Stage 1 (Master Preparation):** Aligns all ground-truth features to delivery start time ($t_{\text{delivery}}$) without baking in artificial shifts.
   * **Stage 2 (Downstream ML Modeling):** Realized generation/demand actuals and balancing reserves flagged with `shift_needed_2h = "x"` in `features.csv` must be lagged by at least 8 quarter-hours (120 minutes) prior to delivery to prevent data leakage.

---

## 4. Execution

To regenerate the master dataset:

```bash
# Execute end-to-end Python pipeline
python create_master_dataset.py

# Alternatively, execute the step-by-step interactive Jupyter notebook
jupyter notebook create_master_dataset.ipynb
```

---

## 5. Repository Structure

```text
├── AGENTS.md                   # Operational guidelines, mapping rules & scientific constraints
├── README.md                   # Project overview & architecture documentation
├── create_master_dataset.py    # Canonical master dataset generation script
├── create_master_dataset.ipynb # Step-by-step interactive generation notebook
├── features.csv                # Version-controlled feature catalog (903 ex-ante/ex-post definitions)
└── Data/
    ├── master_dataset_2021_2024.csv   # Final 114,052 x 962 ML training dataset
    ├── DA Auction Spot Prices/        # Day-Ahead auction processing script & clean CSV
    ├── ID Auction Spot Prices/        # Intraday auction processing script & clean CSV
    ├── EPEX Intraday Continuous/      # 20 raw continuous trading files (DE, DE1-DE4)
    ├── Balancing Energy/              # SRL and MRL operational reserve files
    └── Fundamentals/                  # Energy-Charts generation & load files (LOAD, SOLAR, WIND)
```
