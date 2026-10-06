# Delivery-Area-EPF: Regional Electricity Price Forecasting

Machine Learning pipeline for regional Electricity Price Forecasting (EPF) across the four German Transmission System Operator (TSO) delivery areas:
- **TransnetBW (`DE1`)**
- **Amprion (`DE2`)**
- **TenneT (`DE3`)**
- **50Hertz (`DE4`)**

The study evaluates Model-Agnostic Meta-Learning Neural Networks (**MAML-NN**) against the state-of-the-art benchmark **Corrected Support Vector Regression (cSVR)** (Puć & Janczura, 2024) and **LASSO** across four structured feature sets over the entire calendar year **2024** ($N = 35,132$ continuous 15-minute contracts per zone).

---

## 1. Master Dataset

The master dataset integrates 15-minute continuous electricity contracts covering **October 1, 2021 to December 31, 2024** (Q4 2021 through 2024):
* **Observations ($N$):** 114,060 continuous quarter-hours without any missing intervals.
* **Features ($K$):** 936 aligned cross-domain variables defined in [`features.csv`](features.csv) (338 direct active features for $k=0$, 52 safe ex-post neighbor inputs for $k \in [-4, -1]$ under `only past products`, 520 excluded under `no`).
* **Missing Values:** Zero NaNs across the entire evaluation horizon via hierarchical fallback imputation.
* **Time Standard:** Strictly Pure UTC (`+00:00` / `Z`) across all sources and models.

### Integrated Data Sources

| Domain | Source & Description |
| :--- | :--- |
| **Continuous Intraday** | EPEX continuous 15-min rolling VWAPs, traded volumes, trade counts, and bilateral inter-zonal flow balances across 25 lead-time windows (`345to360` down to `0to30`). |
| **Day-Ahead Spot** | EPEX Day-Ahead hourly auction prices cleared at 12:00 D-1. |
| **Intraday Spot** | EPEX Intraday 15-minute auction spot prices cleared at 15:00 D-1. |
| **Balancing Energy** | Operational activated balancing reserves (SRL / aFRR and MRL / mFRR, positive and negative) mapped to TSO zones. |
| **Fundamentals** | Energy-Charts generation, demand, and cross-border fundamentals across 5 categories: `LOAD`, `SOLAR`, `ONSHORE`, `OFFSHORE`, and `CROSS_BORDER` (actuals, Day-Ahead forecasts, forecast errors, and net cross-border commercial trading flows). |
| **Calendar & Lags** | Wall-clock indicators (`Weekday`, `Hour`, `Quarter`) and historical price lags (`lag_24h`, `lag_48h`, `lag_168h`). |

---

## 2. Methodology & Information Availability

1. **Information Cutoff & Operational Delay:**
   * Nominal trading cutoff: 60 minutes before delivery ($t_{\text{delivery}} - 60\text{ min}$).
   * Operational publication buffer: 30 minutes (2 quarter-hours, rounding up Puć et al.'s 20-minute buffer to the 15-minute grid).
   * Effective ex-ante cutoff: **90 minutes before delivery** ($t_{\text{delivery}} - 90\text{ min}$).
   * Canonical Naive Benchmark: `{zone}_VWAP_90to105`.
2. **Ex-Post Features:**
   * All realized fundamentals (generation, load, balancing activations) are lagged by at least 8 quarter-hours (120 minutes / 2 hours) prior to delivery.
3. **Neighbor Contract Temporal Availability:**
   * For neighbor contract $k \in [-4, +2]$, a window ending $A$ minutes before its delivery is admitted if and only if $A \ge 90 + 15 \cdot k$.

---

## 3. Evaluated Models

The benchmark evaluates multiple paradigm approaches, each trained independently per delivery quarter-hour ($q \in \{0, \dots, 95\}$) on a rolling 822-day historical lookback window (~2.25 years):

### 3.1 LASSO Models
* **Standard LASSO (Literature):** L1-regularized regression with temporal expanding-window cross-validation for alpha selection over the last 14 days of the training window, following Marcjasz et al. (2020).
* **Weighted LASSO (1-SE Rule):** Integrates soft-kernel regime distance weighting ($w \propto \exp(-d/\tau)$) with the 1-Standard-Error rule for alpha selection, avoiding small-sample overfit and reducing MAE by >1.40 EUR/MWh.

### 3.2 cSVR (Corrected Support Vector Regression)
Port of Puć & Janczura (2024, Eq. 12).
* **Kernel:** Laplace kernel on standardized features multiplied by a Gaussian correction kernel on the standardized naive benchmark:
  $$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|_2\right) \cdot \exp\left(-\frac{1}{2\sigma^2} (P_i^{\text{naive}} - P_j^{\text{naive}})^2\right)$$
* **Feature Filtering & Correlation Ablation:** 
  - *Zero-Variance Removal:* Applied to all feature sets.
  - *Correlation Filtering (Puć & Janczura, 2024):* Prunes collinear features on the training set ($|r| \ge 0.80$ on S1, $|r| \ge 0.95$ on S2).
  - *Unfiltered Variant (`CSVR_USE_CORR_FILTER = False`):* Eliminating the correlation filter preserves all 86 S1 features and 31 S2 features. Because SVR regularizes weights ($C=1.0$), avoiding arbitrary column pruning significantly improves accuracy (DE2 MAE drops from 26.2367 to 26.2044 EUR/MWh, Diebold-Mariano Stat = 4.4751, $p < 0.0001$). Toggleable via `CSVR_USE_CORR_FILTER` in [`config.py`](config.py).

### 3.3 Pure MAML-NN & Deep Ensembling
Addresses small-sample limitations ($N \approx 792$ observations per quarter-hour) via meta-learning (Finn et al., 2017) directly on price spreads without linear bypass:
* **Architecture:** 2-layer MLP (`[128, 64]`) with Tanh activation, L1 weight regularization ($10^{-4}$), and proximal soft-thresholding.
* **Regime-Aware Support Sampling:** Fast compiled inner-loop adaptation (8 gradient steps, $\alpha = 0.0025$) on the $K=28$ nearest historical regimes selected via L1 feature distance with soft-kernel weighting.
* **Deep Ensembling (Lago et al., 2021):** Trains multiple independent model instances with distinct random initializations (`seeds = [42, 123, 999]`), averaging forecasts per feature set to eliminate epistemic variance.

---

## 4. Feature Sets & Ensembling

Models are trained on four distinct, non-overlapping feature sets:

| Set | Name | Primary Content |
| :---: | :--- | :--- |
| **S1** | Macro | Own VWAP trajectory, auction prices, auto-regressive lags, calendar dummies |
| **S2** | Neighbor | Leakage-safe price trajectories of adjacent contracts ($k \in [-4, +2]$), cross-zone spreads |
| **S3** | Fundamentals | Wind, solar, load forecasts, balancing reserves, and cross-border commercial trading flows (`DE_cross_border_trading`) |
| **S4** | Zonal Balances | Bilateral intraday commercial flow balances and self-trading positions across delivery zones |

### Rolling Weighted Forecast Averaging
Following Puć & Janczura (2024, Eq. 25) and Marcjasz, Serafin & Weron (2018), predictions across the four sets are combined using rolling ex-ante inverse-MAE weighting:

$$w_j^W = \frac{\frac{1}{\text{MAE}_j^W}}{\sum_{k} \frac{1}{\text{MAE}_k^W}}$$

with historical calibration window $W = 14$ days. In accordance with experimental rules, ensembles are strictly intra-architecture (Pure cSVR Ensemble vs Pure MAML-NN Ensemble). Cross-architecture hybrids are excluded. 

---

## 5. Repository Structure

```text
Delivery-Area-EPF/
├── AGENTS.md                          # Workspace rules, TSO mapping & benchmark ground truth
├── README.md                          # Methodology, architecture & reproduction guide
├── config.py                          # Central configuration and hyperparameters
├── features.csv                       # Feature catalog (910 variable definitions)
├── data_loader.py                     # Rolling QH data loader with ex-ante shifting & scaling
│
├── Data/
│   ├── master_dataset.parquet         # Master dataset (114,060 x 936 float32 columns)
│   └── [Raw Auction & Market Directories]
│
├── Models/
│   ├── __init__.py                    # Module exports
│   ├── linear.py                      # Standard LASSO & Weighted LASSO (1-SE rule)
│   ├── csvr.py                        # cSVR with Laplace and Gaussian correction kernels
│   ├── maml_nn.py                     # MAML-NN engine with fast compiled inner adaptation
│   └── ensemble.py                    # Rolling inverse-MAE weighted forecast averaging
│
├── run_master_benchmark_2024.py       # Unified master runner across all 4 zones & feature sets (S1-S4)
│
├── Evaluation/
│   ├── run_full4sets_evaluation.py    # 15-minute QH curves and rMAE heatmaps generator
│   └── plots/                         # Generated evaluation figures
│
├── Results/
│   ├── NPZ_STRUCTURE.md               # Schema documentation for compressed results (.npz)
│   └── annual_run_2024/               # Canonical 2024 backtest results and metrics
│
└── old/                               # Archived exploratory experiments and legacy logs
```

---

## 6. Reproduction & Execution

### 1. Run Unified Master Benchmark (All 4 Zones, 2024 Full Year):
```bash
python run_master_benchmark_2024.py
# To retrain backbones and recompute from scratch:
python run_master_benchmark_2024.py --force
```

### 2. Generate Master Heatmaps & Visualizations:
```bash
python Evaluation/run_full4sets_evaluation.py
```

### 3. Display Master Benchmark Summary Table:
```bash
python run_master_benchmark_2024.py --summary_only
```

---

## 7. Master 2024 Benchmark Results (Full Out-of-Sample Year, MAE in EUR/MWh)

Evaluation over all 35,132 continuous quarter-hours of calendar year 2024:

| Model / Scenario | DE1 | DE2 (Amprion) | DE3 | DE4 | **Average** | Status / Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Benchmark** (`VWAP_90to105`) | 26.2815 | 26.6818 | 27.2725 | 27.2536 | **26.8724** | Ground Truth Baseline |
| **Scenario 1: Standard LASSO** | 26.6733 | 27.2058 | 27.9223 | 27.8579 | **27.4148** | Linear Reference |
| **Scenario 2: Pure cSVR Ensemble (with Corr Filter)** | 25.9016 | 26.2367 | 26.9297 | 26.7828 | **26.4627** | Puć (2024) Standard |
| **Scenario 2b: Pure cSVR Ensemble (No Corr Filter)** | — | **26.2044** | — | — | — | Statistically Superior ($p < 0.0001$) |
| **Scenario 3: Old MAML (Ridge-Bypass $m=0.15$)** | 25.9000 | 26.2781 | 26.9485 | 26.8383 | **26.4912** | Prior Linear Hybrid Baseline |
| **Scenario 4: Pure MAML-NN (Single-Seed 42)** | 25.8966 | 26.1320 | 26.8130 | 26.7340 | **26.3939** | Strong Baseline |
| **Scenario 5: Pure MAML-NN (Deep Ensemble 3-Seed)** | **25.8694** | **26.1121** | **26.7704** | **26.7068** | **26.3647** | 🏆 **Global Winner** |

---

## 8. References

* Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In *International Conference on Machine Learning (ICML)* (pp. 1126–1135). PMLR.
* Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*, 293, 116983.
* Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
* Marcjasz, G., Serafin, T., & Weron, R. (2018). Selection of calibration windows for day-ahead electricity price forecasting. *Energies*, 11(9), 2364.
* Marcjasz, G., Uniejewski, B., & Weron, R. (2020). Beating the Naïve — Combining LASSO with Naïve Intraday Electricity Price Forecasts. *Energies*, 13(7), 1667.
* Puć, A., & Janczura, J. (2024). Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market. *International Journal of Forecasting* (Preprint: arXiv:2411.16237v1).
