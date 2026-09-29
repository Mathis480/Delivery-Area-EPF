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
* **Observations ($N$):** 114,052 continuous quarter-hours without any missing intervals.
* **Features ($K$):** 962 aligned cross-domain variables defined in [`features.csv`](features.csv).
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

The benchmark evaluates three model architectures, each trained independently per delivery quarter-hour ($q \in \{0, \dots, 95\}$) on a rolling 822-day historical lookback window (~2.25 years):

### 3.1 LASSO
L1-regularized regression with temporal expanding-window cross-validation for alpha selection over the last 14 days of the training window, following Marcjasz et al. (2020).

### 3.2 cSVR (Corrected Support Vector Regression)
Faithful port of Puć & Janczura (2024, Eq. 12).
* **Kernel:** Laplace kernel on standardized features multiplied by a Gaussian correction kernel on the standardized naive benchmark:
  $$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|_2\right) \cdot \exp\left(-\frac{1}{2\sigma^2} (P_i^{\text{naive}} - P_j^{\text{naive}})^2\right)$$
* **Hyperparameters:** $C = 1.0$, $\epsilon = 0.1$, $\gamma$ and $\sigma$ derived adaptively from distance quantiles.
* **Feature Filtering:** Zero-variance removal and optional correlation filtering ($|r| \ge 0.80$ for S1 Macro, $|r| \ge 0.95$ for S2 Neighbor).

### 3.3 MAML-NN (Model-Agnostic Meta-Learning Neural Network)
Addresses small-sample limitations ($N \approx 792$ observations per quarter-hour) via meta-learning (Finn et al., 2017):
* **Architecture:** 2-layer MLP (`[128, 64]`) with Tanh activation, L1 weight regularization ($10^{-4}$), and proximal soft-thresholding.
* **Linear Residual Bypass:** Linear projection with smooth hyperbolic tangent saturation ($M = 0.15\sigma_y$) anchors the non-stationary mean, enabling the neural network to fit nonlinear microstructural deviations.
* **Regime-Aware Support Sampling:** Inner-loop adaptation (8 gradient steps, $\alpha = 0.0025$) on the $K=28$ nearest historical regimes selected via L1 feature distance with soft-kernel weighting.

---

## 4. Feature Sets & Ensembling

Models are trained on four distinct, non-overlapping feature sets:

| Set | Name | Primary Content |
| :---: | :--- | :--- |
| **S1** | Macro | Own VWAP trajectory, auction prices, auto-regressive lags, calendar dummies |
| **S2** | Neighbor | Leakage-safe price trajectories of adjacent contracts ($k \in [-4, +2]$), cross-zone spreads |
| **S3** | Fundamentals | Wind, solar, load forecasts, balancing reserve activations (no price variables) |
| **S4** | Zonal Balances | Bilateral intraday commercial flow balances and self-trading positions across delivery zones |

### Rolling Weighted Forecast Averaging
Following Puć & Janczura (2024, Eq. 25) and Bates & Granger (1969), predictions across the four sets are combined using rolling ex-ante inverse-MAE weighting:

$$w_j = \frac{\left(\text{MAE}_j^W\right)^{-p}}{\sum_{k} \left(\text{MAE}_k^W\right)^{-p}}$$

with calibration window $W = 14$ days and exponent $p = 4.0$. In accordance with experimental rules, ensembles are strictly intra-architecture (Pure cSVR Ensemble vs Pure MAML-NN Ensemble). Cross-architecture hybrids are excluded.

---

## 5. Repository Structure

```text
Delivery-Area-EPF/
├── AGENTS.md                          # Workspace rules, TSO mapping & benchmark ground truth
├── README.md                          # Methodology, architecture & reproduction guide
├── config.py                          # Central configuration and hyperparameters
├── features.csv                       # Feature catalog (962 variable definitions)
├── data_loader.py                     # Rolling QH data loader with ex-ante shifting & scaling
│
├── Data/
│   ├── master_dataset.parquet         # Master dataset (114,052 x 962)
│   ├── convert_to_parquet.py          # CSV to Parquet conversion script
│   └── [Auktions- und Marktdaten-Verzeichnisse]
│
├── Models/
│   ├── __init__.py                    # Module exports
│   ├── linear.py                      # OLS and LASSO (LassoLarsCV)
│   ├── csvr.py                        # cSVR with Laplace and Gaussian correction kernels
│   ├── maml_nn.py                     # MAML-NN with linear bypass and regime adaptation
│   └── ensemble.py                    # Rolling inverse-MAE weighted forecast averaging
│
├── run_full_4sets_backtest_2024.py    # Full 4-set backtest runner (LASSO, cSVR, MAML-NN)
├── run_csvr_corrfilter_2024.py        # cSVR correlation filter ablation runner
│
├── Evaluation/
│   ├── run_full4sets_evaluation.py    # 15-minute QH curves and rMAE heatmaps generator
│   └── plots/                         # Generated evaluation figures
│
├── Results/
│   ├── NPZ_STRUCTURE.md               # Schema documentation for compressed results (.npz)
│   └── annual_run_2024/               # Canonical 2024 backtest results and metrics
│
└── old/                               # Archived exploratory experiments and legacy logs (ignored by git)
```

---

## 6. Reproduction & Execution

### 1. Data Preparation (if converting from CSV):
```bash
python Data/convert_to_parquet.py
```

### 2. Run Full 4-Sets 2024 Backtest:
```bash
python run_full_4sets_backtest_2024.py --zone all
```

### 3. Run cSVR Correlation Filter Ablation:
```bash
python run_csvr_corrfilter_2024.py --zone all
```

### 4. Generate Visualizations & Heatmaps:
```bash
python Evaluation/run_full4sets_evaluation.py
```

---

## 7. Canonical 2024 Results (DE2 Amprion Reference Area)

Full-year out-of-sample evaluation ($N = 35,132$ quarter-hours):

* **Naive Benchmark (`VWAP_90to105`):** `26.6818 EUR/MWh` (rMAE = 1.0000)
* **Pure cSVR Ensemble (S1..S4):** `26.3148 EUR/MWh` (rMAE = **0.9862**)
* **Pure MAML-NN Ensemble (S1..S4):** `26.3069 EUR/MWh` (rMAE = **0.9860**)
* **Diebold-Mariano Test (MAML Ens vs cSVR Ens):** Stat = `0.369`, $p = 0.712$

---

## 8. References

* Bates, J. M., & Granger, C. W. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451–468.
* Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In *International Conference on Machine Learning (ICML)* (pp. 1126–1135). PMLR.
* Marcjasz, G., Uniejewski, B., & Weron, R. (2020). Beating the Naïve — Combining LASSO with Naïve Intraday Electricity Price Forecasts. *Energies*, 13(7), 1667.
* Puć, A., & Janczura, J. (2024). Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market. *International Journal of Forecasting* (Preprint: arXiv:2411.13945).
