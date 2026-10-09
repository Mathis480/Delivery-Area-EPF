# Delivery-Area-EPF: Regional Electricity Price Forecasting

Machine learning pipeline for regional Electricity Price Forecasting (EPF) across the four German TSO delivery areas:
- TransnetBW (`DE1`)
- Amprion (`DE2`)
- TenneT (`DE3`)
- 50Hertz (`DE4`)

The study evaluates Model-Agnostic Meta-Learning Neural Networks (MAML-NN) against Corrected Support Vector Regression (cSVR; Puć & Janczura, 2024) and LASSO across four structured feature sets over calendar year 2024 ($N = 35,132$ continuous 15-minute contracts per zone).

---

## 1. Master Dataset

The master dataset covers October 1, 2021 to December 31, 2024 (Q4 2021 through 2024):
* Observations ($N$): 114,060 continuous quarter-hours without missing intervals.
* Features ($K$): 936 variables defined in [`features.csv`](features.csv) (338 direct active features for $k=0$, 52 safe ex-post neighbor inputs for $k \in [-4, -1]$ under `only past products`, 520 excluded under `no`).
* Missing values: Zero NaNs via hierarchical fallback imputation.
* Time standard: Pure UTC (`+00:00` / `Z`) across all sources and models.

### Integrated Data Sources

| Domain | Source & Description |
| :--- | :--- |
| Continuous Intraday | EPEX continuous 15-min rolling VWAPs, traded volumes, trade counts, and bilateral flow balances across 25 lead-time windows (`345to360` down to `0to30`). |
| Day-Ahead Spot | EPEX Day-Ahead hourly auction prices cleared at 12:00 D-1. |
| Intraday Spot | EPEX Intraday 15-minute auction spot prices cleared at 15:00 D-1. |
| Balancing Energy | Operational activated balancing reserves (aFRR / SRL and mFRR / MRL) from netztransparenz.org mapped to TSO zones. |
| Fundamentals | Total load, solar, wind (onshore/offshore), and cross-border commercial trading flows from Fraunhofer ISE Energy-Charts (ENTSO-E data). |
| Calendar & Lags | Wall-clock indicators (`Weekday`, `Hour`, `Quarter`) and historical price lags (`lag_24h`, `lag_48h`, `lag_168h`). |

---

## 2. Methodology & Information Availability

1. Information cutoff & publication buffer:
   * Nominal trading cutoff: 60 minutes before delivery ($t_{\text{delivery}} - 60\text{ min}$).
   * Publication buffer: 30 minutes (2 quarter-hours).
   * Effective ex-ante cutoff: 90 minutes before delivery ($t_{\text{delivery}} - 90\text{ min}$).
   * Naive benchmark: `{zone}_VWAP_90to105`.
2. Ex-post features:
   * Realized fundamentals (generation, load, balancing activations) are lagged by at least 8 quarter-hours (2 hours) prior to delivery.
3. Neighbor contract availability:
   * For neighbor contract $k \in [-4, +2]$, a window ending $A$ minutes before delivery is admissible if $A \ge 90 + 15 \cdot k$.

---

## 3. Models

Models are trained independently per delivery quarter-hour ($q \in \{0, \dots, 95\}$) using a rolling 822-day historical lookback window (~2.25 years):

### 3.1 LASSO
* Standard LASSO: L1-regularized linear regression with temporal expanding-window cross-validation for alpha selection over the last 14 days of the training window (Marcjasz et al., 2020).
* Weighted LASSO: Soft-kernel regime distance weighting ($w \propto \exp(-d/\tau)$) paired with the 1-standard-error rule for alpha selection.

### 3.2 cSVR (Corrected Support Vector Regression)
Port of Puć & Janczura (2024, Eq. 12).
* Kernel: Product of a Laplace kernel on standardized features and a Gaussian correction kernel on the standardized naive benchmark:
  $$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|_2\right) \cdot \exp\left(-\frac{1}{2\sigma^2} (P_i^{\text{naive}} - P_j^{\text{naive}})^2\right)$$
* Correlation filter: Following Puć & Janczura (2024), collinear features are pruned on the training set ($|r| \ge 0.80$ on S1, $|r| \ge 0.95$ on S2; toggleable via `CSVR_USE_CORR_FILTER` in [`config.py`](config.py)).

### 3.3 MAML-NN
Meta-learning neural network (Finn et al., 2017) trained directly on price spreads:
* Architecture: 2-layer MLP (`[128, 64]`) with Tanh activation, L1 weight regularization ($10^{-4}$), and proximal soft-thresholding.
* Ex-ante tuning: Architecture, support size ($K=28$), inner learning rate ($\alpha = 0.0025$), and kernel scaling ($\tau = 4.0$) tuned on historical 2021–2023 data and rolling 30-day ex-ante validation windows ($T-30$ to $T-1$).
* Support sampling: Inner-loop adaptation (8 gradient steps) on the $K=28$ nearest historical regimes selected via L1 feature distance with soft-kernel weighting.
* Deep ensembling: Averages predictions across multiple random initializations (`seeds = [42, 123, 999]`) per feature set to reduce epistemic variance (Lago et al., 2021).

---

## 4. Feature Sets & Ensembling

Models are trained on four feature sets:

| Set | Name | Primary Content |
| :---: | :--- | :--- |
| S1 | Macro | Own VWAP trajectory, auction prices, auto-regressive lags, calendar dummies |
| S2 | Neighbor | Leakage-safe price trajectories of adjacent contracts ($k \in [-4, +2]$), cross-zone spreads |
| S3 | Fundamentals | Wind, solar, load forecasts, balancing reserves, and cross-border commercial trading flows |
| S4 | Zonal Balances | Bilateral intraday commercial flow balances and self-trading positions across delivery zones |

### Rolling Weighted Forecast Averaging
Predictions across feature sets are combined via ex-ante inverse-MAE weighting (Marcjasz et al., 2018; Puć & Janczura, 2024):

$$w_{j,q}^W = \frac{\frac{1}{\text{MAE}_{j,q}^W}}{\sum_{k} \frac{1}{\text{MAE}_{k,q}^W}}$$

where $\text{MAE}_{j,q}^W$ is computed per quarter-hour $q$ over the preceding $W = 14$ days (`ENSEMBLE_QH_ADAPTIVE = True`). Ensembles are strictly intra-architecture (Pure cSVR Ensemble vs Pure MAML-NN Ensemble).

---

## 5. Repository Structure

```text
Delivery-Area-EPF/
├── AGENTS.md                          # Workspace rules, TSO mapping & benchmark ground truth
├── README.md                          # Methodology, architecture & reproduction guide
├── requirements.txt                   # Project dependencies
├── config.py                          # Central configuration and hyperparameters
├── features.csv                       # Feature catalog (910 variable definitions)
├── data_loader.py                     # Rolling QH data loader with ex-ante shifting & scaling
│
├── Data/
│   ├── master_dataset.parquet         # Master dataset (114,060 x 936 float32 columns)
│   ├── master_dataset_2021_2024.csv   # Dataset CSV archive
│   └── [Raw Auction & Market Directories]
│
├── Models/
│   ├── __init__.py                    # Module exports
│   ├── linear.py                      # Standard LASSO & Weighted LASSO
│   ├── csvr.py                        # cSVR with Laplace and Gaussian correction kernels
│   ├── maml_nn.py                     # MAML-NN engine with fast inner adaptation
│   └── ensemble.py                    # Rolling inverse-MAE forecast averaging
│
├── run_experiment.py                  # Simulation engine (rolling backtest across DE1-DE4)
│
├── Evaluation/
│   ├── run_full4sets_evaluation.py    # 15-minute QH curves and rMAE heatmaps generator
│   └── plots/                         # Generated evaluation figures
│
├── Results/
│   ├── NPZ_STRUCTURE.md               # Schema documentation for compressed results (.npz)
│   └── annual_run_2024/               # 2024 backtest results and metrics
│
└── old/                               # Archived exploratory experiments and legacy logs
```

---

## 6. Reproduction & Execution

### Step 1: Data Preparation
```bash
# Execute the data integration notebook:
jupyter execute create_master_dataset.ipynb
```
Exports `Data/master_dataset_2021_2024.csv` and `Data/master_dataset.parquet`.

### Step 2: Simulation & Backtest
```bash
# Standard run (resumes from checkpoints):
python run_experiment.py

# Force full re-training from scratch:
python run_experiment.py --force

# Summary metrics of existing runs:
python run_experiment.py --summary_only
```

### Step 3: Evaluation & Plots
```bash
python Evaluation/run_full4sets_evaluation.py
```
Generates 96-interval intraday MAE curves and hourly rMAE heatmaps under `Evaluation/plots/`.

---

## 7. Benchmark Results (Calendar Year 2024, MAE in EUR/MWh)

Evaluated over all 35,132 continuous quarter-hours of 2024:

| Model / Configuration | DE1 | DE2 (Amprion) | DE3 | DE4 | Average | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Naive Benchmark (`VWAP_90to105`) | 26.2815 | 26.6818 | 27.2725 | 27.2536 | 26.8724 | Persistent baseline |
| Scenario 1: Standard LASSO | 26.6651 | 27.2077 | 27.9160 | 27.8554 | 27.4111 | L1 regularization (Marcjasz et al., 2020) |
| Scenario 2: Pure cSVR Ensemble (with corr filter) | 25.8960 | 26.2345 | 26.9249 | 26.7819 | 26.4593 | Puć & Janczura (2024) standard benchmark |
| Scenario 2b: Pure cSVR Ensemble (no corr filter) | — | 26.2044 | — | — | — | Full feature retention (ablation study) |
| Scenario 3: MAML-NN with linear bypass ($m=0.15$) | 25.9000 | 26.2781 | 26.9485 | 26.8383 | 26.4912 | Hybrid meta-learning baseline |
| Scenario 4: Pure MAML-NN (Single seed 42) | 25.8803 | 26.1263 | 26.8084 | 26.7293 | 26.3861 | End-to-end meta-learning |
| Scenario 5: Pure MAML-NN (Deep Ensemble, 3 seeds) | **25.8554** | **26.1087** | **26.7675** | **26.7025** | **26.3585** | Deep ensemble (lowest MAE) |

### Statistical Significance (Multivariate Daily HAC Diebold-Mariano Test)
Following Ziel & Weron (2018) and Lago et al. (2021), loss differentials between Pure MAML-NN Deep Ensemble and Pure cSVR Ensemble are tested using daily mean errors with Newey-West HAC estimation ($h = 7$ days):

| Metric | DE1 | DE2 (Amprion) | DE3 | DE4 | Average |
| :--- | :---: | :---: | :---: | :---: | :---: |
| HAC DM Statistic ($d_{\text{MAML}} - d_{\text{cSVR}}$) | -1.0659 | -3.1073 | -4.1124 | -1.9185 | -2.5510 |
| HAC $p$-value | 0.2865 | 0.0019 | < 0.0001 | 0.0550 | — |
| Significance Level | — | $p < 0.01$ (**) | $p < 0.001$ (***) | $p < 0.10$ (*) | Consistent Advantage |

---

## 8. References

* Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In *International Conference on Machine Learning (ICML)* (pp. 1126–1135). PMLR.
* Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*, 293, 116983.
* Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
* Marcjasz, G., Serafin, T., & Weron, R. (2018). Selection of calibration windows for day-ahead electricity price forecasting. *Energies*, 11(9), 2364.
* Marcjasz, G., Uniejewski, B., & Weron, R. (2020). Beating the Naïve — Combining LASSO with Naïve Intraday Electricity Price Forecasts. *Energies*, 13(7), 1667.
* Puć, A., & Janczura, J. (2024). Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market. *International Journal of Forecasting* (Preprint: arXiv:2411.16237v1).
