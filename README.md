# Delivery-Area-EPF: Regional Electricity Price Forecasting Pipeline

Machine Learning pipeline for **regional Electricity Price Forecasting (EPF)** across the four German Transmission System Operator (TSO) delivery areas: **TransnetBW (`DE1`)**, **Amprion (`DE2`)**, **TenneT (`DE3`)**, and **50Hertz (`DE4`)**, alongside the German national benchmark (**`DE`**).

---

## 1. Overview & Master Dataset

The repository builds the training dataset ([`Data/master_dataset_2021_2024.csv`](Data/master_dataset_2021_2024.csv)) for 15-minute continuous electricity contracts covering **October 1, 2021 to December 31, 2024** (Q4 2021 through 2024).

* **Observations ($N$):** 114,052 continuous quarter-hours.
* **Features ($K$):** 962 aligned cross-domain variables.
* **Missing Values:** **0 NaNs (0.00%)** via hierarchical imputation.
* **Time Standard:** Strictly **Pure UTC (`+00:00` / `Z`)** across all sources.

---

## 2. Integrated Data

| Data Domain | Directory | Description & Sources |
| :--- | :--- | :--- |
| **Continuous Intraday** | `Data/EPEX Intraday Continuous/` | EPEX continuous 15-min rolling VWAPs, traded volumes, trade counts, and bilateral inter-zonal flow balances across 25 lead-time windows (`345to360` down to `0to30`). |
| **Day-Ahead Spot** | `Data/DA Auction Spot Prices/` | EPEX Day-Ahead hourly auction prices cleared at 12:00 D-1. |
| **Intraday Spot** | `Data/ID Auction Spot Prices/` | EPEX Intraday 15-minute auction spot prices cleared at 15:00 D-1. |
| **Balancing Energy** | `Data/Balancing Energy/` | Operational activated balancing reserves (SRL / aFRR and MRL / mFRR, positive and negative) mapped to TSO zones. |
| **Fundamentals** | `Data/Fundamentals/` | Energy-Charts generation, demand, and cross-border fundamentals across 5 categories: `LOAD`, `SOLAR`, `ONSHORE`, `OFFSHORE`, and `CROSS_BORDER` (actuals, Day-Ahead forecasts, forecast errors, and net cross-border commercial trading flows). |
| **Calendar Dummies** | *Generated in-pipeline* | Granular wall-clock delivery indicators: `Weekday_1..7`, `Hour_0..23`, and `Quarter_1..4` (Europe/Berlin). |
| **Multi-Day Lags** | *Generated in-pipeline* | Historical multi-day shifts: `lag_24h` (96 steps), `lag_48h` (192 steps), and `lag_168h` (672 steps) for key price benchmarks. |

---

## 3. Forecasting Models

The pipeline evaluates four complementary model classes, each trained independently per quarter-hour (QH 0..95) on a rolling 822-day lookback window:

### 3.1 Linear Models

| Model | Description |
| :--- | :--- |
| **LR (OLS)** | Unregularized linear regression baseline. | -> Not used in final Pipeline
| **LASSO** | L1-regularized regression with `LassoLarsCV` alpha selection using expanding-window temporal cross-validation (last 14 days). |

### 3.2 Nonlinear Kernel Model

| Model | Description |
| :--- | :--- |
| **cSVR** | Corrected SVR with Laplacian kernel following Puć & Janczura (2024). Kernel width computed via quantile-based distance scaling. SOTA non-deep-learning benchmark in continuous intraday EPF. |

### 3.3 Meta-Learning Neural Network

| Model | Description |
| :--- | :--- |
| **MAML-NN** | Model-Agnostic Meta-Learning (Finn et al., 2017) residual MLP. Addresses the small-sample challenge (792 obs/QH) by pre-training shared meta-backbones across all 96 quarter-hours (~76,000 pooled observations), then performing fast inner-loop adaptation (5 gradient steps) per QH on regime-specific support sets. |

**Architecture:** `y_hat = p_linear + MLP(x)` (where linear bypass is modular and toggleable via `MAML_USE_LINEAR_BYPASS`).

Key components:
- **Residual MLP:** 2-layer `[64, 32]` architecture with GELU activation, LayerNorm, Dropout (0.10), L1 shrinkage penalty ($5 \times 10^{-4}$), and proximal soft-thresholding.
- **Bounded Linear Residual Bypass:** LASSO linear projection with smooth `tanh` saturation ($M = 0.10\sigma_y$) anchors the non-stationary mean, letting the network specialize in nonlinear microstructural dynamics.
- **Regime-Aware Support Selection (`regime_l1`):** MAML selects the $K=14$ nearest historical market regimes within a 30-day lookback pool via $L_1$ distance in feature space for inner-loop task adaptation.
- **Quarterly Rolling Meta-Backbone Retraining (4x/Year):** Meta-backbones are re-trained at the start of each quarter (Q1: 01.01, Q2: 01.04, Q3: 01.07, Q4: 01.10) on rolling 822-day windows, preventing distribution shift from new PV capacity without look-ahead bias.
- **Adaptive Delta Anchoring:** Dynamically anchors predicted deltas ($\hat{y} = P^{\text{naive}} + \gamma(t, q) \cdot \hat{\Delta}_{\text{MAML}}$) using ex-ante quarter-hour empirical Bayes confidence, benchmark extremity filters, and dynamic amplitude saturation.
- **Multi-Seed Deep Ensembling (Lakshminarayanan et al., 2017):** Backbones are trained independently across multiple initialization seeds (`MAML_SEEDS = [42, 123, 999]`). Combining predictions across seeds eliminates neural initialization variance and stabilizes out-of-sample forecasts.

### 3.4 Multi-Set Weighted Forecast Averaging

All models are evaluated across **4 thematic feature sets** (see Section 4), producing base predictions per QH. Following the **weighted forecast averaging scheme** of **Puć & Janczura (2024, Eq. 25)** and **Bates & Granger (1969)**, predictions are combined using rolling out-of-sample inverse-MAE weights:

$$w_j = \frac{\left(\text{MAE}_j^W\right)^{-p}}{\sum_{k} \left(\text{MAE}_k^W\right)^{-p}}$$

Where $W$ is the historical calibration window ($W = 28$ days) and $p$ is the power exponent ($p = 2.0$ represents inverse-variance weighting).

Evaluated ensemble variants:
1. **Pure cSVR Ensemble (`csvr_ens_wn`)** — cSVR across all 4 sets + Naive benchmark (direct counterpart to Puć & Janczura's `cSVR_w.avg.`)
2. **Pure MAML-NN Ensemble (`maml_ens_pure`)** — Multi-Seed MAML across all 4 sets (Pure MAML, NO cSVR)
3. **Pure MAML-NN + Naive Ensemble (`maml_ens_wn`)** — Multi-Seed MAML across all 4 sets + Naive benchmark
4. **Pure MAML-NN QH-Adaptive Ensemble (`maml_ens_adapt`)** — Time-of-Day adaptive weighting per 15-minute position

---

### 3.5 Methodological Comparison: Puć & Janczura (2024) vs. This Pipeline

| Dimension | Puć & Janczura (2024) | Delivery-Area-EPF (This Project) |
| :--- | :--- | :--- |
| **Market Scope** | Single aggregate German market (DE-LU) | **4 Regional Delivery Areas** (TransnetBW, Amprion, TenneT, 50Hertz) |
| **Target Variable** | Point transaction price $P_{d,T}(s)$ at minute $s$ | **Quarter-hourly volume-weighted average price** (`VWAP_0to30`) |
| **Look-Ahead & Delay** | 20-minute publication buffer | **30-minute operational delay** (2 QH grid-aligned buffer, cutoff at $t-90$ min) |
| **Feature Sets** | 3 sets: $S^1$ (Trajectory + Exog), $S^2$ (Close Prices), $S^3$ (Pure Exog) | **4 sets:** S1 (Macro), S2 (Neighbor Contracts), S3 (Fundamentals), **S4 (Zonal Flow Balances)** |
| **Evaluated Models** | Naive, SVR, cSVR, LASSO, Random Forest | Naive, LASSO, cSVR, and **MAML-NN (Meta-Learning with GNCL)** |
| **Forecast Averaging** | Weighted average over 3 feature sets + Naive per model (Eq. 25) | Weighted average over 4 feature sets per model + cross-family Hybrid Ensemble |
| **Evaluation Window** | Calendar year 2020 | **Calendar year 2024** (35,132 continuous quarter-hours) |

---

## 4. Feature Sets

Each model is trained on 4 complementary, non-overlapping feature subsets to maximize ensemble diversity:

| Set | Name | Features | Description |
| :---: | :--- | :---: | :--- |
| **S1** | Macro & Fundamentals | ~120 | Own VWAP history, auction prices, AR lags, generation, load, balancing reserves, calendar dummies |
| **S2** | Neighbor Contracts | ~31 | Price trajectories of adjacent QH contracts ($k \in [-4, +2]$), cross-zone spreads, volumes, trades |
| **S3** | Pure Fundamentals | ~40 | Solar, wind, load, balancing reserves, calendar features (no price data) |
| **S4** | Zone Balances | ~91 | Bilateral intraday commercial flow balances and self-trading positions across delivery zones |

---

## 5. Scientific & Methodological Standards

1. **TSO Delivery Area Mapping:**
   * `DE1_` = TransnetBW (DE-ENBW)
   * `DE2_` = Amprion (DE-AMP)
   * `DE3_` = TenneT (DE-TPS)
   * `DE4_` = 50Hertz (DE-50HZ)
   * Raw corporate names are strictly forbidden in feature names and model selectors per [`AGENTS.md`](AGENTS.md).
2. **Canonical Forecasting Targets & Naive Benchmark:**
   * Targets: `DE1_VWAP_0to30`, `DE2_VWAP_0to30`, `DE3_VWAP_0to30`, `DE4_VWAP_0to30`.
   * The national aggregate `VWAP_0to30` serves strictly as an internal fallback for zero-trade regional intervals.
   * The canonical ex-ante benchmark at the effective 90-minute cutoff is `{zone}_VWAP_90to105`.
3. **Three-Stage Look-Ahead Prevention:**
   * **Nominal Cutoff:** 60 minutes before delivery.
   * **Reporting Delay:** 30-minute buffer (following Puć & Janczura's 20-minute operational delay, rounded up to the 15-minute grid).
   * **Effective Cutoff:** 90 minutes before delivery — the last safe window is `VWAP_90to105`.
   * **Ex-Post Features:** All realized fundamentals lagged by 8 QH (2 hours).
4. **Hierarchical Imputation Protocol:**
   * Regional zones: Own trade → contemporaneous national VWAP in the same window.
   * National `VWAP_345to360`: Fallback to Intraday Auction Price; subsequent windows forward-fill.
   * Non-price metrics (volumes, trades, flows): filled with `0.0`.
5. **Experimental Setup & Rolling Backtest:**
   * **Lookback:** Fixed rolling window of 822 days (~2.25 years).
   * **Validation:** Last 30 days of lookback window for parameter selection.
   * **Test Period:** Calendar year 2024 (`2024-01-01` to `2024-12-31`), 35,132 out-of-sample quarter-hours.

---

## 6. Execution

### Generate the master dataset:
```bash
jupyter notebook create_master_dataset.ipynb
```

### Convert to Parquet (one-time):
```bash
python Data/convert_to_parquet.py
```

### Run the full 4-set backtest (2024):
```bash
conda activate deep_learning
python run_full_4sets_backtest_2024.py --zone all
```

### Or update MAML predictions only (fast re-evaluation using cached cSVR/LASSO):
```bash
python update_maml_only_2024.py --zone all
```

### Aggregate results & generate publication plots:
```bash
python aggregate_4sets_results.py
python Evaluation/run_full4sets_evaluation.py
```

---

## 7. Repository Structure

```text
├── AGENTS.md                           # Operational rules, mapping conventions & scientific constraints
├── README.md                           # Project overview & architecture documentation
├── config.py                           # Central paths, lookback (822d), validation (30d), model hyperparameters
├── data_loader.py                      # In-memory QH data provider with 4 feature-set routing
├── evaluate.py                         # MAE, rMAE, and Diebold-Mariano evaluation metrics
├── features.csv                        # Version-controlled feature catalog (962 definitions)
├── create_master_dataset.ipynb         # Interactive master dataset generation notebook
├── run_full_4sets_backtest_2024.py     # Main 4-set annual backtest runner from scratch
├── update_maml_only_2024.py            # Fast MAML-only backtest updater (re-using precomputed cSVR/LASSO)
├── aggregate_4sets_results.py          # Summary aggregation across all 4 zones & 4 feature sets
│
├── Models/
│   ├── __init__.py                     # Package exports for all model classes
│   ├── linear.py                       # OLS and LASSO (LassoLarsCV) implementations
│   ├── csvr.py                         # Corrected SVR with Laplacian kernel (Puć & Janczura 2024)
│   ├── maml_nn.py                      # MAML-NN with GNCL backbone pre-training (TensorFlow/Keras)
│   └── ensemble.py                     # Rolling inverse-MAE weighted forecast averaging (Puć & Janczura, 2024)
│
├── Evaluation/
│   ├── run_full4sets_evaluation.py     # 4-set hourly comparison plots & publication-quality heatmaps
│   └── plots/                          # Generated PNG publication figures
│
├── Results/
│   ├── NPZ_STRUCTURE.md               # NPZ array schema documentation
│   ├── annual_run_2024/               # 2024 backtest outputs (.npz, ensemble weights, summaries)
│   └── maml_hpo/                      # Best HPO configurations and trial logs
│
├── Old/                                # Archived previous runs, intermediate results & legacy scripts
│   ├── scripts/                        # Legacy backtest, runner, and HPO scripts (tune_maml_de2.py etc.)
│   ├── results/                        # Older pre-4sets results & checkpoints
│   ├── logs/                           # Historical execution logs
│   └── plots/                          # Historical plots from preliminary runs
│
└── Data/
    ├── master_dataset_2021_2024.csv    # Final 114,052 × 962 ML training dataset
    ├── master_dataset.parquet          # Fast columnar Parquet format
    ├── convert_to_parquet.py           # One-shot CSV → Parquet conversion utility
    ├── DA Auction Spot Prices/         # Day-Ahead auction processing
    ├── ID Auction Spot Prices/         # Intraday auction processing
    ├── EPEX Intraday Continuous/       # Raw continuous trading data (20 files)
    ├── Balancing Energy/               # SRL and MRL operational reserves
    └── Fundamentals/                   # Generation, load & cross-border data
```

---

## 8. Backtest Output & NPZ Specification

Results from rolling backtests are stored as compressed NumPy archives (`.npz`) in `Results/annual_run_2024/`.
For the exact array schema, shapes, data types, and usage examples, see:
-> [Results/NPZ_STRUCTURE.md](Results/NPZ_STRUCTURE.md)

---

## 9. References

* Bates, J. M., & Granger, C. W. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451–468.
* Buschjäger, S., Pfahler, L., & Morik, K. (2020). Generalized Negative Correlation Learning for Deep Ensembling. *arXiv preprint arXiv:2011.02952*. Official Repository: [sbuschjaeger/gncl](https://github.com/sbuschjaeger/gncl).
* Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In *International Conference on Machine Learning (ICML)* (pp. 1126–1135). PMLR.
* Liu, Y., & Yao, X. (1999). Ensemble learning via negative correlation. *Neural Networks*, 12(10), 1399–1404.
* Marcjasz, G., Uniejewski, B., & Weron, R. (2020). Beating the Naïve — Combining LASSO with Naïve Intraday Electricity Price Forecasts. *Energies*, 13(7), 1667.
* Puć, A., & Janczura, J. (2024). Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market. *International Journal of Forecasting* (Preprint: arXiv:2411.13945).
