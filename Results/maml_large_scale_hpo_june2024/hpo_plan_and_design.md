# Large-Scale Hyperparameter & Architecture Optimization Plan (June 2024 - DE2 Amprion)

## 1. Executive Summary & Objective

This document outlines the architecture, search space, and execution methodology for the large-scale Bayesian Hyperparameter Optimization (HPO) targeting summer intraday electricity price forecasting (EPF) dynamics in **June 2024 for Delivery Area DE2 (Amprion)**.

### Target Performance Benchmarks (Ground Truth June 2024, N = 2,880 QH)
* **Naive Benchmark ($|VWAP_{0\to30} - VWAP_{90\to105}|$):** `48.8115 EUR/MWh` (rMAE: `1.0000`)
* **cSVR Ensemble (+Naive Benchmark):** `48.2286 EUR/MWh` (rMAE: `0.9881`)
* **Primary Objective:** Empirically beat the cSVR ensemble on June 2024 with the 4-set MAML-NN ensemble ($\text{MAE} < 48.2286\text{ EUR/MWh}$, $\text{rMAE} < 0.9881$, statistically validated via Diebold-Mariano test $p < 0.05$).

---

## 2. Mandatory Core Architectural Principles

### 2.1 Strictly All 4 Feature Sets Included
Following methodological integrity and empirical stability, no feature sets are dropped. Every trial trains and ensembles across:
1. **Set 1 (Macro / National Fundamentals):** Continuous intraday volume, auction prices, national load, solar, wind, and calendar cyclics.
2. **Set 2 (Neighbor / Temporal Lead-Lag VWAP):** Ex-ante safe lead-lag contract windows ($k \in [-4, +2]$) with temporal availability shifts ($A \ge 90 + 15k$).
3. **Set 3 (Regional Amprion Fundamentals & Cross-Border):** Regional solar/wind generation, local vertical load, and cross-border physical commercial exchange flows.
4. **Set 4 (Balancing Energy Reserves):** Secondary (SRL) and tertiary (MRL) activated balancing energy volumes and prices (strictly lagged by 2 hours / 8 QH).

### 2.2 Buschjäger et al. (2020) GNCL Difference Objective
Re-integration of Generalized Negative Correlation Learning (GNCL) based on Krogh & Vedelsby's ambiguity decomposition:
$$E_{\text{ens}} = \overline{E} - \lambda \cdot \overline{A}$$
where the diversity / difference penalty is:
$$\mathcal{L}_i(f_i; y) = \text{MAE}(y, f_i(x)) - \lambda_{\text{ncl}} \cdot \text{div}(f_i(x), \bar{f}(x))$$
* **Ambiguity Formulations Evaluated:**
  * **Squared Ambiguity:** $\text{div}(f_i, \bar{f}) = (f_i(x) - \bar{f}(x))^2$ (classic Buschjäger / Brown)
  * **Absolute L1 Ambiguity:** $\text{div}(f_i, \bar{f}) = |f_i(x) - \bar{f}(x)|$ (scale-matched to MAE subgradients)
* **Coupling Intensity $\lambda_{\text{ncl}}$:** $[0.00, 0.05, 0.10, 0.20, 0.35, 0.50]$
* **Comparative Baseline:** Convex joint ensemble loss: $\mathcal{L} = \lambda \text{MAE}(\bar{f}, y) + (1-\lambda) \overline{\text{MAE}}$.

---

## 3. Systematic Search Space Specification

The Bayesian Tree-structured Parzen Estimator (TPE) algorithm explores the following dimensions:

| Component | Parameter | Search Space / Values | Rationale |
| :--- | :--- | :--- | :--- |
| **Model Capacity** | `hidden_sizes` | **Small:** `[32]`, `[48]`, `[32, 16]`, `[48, 24]`, `[64, 32]`<br>**Medium:** `[64, 64]`, `[96, 48]`, `[80, 40]`, `[48, 32, 16]`, `[64, 48, 24]`<br>**Large:** `[128, 64]`, `[128, 128]`, `[160, 80]`, `[96, 64, 32]`, `[128, 64, 32]`, `[128, 96, 48]`, `[192, 96, 48]` | Tests whether higher model capacity captures nonlinear summer solar extremes, or if compact nets prevent overfitting. |
| **Nonlinear Activation** | `activation` | `gelu`, `relu`, `leaky_relu`, `elu`, `tanh` | GELU previously outperformed ReLU; tests smooth transitions vs. piecewise linear. |
| **Regularization** | `dropout`<br>`l1_reg` | $[0.0, 0.05, 0.10, 0.15, 0.20]$<br>$[0.0, 10^{-5}, 5\cdot 10^{-5}, 10^{-4}, 5\cdot 10^{-4}]$ | Sparsity enforcement on high-dimensional multi-set feature spaces. |
| **Buschjäger GNCL** | `ncl_mode`<br>`ambiguity_type`<br>`lambda_ncl` | `['difference', 'convex']`<br>`['squared', 'abs']`<br>$[0.0, 0.05, 0.10, 0.20, 0.35, 0.50]$ | Enforces predictive diversity across the 4 feature sets during shared meta-pretraining. |
| **Inner Meta-Adaptation** | `inner_lr`<br>`inner_steps`<br>`prox_shrink` | Log-uniform $[0.0005, 0.010]$<br>`[1, 2, 3, 5, 8, 10]`<br>$[0.0, 10^{-5}, 5\cdot 10^{-5}, 10^{-4}]$ | Controls the speed and depth of fast-weight updates on the recent support set. |
| **Support Conditioning** | `support_k`<br>`selection_mode` | `[5, 7, 10, 14, 21, 28]` days<br>`['regime_l1', 'chronological']` | Regime-based (L1 nearest neighbor manifold) vs. trailing calendar history conditioning. |
| **Linear Bypass & Saturation** | `use_linear_bypass`<br>`bypass_saturation_m` | `[True, False]`<br>$[0.06, 0.08, 0.10, 0.15, 0.25, 0.40, \text{None}]$ | $\tanh$-based bounded linear bypass preventing catastrophic overshooting during volatile solar hours. |
| **Forecast Ensembling** | `ensemble_power`<br>`calib_window`<br>`qh_adaptive` | `[1.0, 2.0, 3.0, 4.0]`<br>`[7, 14, 21, 28]` days<br>`[False, True]` | Rolling Bates-Granger inverse-MAE weighting; QH-adaptivity dynamically tracks intraday duck-curve volatility. |

---

## 4. Execution Architecture & Performance Optimizations

1. **Persistent Disk & RAM Caching:**
   * All 2,880 quarter-hours $\times$ 4 feature sets ($11,520$ data extractions and LASSO calibration models) are pre-cached in memory and stored to `Results/maml_large_scale_hpo_june2024/backbone_cache/context_cache_DE2_2024-06.pkl`.
   * Any process restart or continuation loads within **1-2 seconds**, avoiding repeated preprocessing.
2. **Compiled TensorFlow Graph Execution:**
   * Inner adaptation loops are executed via `@tf.function(reduce_retracing=True)` using compiled C++ graph operations.
   * Execution time per full 30-day evaluation ($2,880$ quarter-hours across 4 models) is reduced from $\sim 15\text{s}$ down to **$4 - 6\text{s}$ per trial**.
3. **Collinearity Handling:**
   * Safe fallback to `Ridge(alpha=100.0)` in the event that `LassoLarsCV` encounters rank-deficient degenerate covariance matrices.
4. **Dedicated Output Isolation:**
   * Script: `optimize_maml_large_scale_june2024.py`
   * Trials Log: `Results/maml_large_scale_hpo_june2024/trials.csv`
   * Champion Model: `Results/maml_large_scale_hpo_june2024/best_config.json`
   * Backbone Weights Cache: `Results/maml_large_scale_hpo_june2024/backbone_cache/`
5. **Time Budget:**
   * Allocated runtime: **13.8 hours** ($\approx 50,000$ seconds), allowing between **$2,500$ and $4,000$ complete trials**.
