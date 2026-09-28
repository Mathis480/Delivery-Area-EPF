# NPZ Prediction & Result File Specification

This document defines the exact schema, array shapes, and data types stored in the compressed NumPy result archives for each delivery area (`DE1`, `DE2`, `DE3`, `DE4`).

All files are located in `Results/annual_run_2024/` and generated via `np.savez_compressed`.

---

## 1. File Types

| File Pattern | Description |
| :--- | :--- |
| `results_{zone}_2024.npz` | Base results: LR, LASSO, cSVR on Set 1 (Macro) |
| `results_{zone}_multiset_2024.npz` | Extended results: all 4 feature sets, all models, all ensembles |
| `results_{zone}_full4sets_2024.npz` | Full 4-set results (identical to multiset, kept for backward compatibility) |

---

## 2. Array Overview & Keys

### 2.1 Core Arrays (all files)

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `feature_names` | `(n_features,)` | `str / <U32` | Names of all active input features for Set 1 |
| `dates` | `(N,)` | `str / <U10` | Test date string (`"YYYY-MM-DD"`) for each 15-min interval ($N = 35{,}132$) |
| `qh_idx` | `(N,)` | `int16` | Quarter-hour index ($0 \dots 95$) within the delivery day |
| `y_true` | `(N,)` | `float32` | Realized target price $\text{VWAP}_{\text{0to30}}$ in EUR/MWh |
| `benchmark` | `(N,)` | `float32` | Naive benchmark price $\text{VWAP}_{\text{90to105}}$ in EUR/MWh |
| `dates_daily` | `(n_days,)` | `str / <U10` | Unique calendar dates ($n_{\text{days}} = 366$ for leap year 2024) |
| `y_true_2d` | `(n_days, 96)` | `float32` | 2D matrix of realized prices in EUR/MWh |
| `benchmark_2d` | `(n_days, 96)` | `float32` | 2D matrix of naive benchmark prices in EUR/MWh |

### 2.2 Model Predictions — Set 1 (Macro)

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `pred_lr` / `pred_lr_2d` | `(N,)` / `(n_days, 96)` | `float32` | Linear Regression (OLS) predictions |
| `pred_lasso` / `pred_lasso_2d` | `(N,)` / `(n_days, 96)` | `float32` | LASSO predictions |
| `pred_csvr` / `pred_csvr_2d` | `(N,)` / `(n_days, 96)` | `float32` | cSVR predictions |
| `pred_maml` / `pred_maml_s1` | `(N,)` | `float32` | MAML-NN predictions on Set 1 (alias) |
| `pred_maml_s1_2d` | `(n_days, 96)` | `float32` | MAML-NN Set 1 predictions (2D) |

### 2.3 Model Predictions — Sets 2, 3, 4 (multiset/full4sets files only)

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `pred_lasso_s2` / `pred_lasso_s2_2d` | `(N,)` / `(n_days, 96)` | `float32` | LASSO on Set 2 (Neighbor) |
| `pred_csvr_s2` / `pred_csvr_s2_2d` | `(N,)` / `(n_days, 96)` | `float32` | cSVR on Set 2 |
| `pred_maml_s2` / `pred_maml_s2_2d` | `(N,)` / `(n_days, 96)` | `float32` | MAML-NN on Set 2 |
| `pred_lasso_s3` / `pred_lasso_s3_2d` | `(N,)` / `(n_days, 96)` | `float32` | LASSO on Set 3 (Fundamental) |
| `pred_csvr_s3` / `pred_csvr_s3_2d` | `(N,)` / `(n_days, 96)` | `float32` | cSVR on Set 3 |
| `pred_maml_s3` / `pred_maml_s3_2d` | `(N,)` / `(n_days, 96)` | `float32` | MAML-NN on Set 3 |
| `pred_lasso_s4` / `pred_lasso_s4_2d` | `(N,)` / `(n_days, 96)` | `float32` | LASSO on Set 4 (Balance) |
| `pred_csvr_s4` / `pred_csvr_s4_2d` | `(N,)` / `(n_days, 96)` | `float32` | cSVR on Set 4 |
| `pred_maml_s4` / `pred_maml_s4_2d` | `(N,)` / `(n_days, 96)` | `float32` | MAML-NN on Set 4 |

### 2.4 Ensemble Predictions (multiset/full4sets files only)

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `pred_csvr_ens_pure` / `pred_csvr_ens_pure_2d` | `(N,)` / `(n_days, 96)` | `float32/64` | Pure cSVR ensemble (S1+S2+S3+S4) |
| `pred_csvr_ens_wn` / `pred_csvr_ens_wn_2d` | `(N,)` / `(n_days, 96)` | `float32/64` | cSVR ensemble + naive benchmark |
| `pred_maml_ens_pure` / `pred_maml_ens_pure_2d` | `(N,)` / `(n_days, 96)` | `float32/64` | Pure MAML-NN ensemble (S1+S2+S3+S4) |
| `pred_maml_ens_wn` / `pred_maml_ens_wn_2d` | `(N,)` / `(n_days, 96)` | `float32/64` | MAML-NN ensemble + naive benchmark |
| `pred_ens` / `pred_ens_2d` | `(N,)` / `(n_days, 96)` | `float32/64` | Consolidated hybrid ensemble (all 12 models + naive) |

### 2.5 Coefficient Arrays (base files only)

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `coef_lr` | `(N, n_features)` | `float32` | Per-QH OLS regression coefficients |
| `coef_lasso` | `(N, n_features)` | `float32` | Per-QH LASSO regression coefficients |
| `coef_lr_daily` | `(n_days, n_features)` | `float32` | Daily mean OLS coefficients |
| `coef_lasso_daily` | `(n_days, n_features)` | `float32` | Daily mean LASSO coefficients |

---

## 3. Python Usage Examples

### Loading Data
```python
import numpy as np

data = np.load("Results/annual_run_2024/results_DE2_full4sets_2024.npz")

y_true = data["y_true"]
bm = data["benchmark"]
pred_csvr_s1 = data["pred_csvr"]
pred_maml_s1 = data["pred_maml_s1"]
pred_ens = data["pred_ens"]
```

### Computing MAE and rMAE
```python
mae_naive = float(np.mean(np.abs(y_true - bm)))
mae_csvr = float(np.mean(np.abs(y_true - pred_csvr_s1)))
mae_ens = float(np.mean(np.abs(y_true - pred_ens)))

print(f"Naive MAE:    {mae_naive:.4f} EUR/MWh")
print(f"cSVR S1 MAE:  {mae_csvr:.4f} EUR/MWh (rMAE: {mae_csvr / mae_naive:.4f})")
print(f"Ensemble MAE: {mae_ens:.4f} EUR/MWh (rMAE: {mae_ens / mae_naive:.4f})")
```

### Diebold-Mariano Test
```python
from evaluate import dm_test

e_model = y_true - pred_ens
e_naive = y_true - bm

dm_stat, p_val = dm_test(e_model, e_naive)
print(f"DM test: stat={dm_stat:.3f}, p={p_val:.4e}")
```

### Feature Importance Tracking over Time
```python
import matplotlib.pyplot as plt

data_base = np.load("Results/annual_run_2024/results_DE2_2024.npz")
coef_lasso_daily = data_base["coef_lasso_daily"]  # Shape: (366, n_features)
feature_names = data_base["feature_names"]

# Find top-5 features by mean absolute coefficient
mean_abs = np.mean(np.abs(coef_lasso_daily), axis=0)
top5 = np.argsort(mean_abs)[-5:][::-1]

for idx in top5:
    plt.plot(coef_lasso_daily[:, idx], label=feature_names[idx])

plt.legend()
plt.title("LASSO Feature Coefficients over 2024")
plt.show()
```
