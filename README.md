# Delivery-Area-EPF

Price forecasting for the final 30 minutes of the German continuous intraday market, in which trading is restricted to the four control areas (delivery areas):

| Code | Delivery area |
| :--- | :--- |
| `DE1` | `DE-ENBW` |
| `DE2` | `DE-AMP` |
| `DE3` | `DE-TPS` |
| `DE4` | `DE-50HZ` |

Target: zonal VWAP of the window 0–30 min before delivery (`{zone}_VWAP_0to30`), forecast 60 min before delivery with information available 90 min before delivery. Naive benchmark: `{zone}_VWAP_90to105`. Test period: 2024 (35,132 quarter-hours per zone), rolling calibration window of 822 days.

## Models

| Key | Model |
| :--- | :--- |
| `lasso` | LASSO, penalty selected by LARS and expanding-window CV (Puć & Janczura, 2024) |
| `lasso_n` | 50:50 average of LASSO and naive (Marcjasz et al., 2020) |
| `csvr` | Corrected SVR (Puć & Janczura, 2024), main benchmark without correlation filter |
| `csvr_corr` | cSVR with correlation filter (replication) |
| `fann` | Neural network with fast local adaptation (main model) |
| `fomaml`, `fomaml_scratch` | Ablations: first-order MAML initialization |
| `*_bypass` | FA-NN variants fitting the residual of a saturated LASSO forecast |

Each model is estimated per zone, product and feature set (S1–S4); the four set forecasts are combined with rolling inverse-MAE weights.

## Data

The EPEX SPOT transaction data are not public; the dataset cannot be shared. Fundamentals (Fraunhofer ISE energy-charts) and balancing energy (netztransparenz.de) are public.

`features.csv` lists every column of the master dataset once: `tag` (`input`, `excluded`, `label`, `index`), set membership (S1–S4), `shift_needed_2h` (ex-post data lagged by 2 h), `window_end_min` (end of the trading window in minutes before delivery of the target contract) and `derived_from`.

## Reproduction

```bash
pip install -r requirements.txt
jupyter execute create_master_dataset.ipynb   # raw data -> Data/master_dataset.parquet (incl. feature_engineering.py)
python run_experiment.py                       # backtest -> Results/backtest_2024/results_<zone>.npz, runtime_<zone>.csv, summary.csv
python Evaluation/draft_figures.py             # figures -> Evaluation/plots/drafts/
```

All settings are in `config.py`.

## Result files

`Results/backtest_2024/results_<zone>.npz`, arrays of shape (days, 96) unless noted:

| Key | Content |
| :--- | :--- |
| `dates` | delivery days (366) |
| `y_true`, `naive` | target and naive price (EUR/MWh) |
| `pred_<model>` | averaged forecast over the feature sets |
| `pred_<model>_s1` … `_s4` | forecast per feature set |
| `weights_<model>` | daily mean inverse-MAE weight per feature set (days, 4) |

`runtime_<zone>.csv` contains wall-clock seconds per component (`data`, `lasso`, `csvr`, `csvr_corr`, `<fann model>_stage1`, `<fann model>_stage2`).

## Structure

```text
config.py                  settings
create_master_dataset.ipynb  raw data -> master dataset
feature_engineering.py     derived features (S2 neighbors, S4 trade balances)
features.csv               feature catalog
data_loader.py             rolling windows, standardization, variable filters
Models/                    linear.py (LASSO), csvr.py, fann.py, ensemble.py (averaging, DM/GW tests)
run_experiment.py          backtest
Evaluation/                figures
tests/                     consistency and regression tests
old/                       archived code, data and results
```
