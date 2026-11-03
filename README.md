# Delivery-Area-EPF

## Overview

Price forecasting for the final 30 minutes of the German continuous intraday market, where trading is restricted to the four control areas (delivery areas):

| Code | Delivery area |
| :--- | :--- |
| `DE1` | `DE-ENBW` |
| `DE2` | `DE-AMP` |
| `DE3` | `DE-TPS` |
| `DE4` | `DE-50HZ` |

Target: zonal VWAP of the window 0–30 min before delivery (`{zone}_VWAP_0to30`), forecast 60 min before delivery with information available 90 min before delivery. Naive benchmark: `{zone}_VWAP_90to105`. Test period: calendar year 2024 (35,136 quarter-hours per zone), rolling calibration window of 822 days.

## AI Disclosure

This project was created with assistance of generative AI. 

- **Author:** Mathis Wilz. The experiment has existed since early 2023 without AI and has been developed continuously. At their core, idea, code and text come from the author.
- **Claude Opus 5.5 (Anthropic):** code, data analysis and paper revision.
- **Gemini 3.5 to 3.8 Flash (Google):** coding.
- **Earlier models:** DeepSeek Pro v4 and DeepSeek Coder (hosted locally), coding.

## Models

| Key | Model |
| :--- | :--- |
| `lasso` | LASSO, penalty selected by LARS and expanding-window CV (Puć & Janczura, 2024) |
| `lasso_n` | 50:50 average of LASSO and naive (Marcjasz et al., 2020) |
| `csvr_corr` | Corrected SVR (Puć & Janczura, 2024) with correlation filter, main benchmark |
| `csvr` | cSVR without correlation filter (post-hoc modification) |
| `fann` | Neural network with fast local adaptation (main model) |
| `fann_static` | Model variant: FA-NN backbone without local adaptation (G = 0) |
| `fomaml`, `fomaml_scratch` | Model variants: first-order MAML initialization |
| `*_bypass` | FA-NN with LASSO bypass: fits the residual of a saturated LASSO forecast (post-hoc modification) |

Each model is estimated per zone, product and feature set (S1–S4); the four set forecasts are combined with rolling inverse-MAE weights.

## Data

The EPEX SPOT transaction data are not public. The dataset cannot be shared. Fundamentals (Fraunhofer ISE energy-charts) and balancing energy (netztransparenz.de) are public.

`features.csv` lists every column of the master dataset once: `tag` (`input`, `excluded`, `label`, `index`), set membership (S1–S4), `ex_post_lag_qh` (lag of ex-post data in quarter-hours until publication at the forecast time: 6 for balancing energy, 9 for ENTSO-E actuals), `window_end_min` (end of the trading window in minutes before delivery of the target contract) and `derived_from`.

## Reproduction

```bash
pip install -r requirements.txt
jupyter execute create_master_dataset.ipynb   # raw data -> Data/master_dataset.parquet (incl. feature_engineering.py)
python run_experiment.py                       # backtest -> Results/backtest_2024/results_<zone>.npz, runtime_<zone>.csv, summary.csv
python Evaluation/make_tables.py               # tables -> Contribution/tables/*.tex, Results/backtest_2024/tables/*.csv
python Evaluation/draft_figures.py             # figures -> Evaluation/plots/drafts/
python Evaluation/runtime_benchmark.py         # single-thread runtimes -> Results/runtime_benchmark/
python run_experiment.py --models fann --seed 1  # seed robustness -> Results/backtest_2024/seed_1/
```

All settings are in `config.py`.

## Result files

`Results/backtest_2024/results_<zone>.npz`, arrays of shape (days, 100) unless noted: 96 local products plus the second deliveries of 02:00-02:45 on the autumn DST day:

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
Evaluation/                tables, figures, runtime benchmark, 2023 pre-tests
tests/                     consistency and regression tests
old/                       archived code, data and results
```
