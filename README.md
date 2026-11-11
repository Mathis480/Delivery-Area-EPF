# Delivery-Area-EPF

## Overview

Price forecasting for the final 30 minutes of the German continuous intraday market, where trading is restricted to the four control areas (delivery areas):

| Code | Delivery area |
| :--- | :--- |
| `DE1` | `DE-ENBW` |
| `DE2` | `DE-AMP` |
| `DE3` | `DE-TPS` |
| `DE4` | `DE-50HZ` |

Target: zonal VWAP of the window 0–30 min before delivery (`{zone}_VWAP_0to30`), forecast 60 min before delivery using only information published by then, with continuous trades up to 90 min before delivery. Naive benchmark: `{zone}_VWAP_90to105`. Test period: calendar year 2024 (35,136 quarter-hours per zone), rolling calibration window of 822 days.

## AI Disclosure

This project was created with assistance of generative AI. 

- **Author:** Mathis Wilz. The experiment has existed since early 2023 without AI and has been developed continuously. At their core, idea, code and text come from the author.
- **Claude Opus 5.5 (Anthropic):** code, data analysis and paper revision.
- **Argon (Google):** code and paper review.
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
| `fomaml`, `fomaml_pre` | Model variants: first-order MAML without / after supervised pre-training |
| `fann_bypass` | FA-NN with LASSO bypass: fits the residual of a saturated LASSO forecast (post-hoc modification) |

Each model is estimated per zone, product and feature set (S1–S4); the four set forecasts are combined with rolling inverse-MAE weights.

## Data

The EPEX SPOT transaction data are not public. The dataset cannot be shared. Fundamentals (Fraunhofer ISE energy-charts) and balancing energy (netztransparenz.de) are public.

`features.csv` lists every column of the master dataset once: `tag` (`input`, `excluded`, `label`, `index`), set membership (S1–S4), `ex_post_lag_qh` (lag of ex-post data in quarter-hours until publication at the forecast time: 6 for balancing energy, 9 for ENTSO-E actuals, 12 for cross-border trading), `window_end_min` (end of the trading window in minutes before delivery of the target contract) and `derived_from`.

## Reproduction

All results were computed on CPU with Python 3.11.15 and the package versions pinned in `requirements.txt`:

```bash
conda create -n deep_learning python=3.11.15
conda activate deep_learning
pip install -r requirements.txt
```

Run from the repository root, in this order:

```bash
# 1. Data (raw files as described in Data/*/README.md)
python "Data/EPEX Intraday Continuous/calculate_ID_DEx_UTC.py"  # raw EPEX trades -> ID_DelArea_<area>_<year>.csv
jupyter execute create_master_dataset.ipynb   # -> Data/master_dataset.parquet (incl. feature_engineering.py); outputs are not written back
python -m unittest discover -s tests          # consistency tests
python tests/feature_snapshot.py              # loader regression check against hashes in tests/fixtures/ (not in the repository; create with --update)

# 2. Backtest 2024
python run_experiment.py                      # all models -> Results/backtest_2024/results_<zone>.npz, runtime_<zone>.csv, summary.csv
for s in 1 2 3 4; do python run_experiment.py --models fann --seed $s; python run_experiment.py --models fomaml,fomaml_pre --seed $s; done  # seed robustness -> Results/backtest_2024/seed_<s>/

# 3. Pre-tests on 2023 and diagnostics (design decisions, not reported in the paper unless noted)
Q2023=2023-01-01:2023-03-31,2023-04-01:2023-06-30,2023-07-01:2023-09-30,2023-10-01:2023-12-31
python Evaluation/variant_pretest.py --zones DE1,DE2,DE3,DE4 --periods $Q2023 current ln_pre
python Evaluation/variant_pretest.py --mode fomaml --zones DE1,DE2,DE3,DE4 --periods $Q2023 current ln_pre
python Evaluation/variant_pretest.py ln reg ln_reg             # regularization / layer normalization on DE2, Q4 2023
python Evaluation/fomaml_diagnostics.py       # FOMAML diagnostics and sensitivity (Q4 2023) -> Evaluation/plots/drafts/
python Evaluation/runtime_benchmark.py        # single-thread runtimes -> Results/runtime_benchmark/ (paper, Table 3)
# retraining reference (paper, Sec. 5.5): see below

# 4. Paper tables, figures and numbers
python Evaluation/make_tables.py              # -> Contribution/tables/tab_main.tex, tab_runtime.tex
python Evaluation/data_figures.py             # -> Contribution/SVG/Figure_1.pdf, Figure_2.pdf, Figure_target_window.pdf
python Evaluation/text_numbers.py             # numbers quoted in the text that are not in the tables (printed)
python Evaluation/draft_figures.py            # -> Contribution/SVG/Figure_3.pdf and Evaluation/plots/drafts/
```

### Retraining reference (paper, Sec. 5.5)

Two scripts compare the FA-NN with conventional per-forecast retraining on pre-test data of 2023 (DE2), so the test year is not touched:

- `Evaluation/backbone_epochs.py` trains one FA-NN backbone per 2023 quarter and feature set and evaluates snapshots after 6 to 200 epochs, with and without local adaptation. Snapshot `e6` is the backtest configuration and serves as the FA-NN reference.
- `Evaluation/retrain_baseline.py` trains a fresh network of the same architecture for every product, day and feature set (default Keras initialization, Adam, early stopping) on every 13th day of 2023 (28 days) and compares it with the naive benchmark and the `e6` reference. Run `backbone_epochs.py` first.

```bash
python Evaluation/backbone_epochs.py --zones DE2               # -> Results/backbone_epochs/
python Evaluation/retrain_baseline.py --zones DE2 --every 13   # -> Results/retrain_baseline/
python Evaluation/retrain_baseline.py --summary                # MAE relative to naive, days won, DM test
python Evaluation/retrain_baseline.py --timing                 # unloaded single-thread time per fit -> hours per zone and test year
```

The summary is also printed by `Evaluation/text_numbers.py`.

The map of the control areas and the timeline of the forecasting setup are drawn by hand (draw.io), not generated. All settings are in `config.py`.

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
