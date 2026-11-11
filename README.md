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

Python 3.11.15, CPU only, package versions pinned in `requirements.txt`:

```bash
conda create -n deep_learning python=3.11.15 && conda activate deep_learning
pip install -r requirements.txt
bash reproduce.sh            # all stages in order, about 12 h in total
bash reproduce.sh backtest   # or single stages
```

Raw data must be placed as described in `Data/*/README.md` before the `data` stage. Times measured on an AMD Ryzen 5 7600X (6 cores, 12 threads, 64 GB RAM):

| Stage | Content | Time |
| :--- | :--- | :--- |
| `data` | EPEX trades → zonal 15-min windows (`calculate_ID_DEx_UTC.py`), master dataset (`create_master_dataset.ipynb`), unit tests | 2–3 h |
| `backtest` | all models, four zones in parallel → `Results/backtest_2024/` | 3.7 h |
| `seeds` | four further seeds of FA-NN and FOMAML → `Results/backtest_2024/seed_<s>/` | 3 h |
| `retrain` | per-forecast retraining reference on 2023, DE2 (paper Sec. 5.5, see below) | 1.75 h |
| `runtime` | single-thread runtime benchmark (paper Table 3) | 10 min |
| `paper` | tables (`Contribution/tables/`), figures (`Contribution/SVG/`), numbers quoted in the text (printed) | minutes |

Memory: about 2.4 GB per worker; `backtest` uses 4 workers, `retrain` 8. `python tests/feature_snapshot.py` checks the data loader against reference hashes (create them once with `--update`).

### Retraining reference (paper, Sec. 5.5)

Two scripts compare the FA-NN with conventional per-forecast retraining on pre-test data of 2023 (DE2), so the test year is not touched:

- `Evaluation/backbone_epochs.py` trains one FA-NN backbone per 2023 quarter and feature set and evaluates snapshots after 6 to 200 epochs, with and without local adaptation. Snapshot `e6` is the backtest configuration and serves as the FA-NN reference (15 min).
- `Evaluation/retrain_baseline.py` trains a fresh network of the same architecture for every product, day and feature set (default Keras initialization, Adam, early stopping) on every 13th day of 2023 (28 days) and compares it with the naive benchmark and the `e6` reference (1.5 h). `--summary` prints MAE relative to naive and the DM test, `--timing` the unloaded single-thread time per fit. Run `backbone_epochs.py` first.

### Optional: design pre-tests (not reported in the paper)

```bash
Q2023=2023-01-01:2023-03-31,2023-04-01:2023-06-30,2023-07-01:2023-09-30,2023-10-01:2023-12-31
python Evaluation/variant_pretest.py --zones DE1,DE2,DE3,DE4 --periods $Q2023 current ln_pre             # layer normalization, FA-NN
python Evaluation/variant_pretest.py --mode fomaml --zones DE1,DE2,DE3,DE4 --periods $Q2023 current ln_pre   # same for FOMAML
python Evaluation/variant_pretest.py ln reg ln_reg                                                      # regularization, DE2 Q4 2023
python Evaluation/fomaml_diagnostics.py                                                                 # FOMAML diagnostics, Q4 2023
```

The map of the control areas and the timeline of the forecasting setup are drawn by hand (draw.io). All settings are in `config.py`.

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
reproduce.sh               all stages in order (see Reproduction)
Evaluation/                tables, figures, runtime benchmark, 2023 pre-tests
tests/                     consistency and regression tests
old/                       archived code, data and results
```
