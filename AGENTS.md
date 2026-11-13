# Agent Instructions & Workspace Rules

> Keep text minimal, stick to the point, and avoid lengthy explanations.

## 1. Language & Literature
* Chat: German or English, matching the user.
* Code & Docs: Strictly English.
* Literature ground truth: Consult `old/LITERATURE_KNOWLEDGE_BASE.md` (local archive, git-ignored) before citing formulas or papers.
  * Marcjasz et al. (2018): Inverse-MAE window weighting ($w \propto 1/\text{MAE}$, Eq. 5).
  * Marcjasz et al. (2020): 50:50 arithmetic average with naive baseline ($\text{ens} = 0.5\hat{y} + 0.5y_{\text{naive}}$, Eq. 15).
  * Puć & Janczura (2026, IJF 42(3), bib key `Puc.2024`): Linear inverse-MAE rolling weighting ($p=1.0$, Eq. 18; Eq. 25 in arXiv v1). Cite equation numbers of the published version (`Contribution/Literature/pdf/Puc.2024.pdf`).

## 2. Formula Rendering (Chat Readability)
* No raw LaTeX math tags (`$...$` or `$$...$$`) in chat responses (IDE markdown renderer does not parse them).
* Render non-trivial equations as PNG via matplotlib (`dpi=250`, `facecolor='white'`, `pad_inches=0.15`, `bbox_inches='tight'`) in scratch/artifacts and embed via markdown image syntax.
* Short expressions can use standard fenced ````math ```` blocks.

## 3. Core Methodological Rules
* Grid: Continuous 15-minute intervals in pure UTC (`+00:00`). Never drop rows.
* Products and days: Local delivery time (`Europe/Berlin`). Products 02:00-02:45 are missing on the spring DST day and delivered twice on the autumn DST day (second delivery: slots 96-99 in result arrays, shape days x 100).
* Regional targets: `DE1_VWAP_0to30`, `DE2_VWAP_0to30`, `DE3_VWAP_0to30`, `DE4_VWAP_0to30`.
* Primary reference area: Prioritize DE2 (Amprion) for detailed model comparisons.
* Naive benchmark: `{zone}_VWAP_90to105` (last observed window at the 90-minute pre-delivery cutoff).
* National `VWAP_0to30`: Internal imputation fallback only. Never use as target, feature, or benchmark.
* Information set: forecast issued 60 min before delivery + 30 min publication buffer = continuous-trade cutoff 90 min before delivery.
* Neighbor contracts: Window ending $A$ minutes before delivery is safe if $A \ge 90 + 15 \cdot k$ for $k \in [-4, +2]$.
* Ex-post features: Realized values must be published by the forecast time (60 min before delivery); delay counts from the end of the quarter-hour. Balancing energy (netztransparenz, 15 min delay): lag 6 QH. ENTSO-E actuals (load, generation; 60 min delay): lag 9 QH. Cross-border scheduled exchanges (updates up to 2 h after each intraday nomination): lag 12 QH. Lags per column in `features.csv` (`ex_post_lag_qh`).
* No backward fill: `bfill` is prohibited.
* Feature catalog (`features.csv`): one row per master-dataset column (1,221).
  * `tag`: `input` (961), `excluded` (254, e.g. windows after the 90-min cutoff), `label` (5), `index` (1).
  * `window_end_min`: window end in minutes before delivery of the target contract (neighbors: own window end − 15k); every `input` must be ≥ 90.
  * Sets S1–S4 are selected from the catalog only; derived S2/S4 columns are built in `feature_engineering.py`, never in the loader.

## 4. TSO Delivery Area Mapping
Map German TSOs 1:1 to official EPEX delivery areas:
* `DE1_`: TransnetBW (`DE-ENBW`)
* `DE2_`: Amprion (`DE-AMP`)
* `DE3_`: TenneT (`DE-TPS`)
* `DE4_`: 50Hertz (`DE-50HZ`)
* `DE_`: National aggregate

Never use raw company names in code, features, or model configs.

## 5. Missing Data Imputation
* Regional prices: Own window first, then the preceding zonal window of the same product; if no zonal trade has occurred yet, the national `VWAP_{w}` of the same window.
* National prices: Window `345to360` falls back to Intraday Auction (or Day-Ahead Auction if missing); subsequent windows forward-fill horizontally; window `0to30` falls back to `VWAP_0to15` / `VWAP_15to30`.
* Volumes, trade counts, net flows: Fill missing values with `0.0`.

## 6. Backtesting Protocol
* Rolling window: Fixed 822 days (~2.25 years).
* No separate validation window. Rolling windows within the calibration window:
  * LASSO penalty: expanding-window CV over the last 14 calibration days (`LASSO_CV_DAYS`, as in Puć & Janczura, 2024).
  * Forecast averaging: per-product inverse-MAE weights over the 14 days before the test day (`ENSEMBLE_CALIB_DAYS`; Puć & Janczura use 7).
  * FA-NN/FOMAML: fixed hyperparameters, no rolling validation. Design decisions (e.g. layer normalization) use pre-test blocks of 2023 (`Evaluation/variant_pretest.py`).
* Test horizon: Calendar year 2024 in local time (35,136 quarter-hours, zero NaNs).
* Canonical 2024 Naive MAE baselines:
  * DE1: `26.2807 EUR/MWh`
  * DE2: `26.6803 EUR/MWh` (RMSE: `133.8570`, Median: `13.9600`)
  * DE3: `27.2706 EUR/MWh`
  * DE4: `27.2518 EUR/MWh`
* Competition policy: Evaluate every model on its own first (FA-NN vs cSVR 1:1). Cross-architecture ensembles are allowed only as a final step after the separate evaluation, never inside the experiment or model development.
